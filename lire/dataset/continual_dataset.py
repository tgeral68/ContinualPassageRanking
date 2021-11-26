import os
from os.path import join, basename, exists
from os import makedirs, rename

import json

import torch
import numpy as np
import pandas as pd


import ir_datasets

class MSMarcoBaseDataset():
    def __init__(self, load_triplets=False,
                rerank_path=None, seed=42, subrank=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.load_triplets = load_triplets
        dataset = ir_datasets.load('msmarco-passage/train/judged')
        # load the tsv files
        print("Loading qrels")
        self.qrels_collection =\
            pd.DataFrame.from_records(dataset.qrels_iter(),
                                        columns=['query_id', 'doc_id', 'relevance','iteration'],
                                        index='query_id')
        print("Loading documents")
        self.documents_collection =\
            pd.DataFrame.from_records(dataset.docs_iter(),
                                        columns=['doc_id', 'text'],
                                        index='doc_id')
        print("Loading queries")
        self.queries_collection =\
            pd.DataFrame.from_records(dataset.queries_iter(),
                                        columns=['query_id', 'text'],
                                        index='query_id')
        self.return_negative_docs = False
        self.triplets_collection = None
        if self.load_triplets:
            self.return_negative_docs = True
            self.triplets_collection_raw =\
                pd.DataFrame.from_records(dataset.docpairs_iter(), 
                                            columns=['query_id', 'relevant_doc', 'irrelevant_doc'])
            pd.read_csv(self.triplets_path, sep='\t', header=None)
            self.triplets_collection = self.triplets_collection_raw.groupby('query_id')
            self.group_keys = set(self.triplets_collection.groups.keys())

        self.queries_index = self.queries_collection.index
        self.n_sample_negative = None
        self.uniform_negative_sampling = True
        if rerank_path is not None:
            self.rerank_collection = pd.read_csv(rerank_path, sep='\t', header=None)
            if subrank is not None:
                self.rerank_collection = self.rerank_collection.loc[self.rerank_collection.groupby([0])[2].nlargest(subrank).reset_index["level_1"]]
            self.rerank_groups = self.rerank_collection.groupby(0)

    def get_rerank_group_by_query_id(self, query_id):
        query_text = self.queries_collection.loc[query_index]['text']    
        documents_index = self.rerank_groups.get_group(query_id)[1].tolist()

        if not isinstance(documents_index, list):
            documents_index = [documents_index]
        documents_text = self.documents_collection.loc[documents_index]['text'].tolist()

        return {'qid':query_id, 'qtxt': query_text, 'pdid': documents_index, 'pdtxt': document_text}

    def filter_rerank_group(self, queries_index):
        return self.rerank_collection[self.rerank_collection[0].isin(queries_index)]


    def get_relevant_documents_ids(self, queries_index=None):
        queries_index = self.queries_index if(queries_index is None) else queries_index
        # to faster collection of data we temporary remove ngeative docs
        temp_return_negative_docs = self.return_negative_docs
        self.return_negative_docs = False 
        relevant_documents = {}
        for qid in queries_index:
            sample = self.get_data_by_query_id(qid)
            if sample['pdid'] is not None:
                relevant_documents[qid] = {}
                for did in  sample['pdid']:
                    relevant_documents[qid][did] = 1
        self.return_negative_docs = temp_return_negative_docs
        return relevant_documents

    def get_rerank_collection(self, queries_index=None, subrank=None):
        rerank_df =\
            self.filter_rerank_group(self.queries_index) if(queries_index is None)\
                else self.filter_rerank_group(queries_index)
        if subrank is not None:
            rerank_df = rerank_df.loc[rerank_df.groupby([0])[2].nlargest(subrank).reset_index()["level_1"]]

        
        class RerankCorpus():
            def __init__(self, queries_collection, documents_collection, rerank_collection):
                self.queries_collection = queries_collection
                self.documents_collection = documents_collection
                self.rerank_collection = rerank_collection

            def __len__(self):
                return len(self.rerank_collection)

            def __getitem__(self, index):
                rerank_serie = self.rerank_collection.iloc[index]
                init_score = rerank_serie[2]
                query_index, document_index = rerank_serie[0], rerank_serie[1]
                query_txt  = self.queries_collection.loc[query_index]["text"]
                document_txt = self.documents_collection.loc[document_index]["text"]

                return {
                    'qid':int(query_index),
                    'qtxt':query_txt,
                    'did':int(document_index),
                    'dtxt':document_txt,
                    'score': init_score
                }
        return RerankCorpus(self.queries_collection, self.documents_collection, rerank_df)

    def get_query(self, query_index):
        return self.queries_collection.loc[query_index]["text"]
    
    def get_document(self, document_index):
        return self.documents_collection.loc[document_index]["text"]

    def get_data_by_query_id(self, query_index):
        query_text = self.queries_collection.loc[query_index]["text"]
        try:
            documents_index = self.qrels_collection.loc[query_index]["doc_id"]

            if isinstance(documents_index, str):
                documents_index = [documents_index]
            else: 
                documents_index = documents_index.tolist()
            documents_text = self.documents_collection.loc[documents_index]["text"].tolist()
        except KeyError as err:
            documents_index, documents_text = None, None
        output_dict = {"qid":query_index, "qtxt":query_text, "pdid":documents_index, "pdtxt":documents_text}
        if self.return_negative_docs and self.load_triplets:
            try:
                negative_documents_index = self.triplets_collection.get_group(query_index)["irrelevant_doc"]

                if isinstance(documents_index, str):
                    negative_documents_index = [negative_documents_index]
                else: 
                    negative_documents_index = negative_documents_index.tolist()

                if self.n_sample_negative is not None:
                    rp = torch.randperm(len(negative_documents_index))
                    if self.uniform_negative_sampling:
                        negative_documents_index =\
                            torch.LongTensor(negative_documents_index)[rp][:self.n_sample_negative].tolist()
                    else:
                        negative_documents_index =\
                            torch.LongTensor(negative_documents_index)[:self.n_sample_negative].tolist()            

                negative_documents_text = self.documents_collection.loc[negative_documents_index]["text"].tolist()
            except KeyError as err:
                negative_documents_index, negative_documents_text = None, None
            output_dict["ndid"] = negative_documents_index
            output_dict["ndtxt"] = negative_documents_text
        return output_dict

    def set_uniform_negative_sampling(self, boolean_value):
        self.uniform_negative_sampling = boolean_value

    def __getitem__(self, index):
        query_index = self.queries_index[index]
        return self.get_data_by_query_id(query_index)

    def set_n_sample_negative(self, max_negative):
        self.n_sample_negative = max_negative

    def set_negative_docs_return(self, return_negative_docs):
        if self.return_negative_docs and not self.load_triplets:
            print('Provide triplet file when init the dataset')
            return
        self.return_negative_docs = return_negative_docs

    def set_queries_index(self, queries_ids):
        self.queries_index = queries_ids
        if self.triplets_collection is not None:
            self.queries_index = [query_id for query_id in queries_ids if query_id in self.group_keys]


    def get_queries_index(self):
        return copy.deepcopy(self.queries_index)

    def __len__(self):
        return len(self.queries_index)

    def clone(self):
        return copy.copy(self)


class MSMarcoRankingDataset(MSMarcoBaseDataset):
    def __init__(self, topics_folder_path, 
                 load_triplets=False, seed=42, 
                 rerank_path=None, **args):
        super().__init__(load_triplets, rerank_path=rerank_path, seed=seed)
        self.data = {}
        for filename in ("train", "val", "test"):
            with open(os.path.join(topics_folder_path, 'topics.'+filename+'.json'), 'r') as topic_file:
                self.data[filename] = [[str(query) for query in topic] for topic in json.load(topic_file)]

        test_assert = [{q for t in v for q in t} for k, v in self.data.items()]
        assert(len(test_assert[0].intersection(test_assert[1])) == 0)
        assert(len(test_assert[0].intersection(test_assert[2])) == 0)
        assert(len(test_assert[1].intersection(test_assert[2])) == 0)
        
        self.working_task = "all"
        self.working_split = "all"
        self.set_queries_index(self._get_queryids_task_split("all", "all"))
        self.query2task_collection = {q:i for k, v in self.data.items() 
                                      for i,t in enumerate(v) 
                                      for q in t }

        self.seed = seed
        MSMarcoRankingDataset.set_split(self, 'all')


    def clone(self):
        return copy.copy(self)

    def _get_queryids_task_split(self, task_id, split_name):
        if isinstance(split_name, str):
            if split_name == 'all':
                split_name_list = list(self.data.keys())
            else:
                split_name_list = [split_name]
        else: 
            split_name_list = split_name
        task_query = []
        for split_key in split_name_list:
            for task_key, queries_id in enumerate(self.data[split_key]):
                if len(task_query) <= task_key:
                    task_query.append([])
                task_query[task_key] += queries_id

        if task_id == 'all':
            task_id_list = list(range(len(task_query)))
        elif isinstance(task_id, int):
            task_id_list = [task_id]
        else: 
            task_id_list = task_id

        return sum([task_query[i] for i in task_id_list],[])

    def set_split(self, split_name):
        self.set_queries_index(self._get_queryids_task_split(self.working_task, split_name))
        self.working_split = split_name
        
    def set_task(self, task_id):
        self.set_queries_index(self._get_queryids_task_split(task_id, self.working_split))
        self.working_task = task_id

    def get_task_ids(self):
        return [i for i in range(len(list(self.data.values())[0]))]

    def get_nb_tasks(self):
        return len(list(self.data.values())[0])

    def __getitem__(self, index):
        output_dict = super().__getitem__(index)
        output_dict["tid"] = self.query2task_collection[output_dict["qid"]]
        return output_dict
    
    # deprecated
    def set_current_task_by_id(self, task_id):
        self.set_task(task_id)


class MSMarcoRankingDatasetAllTasks(MSMarcoRankingDataset):
    def __init__(self,  topics_folder_path,
                 load_triplets=False, seed=42, 
                 rerank_path=None, **args):
        super().__init__( topics_folder_path, load_triplets=load_triplets,  seed=seed, rerank_path=rerank_path)

    def get_task_ids(self):
        return [0]

    def get_nb_tasks(self):
        return 1

    def set_task(self, task_id):
        self.set_queries_index(self._get_queryids_task_split('all', self.working_split))
        self.working_task = 'all'