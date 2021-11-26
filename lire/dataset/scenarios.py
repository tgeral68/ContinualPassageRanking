import torch
import numpy as np
import json
import tqdm
import copy

from lire.dataset.continual_dataset import MSMarcoRankingDataset

from sentence_transformers import SentenceTransformer, util
from k_means_constrained import KMeansConstrained

class MSMarcoIULDBaselineDataset(MSMarcoRankingDataset):
    def __init__(self, data_folder_path, topics_folder_path,
                 load_triplets=True, rerank_path=None, seed=42, nb_init_task=5, nb_evaluated_task=2, switch=False):
        super().__init__(data_folder_path, topics_folder_path, 
                         load_triplets=load_triplets, rerank_path=rerank_path, seed=seed)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        task_ids = torch.LongTensor(super().get_task_ids())
        task_permutation = torch.randperm(len(task_ids))
        permuted_task_ids = task_ids[task_permutation]
        self.scenario_task = 0
        self.scenario_tasks =\
            [ permuted_task_ids[0:nb_init_task].tolist() ,
             permuted_task_ids[nb_init_task:nb_init_task+nb_evaluated_task].tolist()]
        self._build_scenario(self.scenario_tasks[1])

        self.set_task(0)
        if(load_triplets):
            self.set_negative_docs_return(True)

    def set_split(self, split_name):
        self.set_queries_index(self._get_queryids_task_split(self.working_task, split_name))
        self.working_split = split_name

    def set_task(self, task_id):
        task_id_list = self.scenario_tasks[task_id]
        self.working_task = task_id_list
        self.scenario_task = task_id
        self.set_split(self.working_split)

    def __getitem__(self, index):
        output_dict = super().__getitem__(index)
        return output_dict

    def get_task_ids(self):
        return [0, 1]
    
    def get_nb_tasks(self):
        return 2

    def _build_scenario(self, topics_scenario):
        pass

class MSMarcoInformationUpdateDataset(MSMarcoRankingDataset):
    def __init__(self, data_folder_path, topics_folder_path,
                 load_triplets=True, rerank_path=None, seed=42, nb_init_task=5, nb_evaluated_task=2, switch=False):
        super().__init__(data_folder_path, topics_folder_path, 
                         load_triplets=load_triplets, rerank_path=rerank_path, seed=seed)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        task_ids = torch.LongTensor(super().get_task_ids())
        task_permutation = torch.randperm(len(task_ids))
        permuted_task_ids = task_ids[task_permutation]
        self.scenario_task = 0
        self.scenario_tasks =\
            [ permuted_task_ids[0:nb_init_task].tolist() ,
             permuted_task_ids[nb_init_task:nb_init_task+nb_evaluated_task].tolist(),
             permuted_task_ids[nb_init_task:nb_init_task+nb_evaluated_task].tolist()]


        self._build_scenario(self.scenario_tasks[1], switch=switch)

        self.set_task(0)
        if(load_triplets):
            self.set_negative_docs_return(True)

    def _doc_embedding_kmeans(self, embedding_model, switch=False):
        qids, dids, dtxts = [], [], []
        for sample_idx in tqdm.trange(super().__len__()):
            sample = super().__getitem__(sample_idx)
            qids.append(sample["qid"])
            dids.append(sample["pdid"][0])
            dtxts.append(sample["pdtxt"][0])
        qids = torch.LongTensor(qids)
        dids = torch.LongTensor(dids)
        documents_embedding =\
            embedding_model.encode(dtxts, show_progress_bar=True, convert_to_numpy=True)
        length = np.sqrt((documents_embedding**2).sum(axis=1))[:,None]
        documents_embedding_cos = documents_embedding / length
        
        kmeans_documents = KMeansConstrained(size_min=len(documents_embedding_cos)//2.1,n_clusters=2, random_state=5).fit_predict(documents_embedding_cos)
        torch_embeddings = torch.Tensor(documents_embedding)
        cluster_1, cluster_2 = np.where(kmeans_documents == 0)[0], np.where(kmeans_documents == 1)[0]
        if switch:
            cluster_temp = cluster_1
            cluster_1 = cluster_2
            cluster_2 = cluster_temp
        qids_cluster_1, qids_cluster_2 = qids[cluster_1], qids[cluster_2]
        dids_cluster_1 = dids[cluster_1]

        query_dev = self._get_queryids_task_split(self.working_task, "dev")
        nb_cluster_1_dev = len(set(query_dev).intersection(set(qids_cluster_1.tolist())))
        nb_cluster_2_dev = len(set(query_dev).intersection(set(qids_cluster_2.tolist())))

        cluster_2_to_cluster_1 = []
        embedding_cluster_1, embedding_cluster_2 = torch_embeddings[cluster_1], torch_embeddings[cluster_2]
        for id_cluster_2 in range(len(cluster_2)):
            _, reference_to_cluster_1 =\
                torch.cosine_similarity(embedding_cluster_2[id_cluster_2].unsqueeze(0), embedding_cluster_1).max(-1)
            cluster_2_to_cluster_1.append(dids_cluster_1[reference_to_cluster_1].item())

        return cluster_2_to_cluster_1, qids_cluster_1.tolist(), qids_cluster_2.tolist()

    def _build_scenario(self, topics_scenario, switch=False):
        embedding_model = SentenceTransformer('stsb-roberta-large')
        self.topics_scenario = topics_scenario
        self.set_negative_docs_return(False)
        self.dids_task_1 = []
        self.qids_scenario = [[], [], []]
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        for topic_scenario in topics_scenario:
            super().set_task(topic_scenario)
            dids, qids_task_1, qids_task_2 = self._doc_embedding_kmeans(embedding_model, switch=switch)
            self.dids_task_1 += dids
            self.qids_scenario[1] += qids_task_1
            self.qids_scenario[2] += qids_task_2
            assert(len(self.qids_scenario[2]) == len(self.dids_task_1))
        self.reversed_cluster_1 =\
            {self.qids_scenario[2][i]: i
             for i in range(len(self.dids_task_1))}

    def clone(self):
        return copy.copy(self)

    def get_task_ids(self):
        return [0, 1, 2]
    
    def get_nb_tasks(self):
        return 3

    def set_split(self, split_name):
        self.set_queries_index(self._get_queryids_task_split(self.working_task, split_name))
        self.working_split = split_name
        if (split_name == "dev" and self.scenario_task != 0) or self.scenario_task == 2:
            filtering_query = set(self.qids_scenario[self.scenario_task])
            self.set_queries_index([qid for qid in self.queries_index if qid in filtering_query])

    def get_generated_documents_ids(self):
        res = {}
        for qid, did_remap_index in self.reversed_cluster_1.items():
            did = self.dids_task_1[did_remap_index]
            res[qid] = {did:1}
        return res



    def set_task(self, task_id):
        task_id_list = self.scenario_tasks[task_id]
        self.working_task = task_id_list
        self.scenario_task = task_id
        self.set_split(self.working_split)


    def __getitem__(self, index):
        output_dict = super().__getitem__(index)
        query_index = output_dict["qid"]
        if self.scenario_task == 1 and query_index in self.reversed_cluster_1:
            try:
                documents_index = self.dids_task_1[self.reversed_cluster_1[query_index]]
                if not isinstance(documents_index, list):
                    documents_index = [documents_index]
                documents_text = self.documents_collection.loc[documents_index][1].tolist()
            except KeyError as err:
                documents_index, documents_text = None, None
            
            output_dict["pdid"] = documents_index
            output_dict["pdtxt"] = documents_text

        return output_dict

class MSMarcoLanguageDriftDataset(MSMarcoInformationUpdateDataset):


    def _doc_embedding_kmeans(self, embedding_model, switch=False):
        qids, qtxt = [], []
        for sample_idx in tqdm.trange(MSMarcoRankingDataset.__len__(self)):
            sample = MSMarcoRankingDataset.__getitem__(self, sample_idx)
            qids.append(sample['qid'])
            qtxt.append(sample['qtxt'])

        qids = torch.LongTensor(qids)
        sts_embedding =\
            embedding_model.encode(qtxt, show_progress_bar=True, convert_to_numpy=True)
        
        # Using cosine similarity
        length = np.sqrt((sts_embedding**2).sum(axis=1))[:,None]
        sts_embedding_cos = sts_embedding / length
        pkmeans = KMeansConstrained(size_min=len(sts_embedding_cos)//2.1,n_clusters=2, random_state=5).fit_predict(sts_embedding_cos)
        torch_embeddings = torch.Tensor(sts_embedding)

        cluster_1, cluster_2 = np.where(pkmeans == 1)[0], np.where(pkmeans == 0)[0]
        if switch:
            cluster_temp = cluster_1
            cluster_1 = cluster_2
            cluster_2 = cluster_temp
        qids_cluster_1, qids_cluster_2 = qids[cluster_1], qids[cluster_2]
        query_dev = self._get_queryids_task_split(self.working_task, "dev")
        nb_cluster_1_dev = len(set(query_dev).intersection(set(qids_cluster_1.tolist())))
        nb_cluster_2_dev = len(set(query_dev).intersection(set(qids_cluster_2.tolist())))

        cluster_2_to_cluster_1 = []
        embedding_cluster_1, embedding_cluster_2 = torch_embeddings[cluster_1], torch_embeddings[cluster_2]
        for id_cluster_2 in range(len(cluster_2)):
            _, reference_to_cluster_1 =\
                torch.cosine_similarity(embedding_cluster_2[id_cluster_2].unsqueeze(0), embedding_cluster_1).max(-1)

            cluster_2_to_cluster_1.append(qids_cluster_1[reference_to_cluster_1].item())

        return cluster_2_to_cluster_1, qids_cluster_1.tolist(), qids_cluster_2.tolist()

    def _build_scenario(self, topics_scenario, switch=False):
        embedding_model = SentenceTransformer('stsb-roberta-large')
        self.topics_scenario = topics_scenario
        self.set_negative_docs_return(False)
        self.dids_task_1 = []
        self.qids_scenario = [[], [], []]
        self.qids_task_1 = []
        for topic_scenario in topics_scenario:
            MSMarcoRankingDataset.set_task(self, topic_scenario)
            qids, qids_task_1, qids_task_2 = self._doc_embedding_kmeans(embedding_model, switch=switch)
            self.qids_task_1 += qids
            self.qids_scenario[1] += qids_task_1
            self.qids_scenario[2] += qids_task_2
            assert(len(self.qids_scenario[2]) == len(self.qids_task_1))
        self.reversed_cluster_1 =\
            {self.qids_scenario[2][i]: i
                for i in range(len(self.qids_task_1))}

        

    def set_split(self, split_name):
        self.set_queries_index(self._get_queryids_task_split(self.working_task, split_name))
        self.working_split = split_name
        if (split_name == "dev" and self.scenario_task != 0) or self.scenario_task == 2:
            filtering_query = set(self.qids_scenario[self.scenario_task])
            queries_index = [qid for qid in self.queries_index if qid in filtering_query]
            self.set_queries_index(queries_index)

    def get_generated_documents_ids(self):
        rqdids = MSMarcoRankingDataset.get_relevant_documents_ids(self, queries_index=self.qids_scenario[2])
        res = {}
        for qid, index_map_qid in self.reversed_cluster_1.items():
            res[self.qids_task_1[index_map_qid]] = rqdids[qid]
        return res

    def __getitem__(self, index):
        output_dict = super(MSMarcoInformationUpdateDataset, self).__getitem__(index)
        query_index = output_dict['qid']
        if self.scenario_task == 1 and query_index in self.reversed_cluster_1:
            try:
                queries_index = self.qids_task_1[self.reversed_cluster_1[query_index]]
                if not isinstance(queries_index, list):
                    queries_index = [queries_index]
                queries_text = self.queries_collection.loc[queries_index][1].tolist()[0]
            except KeyError as err:
                queries_index, queries_text = "None", "None"
            
            output_dict["qid"] = queries_index
            output_dict["qtxt"] = queries_text
        return output_dict

class MSMarcoDirectTransferDataset(MSMarcoRankingDataset):
    def __init__(self, data_folder_path, topics_folder_path,
                 load_triplets=True, rerank_path=None, seed=42, nb_init_task=5, nb_evaluated_task=2, switch=False):

        super().__init__(data_folder_path, topics_folder_path, 
                         load_triplets=load_triplets, rerank_path=rerank_path, seed=seed)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        task_ids = torch.LongTensor(super().get_task_ids())
        task_permutation = torch.randperm(len(task_ids))
        permuted_task_ids = task_ids[task_permutation]
        self.scenario_task = 0
        self.scenario_tasks =\
            [ permuted_task_ids[0:nb_init_task].tolist() ,
             permuted_task_ids[nb_init_task:nb_init_task+nb_evaluated_task].tolist(),
             [permuted_task_ids[nb_init_task+nb_evaluated_task].item()],
             permuted_task_ids[nb_init_task:nb_init_task+nb_evaluated_task].tolist()]


        self._build_scenario(self.scenario_tasks[1])

        self.set_task(0)
        if(load_triplets):
            self.set_negative_docs_return(True)

    def _split_task(self, embedding_model):
        qids, dids, qtxt = [], [], []
        for sample_idx in tqdm.trange(super().__len__()):
            d = super().__getitem__(sample_idx)
            qids.append(d['qid'])
        qids = torch.LongTensor(qids)
        perm = torch.randperm(len(qids))
        qids_cluster_1 = qids[perm[:int(len(perm) * 0.75)]]
        qids_cluster_2 = qids[perm[int(len(perm) * 0.75):]]
        return qids_cluster_1.tolist(), qids_cluster_2.tolist()

    def _build_scenario(self, topics_scenario):
        embedding_model = SentenceTransformer('stsb-roberta-large')
        self.topics_scenario = topics_scenario
        self.set_negative_docs_return(False)
        self.dids_task_1 = []
        self.qids_scenario = [[], [], [], []]
        for topic_scenario in topics_scenario:
            super().set_task(topic_scenario)
            qids_task_1, qids_task_2 = self._split_task(embedding_model)
            print("TASK SPLIT SIZE",len(qids_task_1), len(qids_task_2))
            self.qids_scenario[1] += qids_task_1
            self.qids_scenario[3] += qids_task_2

    def set_split(self, split_name):
        queries_split = self._get_queryids_task_split(self.working_task, split_name)
        self.set_queries_index(queries_split)
        self.working_split = split_name
        if self.scenario_task == 1 or self.scenario_task == 3:
            filtering_query = set(self.qids_scenario[self.scenario_task])
            queries_index = [qid for qid in queries_split if qid in filtering_query]
            self.set_queries_index(queries_index)
            print(self.scenario_task, ",", split_name, "->", len(self))
        

    def get_task_ids(self):
        return [0, 1, 2, 3 ]
    
    def get_nb_tasks(self):
        return 4

    def set_task(self, task_id):
        task_id_list = self.scenario_tasks[task_id]
        self.working_task = task_id_list
        self.scenario_task = task_id
        self.set_split(self.working_split)