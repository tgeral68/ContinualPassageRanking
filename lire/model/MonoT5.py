

import torch
import torch.nn

from lire.experiment_tools.experiment_template import DDPLIReExperiment, dict_collate_fn
from lire.modules import word_embedding as we
from lire.models import knrm
from lire.data_tools.dataset import scenarios as sc
from lire.data_tools.dataset import MSMarco as ms


from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5RankerTrainDataset():
    def __init__(self, dataset,
                    positive_token='true',
                    negative_token='false'):
        self.dataset = dataset
        self.positive_token = positive_token
        self.negative_token = negative_token
    
    def __getitem__(self, index):
        query_id, document_id, query_document = self.dataset[index]
        query, document = query_document
        return (query_id, document_id, 'Query: ' + query +' Document: '+ document + ' Relevant:')

    def __len__(self):
        return len(self.dataset)

    def set_current_task_by_id(self, task_id):
        self.dataset.set_current_task_by_id(task_id)

    def set_task(self, task_id):
        self.dataset.set_task(task_id)

    def get_nb_tasks(self):
        return self.dataset.get_nb_tasks()

class MonoT5TrainingDataset():
    def __init__(self, dataset,
                    positive_token='true',
                    negative_token='false'):
        self.dataset = dataset

        self.positive_token = positive_token
        self.negative_token = negative_token
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["pinput"] = 'Query:' + sample['qtxt'] +' Document:'+ sample['pdtxt'][0] + ' Relevant:'
        sample["ninput"] = 'Query:' + sample['qtxt'] +' Document:'+ sample['ndtxt'][0] + ' Relevant:'
        sample["spdid"] = sample["pdid"][0]
        sample["npdid"] = sample["ndid"][0]
        return sample
    def __len__(self):
        return len(self.dataset)

    def set_current_task_by_id(self, task_id):
        self.dataset.set_current_task_by_id(task_id)

    def set_task(self, task_id):
        self.dataset.set_task(task_id)

    def get_nb_tasks(self):
        return self.dataset.get_nb_tasks()

class MonoT5EvaluationDataset():
    def __init__(self, dataset,
                    positive_token='true',
                    negative_token='false'):
        self.dataset = dataset
        self.rerank_collection = None
        self.positive_token = positive_token
        self.negative_token = negative_token
    
    def __getitem__(self, index):
        sample = self.rerank_collection[index]
        sample["input"] = 'Query:' + sample['qtxt'] +' Document:'+ sample['dtxt'] + ' Relevant:'
        return sample
    def __len__(self):
        return len(self.rerank_collection)

    def get_relevant_documents_ids(self):
        return self.dataset.get_relevant_documents_ids()

    def set_current_task_by_id(self, task_id, subrank=None):
        self.dataset.set_task(task_id)
        self.rerank_collection = self.dataset.get_rerank_collection(subrank=subrank)


    def set_task(self, task_id, subrank=None):
        self.dataset.set_task(task_id)
        self.rerank_collection = self.dataset.get_rerank_collection(subrank=subrank)

    def get_nb_tasks(self):
        return self.dataset.get_nb_tasks()



class DDPMonoT5(DDPLIReExperiment):
    def __init__(self,
                 n_gpus=1,
                 n_nodes=1,
                 node=0,
                 master_address='localhost',
                 master_port='8879', 
                 gpu_shift=0,
                 init_method='env://',
                 gpus=None,
                 hyperparameters=None,
                 logger=None):
        super().__init__(n_gpus, n_nodes, node, master_address,
                                master_port, gpu_shift, init_method, gpus,
                                hyperparameters)
        self.dataset = None
        self.plabels = None
        self.nlabels = None
        self.start_decode_id = None
        self.positive_token = 'true'
        self.negative_token = 'false'

    def build_dataset(self):
        dataset_types = {
            'scenario_informationUpdate': sc.MSMarcoInformationUpdateDataset,
            'scenario_languageDrift': sc.MSMarcoLanguageDriftDataset,
            'scenario_directTransfer': sc.MSMarcoDirectTransferDataset,
            'scenario_iudtbaseline':sc.MSMarcoIULDBaselineDataset,
            'topics': ms.MSMarcoRankingDataset,
            'all': ms.MSMarcoRankingDatasetAllTasks
            }
        if self.dataset is None:
            self.dataset = dataset_types[self.hyperparameters['dataset']](
                self.hyperparameters["data_folder_path"],
                self.hyperparameters["topics_folder_path"],
                load_triplets=True, 
                rerank_path=self.hyperparameters["rerank_path"],
                seed=self.hyperparameters["seed"],
                nb_init_task=self.hyperparameters["nb_init_task"], 
                nb_evaluated_task=self.hyperparameters["nb_evaluated_task"],
                switch=self.hyperparameters["switch"]
                )
            if hasattr(self.dataset, "scenario_tasks"):
                self.hyperparameters["grouped_scenario_task"] = self.dataset.scenario_tasks

    def load_training_set(self):
        self.build_dataset()
        self.train_set = MonoT5TrainingDataset(self.dataset.clone())
        self.train_set.dataset.set_n_sample_negative(1)
        self.train_set.dataset.set_split('train')

    def load_validation_set(self):
        self.build_dataset()
        self.val_set = MonoT5TrainingDataset(self.dataset.clone())
        self.val_set.dataset.set_n_sample_negative(1)
        self.val_set.dataset.set_uniform_negative_sampling(False)
        self.val_set.dataset.set_split('val')

    def load_testing_set(self):
        self.build_dataset()
        self.test_set = MonoT5EvaluationDataset(self.dataset.clone())
        self.test_set.dataset.set_n_sample_negative(1)
        self.test_set.dataset.set_split('dev')



    def init_optim(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.optimizer = optimizer

    def init_model(self):
        self.model = T5ForConditionalGeneration.from_pretrained(self.hyperparameters['pretrained_name'])
        self.tokenizer = T5Tokenizer.from_pretrained(self.hyperparameters['pretrained_name'])
        self.model_config = self.model.config
        print('Initial tokenizer size', len(self.tokenizer))
        self.tokenizer.add_tokens(['Query:', 'Document:', 'Relevant:'])
        torch.manual_seed(1)
        print('Final size', len(self.tokenizer))
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    def batch_transform(self, batch):
        
        input_tokens_index =\
            self.tokenizer(batch["pinput"] + batch["ninput"], return_tensors="pt",
                           max_length=512, padding=True, truncation=True)
        input_ids = input_tokens_index.input_ids
        attention_mask = input_tokens_index.attention_mask

        if self.plabels is None:
            self.plabels =\
                self.tokenizer(self.positive_token, return_tensors="pt").input_ids.squeeze().unsqueeze(0) 
        if self.nlabels is None:
            self.nlabels =\
                self.tokenizer(self.negative_token, return_tensors="pt").input_ids.squeeze().unsqueeze(0)   

        labels =\
            torch.cat((self.plabels.expand(len(batch["pinput"]), self.plabels.shape[-1]),
                      self.nlabels.expand(len(batch["ninput"]), self.nlabels.shape[-1])), 0)
        return {"input_ids":input_ids.to(self.device), 'attention_mask':attention_mask.to(self.device),
                 'labels':labels.to(self.device)}

    def evaluation_batch_transform(self, batch):
        input_tokens_index =\
            self.tokenizer(batch["input"], return_tensors="pt",
                            padding=True, truncation=True, max_length=512)
        input_ids = input_tokens_index.input_ids
        attention_mask = input_tokens_index.attention_mask

        return {"input_ids":input_ids.to(self.device), 'attention_mask':attention_mask.to(self.device)}       

    def train_step(self, batch, batch_idx):
        mt5_input = self.batch_transform(batch)
        outputs_prediction = self.model(**mt5_input)
        loss = outputs_prediction.loss
        return loss

    def rank_step(self, batch, batch_idx):
        mt5_input = self.evaluation_batch_transform(batch)
        if(self.start_decode_id is None):
            # retrieve the start decoder token (often 0)
            # retrieve the start decoder token (often 0)
            self.start_decode_id = torch.rand(1,1).to(self.device).long()
            self.start_decode_id[0,0] = self.model_config.decoder_start_token_id
            
            self.positive_token_id =\
                self.tokenizer(self.positive_token)
            self.positive_token_id = self.positive_token_id.input_ids[0]
            self.negative_token_id =\
                self.tokenizer(self.negative_token)
            self.negative_token_id  = self.negative_token_id.input_ids[0]


        # apply one decoding step (see hugging face transformer forward)
        output = self.model(**{**mt5_input, 
                            'decoder_input_ids':self.start_decode_id.expand(len(batch['qid']), 1)})
        # get logits on tokens
        out_logits = output['logits']
        # compute log_softmax on [positive_token, negative_token] for each batch elements
        pn_score = out_logits[:, -1, [self.positive_token_id, self.negative_token_id]]

        n_log_prob = torch.nn.functional.log_softmax(pn_score, dim=1)
        # get and return the score for positive token
        return batch['qid'], batch['did'], torch.exp(n_log_prob[:,0]).squeeze()