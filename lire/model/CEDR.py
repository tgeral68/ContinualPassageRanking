import torch
import torch.nn

from lire.experiment_tools.experiment_template import DDPLIReExperiment
from lire.models import cedr
from lire.data_tools.dataset import scenarios as sc
from lire.data_tools.dataset import scenarios as sc
from lire.data_tools.dataset import MSMarco as ms


class MonoT5EvaluationDataset():
    def __init__(self, dataset,
                    positive_token='true',
                    negative_token='false'):
        self.dataset = dataset
        self.rerank_collection = None
        self.positive_token = positive_token
        self.negative_token = negative_token
    
    def __getitem__(self, index):
        return self.rerank_collection[index]

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

class DDPCEDRVanilla(DDPLIReExperiment):
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
                                master_port, gpu_shift, init_method, gpus, hyperparameters)
        self.loss_func = torch.nn.MarginRankingLoss(margin=1)
        self.dataset = None
        self.log_train_first = 0
        self.log_test_first = 0

    @staticmethod
    def padding_tensor(sequences):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        mask = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1
        return out_tensor, mask

    def init_optim(self):
        params = [(k, v) for k, v in self.model.named_parameters() if(v.requires_grad )]

        non_bert_params = {'params': [v for k, v in params if not k.startswith('module.bert.')]}

        bert_params = {'params': [v for k, v in params if k.startswith('module.bert.')], 'lr': 2e-5}
        # print('KEYS', [k for k, v in params])
        print("BERT PARAMS ", len(bert_params['params']))
        print("NB PARAMS", len(non_bert_params['params']))
        self.optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=1e-3)

    def init_model(self):
        # tokenizer + embedding

        self.model = cedr.VanillaBertRanker()
        self.tokenizer = self.model
        batch = {"qtxt":['tes t', 'te st'],"dtxt":['te st', 'test']}
        print("Some parameters in this model are unused with the current framework.\
             We proceed to a forward pass to check which are used")
        qids, qmask, dids, dmask = self.batch_transform(batch)
        scores = self.model(qids, qmask, dids, dmask)
        scores.sum().backward()

        initial_pg =  len([(k, v) for k, v in self.model.named_parameters() if(v.requires_grad)])
        no_grad_set =  {"bert.pooler.dense.weight","bert.pooler.dense.bias"}
        for k, v in self.model.named_parameters():
            if k in no_grad_set:
                print("set  no grad for defined ", k)
                v.requires_grad = False
            if v.grad is None:
                print("set  no grad for ", k)
                v.requires_grad = False
        for v in self.model.parameters():
            if v.grad is None:
                print("set  no grad for ", k)
                v.requires_grad = False
        batch = {"qtxt":['tes t', 'te st'],"dtxt":['te st', 'test']}
        print("Second pass to verify")
        qids, qmask, dids, dmask = self.batch_transform(batch)
        scores = self.model(qids, qmask, dids, dmask)
        scores.sum().backward()

        initial_pg =  len([(k, v) for k, v in self.model.named_parameters() if(v.requires_grad)])
        no_grad_set =  {"bert.pooler.dense.weight","bert.pooler.dense.bias"}
        for k, v in self.model.named_parameters():
            if k in no_grad_set:
                print("S2 set  no grad for defined ", k)
                v.requires_grad = False
            if v.grad is None:
                print("S2 set  no grad for ", k)
                v.requires_grad = False


    def build_dataset(self):
        dataset_types = {
            'scenario_informationUpdate': sc.MSMarcoInformationUpdateDataset,
            'scenario_languageDrift': sc.MSMarcoLanguageDriftDataset,
            'scenario_directTransfer': sc.MSMarcoDirectTransferDataset,
            'scenario_iudtbaseline':sc.MSMarcoIULDBaselineDataset,
            'topics': ms.MSMarcoRankingDataset,
            'all': ms.MSMarcoRankingDatasetAllTasks,
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
        self.train_set = self.dataset.clone()
        self.train_set.set_n_sample_negative(1)
        self.train_set.set_split('train')

    def load_validation_set(self):
        self.build_dataset()
        self.val_set = self.dataset.clone()
        self.val_set.set_n_sample_negative(1)
        self.val_set.set_uniform_negative_sampling(False)
        self.val_set.set_split('val')


    def load_testing_set(self):
        self.build_dataset()
        self.test_set = MonoT5EvaluationDataset(self.dataset.clone())
        self.test_set.dataset.set_n_sample_negative(1)
        self.test_set.dataset.set_split('dev')


    def batch_transform(self, batch):

        queries = [torch.LongTensor(self.tokenizer.tokenize(query)) for query in batch['qtxt']]
        if 'pdtxt' in batch:
            docs =\
                [torch.LongTensor(self.tokenizer.tokenize(d))[:512] for d in ([dp[0] for dp in batch['pdtxt']] + [dn[0] for dn in batch['ndtxt']])]
        else:
            docs = [torch.LongTensor(self.tokenizer.tokenize(d))[:512] for d in batch['dtxt']]
        return (*DDPCEDRVanilla.padding_tensor(queries), *DDPCEDRVanilla.padding_tensor(docs))

    def train_step(self, batch, batch_idx):
        qids, qmask, dids, dmask = self.batch_transform(batch)
        if self.log_train_first < 10:
            print("TEST:",qids.shape, qmask.shape, dids.shape, dmask.shape)
            self.log_train_first += 1
        qids = torch.cat((qids, qids), 0).to(self.device)
        qmask = torch.cat((qmask, qmask), 0).to(self.device)
        scores = self.model(qids, qmask, dids.to(self.device), dmask.to(self.device))

        # get relevant, irrelevant 
        scores = scores.view(len(qids), 1)
        scores = torch.cat((scores[:len(qids)//2], scores[len(qids)//2:]), -1)

        # compute loss
        loss = torch.mean(1. - scores.softmax(dim=1)[:, 0])
        return loss

    def rank_step(self, batch, batch_idx):
        qids, qmask, dids, dmask = self.batch_transform(batch)

        if self.log_test_first < 10:
            print("TEST:",qids.shape, qmask.shape, dids.shape, dmask.shape)
            self.log_test_first += 1
        score = self.model(qids.to(self.device), qmask.to(self.device), dids.to(self.device), dmask.to(self.device))
        return batch['qid'], batch['did'], score.squeeze()

    @staticmethod
    def test():
        hyperparameters = {
            "data_folder_path": "/net/cal/gerald/CPD/total_data",
            "topics_folder_path": "/net/cal/gerald/CPD/data/MSMarco-task-clustering-74-0.75-0.55/",
            # "data_folder_path": "/net/cal/gerald/CPD/small_data",
            # "topics_folder_path": "/net/cal/gerald/CPD/small_data",
            "rerank_path":"/net/cal/gerald/CPD/data/MSMarco-task-clustering-74-0.75-0.55/dev-bm25_top_1000.tsv",
            "patience":3,
            "ranking_examples":[1, 2],
            "log_folder": "/local/gerald/test",
            "task_perm_seed": -1,
            "lr":2e-5,
            "n_epoch": 10,
            "batch_size":8,
            "pretrained_name":"t5-small",
            "nb_init_task": 2,
            "dataset": "scenario_informationUpdate",
            "seed":42,
            "nb_evaluated_task": 2
        }
        hyperparameters['ranking_examples'] = [0, 1, 2]
        hyperparameters['log_folder'] = "/local/gerald/test"
        hyperparameters['task_perm_seed'] = -1
        hyperparameters['lr'] = 2e-5
        hyperparameters['n_epoch'] = 5
        hyperparameters['batch_size'] = 8
        hyperparameters['pretrained_name'] = 't5-small'

        experiment = DDPCEDRVanilla(n_gpus=1, 
                               n_nodes=1,
                               node=0, 
                               master_address="localhost", 
                               master_port="12346",
                               gpus=[0],
                               hyperparameters=hyperparameters)
        #experiment._run(0, experiment, DDPCEDRVanilla.train)
        experiment.run(DDPCEDRVanilla.train)

class DDPCEDRKNRM(DDPLIReExperiment):
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
                                master_port, gpu_shift, init_method, gpus, hyperparameters)
        self.loss_func = torch.nn.MarginRankingLoss(margin=1)
        self.dataset = None
        self.log_train_first = 0
        self.log_test_first = 0

    @staticmethod
    def padding_tensor(sequences):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        mask = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1
        return out_tensor, mask

    def init_optim(self):


        params = [(k, v) for k, v in self.model.named_parameters() if(v.requires_grad )]
        util_parameters  = [v for v in self.model.parameters() if(v.requires_grad )]
        non_bert_params = {'params': [v for k, v in params if not k.startswith('module.bert.')]}

        bert_params = {'params': [v for k, v in params if k.startswith('module.bert.')], 'lr': 2e-5}
        # print('KEYS', [k for k, v in params])
        print("BERT PARAMS ", len(bert_params['params']))
        print("NB PARAMS", len(non_bert_params['params']))
        print("NB UTILS PARAMETERS", len(util_parameters))


        self.optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=5e-4)

    def init_model(self):
        # tokenizer + embedding

        self.model = cedr.CedrKnrmRanker()
        self.tokenizer = self.model
        batch = {"qtxt":['tes t', 'te st'],"dtxt":['te st', 'test']}
        print("Some parameters in this model are unused with the current framework.\
             We proceed to a forward pass to check which are used")
        qids, qmask, dids, dmask = self.batch_transform(batch)
        scores = self.model(qids, qmask, dids, dmask)
        scores.sum().backward()

        initial_pg =  len([(k, v) for k, v in self.model.named_parameters() if(v.requires_grad)])
        no_grad_set =  {"bert.pooler.dense.weight","bert.pooler.dense.bias"}
        for k, v in self.model.named_parameters():
            if k in no_grad_set:
                print("set  no grad for defined ", k)
                v.requires_grad = False
            if v.grad is None:
                print("set  no grad for ", k)
                v.requires_grad = False
        for v in self.model.parameters():
            if v.grad is None:
                print("set  no grad for ", k)
                v.requires_grad = False
        batch = {"qtxt":['tes t', 'te st'],"dtxt":['te st', 'test']}
        print("Second pass to verify")
        qids, qmask, dids, dmask = self.batch_transform(batch)
        scores = self.model(qids, qmask, dids, dmask)
        scores.sum().backward()

        initial_pg =  len([(k, v) for k, v in self.model.named_parameters() if(v.requires_grad)])
        no_grad_set =  {"bert.pooler.dense.weight","bert.pooler.dense.bias"}
        for k, v in self.model.named_parameters():
            if k in no_grad_set:
                print("S2 set  no grad for defined ", k)
                v.requires_grad = False
            if v.grad is None:
                print("S2 set  no grad for ", k)
                v.requires_grad = False

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
                True, 
                rerank_path=self.hyperparameters["rerank_path"],
                seed=self.hyperparameters["seed"],
                nb_init_task=self.hyperparameters["nb_init_task"], 
                nb_evaluated_task=self.hyperparameters["nb_evaluated_task"])
            if hasattr(self.dataset, "scenario_tasks"):
                self.hyperparameters["grouped_scenario_task"] = self.dataset.scenario_tasks

    def load_training_set(self):
        self.build_dataset()
        self.train_set = self.dataset.clone()
        self.train_set.set_n_sample_negative(1)
        self.train_set.set_split('train')

    def load_validation_set(self):
        self.build_dataset()
        self.val_set = self.dataset.clone()
        self.val_set.set_n_sample_negative(1)
        self.val_set.set_uniform_negative_sampling(False)
        self.val_set.set_split('val')


    def load_testing_set(self):
        self.build_dataset()
        self.test_set = MonoT5EvaluationDataset(self.dataset.clone())
        self.test_set.dataset.set_n_sample_negative(1)
        self.test_set.dataset.set_split('dev')


    def batch_transform(self, batch):

        queries = [torch.LongTensor(self.tokenizer.tokenize(query)) for query in batch['qtxt']]
        if 'pdtxt' in batch:
            docs =\
                [torch.LongTensor(self.tokenizer.tokenize(d))[:512] for d in ([dp[0] for dp in batch['pdtxt']] + [dn[0] for dn in batch['ndtxt']])]
        else:
            docs = [torch.LongTensor(self.tokenizer.tokenize(d))[:512] for d in batch['dtxt']]
        return (*DDPCEDRVanilla.padding_tensor(queries), *DDPCEDRVanilla.padding_tensor(docs))

    def train_step(self, batch, batch_idx):
        qids, qmask, dids, dmask = self.batch_transform(batch)
        if self.log_train_first < 10:
            print("TEST:",qids.shape, qmask.shape, dids.shape, dmask.shape)
            self.log_train_first += 1
        qids = torch.cat((qids, qids), 0).to(self.device)
        qmask = torch.cat((qmask, qmask), 0).to(self.device)
        scores = self.model(qids, qmask, dids.to(self.device), dmask.to(self.device))

        # get relevant, irrelevant 
        scores = scores.view(len(qids), 1)
        scores = torch.cat((scores[:len(qids)//2], scores[len(qids)//2:]), -1)

        # compute loss
        loss = torch.mean(1. - scores.softmax(dim=1)[:, 0])
        return loss

    def rank_step(self, batch, batch_idx):
        qids, qmask, dids, dmask = self.batch_transform(batch)

        if self.log_test_first < 10:
            print("TEST:",qids.shape, qmask.shape, dids.shape, dmask.shape)
            self.log_test_first += 1
        score = self.model(qids.to(self.device), qmask.to(self.device), dids.to(self.device), dmask.to(self.device))
        return batch['qid'], batch['did'], score.squeeze()


    @staticmethod
    def test():
        hyperparameters = {
            "data_folder_path": "/net/cal/gerald/CPD/total_data",
            "topics_folder_path": "/net/cal/gerald/CPD/data/MSMarco-task-clustering-74-0.75-0.55/",
            # "data_folder_path": "/net/cal/gerald/CPD/small_data",
            # "topics_folder_path": "/net/cal/gerald/CPD/small_data",
            "rerank_path":"/net/cal/gerald/CPD/data/MSMarco-task-clustering-74-0.75-0.55/dev-bm25_top_1000.tsv",
            "patience":3,
            "ranking_examples":[1, 2],
            "log_folder": "/local/gerald/test",
            "task_perm_seed": -1,
            "lr":2e-5,
            "n_epoch": 10,
            "batch_size":8,
            "pretrained_name":"t5-small",
            "nb_init_task": 2,
            "dataset": "scenario_informationUpdate",
            "seed":42,
            "nb_evaluated_task": 2
        }
        hyperparameters['ranking_examples'] = [0, 1, 2]
        hyperparameters['log_folder'] = "/local/gerald/test"
        hyperparameters['task_perm_seed'] = -1
        hyperparameters['lr'] = 2e-5
        hyperparameters['n_epoch'] = 5
        hyperparameters['batch_size'] = 8
        hyperparameters['pretrained_name'] = 't5-small'

        experiment = DDPCEDRVanilla(n_gpus=1, 
                               n_nodes=1,
                               node=0, 
                               master_address="localhost", 
                               master_port="12346",
                               gpus=[0],
                               hyperparameters=hyperparameters)
        #experiment._run(0, experiment, DDPCEDRVanilla.train)
        experiment.run(DDPCEDRVanilla.train)

if __name__ == '__main__':
    import os
    os.environ['WANDB_SILENT'] = 'true'
    #os.environ['TOKENIZERS_PARALLELISM']='false'
    DDPCEDRVanilla.test()

