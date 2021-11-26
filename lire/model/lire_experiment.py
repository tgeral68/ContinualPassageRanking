'''An experiment template for lire for mono-node multi-node training.
'''
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
import argparse
import os
import tqdm
from lire.data_tools.dataset import NamedDataset
import time
import datetime
import copy
import wandb
import json
import numpy as np
import sys
import pytrec_eval
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard.writer import SummaryWriter
def dict_collate_fn(batch):
    dct = {}
    for dictionary_sample in batch:
        for key, value in dictionary_sample.items():
            if key not in dct:
                dct[key] = []
            dct[key].append(value)
    return dct
def sum_by_key(values):
    sum_on_key = {}
    for value in values:
        for key, score in value.items():
            if(key not in sum_on_key):
                sum_on_key[key] = 0
            sum_on_key[key] += score/len(values)
    return sum_on_key
def evaluation(prediction, ground_truth, metrics={'map', 'ndcg', 'recip_rank'}):
    # ensure same queries are availabel in both prediction and ground truth
    gt_final = {}
    pd_final = {}
    for query in prediction:
        if(query not in ground_truth):
            print("The query ", query, " is not available in ground truth set")
        else:
            gt_final[query] = ground_truth[query]
            pd_final[query] = prediction[query]
    print('evaluate ', len(pd_final), ' queries')
    pytrec_evaluator = pytrec_eval.RelevanceEvaluator(gt_final, metrics)
    scores = pytrec_evaluator.evaluate(pd_final)
    mean_score = sum_by_key(scores.values())
    return scores, mean_score

def subrank(pred, at):
    new_pred = {}
    for query, document in  pred.items():
        tuple_d_v = [(k,v) for k,v in document.items()]
        d = [k for k,v in document.items()]
        s = [v for k,v in document.items()]
        index_at_score = (-np.array(s)).argsort()[:min(at, len(s))]
        new_pred[query] = {d[i]:s[i] for i in index_at_score}
    return new_pred

def evaluate(prediction, ground_truth):
    ground_truth = { str(q_id):{str(d_id): v for d_id, v in ground_truth[q_id].items() }
                    for q_id in ground_truth }
    prediction = { str(q_id):{str(d_id): v for d_id, v in prediction[q_id].items() }
                    for q_id in prediction}
    _, mean_score = evaluation(subrank(prediction, 10), ground_truth)
    return mean_score

class LIReExperiment(object):
    def __init__(self, options, dataset):
        self.dataset = dataset
        self.options = options
        self.current_state = {}

    @property
    def state_dict(self):
        ''' Get metadata from experiments.
        get metadata of experiments object
        with important informations to continue
        experiments at this step. It also recursivelly call
        the state_dict property of contained object.
        '''
        state_dict = {}
        for k, v in self.__dict__.items():
            if(hasattr(v, "load_state_dict") and hasattr(v, 'state_dict')):
                state_dict[k] = v.state_dict()
            elif(hasattr(v, 'state_dict')):
                state_dict[k] = v.state_dict
        state_dict["options"] = self.options
        state_dict["experiment_state"] = self.current_state
        return state_dict

    @state_dict.setter
    def state_dict(self, value):
        ''' set metadata from experiments.
        set metadata of experiments object.
        Allow to load an experiment checkpoint
        '''
        for k, v in self.__dict__.items():
            if hasattr(v, "load_state_dict"):
                v.load_state_dict(value[k])
            elif hasattr(v, 'state_dict'):
                v.state_dict = value[k]

        self.options = value["options"]
        self.current_state = value["experiment_state"]

    def save_experiment(self, filepath):
        torch.save(self.state_dict, filepath)
    
    def __getitem__(self, key):
        return self.options.__dict__[key]
    
    def train(self):
        raise NotImplementedError
    
    def ending_task(self):
        raise NotImplementedError

    def begin_task(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

class DDPExperiment(object):
    def __init__(self, n_gpus=1, n_nodes=1, node=0,
                master_address='localhost', master_port='8879', 
                gpu_shift=0, init_method='env://', gpus=None):
        self.n_gpus = n_gpus
        self.n_nodes = n_nodes
        self.node = node
        self.world_size = n_gpus * n_nodes
        self.master_process = False
        self.log_data = {}
        self.gpu_shift = gpu_shift
        self.master_address = master_address
        self.master_port = master_port
        self.citer = 0
        self.init_method = init_method
        self.print_redirection = None
        if gpus is not None:
            self.gpus = gpus
        else:
            self.gpus = [i+self.gpu_shift for i in range(n_gpus)]
        os.environ['MASTER_ADDR'] = master_address
        os.environ['MASTER_PORT'] = master_port
        print('DDPLIReExperiment using master node ', os.environ['MASTER_ADDR'], 'on', os.environ['MASTER_PORT'])

    def init_model(self, args):
        pass        
    def load_training_set(self, args):
        pass
    def load_validation_set(self, args):
        pass
    def load_testing_set(self, args):
        pass
    def train(self):
        pass
    def is_master(self):
        return self.master_process
    
    def log(self, key, value):
        if key not in self.log_data:
            self.log_data[key] = []
        self.log_data[key].append([None for i in range(self.world_size)])
        dist.all_gather_object(self.log_data[key][-1], value)
        dist.barrier()

    def set_print_redirection(self, path):
        self.print_redirection = path

    def print(self, *args):
        if self.is_master():
            if self.print_redirection is not None:
                with open(self.print_redirection, mode='a') as file_object:
                    print(*args, file=file_object)
            else:
                print(*args)

    def print_log(self, key):
        self.print(self.log_data[key])

    def _multiprocessing_start_routine(self, local_rank):
        # rank is also the gpu used on the local machine
        print('GPU used', self.gpus)
        self.device = self.gpus[local_rank]
        print('DEVICE used', self.device)
        self.local_rank = local_rank
        self.rank = self.node * self.n_gpus + self.local_rank
        print(torch.cuda.device_count())
        # Setting the cuda device using naccl backend is mandatory
        torch.cuda.set_device(self.device)

        print('Init process of rank ', self.rank, "on", self.world_size)
        if self.world_size > 1:
            dist.init_process_group(                  
                backend='nccl',                                  
                world_size=self.world_size,                              
                rank=self.rank
                        
            )
   
    def _multiprocessing_stop_routine(self):
        dist.destroy_process_group()

    @staticmethod
    def _run(rank, xp, main_func):
        print("start routine rank ", rank)
        xp._multiprocessing_start_routine(rank)
        print('rank, node:',rank, xp.node)
        if(rank == 0 and xp.node == 0):
            print('It is the master')
            xp.master_process = True
        print("Start main func")
        print("ALL IS OK (printing on stderr due to bufferization of stdout)", file=sys.stderr)
        main_func(xp)
        xp._multiprocessing_stop_routine()

    def run(self, main_func=None):
        print('start launching')
        if main_func is None:
            main_func = type(self).fit
        if self.world_size == 1:
            type(self)._run(0, self, main_func)
        else:
            mp.spawn(type(self)._run, args=(self, main_func), nprocs=self.n_gpus, join=True)

    @staticmethod
    def test_ddp(args):
        gpus = None
        if args.gpus is not None:
            gpus = [int(gpu) for gpu in gpus.split(',')]
        else:
            gpus = [i for i in range(args.n_gpus)]

        ddp_test = DDPTEST(args.n_gpus, args.n_nodes, args.node, args.master_addr, args.master_port, 
                          args.gpu_shift, args.init_method, gpus)
        ddp_test.run()

    def gather(self, tensor):
        '''
            Gather tensor from all process to the master one.

            Use all_gather instead as not supported by nccl, which is clearly
            unefficient in term of bandwidth. It will be updated when the backend 
            will supports it. Notice that gpu tensor is mandatory in case of nccl 
            backend.
            Morever notice that only the master process will get the real data
            for other nones is returned.
        '''
        gathered_list = [copy.deepcopy(tensor) for i in range(self.world_size)]

        dist.all_gather(gathered_list, tensor)
        if self.is_master():
            return torch.cat(gathered_list, 0)
        else:
            return None

    def gather_safe(self, tensor):
        ''' similar to gather additionally ensure tensor have the same first shape.
        '''
        if self.world_size <= 1:
            return tensor
        tensor_shape = torch.LongTensor([tensor.shape[0]]).to(self.device)
        gathered_list = [copy.deepcopy(tensor_shape) for i in range(self.world_size)]
        dist.all_gather(gathered_list, tensor_shape)
        maximum_shape = 0
        for ts in gathered_list:
            if ts.item() > maximum_shape:
                maximum_shape = ts.item()
        if tensor.shape[0] < maximum_shape:
            add_zero = x.new(maximum_shape - tensor.shape[0], *tensor.shape[1:])
            gathered_tensor = torch.cat((tensor, add_zero))
        else:
            gathered_tensor = tensor
        gathered_list_final = [copy.deepcopy(gathered_tensor) for i in range(self.world_size)]
        dist.all_gather(gathered_list_final, gathered_tensor)

        if self.is_master():
            return torch.cat([gathered_list_final[i][:gathered_list[i]] 
                              for i in range(len(gathered_list_final))]
                             , 0)

    def log(self, data, path='~/.runs'):
        if self.is_master():
            if not hasattr(self, 'is_log_init'):
                self.writer = SummaryWriter(path)
                self.is_log_init = True
            for k, v in data.items():
                self.writer.add_scalar(k, v, self.citer)



class DDPLIReExperiment(DDPExperiment):
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
                 logger=None, 
                 print_redirection=None):
        super().__init__(n_gpus, n_nodes, node, master_address,
                                master_port, gpu_shift, init_method, gpus)
        
        ## def init word embedding
        self.hyperparameters = hyperparameters
        self.xp_name = self.__class__.__name__
        print('checking data_folder')
        if 'rerank_path' in self.hyperparameters:
            self.hyperparameters['rerank_path'] =\
                os.path.expanduser(os.path.expandvars(self.hyperparameters['rerank_path']))
        if 'data_folder_path' in self.hyperparameters:
            self.hyperparameters['data_folder_path'] =\
                os.path.expanduser(os.path.expandvars(self.hyperparameters['data_folder_path']))
        if 'topics_folder_path' in self.hyperparameters:
            self.hyperparameters['topics_folder_path'] =\
                os.path.expanduser(os.path.expandvars(self.hyperparameters['topics_folder_path']))
            print(self.hyperparameters['topics_folder_path'])
        if 'log_folder_run' not in self.hyperparameters:
            assert('log_folder' in self.hyperparameters)

            lf = self.hyperparameters['log_folder']
            # make def instead of running this code here
            lf_exp = os.path.expandvars(os.path.expanduser(lf))
            lf_exp = os.path.join(lf_exp, self.xp_name+"-"+self.hyperparameters["dataset"])
            lf_exp += "-switch" if(self.hyperparameters["switch"]) else ""
            if not os.path.exists(lf_exp):
                os.makedirs(lf_exp)
            lf_dir = set(os.listdir(lf_exp))
            lfr_exist = True
            lfr_cpt = 0
            lfr_name = 'run_0'
            while(lfr_exist):
                lfr_name = 'run_'+str(lfr_cpt)
                if os.path.exists(os.path.join(lf_exp, lfr_name)):
                    lfr_cpt += 1
                else:
                    lfr_exist = False
            self.experiment_name  = lfr_name
            self.hyperparameters['log_folder_run'] = os.path.join(lf_exp, self.experiment_name)
            os.makedirs(self.hyperparameters['log_folder_run'])
            print('Logs will be find at ', self.hyperparameters['log_folder_run'])
            if print_redirection is None:
                self.set_print_redirection(os.path.join(self.hyperparameters['log_folder_run'], 'log.txt'))

    def init_model(self):
        pass

    def save_checkpoint(self, name='default'):
        if self.is_master():
            checkpoint_path = os.path.join(self.hyperparameters['log_folder_run'], name+'-model.pth')
            model_state_dict = self.model.state_dict()
            torch.save(model_state_dict, checkpoint_path)
            optim_state_dict = self.optimizer.state_dict()
            checkpoint_path = os.path.join(self.hyperparameters['log_folder_run'], name+'-optim.pth')
            torch.save(optim_state_dict, checkpoint_path)
        dist.barrier()

    def load_checkpoint(self, name='default'):
        dist.barrier()
        checkpoint_path = os.path.join(self.hyperparameters['log_folder_run'], name+'-model.pth')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:'+ str(self.device)))
        checkpoint_path = os.path.join(self.hyperparameters['log_folder_run'], name+'-optim.pth')
        self.optimizer.load_state_dict(torch.load(checkpoint_path))

        

    def load_training_set(self):
        # training set
        self.train_set = NamedDataset[self.hyperparameters.dataset['type']](self.hyperparameters['data_folder'],
                                        set_name='train',
                                        mode='training',
                                        ressource_file=os.path.join('ressource',
                                                                    self.hyperparameters.dataset['ressource']+'.json'))
    def load_validation_set(self):
        self.val_set = NamedDataset[self.hyperparameters.dataset['type']](self.hyperparameters['data_folder'],
                                        set_name='val',
                                        mode='training',
                                        ressource_file=os.path.join('ressource',
                                                                    self.hyperparameters.dataset['ressource']+'.json'))
    def load_testing_set(self):
        self.test_set = NamedDataset[self.hyperparameters.dataset['type']](self.hyperparameters['data_folder'],
                                        set_name='dev',
                                        mode='test',
                                        ressource_file=os.path.join('ressource',
                                                                    self.hyperparameters.dataset['ressource']+'.json'))

    def init_log_folder(self):
        pass


    def init_optim(self):
        pass

    def init_finetune(self):
        self.init_model()
        self.model.to(self.device)
        self.initial_model = self.model
        self.model = DDP(self.model, device_ids=[self.device])
        self.init_optim()

        if 'checkpoint' in self.hyperparameters:
            dist.barrier()
            self.model.load_state_dict(torch.load(self.hyperparameters['checkpoint-model'], 
                                              map_location='cuda:' + str(self.device))
                                             )
            # init optim here to refer optimizer to current model weights
            self.optimizer.load_state_dict(torch.load(checkpoint_data['optim']))
        self.loss_func = torch.nn.MarginRankingLoss(margin=1)

    def init_task_order(self):
        self.nb_tasks = self.train_set.get_nb_tasks()

        if self.hyperparameters['task_perm_seed'] >= 0 :
            torch.manual_seed(self.hyperparameters['task_perm_seed'])
            self.task_perm = torch.randperm(self.nb_tasks)

        else: 
            self.task_perm = torch.arange(self.nb_tasks)
        self.hyperparameters['task_perm_list'] = self.task_perm.tolist()

    def batch_transform(self, batch):
        pass

    def train_step(self, batch, batch_idx):
        pass


    def validate(self, task_id, metric="recip_rank"):
        with torch.no_grad():
            prediction = self.rank_task(task_id, self.val_set, subrank=self.hyperparameters['subrank-val'])
            ground_truth = self.val_set.get_relevant_documents_ids()
            mean_score = evaluate(prediction, ground_truth)
            print('Mean Score :', mean_score)
        return {"v-score": -mean_score[metric]}

    def init_train(self):
        self.print("Initialization: please wait until training start")
        self.init_finetune()
        self.load_training_set()
        self.load_validation_set()
        self.init_task_order()
        self.max_patience = self.hyperparameters['patience']
        self.default_gt = torch.ones(1).to(self.device)
        if 'ranking_examples' in self.hyperparameters:
            self.print("Load testing set to compute performances for:", self.hyperparameters["ranking_examples"])
            self.load_testing_set()
        with open(os.path.join(self.hyperparameters['log_folder_run'],'configuration.json'), 'w') as conf_file:
            json.dump(self.hyperparameters, conf_file, indent=4)

        if hasattr(self.dataset, "get_generated_documents_ids"):
            print("Saving special tasks")
            with open(os.path.join(self.hyperparameters['log_folder_run'],'special-task.json'), 'w') as st_file:
                json.dump(self.dataset.get_generated_documents_ids(), st_file, indent=4)
    


    def train_epoch(self, report_loss=100):
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set, 
                                                                             num_replicas=self.world_size,
                                                                             rank=self.rank,
                                                                             seed=self.training_epoch)
        train_loader = torch.utils.data.DataLoader(self.train_set, num_workers=0, batch_size=self.hyperparameters['batch_size'],
                                pin_memory=True, sampler=self.train_sampler, collate_fn=dict_collate_fn)
        self.model.train()
        epoch_loss = 0
        log_loss = 0

        self.training_epoch += 1
        acc = 0
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            self.citer += 1
            loss = self.train_step(batch, batch_idx)
            loss.backward()
            epoch_loss += loss
            log_loss += loss
            if acc % self.hyperparameters['accumulation'] == 0 and acc != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            acc += 1
            if batch_idx % report_loss == 0  and batch_idx != 0:
                dist.reduce(log_loss, 0, op=dist.ReduceOp.SUM)
                self.log({'train_step_loss': log_loss.item() / (self.world_size * report_loss)}, os.path.join(self.hyperparameters['log_folder_run'], 'log'))
                print({'train_step_loss': log_loss.item() / (self.world_size * report_loss)})
                log_loss = 0
        if acc % self.hyperparameters['accumulation'] != 0 :
            # we finally not update here may cause issue in the gradient value and
            # thus not consider last batch-size
            # self.optimizer.step()
            self.optimizer.zero_grad()
        
        dist.reduce(epoch_loss, 0, op=dist.ReduceOp.SUM)
        train_loss = epoch_loss.item() / (self.world_size * len(train_loader))
        return {"training_loss": train_loss}


    def checkpoint_routine(self, validation_score):
        ''' Save checkpoint according to certain conditions
        '''
        if self.training_best_validation > validation_score['v-score']:
            self.print('best validation reached ('+str(training_best_validation)+'->'
                       +str(validation_score['v-score'])+') saving the model')
            self.training_best_validation = validation_score['v-score']
            self.save_checkpoint()
            self.training_patience = 0
            self.print("VAL", self.citer, validation_score['v-score'])

    def break_routine(self, validation_score):
        ''' Return if the model have to stop training
        '''
        if self.training_patience > self.max_patience:
            self.print('Stopping at epoch', epoch, 'for task', i)
            self.load_checkpoint()
            return True
        return False
          
    def test(self, end_training=False):
        if 'ranking_examples' not in self.hyperparameters and not end_training:
            return
        elif not end_training: 
            prefix = ""
            testing_task = self.hyperparameters['ranking_examples']
        elif 'end_eval' not in self.hyperparameters:
            return
        else:
            prefix = "FINAL-"
            testing_task = self.hyperparameters['end_eval']    

        with torch.no_grad():
            self.print('Evaluate the tasks:', self.hyperparameters['ranking_examples'])
            res = {}
            for ttid in self.hyperparameters['ranking_examples']:
                res[ttid] = self.rank_task(ttid, self.test_set, subrank=self.hyperparameters['subrank'])
            # only master process got the complete results
            if self.is_master():

                result_set = {
                    "current_task_id":self.training_task_idn,
                    "current_task":self.training_task_idx,
                    "results":res
                    }
                tracked_task_path = os.path.join(self.hyperparameters['log_folder_run'], 'tracked_task')
                os.makedirs(tracked_task_path, exist_ok=True)

                with open(os.path.join(tracked_task_path, prefix+'pred-'+str(self.training_task_idx)+'-best.json'), 'w') as tracked_task_file:
                    json.dump(fres, tracked_task_file, indent=4)
                
                for k, prediction in res.items():
                    qualitative_results = {}
                    for query, document in  prediction.items():
                        tuple_d_v = [(k,v) for k,v in document.items()]
                        d = [k for k,v in document.items()]
                        s = [v for k,v in document.items()]
                        index_at_score = (-np.array(s)).argsort()
                        qualitative_results[query] =\
                            [self.test_set.dataset.get_query(query),
                                d[index_at_score[0]],
                                self.test_set.dataset.get_document(d[index_at_score[0]]),
                                d[index_at_score[-1]],
                                self.test_set.dataset.get_document(d[index_at_score[-1]])
                            ]


                    with open(os.path.join(tracked_task_path, prefix+'text-'+str(self.training_task_idx)+'-best-'+str(k)+'.json'), 'w') as tracked_task_file:
                        json.dump(qualitative_results, tracked_task_file, indent=4)
                    self.test_set.set_task(k)
                    ground_truth = self.test_set.get_relevant_documents_ids()


                    mean_score = evaluate(prediction, ground_truth)

                    with open(os.path.join(tracked_task_path, prefix+'score-'+str(self.training_task_idx)+'-best-'+str(k)+'-'+str(len(ground_truth))+'.json'), 'w') as tracked_task_file:
                        json.dump(mean_score, tracked_task_file, indent=4)
                    self.print(mean_score)
                    self.log({'MRR_T'+str(k): mean_score['recip_rank']})
                del fres

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def fit(self):
        self.init_train()
        # load start time for log
        start_time = time.time()
        # var for total iteration
        self.citer = 0
        if not hasattr(self, "training_task_idx"):
            self.training_task_idx = 0
        if not hasattr(self, "training_patience"):
            self.training_patience = 0
        if not hasattr(self, "training_best_validation"):
            self.training_best_validation = float("inf")
        if not hasattr(self, "training_epoch"):
            self.training_epoch = 0
        print('Start training')
        for tr_index, task_idx in enumerate(range(self.training_task_idx, len(self.task_perm))):
            if tr_index != 0:
                self.training_task_idx = task_idx
                self.training_patience = 0
                self.training_best_validation = float("inf")
                self.training_epoch = 0
            
            if tr_index != 0:
                self.optimizer.zero_grad()
                self.load_checkpoint()

            self.training_task_idn = self.task_perm[task_idx].item()

            self.print("setting train/validation set to task: (%d,%d)",
                       self.training_task_idx,
                       self.training_task_idn)

            self.train_set.set_task(self.training_task_idn)
            self.val_set.set_task(self.training_task_idn)

            print("Training epoch %d", self.hyperparameters['n_epoch'])
            for ep_index, tr_epoch in enumerate(range(self.hyperparameters['n_epoch'])):
                self.train()
                epoch_loss = self.train_epoch(report_loss=40)
                self.eval()
                
                validation_score = self.validate(self.training_task_idn)
                print(validation_score)
                self.checkpoint_routine(validation_score)
                self.end_epoch_routine(dict(**epoch_loss, **validation_score))
                do_break = self.break_routine(validation_score)
                self.training_patience += 1
                if do_break:
                    break
            self.load_checkpoint()
            self.test(False)
        self.test(True)

    def rank_step(self, batch, batch_idx):
        pass

    def rank_task(self, ranking_set, task_id, subrank=None):
        self.model.eval()
        self.print('start rank_task', task_id)
        ranking_set.set_task(task_id, subrank=subrank)

        sampler = torch.utils.data.distributed.DistributedSampler(ranking_set, 
                                                                 num_replicas=self.world_size,
                                                                 rank=self.rank)
        dataloader = torch.utils.data.DataLoader(ranking_set, num_workers=0,
                                                 batch_size=self.hyperparameters['batch_size']*2,
                                                 pin_memory=True, sampler=sampler, collate_fn=dict_collate_fn) 
        progress_bar = range(len(dataloader))
        queries_id, documents_id, scores = [], [], []

        for batch_idx, batch in zip(progress_bar, dataloader):
            query_id, document_id, score = self.rank_step(batch, batch_idx)
            queries_id.append(torch.LongTensor(query_id))
            documents_id.append(torch.LongTensor(document_id))
            if score.dim() == 0:
                score = score.unsqueeze(0)
            scores.append(score)
            if batch_idx % 100 == 0:
                self.print('prediction completed at: '+str(batch_idx/len(dataloader) * 1000 // 1 /10)+'%')
        
        queries_id = self.gather_safe(torch.cat(queries_id, 0).to(self.device))
        documents_id = self.gather_safe(torch.cat(documents_id, 0).to(self.device))
        scores = self.gather_safe(torch.cat(scores, 0))

        prediction_set = {}
        if self.is_master():
            prediction_set = {}
            for i in range(len(queries_id)):
                qid = queries_id[i].item()
                did = documents_id[i].item()
                s = scores[i].item()
                if qid not in prediction_set:
                    prediction_set[qid] = {}
                prediction_set[qid][did] = s
        return prediction_set

    def rank(self, verbose=True, output_file=None):
        self.init_finetune()
        self.max_patience = 2
        self.load_testing_set()
        self.load_checkpoint()
        self.model.eval()
        with torch.no_grad():
            task_ids = self.test_set.dataset.get_task_ids()
            prediction_set = self.rank_task(task_ids)
            with open(os.path.join(tracked_task_path, 'rank_all_prediction.json'), 'w') as tracked_task_file:
                json.dump(prediction_set, tracked_task_file, indent=4)
            ground_truth = self.test_set.get_relevant_documents_ids()

            mean_score = evaluate(prediction, ground_truth)
            with open(os.path.join(tracked_task_path, 'rank_all_scores.json'), 'w') as tracked_task_file:
                json.dump(mean_score, tracked_task_file, indent=4)