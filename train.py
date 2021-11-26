from lire.experiment_tools import knrm
from lire.experiment_tools import monoT5
from lire.experiment_tools import cedr
import argparse
import json
import os

# testing command 
# python experiments_scripts/knrm_ddp_training.py  --master-addr localhost --master-port 12345 --node 0 --n-nodes 1 --n-gpus 1 --data-folder /net/cal/gerald/CPD/data --xp-folder /net/cal/gerald/ddp-lire-xps
parser = argparse.ArgumentParser(description='Experiments launcher')

# multi-gpu options

parser.add_argument('--master-addr', type=str, dest='master_addr', default='localhost',
                    help='The address of the master process should be launched first')
parser.add_argument('--master-port', type=str, dest='master_port',  default='123456',
                    help='The port of the master process')
parser.add_argument('--node', type=int, dest='node', default=0,
                    help='Where the xps files will be saved')
parser.add_argument('--n-nodes', type=int, dest='n_nodes', default=1,
                    help='Where the xps files will be saved')
parser.add_argument('--n-gpus', type=int, dest='n_gpus', default=1,
                    help='Where the xps files will be saved')
parser.add_argument('--gpus', type=str, dest='gpus', default=None,
                    help='Where the xps files will be saved')
parser.add_argument('--init-method', type=str, dest='init_method', default='env://',
                    help='Where the xps files will be saved')


# xp options
parser.add_argument('--xp-config', type=str, dest='xp_config', default='experiments_configs/test.json',
                    help='The location of the experiment configuration file\
                    (a json file see experiments/parameters_search.json for example)')
parser.add_argument('--data-folder', type=str, dest='data_folder',  default='/net/cal/gerald/CPD/data',
                    help='The data where saving, finding or downloading the data')
parser.add_argument('--xp-folder', type=str, dest='xp_folder', default='/net/sundays/gerald/xps',
                    help='Where the xps files will be saved')
parser.add_argument('--tokenizer_path', type=str, dest='tokenizer_path', 
                    default='/net/sundays/gerald/data/word_embedding/wiki.en/wiki.en.bin',
                    help='Where the xps files will be saved')
parser.add_argument('--model', type=str, dest='model', 
                    default='MONOT5',
                    help='The ranker to use [KNRM, MONOT5, CEDR]')
parser.add_argument('--batch-size', dest='batch_size',type=int, default=None)
parser.add_argument('--accumulation', dest='accumulation',type=int, default=1)

parser.add_argument('--switch', dest='switch', default=False, action='store_true')



args = parser.parse_args()

MODELS = {
    "KNRM": knrm.DDPKNRM,
    "MONOT5": monoT5.DDPMonoT5,
    "VBERT": cedr.DDPCEDRVanilla,
    "CEDRKNRM": cedr.DDPCEDRKNRM
}



class struct(dict):
    ''' A structure for access dictionary like python objects
    '''
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)



if __name__ == "__main__":
    if args.gpus is not None:
        gpus = [int(gpu) for gpu in args.gpus.split(',')]
    else:
        gpus = [i for i in range(args.n_gpus)]
    with open(args.xp_config, 'r') as f:
        options = json.load(f)
    print(gpus)
    for k in options:
        if isinstance(options[k],list):
            print("list",options[k])
            nl = []
            for l in options[k]:
                try:
                    r = l
                    if '$' in r:
                        r = os.path.expandvars(r) 
                    if '~' in r:
                        r = os.path.expanduser(r)
                    nl.append(r)
                except TypeError:
                    nl.append(l)
            options[k] = nl
        else:
            try:
                if '$' in options[k]:
                    options[k] = os.path.expandvars(options[k]) 
                if '~' in options[k]:
                    options[k] = os.path.expanduser(options[k])
            except TypeError:
                pass
    
    if args.batch_size is not None:
        options["batch_size"] = args.batch_size
    
    if "subrank" not in options:
        options["subrank"] = None 
    if "subrank" not in options:
        options["subrank"] = 100
    options["accumulation"] = args.accumulation
    options["switch"] = args.switch

    for i in options["topics_folder_path"]:
        hyperparameters = struct(options)
        hyperparameters["topics_folder_path"] = i
        hyperparameters["rerank_path"] = os.path.join(i, hyperparameters["rerank_path"])
        hyperparameters["model"] = args.model
        print("batch process ", i, "->", hyperparameters)
        experiment =\
            MODELS[args.model](n_gpus=args.n_gpus, 
                                n_nodes=args.n_nodes,
                                node=args.node, 
                                master_address=args.master_addr, 
                                master_port=args.master_port,
                                gpus=gpus,
                                hyperparameters=hyperparameters)
        experiment.run(main_func=MODELS[args.model].fit)

# python experiments_scripts/ddp_training.py --xp-config experiments_configs/original/monot5_all.json --n-gpus 1 --gpus 1 --master-port 12345 --model MONOT5
# python experiments_scripts/ddp_training.py --xp-config experiments_configs/original/monot5_all.json --n-gpus 1 --gpus 2 --master-port 12346 --model VBERT
# python experiments_scripts/ddp_training.py --xp-config experiments_configs/original/monot5.json --n-gpus 1 --gpus 4 --master-port 12347 --model MONOT5
# python experiments_scripts/ddp_training.py --xp-config experiments_configs/original/vbert.json --n-gpus 1 --gpus 4 --master-port 12348 --model VBERT
# python experiments_scripts/ddp_training.py --xp-config experiments_configs/scenarios/informationupdate_cedr.json --n-gpus 2 --gpus 5,6 --master-port 12349 --model CEDRKNRM
# python experiments_scripts/ddp_training.py --xp-config experiments_configs/scenarios/informationupdate_vbert.json --n-gpus 2 --gpus 5,6 --master-port 12350 --model VBERT