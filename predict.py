import argparse
import os
import time
import datetime
import json


import pytrec_eval

# ray ressources
from ray import tune
from ray.tune import run




import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from lire import data_tools
from lire.log_tools.logger import ConfigurationFile
from lire.misc.class_tools import struct
from lire import experiment_tools 
from lire.data_tools.dataset import NamedDataset
from lire.experiment_tools import ranker_v2 as ranker
from lire.data_tools.dataset import MSMarco

from experiments_scripts.evaluation import evaluation

parser = argparse.ArgumentParser(description='Performs prediction on all finished experiments inside a folder')

parser.add_argument('--data-folder', type=str, dest='data_folder',  default='/net/cal/gerald/CPD/data',
                    help='The data where saving, finding or downloading the data')
parser.add_argument('--xp-folder', type=str, dest='xp_folder', default='/net/sundays/gerald/xps',
                    help='Where the xps files will be saved')

args = parser.parse_args()



def evaluate(evaluation_set, model):
    queries_id = evaluation_set.get_queries_id()
    ground_truth = evaluation_set.qrels.get_dictionary()
    ground_truth = { str(q_id):{str(d_id): v for d_id, v in ground_truth[str(q_id)].items() }
                    for q_id in queries_id }

    prediction = model.rank(model.prepare_evaluation_dataset(evaluation_set))
    prediction = { str(q_id):{str(d_id): v for d_id, v in prediction[q_id].items() }
                    for q_id in queries_id }

    _, mean_score = evaluation.evaluation(evaluation.subrank(prediction, 10), ground_truth)
    return prediction, mean_score



def ray_evaluation(config, checkpoint_dir=None):
    print("Evaluating the following config: ", config)

    print("Loading the dataset")
    dataset = MSMarco.MSMarcoPassageRanking(config['data_folder'],
                                            set_name='dev',
                                            mode='eval',
                                            eval_key = config["eval_key"],
                                            ressource_file=os.path.join('ressource',
                                                config['dataset']['ressource']+'.json'))
    print("Loading the model")
    model = ranker.models[config['model']['model_name']](config['trial_folder'], "best_model_validation.pth",
                                                         config)
    model.init_optimizer()
    model._load_checkpoint(os.path.join(config['trial_folder'], "best_model_validation.pth"))

    prediction, average_performances = evaluate(dataset, model)
    torch.save(prediction, os.path.join(config['trial_folder'], 'prediction.pkl'))
    with open(os.path.join(config['trial_folder'], 'prediction.json'), 'w') as prediction_file:
        json.dumps({"prediction":prediction, 'average_performances': average_performances}, prediction_file)

if __name__ == "__main__":
    os.environ["SLURM_JOB_NAME"] = "bash"
    print("Finiding experiments inside the folder")
    experiments_folder = os.listdir(args.xp_folder)

    print("Building ray config from experiments")
    ray_config = {"grid_search":[]}

    for folder_xp in experiments_folder:
        ray_xp_config = {}
        configuration_filepath = os.path.join(args.xp_folder, folder_xp,'configuration.json')
        if not os.path.exists(configuration_filepath):
            continue
        # reading configuration
        with open(configuration_filepath, 'r') as configuration_file:
            configuration = json.load(configuration_file)
        # ensure xp is finished
        if("ending" in configuration and configuration["ending"]):
            ray_xp_config['dataset'] = configuration['dataset']
            ray_xp_config['model'] = configuration['model']
            ray_xp_config['trial_folder'] = os.path.join(args.xp_folder, folder_xp)
            ray_xp_config['data_folder'] = args.data_folder
            ray_xp_config['eval_key'] = configuration['eval_key']

            ray_config['grid_search'].append(ray_xp_config)
    print("The following experiments will be evaluated : ", ray_config)


    es = {
        'name': "evaluation",
        'run': ray_evaluation,
        'config': ray_config,
        'local_dir': args.xp_folder,
        'resources_per_trial':{
            'cpu':4, 'gpu': 1
        }
    }
    tune.run(es['run'], es['name'], config=es["config"], local_dir=args.xp_folder,
             resources_per_trial=es['resources_per_trial'])