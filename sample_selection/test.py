from ssutil import write_results
from ENV import ADEnv
from DQNSS import DQNSS
import torch
import os
import pandas as pd
import getpass
import argparse
import time
from testbed.utils import import_ts_data_unsupervised

dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default='@results/',
                    help="the output file path")
parser.add_argument("--model_dir", type=str, default='@models/',
                    help="the output file path")
parser.add_argument("--dataset", type=str,
                    default='MSL',
                    help='ASD,SMAP,MSL,SWaT_cut'
                    )
parser.add_argument("--entities", type=str,
                    default='FULL',
                    help='FULL represents all the csv file in the folder, '
                         'or a list of entity names split by comma '    # ['D-14', 'D-15']
                    )

parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--num_sample", type=int, default=1000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_filename = os.path.join(args.output_dir, 'results.csv')
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
with open(results_filename, 'w') as f:
    f.write('dataset,subset,pr_mean,pr_std,roc_mean,roc_std\n')


dataset_name_lst = args.dataset.split(',')
for dataset in dataset_name_lst:
    # # import data
    data_pkg = import_ts_data_unsupervised(dataset_root,
                                           dataset, entities=args.entities,
                                           combine=args.entity_combined)
    train_lst, test_lst, label_lst, name_lst = data_pkg

    entity_metric_lst = []
    entity_metric_std_lst = []
    for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):

        entries = []
        t_lst = []
        pr_auc_history = []
        roc_auc_history = []

        for i in range(args.runs):
            print(f'Running {dataset} {i}...')
            model_id = f'_run_{i}'

            env = ADEnv(
                dataset=train_data,
                num_sample=args.num_sample
            )

            dplan = DQNSS(
                env=env,
                test_X=test_data,
                test_Y=labels,
                destination_path=args.model_dir,
                double_dqn=False
            )
            dplan.fit(reset_nets=True)
            dplan.show_results()
            # roc, pr = dplan.model_performance(on_test_set=True)
            roc, pr = dplan.model_performance()
            print(f'Finished run {i} with pr: {pr} and auc-roc: {roc}...')
            pr_auc_history.append(pr)
            roc_auc_history.append(roc)

            destination_filename = dataset_name + '_' + model_id + '.pth'
            dplan.save_model(destination_filename)
            print()
            print('--------------------------------------------------\n')

        print(f'Finished {dataset_name}...')
        print('--------------------------------------------------\n')
        write_results(pr_auc_history, roc_auc_history, dataset_name, results_filename)
