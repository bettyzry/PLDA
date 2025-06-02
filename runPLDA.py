import matplotlib.pyplot as plt
import os
import argparse
import getpass
import yaml
import time
import importlib as imp
import numpy as np
from testbed.utils import import_ts_data_unsupervised, get_lr, generate_data
from deepod.metrics import ts_metrics, point_adjustment
import pandas as pd
from sample_selection.PLDA import PLDA
from sample_selection.QSS import QSS
from sample_selection.ENV import ADEnv
from deepod.utils.utility import insert_pollution, insert_pollution_seq, insert_pollution_new, split_pollution, \
    insert_pollution_from_test

dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'
dataset_root_DC = f'/home/{getpass.getuser()}/dataset/5-TSdata/_DCDetector/'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--output_dir", type=str, default='@records/',  # records
                    help="the output file path")
parser.add_argument("--trainsets_dir", type=str, default='@trainsets/',
                    help="the output file path")

parser.add_argument("--dataset", type=str,
                    default='PUMP',
                    help='ASD,MSL,SMAP,SMD,SWaT_cut,PUMP,DASADS,UCR_natural_fault,UCR_natural_gait,UCR_natural_heart_sbeat',
                    # help='WADI,PUMP,PSM,ASD,SWaT_cut,DASADS,EP,UCR_natural_mars,UCR_natural_insect,UCR_natural_heart_vbeat2,'
                    #      'UCR_natural_heart_vbeat,UCR_natural_heart_sbeat,UCR_natural_gait,UCR_natural_fault'
                    )
parser.add_argument("--entities", type=str,
                    default='FULL',  # ['C-1', 'C-2', 'F-4']
                    help='FULL represents all the csv file in the folder, '
                         'or a list of entity names split by comma '  # ['D-14', 'D-15'], ['D-14']
                    )
parser.add_argument("--entity_combined", type=int, default=1, help='1:merge, 0: not merge')
parser.add_argument("--model", type=str, default='TcnED',
                    help="TcnED, TranAD, NCAD, NeuTraLTS, LSTMED, TimesNet, AnomalyTransformer"
                    )

parser.add_argument('--silent_header', type=bool, default=False)
parser.add_argument("--flag", type=str, default='')
parser.add_argument("--note", type=str, default='')

parser.add_argument('--seq_len', type=int, default=30)
parser.add_argument('--stride', type=int, default=1)

parser.add_argument('--sample_selection', type=int, default=0)  # data augmentation functions
parser.add_argument('--insert_outlier', type=int, default=0)    # 0: insert anomaly contamination，1:do not insert
parser.add_argument('--rate', type=float, default=10)           # contamination rate 10%
args = parser.parse_args()


module = imp.import_module('deepod.models')
model_class = getattr(module, args.model)

path = 'testbed/configs.yaml'
with open(path) as f:
    d = yaml.safe_load(f)
    try:
        model_configs = d[args.model]
    except KeyError:
        print(f'config file does not contain default parameter settings of {args.model}')
        model_configs = {}
model_configs['seq_len'] = args.seq_len
model_configs['stride'] = args.stride


def main(ss_epoch=100, a=0.5, p=1000):
    # # setting result file/folder path
    cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f'{args.model}.{args.flag}.csv')
    # # setting loss file/folder path
    funcs = ['norm', 'PI', 'LOSS', 'PLDA', 'DQN-PLDA']
    trainsets_dir = f'{args.trainsets_dir}/{args.model}.{args.flag}/'
    os.makedirs(trainsets_dir, exist_ok=True)

    # # print header in the result file
    if not args.silent_header:
        f = open(result_file, 'a')
        print('\n---------------------------------------------------------', file=f)
        print(f'model: {args.model}, dataset: {args.dataset}, '
              f'{args.runs}runs, {cur_time}', file=f)
        for k in model_configs.keys():
            print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
        print(f'Parameters,\t [funcs], \t\t  {funcs[args.sample_selection]}', file=f)
        print(f'Note: {args.note}', file=f)
        print(f'---------------------------------------------------------', file=f)
        print(f'data, adj_auroc, std, adj_ap, std, adj_f1, std, adj_p, std, adj_r, std, time, model', file=f)
        f.close()
        print('write')
        print(args.insert_outlier, args.rate, args.sample_selection, ss_epoch, a)

    dataset_name_lst = args.dataset.split(',')

    for dataset in dataset_name_lst:

        data_pkg = import_ts_data_unsupervised(dataset_root, dataset_root_DC,
                                               dataset, entities=args.entities,
                                               combine=args.entity_combined)
        train_lst, test_lst, label_lst, name_lst = data_pkg

        entity_metric_lst = []
        entity_metric_std_lst = []
        entity_t_lst = []
        for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
            if args.insert_outlier:
                train_data, train_labels = insert_pollution_new(train_data, test_data, labels, args.rate)  # 插入完整异常序列

            entries = []
            t_lst = []
            lr, epoch = get_lr(dataset_name, args.model, args.insert_outlier, model_configs['lr'],
                               model_configs['epochs'])
            model_configs['lr'] = lr
            model_configs['epochs'] = epoch
            print(f'Model Configs: {model_configs}')
            for i in range(args.runs):
                print(
                    f'\nRunning [{i + 1}/{args.runs}] of [{args.model}] [{funcs[args.sample_selection]}] on Dataset [{dataset_name}]')
                print(f'\ninsert outlier [{args.insert_outlier}] with pollution rate [{args.rate}]')

                t1 = time.time()
                clf = model_class(**model_configs, random_state=83 + i)
                clf.sample_selection = args.sample_selection
                clf.ss_epoch = ss_epoch
                clf.a = a
                clf.p = p
                if args.sample_selection != 4:
                    clf.fit(train_data)  # 简化版本
                else:
                    env = ADEnv(
                            dataset=train_data,
                            y=None,
                            clf=clf
                        )
                    # plda = PLDA(env)
                    plda = QSS(env)
                    plda.OD_fit()

                t = time.time() - t1

                scores = clf.decision_function(test_data)
                eval_metrics = ts_metrics(labels, scores)
                adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))

                # print single results
                txt = f'{dataset_name},'
                txt += ', '.join(['%.4f' % a for a in eval_metrics]) + \
                       ', pa, ' + \
                       ', '.join(['%.4f' % a for a in adj_eval_metrics])
                txt += f', model, {args.model}, time, {t:.1f}, runs, {i + 1}/{args.runs}, {funcs[args.sample_selection]}'
                print(txt)

                entries.append(adj_eval_metrics)
                t_lst.append(t)

            avg_entry = np.average(np.array(entries), axis=0)
            std_entry = np.std(np.array(entries), axis=0)
            avg_t = np.average(t_lst)

            entity_metric_lst.append(avg_entry)
            entity_metric_std_lst.append(std_entry)
            entity_t_lst.append(avg_t)

            if 'UCR' not in dataset_name and args.entity_combined == 1:
                txt = '%s, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
                      '%.4f, %.4f, %.4f, %.4f, %.1f, %s, %f ' % \
                      (dataset_name,
                       avg_entry[0], std_entry[0], avg_entry[1], std_entry[1],
                       avg_entry[2], std_entry[2], avg_entry[3], std_entry[3],
                       avg_entry[4], std_entry[4],
                       avg_t, args.model + '-' + funcs[args.sample_selection] + str(
                          args.insert_outlier * args.rate) + '_' + str(ss_epoch) + "_" + str(a), model_configs['lr'])
                print(txt)

                if not args.silent_header:
                    f = open(result_file, 'a')
                    print(txt, file=f)
                    f.close()

        if 'UCR' in dataset or args.entity_combined == 0:
            entity_avg_mean = np.average(np.array(entity_metric_lst), axis=0)
            entity_std_mean = np.average(np.array(entity_metric_std_lst), axis=0)

            txt = '%s, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
                  '%.4f, %.4f, %.4f, %.4f, %.1f, %s, %f ' % \
                  (dataset,
                   entity_avg_mean[0], entity_std_mean[0], entity_avg_mean[1], entity_std_mean[1],
                   entity_avg_mean[2], entity_std_mean[2], entity_avg_mean[3], entity_std_mean[3],
                   entity_avg_mean[4], entity_std_mean[4],
                   np.sum(np.array(entity_t_lst)),
                   args.model + '-' + funcs[args.sample_selection] + str(args.insert_outlier * args.rate) + '_' + str(
                       ss_epoch) + "_" + str(a),
                   model_configs['lr'])
            print(txt)

            if not args.silent_header:
                f = open(result_file, 'a')
                print(txt, file=f)
                f.close()


def main_scalability(l=100000, d=16):
    # # setting result file/folder path
    cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
    os.makedirs(args.output_dir, exist_ok=True)
    scalability_file = os.path.join('plotsource/', 'Scalability.csv')
    # # setting loss file/folder path
    funcs = ['norm', 'expand', 'delete', 'both', '', 'ICLM21', 'Arxiv22', 'myfunc', 'DQN-myfunc']

    train_data = generate_data(d, l)

    for sample_selection in [7, 0, 5, 6]:
        t1 = time.time()
        clf = model_class(**model_configs, random_state=83)
        clf.sample_selection = sample_selection
        clf.es = False
        clf.fit(train_data)
        t = time.time() - t1

        txt = '%s, %s, %.4f, %d, %d' % \
              (args.model, funcs[sample_selection], t, d, l)
        print(txt)
        if not args.silent_header:
            f = open(scalability_file, 'a')
            print(txt, file=f)
            f.close()


if __name__ == '__main__':
    func_list = {'ORIG':0, 'PI':1, 'LOSS':2, 'PLDA':3, 'RPLDA':4 }
    func = 'PLDA'
    args.sample_selection = func_list[func]     # Data Argumentation Methods
    args.dataset = 'PUMP'                       # Dataset name
    args.model = 'TcnED'                        # Benchmark Unsupervised Deep TSAD Models
    args.insert_outlier = 1                     # 0: insert anomaly contamination，1:do not insert
    args.rate = 10                              # Contamination rate 10%
    main()
