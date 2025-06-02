import os
import glob
import random

import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# --------------------------- data preprocessing --------------------------- #

def get_lr(dataset_name, model_name, insert_outlier, ori_lr, ori_epoch):
    # a = 0.5
    lr = ori_lr
    epoch = ori_epoch
    if model_name == 'NCAD':
        lr = 0.0000002
        if 'ASD' in dataset_name:
            lr = 0.000002

        elif 'DASADS' in dataset_name:
            lr = 0.000002           # 0.0000002
        elif 'PUMP' in dataset_name:
            lr = 0.000002           # 0.0000002
        elif 'SWaT' in dataset_name:
            lr = 0.000002           # 0.0002
        elif 'SMAP' in dataset_name:
            lr = 0.000002

        elif 'heart_vbeat-' in dataset_name:
            lr = 0.00000002
        elif 'heart_vbeat2-' in dataset_name:
            lr = 0.0000002
        elif 'fault-' in dataset_name or 'gait-' in dataset_name:
            lr = 0.002
        elif 'sbeat-' in dataset_name:
            lr = 0.00002
        elif 'insect-' in dataset_name or 'mars-' in dataset_name:
            lr = 0.00002
        elif 'MSL' in dataset_name:
            lr = 0.0000002

        elif 'SWaT_cut' in dataset_name:    # 有问题
            lr = 0.0002

    if model_name == 'NeuTraLTS':
        lr = 0.001
        if 'gait-' in dataset_name:
            lr = 0.001
        elif 'fault-' in dataset_name:
            lr = 0.00001
        elif 'heart_sbeat-' in dataset_name:
            lr = 0.000001
        elif 'heart_vbeat-' in dataset_name or 'insect-' in dataset_name:
            lr = 0.0001     # 不太好
        elif 'SWaT' in dataset_name:
            lr = 0.0001
        elif 'ASD' in dataset_name:
            lr = 0.1      # 0.001
        elif 'PUMP' in dataset_name:
            lr = 0.001
        elif 'MSL' in dataset_name or 'SMAP' in dataset_name:
            lr = 0.0001
        elif 'DASADS' in dataset_name:
            lr = 0.0001
        elif 'SMD' in dataset_name:
            lr = 0.0001  # ?

    if model_name == 'TranAD':
        lr = 0.000001
        if 'mars-' in dataset_name:
            lr = 0.0001
        elif 'gait-' in dataset_name or 'heart_sbeat-' in dataset_name or 'heart_vbeat-' in dataset_name\
            or 'heart_vbeat2-' in dataset_name or 'insect-' in dataset_name:
            lr = 0.0001
        elif 'fault-' in dataset_name:
            lr = 0.00001    # 未定

        elif 'ASD' in dataset_name:
            lr = 0.001
        elif 'DASADS' in dataset_name:
            lr = 0.0001
        elif 'MSL' in dataset_name:
            lr = 0.00001
        elif 'SMAP' in dataset_name:
            lr = 0.00001
        elif 'SMD' in dataset_name:
            lr = 0.0001

        elif 'PUMP' in dataset_name:
            lr = 0.001

    if model_name == 'TcnED':
        lr = 0.00015

        if 'heart_sbeat-' in dataset_name:
            lr = 0.00015
        elif 'ASD' in dataset_name or 'fault-' in dataset_name:
            lr = 0.0015
        elif 'insect-' in dataset_name:
            lr = 0.05
        elif 'gait-' in dataset_name:
            lr = 0.05
        elif 'heart_vbeat-' in dataset_name:
            lr = 0.01
        elif 'DASADS' in dataset_name or 'PUMP' in dataset_name:
            lr = 0.00015
        elif 'SMAP' in dataset_name:
            lr = 0.015
        elif 'SMD' in dataset_name:
            lr = 0.00015

        elif 'MSL' in dataset_name:
            lr = 0.0015
        elif 'SWaT' in dataset_name:
            lr = 0.0015

    return lr, epoch


def data_standardize_xhz(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
    mini, maxi = X_train.min(), X_train.max()
    for col in X_train.columns:
        if maxi[col] != mini[col]:
            X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
        else:
            assert X_train[col].nunique() == 1
            if remove:
                if verbose:
                    print("Column {} has the same min and max value in train. Will remove this column".format(col))
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
            else:
                if verbose:
                    print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                if mini[col] != 0:
                    X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                    X_test[col] = X_test[col] / mini[col]
                if verbose:
                    print("After transformation, train unique vals: {}, test unique vals: {}".format(
                    X_train[col].unique(),
                    X_test[col].unique()))
    X_train = X_train.values
    X_test = X_test.values

    return X_train, X_test


def data_standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
    mini, maxi = np.min(X_train, axis=0), np.max(X_train, axis=0)
    feature = X_train.shape[1]

    for col in range(feature):
        if maxi[col] != mini[col]:
            X_train[:, col] = (X_train[:, col] - mini[col]) / (maxi[col] - mini[col])
            X_test[:, col] = (X_test[:, col] - mini[col]) / (maxi[col] - mini[col])
            X_test[:, col] = np.clip(X_test[:, col], a_min=min_clip, a_max=max_clip)
        else:
            if mini[col] != 0:
                X_train[:, col] = X_train[:, col] / mini[col]  # Redundant operation, just for consistency
                X_test[:, col] = X_test[:, col] / mini[col]
    return X_train, X_test


def data_normalization(x_train, x_test):
    x_train = x_train.values
    x_test = x_test.values

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


# --------------------------- data loading --------------------------- #

def get_data_lst(dataset_dir, dataset):
    if dataset == 'FULL':
        print(os.path.join(dataset_dir, '*.*'))
        data_lst = glob.glob(os.path.join(dataset_dir, '*.*'))
    else:
        name_lst = dataset.split(',')
        data_lst = []
        for d in name_lst:
            data_lst.extend(glob.glob(os.path.join(dataset_dir, d + '.*')))
    data_lst = sorted(data_lst)
    return data_lst


def import_data(file):
    if file.endswith('.npz'):
        data = np.load(file, allow_pickle=True)
        x, y = data['X'], data['y']
        y = np.array(y, dtype=int)
    else:
        if file.endswith('pkl'):
            func = pd.read_pickle
        elif file.endswith('csv'):
            func = pd.read_csv
        elif file.endswith('arff'):
            def func(f):
                df_ = pd.DataFrame(arff.loadarff(f)[0])
                df_ = df_.replace({b'no': 0, b'yes': 1})
                df_ = df_.drop('id', axis=1)
                return df_
        else:
            raise NotImplementedError('')

        df = func(file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        x = df.values[:, :-1]
        y = np.array(df.values[:, -1], dtype=int)
    return x, y


def read_data(file, split='50%-normal', normalization='z-score', seed=42):
    """
    read data from files, normalization, and perform train-test splitting

    Parameters
    ----------
    file: str
        file path of dataset

    split: str (default='50%-normal', choice=['50%-normal', '60%', 'none'])
        training-testing set splitting methods:
            - if '50%-normal': use half of the normal data as training set,
                and the other half attached with anomalies as testing set,
                this splitting method is used in self-supervised studies GOAD [ICLR'20], NeuTraL [ICML'21]
            - if 'none': use the whole set as both the training and testing set
                This is commonly used in traditional methods.
            - if '60%': use 60% data during training and the rest 40% data in testing,
                while keeping the original anomaly ratio.

    normalization: str (default='z-score', choice=['z-score', 'min-max'])

    seed: int (default=42)
        random seed
    """

    x, y = import_data(file)

    # train-test splitting
    if split == '50%-normal':
        rng = np.random.RandomState(seed)
        idx = rng.permutation(np.arange(len(x)))
        x, y = x[idx], y[idx]

        norm_idx = np.where(y==0)[0]
        anom_idx = np.where(y==1)[0]
        split = int(0.5 * len(norm_idx))
        train_norm_idx, test_norm_idx = norm_idx[:split], norm_idx[split:]

        x_train = x[train_norm_idx]
        y_train = y[train_norm_idx]

        x_test = x[np.hstack([test_norm_idx, anom_idx])]
        y_test = y[np.hstack([test_norm_idx, anom_idx])]

        print(f'Original size: [{x.shape}], Normal/Anomaly: [{len(norm_idx)}/{len(anom_idx)}] \n'
              f'After splitting: training/testing [{len(x_train)}/{len(x_test)}]')

    elif split == '60%':
        x_train, y_train, x_test, y_test = train_test_split(x, y, shuffle=True, random_state=seed,
                                                            test_size=0.4, stratify=y)

    else:
        x_train, x_test = x.copy(), x.copy()
        y_train, y_test = y.copy(), y.copy()

    # normalization
    if normalization == 'min-max':
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(x_train)
        x_train = minmax_scaler.transform(x_train)
        x_test = minmax_scaler.transform(x_test)

    elif normalization == 'z-score':
        mus = np.mean(x_train, axis=0)
        sds = np.std(x_train, axis=0)
        sds[sds == 0] = 1
        x_train = np.array([(xx - mus) / sds for xx in x_train])
        x_test = np.array([(xx - mus) / sds for xx in x_test])

    elif normalization == 'scale':
        x_train = x_train / 255
        x_test = x_test / 255

    return x_train, y_train, x_test, y_test


def generate_data(d, l):
    rd = np.random.RandomState(88)
    matrix_uniform = rd.uniform(-2, 3, (l, d))
    return matrix_uniform


def import_ts_data_unsupervised(data_root1, data_root2, data, entities=None, combine=False):
    if data in ['MSL', 'SMAP', 'SMD']:
        train = np.load(os.path.join(data_root2, data, data + '_train.npy'))
        label = np.load(os.path.join(data_root2, data, data + '_test_label.npy'))
        test = np.load(os.path.join(data_root2, data, data + '_test.npy'))

        train, test = data_standardize(train, test)
        train_lst = [train]
        test_lst = [test]
        label_lst = [label]
        name_lst = [data]
    else:
        if data[:3] == 'UCR':
            combine = False
        else:
            combine = combine

        if type(entities) == str:
            entities_lst = entities.split(',')
        elif type(entities) == list:
            entities_lst = entities
        else:
            raise ValueError('wrong entities')

        name_lst = []
        train_lst = []
        test_lst = []
        label_lst = []

        if len(glob.glob(os.path.join(data_root1, data) + '/*.csv')) == 0:
            machine_lst = os.listdir(data_root1 + data + '/')
            for m in sorted(machine_lst):
                if entities != 'FULL' and m not in entities_lst:
                    continue
                train_path = glob.glob(os.path.join(data_root1, data, m, '*train*.csv'))
                test_path = glob.glob(os.path.join(data_root1, data, m, '*test*.csv'))

                assert len(train_path) == 1 and len(test_path) == 1, f'{m}'
                train_path, test_path = train_path[0], test_path[0]

                train_df = pd.read_csv(train_path, sep=',', index_col=0)
                test_df = pd.read_csv(test_path, sep=',', index_col=0)
                labels = test_df['label'].values
                train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

                # normalization
                train, test = train_df.values, test_df.values
                train, test = data_standardize(train, test)
                # train, test = train_df.values, test_df.values

                train_lst.append(train)
                test_lst.append(test)
                label_lst.append(labels)
                name_lst.append(data+'-'+m)

            if combine:
                train_lst = [np.concatenate(train_lst)]
                test_lst = [np.concatenate(test_lst)]
                label_lst = [np.concatenate(label_lst)]
                name_lst = [data + '_combined']

        else:
            train_df = pd.read_csv(f'{data_root1}{data}/{data}_train.csv', sep=',', index_col=0)
            test_df = pd.read_csv(f'{data_root1}{data}/{data}_test.csv', sep=',', index_col=0)
            labels = test_df['label'].values
            train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)
            train, test = train_df.values, test_df.values
            train, test = data_standardize(train, test)

            train_lst.append(train)
            test_lst.append(test)
            label_lst.append(labels)
            name_lst.append(data)

        # for ii, train in enumerate(train_lst):
        #     test = test_lst[ii]
        #     train, test = data_standardize(train, test)
        #     train_lst[ii] = train
        #     test_lst[ii] = test

    return train_lst, test_lst, label_lst, name_lst


def get_anom_pairs(y):
    anom_pairs = []
    anom_index = np.where(y == 1)[0]
    tmp_seg = []
    for i in anom_index:
        tmp_seg.append(i)
        if i + 1 not in anom_index:
            anom_pairs.append((tmp_seg[0], tmp_seg[-1]))
            tmp_seg = []
    return anom_pairs


def split_train_test_val(x, y, test_ratio, val_ratio, random_state=None, del_features=True):
    idx_norm = y == 0
    idx_out = y == 1

    n_f = x.shape[1]

    if del_features:
        del_list = []
        for i in range(n_f):
            if np.std(x[:, i]) == 0:
                del_list.append(i)
        if len(del_list) > 0:
            print("Pre-process: Delete %d features as every instances have the same behaviour: " % len(del_list))
            x = np.delete(x, del_list, axis=1)

    # keep outlier ratio, norm is normal out is outlier
    x_train_norm, x_teval_norm, y_train_norm, y_teval_norm = train_test_split(x[idx_norm], y[idx_norm],
                                                                              test_size=test_ratio + val_ratio,
                                                                              random_state=random_state)
    x_train_out, x_teval_out, y_train_out, y_teval_out = train_test_split(x[idx_out], y[idx_out],
                                                                          test_size=test_ratio + val_ratio,
                                                                          random_state=random_state)

    x_test_norm, x_val_norm, y_test_norm, y_val_norm = train_test_split(x_teval_norm, y_teval_norm,
                                                                        test_size=val_ratio / (test_ratio + val_ratio),
                                                                        random_state=random_state)
    x_test_out, x_val_out, y_test_out, y_val_out = train_test_split(x_teval_out, y_teval_out,
                                                                    test_size=val_ratio / (test_ratio + val_ratio),
                                                                    random_state=random_state)

    x_train = np.concatenate((x_train_norm, x_train_out))
    x_test = np.concatenate((x_test_norm, x_test_out))
    x_val = np.concatenate((x_val_norm, x_val_out))
    y_train = np.concatenate((y_train_norm, y_train_out))
    y_test = np.concatenate((y_test_norm, y_test_out))
    y_val = np.concatenate((y_val_norm, y_val_out))

    from collections import Counter
    print('train counter', Counter(y_train))
    print('val counter  ', Counter(y_val))
    print('test counter ', Counter(y_test))

    # # Scale to range [0,1]
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(x_train)
    x_train = minmax_scaler.transform(x_train)
    x_test = minmax_scaler.transform(x_test)
    x_val = minmax_scaler.transform(x_val)

    return x_train, y_train, x_test, y_test, x_val, y_val


def semi_setting(y_train, n_known_outliers=30):
    """
    default: using ratio to get known outliers, also can using n_known_outliers to get semi-y
    use the first k outlier as known
    :param y_train:
    :param n_known_outliers:
    :return:
    """
    outlier_indices = np.where(y_train == 1)[0]
    n_outliers = len(outlier_indices)
    n_known_outliers = min(n_known_outliers, n_outliers)

    # rng = np.random.RandomState(random_state)
    # known_idx = rng.choice(outlier_indices, n_known_outliers, replace=False)
    known_idx = outlier_indices[:n_known_outliers]

    new_y_train = np.zeros_like(y_train, dtype=int)
    new_y_train[known_idx] = 1
    return new_y_train


def split_build(num):
    matrix = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
              101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    dic = [[]for i in range(201)]
    for i in matrix:
        for j in matrix:
            if i != j and i+j <= 200:
                # dic[i+j].append([i,j])
                if len(dic[i+j]) == 0:
                    dic[i+j] = [i, j]
                elif abs(dic[i+j][0] - dic[i+j][1]) > abs(i-j):
                    dic[i+j] = [i, j]
                else:
                    pass
    print(dic)
    print(dic[num])


def split(num):
    dic = [[], [], [], [], [], [], [], [], [3, 5], [], [3, 7], [], [5, 7], [], [3, 11], [], [5, 11], [], [7, 11], [], [7, 13],
     [], [5, 17], [], [11, 13], [], [7, 19], [], [11, 17], [], [13, 17], [], [13, 19], [], [11, 23], [], [17, 19], [],
     [7, 31], [], [17, 23], [], [19, 23], [], [13, 31], [], [17, 29], [], [19, 29], [], [19, 31], [], [23, 29], [],
     [23, 31], [], [19, 37], [], [17, 41], [], [29, 31], [], [19, 43], [], [23, 41], [], [29, 37], [], [31, 37], [],
     [29, 41], [], [31, 41], [], [31, 43], [], [29, 47], [], [37, 41], [], [37, 43], [], [29, 53], [], [41, 43], [],
     [19, 67], [], [41, 47], [], [43, 47], [], [31, 61], [], [41, 53], [], [43, 53], [], [37, 61], [], [47, 53], [],
     [43, 59], [], [43, 61], [], [47, 59], [], [47, 61], [], [43, 67], [], [53, 59], [], [53, 61], [], [43, 73], [],
     [47, 71], [], [59, 61], [], [43, 79], [], [53, 71], [], [59, 67], [], [61, 67], [], [59, 71], [], [61, 71], [],
     [61, 73], [], [53, 83], [], [67, 71], [], [67, 73], [], [59, 83], [], [71, 73], [], [67, 79], [], [59, 89], [],
     [71, 79], [], [73, 79], [], [71, 83], [], [73, 83], [], [61, 97], [], [71, 89], [], [79, 83], [], [67, 97], [],
     [59, 107], [], [79, 89], [], [73, 97], [], [83, 89], [], [73, 101], [], [79, 97], [], [71, 107], [], [83, 97], [],
     [79, 103], [], [83, 101], [], [89, 97], [], [79, 109], [], [89, 101], [], [89, 103], [], [67, 127], [], [89, 107],
     [], [97, 101], [], [97, 103]]
    return dic[num]


def to_csv(lst):
    df = pd.DataFrame()
    for l in lst:
        df = df.append(pd.DataFrame(l.T).transpose())
    return df


split(30)
