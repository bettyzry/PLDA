import numpy as np
from sklearn import metrics
import random
import matplotlib.pyplot as plt


def get_sub_seqs(x_arr, seq_len=100, stride=1):
    """

    Parameters
    ----------
    x_arr: np.array, required
        input original data with shape [time_length, channels]

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences

    Returns
    -------
    x_seqs: np.array
        Split sub-sequences of input time-series data
    """

    seq_starts = np.arange(0, x_arr.shape[0] - seq_len + 1, stride)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])

    return x_seqs


def get_sub_seqs_label(y, seq_len=100, stride=1):
    """

    Parameters
    ----------
    y: np.array, required
        data labels

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences

    Returns
    -------
    y_seqs: np.array
        Split label of each sequence
    """
    seq_starts = np.arange(0, y.shape[0] - seq_len + 1, stride)
    ys = np.array([y[i:i + seq_len] for i in seq_starts])
    y_binary = np.zeros(len(ys))
    for ii, y in enumerate(ys):
        if 1 in y:
            y_binary[ii] = 1
        elif 2 in y:
            y_binary[ii] = 2
    # y_binary = np.max(ys, axis=1)
    # y_binary = np.zeros_like(y)
    # y_binary[np.where(y == 1)[0]] = 1
    # y_binary[np.where(y != 0)[0]] = 1
    # y_binary[np.where(y >= seq_len/3)[0]] = 2
    # y_binary[np.where(y >= 2*seq_len/3)[0]] = 3
    # y_binary[np.where(y == seq_len)[0]] = 4
    # y_binary[np.where(y == 0)[0]] = 0
    return y_binary


def get_sub_seqs_label_old(y, seq_len=100, stride=1):
    """

    Parameters
    ----------
    y: np.array, required
        data labels

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences

    Returns
    -------
    y_seqs: np.array
        Split label of each sequence
    """
    seq_starts = np.arange(0, y.shape[0] - seq_len + 1, stride)
    ys = np.array([y[i:i + seq_len] for i in seq_starts])
    y = np.sum(ys, axis=1)

    y_binary = np.zeros_like(y)
    y_binary[np.where(y != 0)[0]] = 1
    # y_binary[np.where(y != 0)[0]] = 1
    # y_binary[np.where(y >= seq_len/3)[0]] = 2
    # y_binary[np.where(y >= 2*seq_len/3)[0]] = 3
    # y_binary[np.where(y == seq_len)[0]] = 4
    # y_binary[np.where(y == 0)[0]] = 0
    return y_binary


def get_sub_seqs_label2(y, seq_starts, seq_len):
    """

    Parameters
    ----------
    y: np.array, required
        data labels

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences

    Returns
    -------
    y_seqs: np.array
        Split label of each sequence
    """

    ys = np.array([y[i:i + seq_len] for i in seq_starts])
    # yy = np.sum(ys, axis=1)

    y_binary = np.zeros_like(seq_starts)
    for ii, y in enumerate(ys):
        if 1 in y:
            y_binary[ii] = 1
        elif 2 in y:
            y_binary[ii] = 2

    # y_binary = np.zeros_like(y)
    # y_binary[np.where(y != 0)[0]] = 1
    return y_binary


def insert_pollution(train_data, test_data, labels, rate, seq_len):
    test_seq = get_sub_seqs(test_data, seq_len=seq_len, stride=1)
    y_seqs = get_sub_seqs_label(labels, seq_len=seq_len, stride=1)
    oseqs = np.where(y_seqs == 1)[0]
    okinds = len(oseqs)
    datasize = int(len(train_data) / seq_len)
    rate = rate / 100
    onum = int(datasize * rate)
    ostarts = random.sample(range(0, datasize - 1), onum)
    train_labels = np.zeros(len(train_data))
    for ostart in ostarts:
        index = random.randint(0, okinds - 1)
        train_data[ostart * seq_len: (ostart + 1) * seq_len] = test_seq[oseqs[index]]
        train_labels[ostart * seq_len: (ostart + 1) * seq_len] = 1
    plt.plot(train_data[:, 4])
    plt.show()
    return train_data, train_labels


def split_pollution(test_data, labels):
    num = int(0.6 * len(test_data))
    train_data, train_labels = test_data[:num], labels[:num]
    test_data, labels = test_data[num:], labels[num:]
    return train_data, train_labels, test_data, labels


def insert_pollution_new(train_data, test_data, labels, rate):
    if rate == 0:
        train_labels = np.zeros(len(train_data))
        return train_data, train_labels
    # plt.plot(train_data[:, 0])
    # plt.show()
    # 将一个序列的outlier插入
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    splits = np.concatenate([[0], splits])
    is_anomaly = labels[0] == 1
    data_splits = [test_data[sp: splits[ii + 1]] for ii, sp in enumerate(splits[:-1])]
    outliers = [sp for ii, sp in enumerate(data_splits) if ii % 2 != is_anomaly]

    length = len(train_data)
    onum = len(outliers)
    train = np.array(outliers[0])
    i = 0
    start = 0
    rate = rate/100
    k = int(1/rate-1)
    train_labels = np.zeros(length)
    while len(train) < length:
        end = start + len((outliers[i]))*k
        train = np.concatenate([train, outliers[i]], axis=0)
        train = np.concatenate([train, train_data[start+len(outliers[i]): end]], axis=0)
        train_labels[start: start+len((outliers[i]))] = 1
        i += 1
        i %= onum
        start = end
    train_data = train[:length]

    #
    # length = [len(o) for o in outliers]
    # timestamp = int(np.average(length))  # 平均异常长度
    # max_ts = np.max(length)
    # train_labels = np.zeros(len(train_data))
    #
    # Onum_all = int(len(train_data) * 0.8)  # 最多这么多异常点
    # N_all = int(Onum_all / timestamp) + 1  # 最多插入这么多次异常段
    # sep_all = int((len(train_data) - max_ts - timestamp) / N_all)  # 每个异常段的最小间隔长度
    # locs_all = [i for i in range(timestamp, len(train_data) - max_ts, sep_all)]  # 有这些位置可能插入异常
    #
    # rate = rate / 100
    # Onum = int(len(train_data) * rate)
    # N = int(Onum / timestamp) + 1  # 总异常数              # 实际异常段数目
    #
    # def search(l):
    #     newl = [i for i in l]
    #     for start in range(len(l)-1):
    #         for end in range(len(l)-1, start, -1):
    #             k = int((l[start]+l[end])/2)
    #             if k not in newl:
    #                 newl.append(k)
    #     return newl
    #
    # order = [0, len(locs_all) - 1]
    # while len(order) != len(locs_all):
    #     order = search(order)
    #
    # locs = [locs_all[order[i]] for i in range(N)]
    # okinds = len(outliers)
    # count = 0
    # for start in locs:
    #     train_data[start: len(outliers[count]) + start] = outliers[count]
    #     train_labels[start: len(outliers[count]) + start] = 1
    #     count += 1
    #     count = count % okinds

    # plt.plot(train_data[:, 0])
    # plt.plot(train_labels)
    # plt.show()
    #
    # plt.plot(test_data[:, 0])
    # plt.plot(labels)
    # plt.show()
    return train_data, train_labels


def insert_pollution_from_test(test_data, labels, rate):
    rate = rate/100
    # 将一个序列的outlier插入
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    splits = np.concatenate([[0], splits])
    is_anomaly = labels[0] == 1
    data_splits = [test_data[sp: splits[ii+1]] for ii, sp in enumerate(splits[:-1])]
    outliers = [data_splits[ii] for ii, sp in enumerate(data_splits) if ii % 2 != is_anomaly]

    split = 0.6
    train_num = int(len(test_data)*split)
    train_data = test_data[:train_num]
    train_l = labels[:train_num]

    ii = 0
    train_data_o = np.array([])
    train_labels = np.array([])
    train_num_o = train_num
    while len(train_data_o) <= train_num_o:
        N = len(train_data_o)
        No = sum(train_labels)
        if N == 0:
            oindex = random.randint(0, len(outliers)-1)
            train_data_o = outliers[oindex]
            train_labels = np.ones(len(outliers[oindex]))
        elif No/N <= rate:
            oindex = random.randint(0, len(outliers)-1)
            train_data_o = np.insert(train_data_o, N, outliers[oindex], axis=0)
            train_labels = np.concatenate([train_labels, np.ones(len(outliers[oindex]))])
        else:       # 插入异常,把一整个序列装进去
            while train_l[ii % len(train_l)] == 1:
                ii += 1
            train_data_o = np.insert(train_data_o, N, train_data[ii], axis=0)
            train_labels = np.concatenate([train_labels, [0]])
            ii += 1

    test_data = test_data[train_num:]
    labels = labels[train_num:]
    return train_data_o, train_labels, test_data, labels


def insert_pollution_seq(test_data, labels, rate, seq_len):
    # 插入一长序列
    ori_seq = get_sub_seqs(test_data, seq_len=seq_len, stride=1)
    oriy_seq = get_sub_seqs_label2(labels, seq_len=seq_len, stride=1)

    split = 0.6
    train_num = int(len(ori_seq) * split)
    train_seq = ori_seq[:train_num]
    train_l = oriy_seq[:train_num]

    oseqs = np.where(train_l == 1)[0]

    ii = 0
    jj = 0
    train_seq_o = []
    train_labels = []
    train_num_o = train_num
    while len(train_seq_o) <= train_num_o:
        l = random.random()
        if l <= rate:  # 插入异常
            train_seq_o.append(ori_seq[oseqs[jj % len(oseqs)]])
            train_labels.append(1)
            jj += 7
        else:  # 插入正常
            while train_l[ii % len(train_l)] > 0:
                ii += 1
            train_seq_o.append(train_seq[ii % len(train_l)])
            train_labels.append(0)
            ii += 7
    train_seq_o = np.array(train_seq_o)
    train_labels = np.array(train_labels)

    test_data = test_data[train_num:]
    labels = labels[train_num:]
    return train_seq_o, train_labels, test_data, labels
