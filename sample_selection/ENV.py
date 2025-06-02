import random
import numpy as np
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from torch.utils.data import DataLoader


class ADEnv():
    """
    Customized environment for anomaly detection
    """

    def __init__(self, dataset: np.ndarray, clf, device='cpu', y=None, seq_len=30, stride=30,
                 num_sample=1000):
        """
        Initialize anomaly environment for DPLAN algorithm.
        :param num_sample: Number of sampling for the generator g_u
        """
        self.param = None
        self.e = 0.5
        self.device = device
        self.seq_len = seq_len
        self.stride = stride

        self.clf = clf

        # Dataset infos
        valid_num = int(len(dataset) * self.clf.valid_rate)
        valid_x = dataset[-valid_num:]
        dataset = dataset[:-valid_num]
        valid_seqs = get_sub_seqs(valid_x, seq_len=self.seq_len, stride=self.stride)
        self.clf.valid_loader = DataLoader(valid_seqs, batch_size=self.clf.batch_size, shuffle=True, drop_last=False)

        self.n_samples, self.n_feature = dataset.shape
        self.x = dataset  # 原始数据
        self.train_start = np.arange(0, self.n_samples - seq_len + 1, stride)  # 训练集序列的索引标签（初始化为无重复的序列开头）
        self.train_seqs = np.array([self.x[i:i + self.seq_len] for i in self.train_start])  # 当前的训练集序列（初始化为无重复的序列）

        self.y = y
        self.train_label = get_sub_seqs_label(y, seq_len=self.seq_len, stride=stride) if y is not None else None

        # hyper parameter
        # 贪婪算法选择行动时，为了提高效率进行了采样。如果训练集少于numsample，则使用全集
        self.num_sample = min(num_sample, len(self.train_start))

        # action space: # 0扩展，1保持，2删除
        # self.action_space = spaces.Discrete(3)  # 0扩展，1保持，2删除.0扩展1删除
        self.action_space = 3  # 0扩展，1保持，2删除.0扩展1删除

        # initial state
        self.state_a = None  # state in all data 当前状态在全部数据中的索引值
        self.state_t = None  # state in training data当前状态在训练集里的索引值
        self.DQN = None
        self.loss = []

        self.init_clf()

    def init_clf(self):
        self.clf.trainsets['seqstarts0'] = self.train_start
        self.clf.n_samples, self.clf.n_features = self.train_seqs.shape[0], self.train_seqs.shape[2]
        if self.y is not None:
            self.clf.trainsets['yseq0'] = self.train_label

    def from_sa2st(self, sa):
        st = np.where(self.train_start == sa)[0][0]
        return st

    def from_st2sa(self, st):
        sa = self.train_start[st]
        return sa

    def generater(self, action, s_a, *args, **kwargs):
        if action == 0:    # expand
            next_sa = []
            if s_a - self.clf.split[0] >= 0:
                next_sa.append(s_a - self.clf.split[0])
            if s_a + self.clf.split[1] < len(self.x) - self.seq_len + 1:
                next_sa.append(s_a + self.clf.split[1])
            if s_a - self.clf.split[1] > 0:
                next_sa.append(s_a - self.clf.split[1])
            if s_a + self.clf.split[0] <= len(self.x) - self.seq_len + 1:
                next_sa.append(s_a + self.clf.split[0])
            index = np.random.choice(next_sa)
            # index = next_sa
        elif action == 1:   # save
            # index = np.random.choice(self.train_start)
            index = s_a
            # index = s_a
        else:       # delete
            # index = np.random.choice(self.train_start)
            index = s_a
        return index

    def generater_r(self, *args, **kwargs):  # 删除
        # sampling function for D_a
        index = np.random.choice(self.train_start)
        return index

    def generate_a(self, action, s_a):
        # 对状态s_t要执行action操作
        # sampling function for D_u
        # 在所有的训练集中随机采num_S个，S为采样数据in all data的索引号               # 为了效率
        S = np.random.choice(range(len(self.train_seqs)), self.num_sample)
        # calculate distance in the space of last hidden layer of DQN
        # all_x = self.train_seqs[S].append(self.x[s_a: s_a + self.seq_len])  # 提取全部采样点+当前位置，对应的数据的值
        all_x = np.concatenate((self.train_seqs[S], [self.x[s_a: s_a + self.seq_len]]), axis=0)

        all_dqn_s = self.DQN.get_latent(all_x)  # 提取数据的表征
        all_dqn_s = all_dqn_s.cpu().detach().numpy()
        dqn_s = all_dqn_s[:-1]
        dqn_st = all_dqn_s[-1]

        dist = np.linalg.norm(dqn_s - dqn_st, axis=1)  # 采样数据点与当前状态st的距离
        dist = np.average(dist, axis=1)

        # 0扩展，1保持，2删除
        if action == 0:  # 扩展该数据
            loc = np.argmin(dist)  # 找最像的
        elif action == 1:
            loc = np.argmax(dist)    # 找不最像的
        else:  # action == 2 # 删除该数据
            loc = np.argmin(dist)  # 找最像的
        state_t = S[loc]
        state_a = self.from_st2sa(state_t)
        return state_a       # 返回state_a

    def step(self, action, state_a, state_t):
        g = np.random.choice([self.generater_r, self.generate_a], p=[0.5, 0.5])
        state_a1 = g(action, state_a)  # 找到下一个要探索的点
        reward = self.param[state_t]           # 当前的行为能获得多大的收益
        return state_a1, reward

    def reset_state(self):
        # reset the status of environment
        # the first observation is uniformly sampled from the D_u
        self.state_a = np.random.choice(self.train_start)
        self.state_t = self.from_sa2st(self.state_a)

        return self.state_a, self.state_t
