from collections import namedtuple, deque
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import normalize

import pandas as pd

from sample_selection.ssutil import DQN_iforest, get_total_reward, test_model
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label, get_sub_seqs_label2
from sample_selection.ENV import ADEnv

import torch
import torch.nn as nn
import torch.optim as optim
from sample_selection.DQN import DQN
from sample_selection.RelayMemory import ReplayMemory
from deepod.metrics import ts_metrics, point_adjustment
import time
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

Transition = namedtuple('Transition',
                        ('data', 'action', 'next_data', 'reward', 'state_t', 'next_state_t'))
# self.memory.push(data, torch.tensor([[action]], device=self.device), next_data, reward, state_t, next_state_t)


class QSS():
    """
    DPLAN agent that encapsulates the training and testing of the DQN
    """

    def __init__(self, env, rate=0.1, device='cpu',
                 n_episodes=6, steps_per_episode=2000, max_memory=10000, eps_max=1, eps_min=0.1,
                 eps_decay=5000, hidden_size=10, learning_rate=0.25e-4, momentum=0.95,
                 min_squared_gradient=0.1, warmup_steps=1, gamma=0.1, batch_size=64,
                 target_update=2000, theta_update=100, weight_decay=1e-3, a=0.2):
        """
        Initialize the DPLAN agent
        :param env: the environment
        :param validation_set: the validation set
        :param test_set: the test set
        :param destination_path: the path where to save the model
        :param device: the device to use for training
        """
        self.policy_reward = None
        self.target_reward = None
        self.reward = None
        self.device = device
        self.env = env
        self.rate = (1-rate)*100

        # hyperparameters setup
        self.hidden_size = hidden_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_max
        self.EPS_END = eps_min
        self.EPS_DECAY = eps_decay
        self.LR = learning_rate
        self.momentum = momentum
        self.min_squared_gradient = min_squared_gradient
        self.num_episodes = n_episodes
        self.num_warmup_steps = warmup_steps
        self.steps_per_episode = steps_per_episode
        self.max_memory_size = max_memory
        self.target_update = target_update
        self.theta_update = theta_update
        self.weight_decay = weight_decay

        self.x_tensor = None
        self.n_actions = None
        self.n_feature = None

        # tensor rapresentation of the dataset used in the intrinsic reward
        self.x_tensor = torch.tensor(self.env.train_seqs, dtype=torch.float32, device=self.device)
        #  n actions and n observations
        self.n_actions = self.env.action_space  # 可以执行的行动的数量
        self.n_feature = self.env.n_feature  # 有错？？
        self.reset()

        self.a = a

    def reset(self):
        self.reset_counters()
        self.reset_memory()
        self.reset_nets()

    def reset_memory(self):
        self.memory = ReplayMemory(self.max_memory_size)

    def reset_counters(self):
        # training counters and utils
        self.num_steps_done = 0
        self.episodes_total_reward = []

    def reset_nets(self):
        # net definition
        self.policy_net = DQN(self.n_feature, self.env.clf.seq_len, self.hidden_size, self.n_actions, device=self.device).to(self.device)
        # setting up the environment's DQN
        self.env.DQN = self.policy_net

    def select_action(self, state_t, steps_done):
        """
        Select an action using the epsilon-greedy policy
        :param state: the current state
        :param steps_done: the number of steps done
        :return: the action
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
             with torch.no_grad():
                return np.argmax(self.reward[state_t])
        else:
            return random.randint(0, self.env.action_space-1)

    def get_param(self):
        self.env.clf.net.eval()
        param_metrics = []
        losses = []
        for ii, batch_x in enumerate(self.env.clf.train_loader):
            metric, loss = self.env.clf.get_importance_dL(batch_x)
            if ii == 0:
                param_metrics = metric
                losses = loss
            else:
                param_metrics = np.concatenate((param_metrics, metric), axis=0)
                losses = np.concatenate((losses, loss), axis=0)

        _range = np.max(losses) - np.min(losses)
        losses = (losses - np.min(losses)) / _range
        self.losses = losses
        self.l = np.percentile(losses, self.rate)

        mean = np.mean(param_metrics, axis=0)
        param_metrics = np.divide(param_metrics, mean, out=np.zeros_like(param_metrics, dtype=np.float64), where=mean != 0)
        param_metrics = normalize(param_metrics, axis=1, norm='l2')  # 对metric按行进行归一化

        importance = np.mean(param_metrics, axis=0)
        param = np.linalg.norm(importance - param_metrics, axis=1, ord=np.Inf)

        param = (param - np.min(param)) / (np.max(param) - np.min(param))
        self.env.param = param
        self.env.e = np.percentile(param, self.rate)

    def Init_params(self):
        """
        Implement the warmup steps to fill the replay memory using random actions
        核心目的是获得memory
        self.memory.push(data, torch.tensor([[action]], device=self.device), next_data, reward, state_t, next_state_t)
        """
        # 确定 param_musk，params
        self.env.clf.init_param()
        self.env.clf.init_param_musk()

    def OD_fit(self, Xtest=None, Ytest=None):
        self.env.clf.training_prepare(self.env.train_seqs, y=self.env.train_label)

        self.env.clf.net.train()
        for epoch in range(self.env.clf.epochs):
            t1 = time.time()
            loss = self.env.clf.training(epoch)
            if epoch < 10:
                self.SS_fit(epoch)
                self.sample_selection(epoch)

            print(f'epoch{epoch + 1:3d}, '
                  f'training loss: {loss:.6f}, '
                  f'time: {time.time() - t1:.1f}s')

            if Xtest is not None and Ytest is not None:
                self.env.clf.net.eval()
                scores = self.env.clf.decision_function(Xtest)
                eval_metrics = ts_metrics(Ytest, scores)
                adj_eval_metrics = ts_metrics(Ytest, point_adjustment(Ytest, scores))
                result = [eval_metrics[0], eval_metrics[1], eval_metrics[2], adj_eval_metrics[0], adj_eval_metrics[1],
                          adj_eval_metrics[2]]
                print(result)
                self.env.clf.result_detail.append(result)
                self.env.clf.net.train()
        return

    def SS_fit(self, epoch):
        """
        Fit the model according to the dataset and hyperparameters. The best model is obtained by using
        the best auc-pr score with the validation set.
        :param reset_nets: whether to reset the networks
        """
        # 初始化参数、关键参数
        if epoch == 0:
            self.Init_params()
        self.get_param()
        self.reset_counters()

        reward = pd.DataFrame()
        reward['0'] = self.a * self.losses + (1 - self.a) * (1 - self.env.param)
        reward['1'] = self.a * (1 - self.losses) + (1 - self.a) * (1 - self.env.param)
        reward['2'] = self.a * self.losses + (1 - self.a) * self.env.param

        self.reward = reward.values
        self.policy_reward = reward.values
        self.target_reward = self.policy_reward
        index = np.argmax(self.target_reward, axis=1)

        for i_episode in range(self.num_episodes):

            actions = np.argmax(self.target_reward, axis=1)
            # Initialize the environment and get it's state
            reward_history = []

            state_a, state_t = self.env.reset_state()
            for t in range(self.steps_per_episode):
                self.num_steps_done += 1
                action = self.select_action(state_t, self.num_steps_done)
                next_state_a, _ = self.env.step(action, state_a, state_t)
                next_state_t = self.env.from_sa2st(next_state_a)
                # reward = get_total_reward(action, reward, self.losses, state_t, d=self.env.e, o=self.l, a=self.a)
                reward = self.reward[state_t][action]
                self.policy_reward[state_t][action] = self.reward[state_t][action] + self.GAMMA*np.max(self.target_reward[next_state_t])
                reward_history.append(reward)
                state_a = next_state_a
                state_t = self.env.from_sa2st(next_state_a)
            # because the theta^e update is equal to the duration of the episode we can update the theta^e here
            self.episodes_total_reward.append(sum(reward_history))

            if self.num_steps_done % self.target_update == 0:
                self.target_reward = self.policy_reward
            # print the results at the end of the episode
            avg_reward = np.average(reward_history)
            print('Episode: {} \t Steps: {} \t Average episode Reward: {}'.format(i_episode, t + 1, avg_reward))
        print('Complete')

    def sample_selection(self, epoch):
        actions = np.argmax(self.target_reward, axis=1)

        add_index = np.where(actions == 0)[0]
        add_seq_starts = self.env.train_start[add_index]
        add_seq_starts = np.sort(add_seq_starts)

        delet_index = np.where(actions == 2)[0]
        self.env.train_start = np.delete(self.env.train_start, delet_index, axis=0)

        for add_seq_start in add_seq_starts:
            if add_seq_start - self.env.clf.split[0] >= 0:
                self.env.train_start = np.append(self.env.train_start, add_seq_start - self.env.clf.split[0])
            if add_seq_start + self.env.clf.split[1] < len(self.env.x) - self.env.clf.seq_len + 1:
                self.env.train_start = np.append(self.env.train_start, add_seq_start + self.env.clf.split[1])
            if add_seq_start - self.env.clf.split[1] >= 0:
                self.env.train_start = np.append(self.env.train_start, add_seq_start - self.env.clf.split[1])
            if add_seq_start + self.env.clf.split[0] < len(self.env.x) - self.env.clf.seq_len + 1:
                self.env.train_start = np.append(self.env.train_start, add_seq_start + self.env.clf.split[0])

        self.env.train_start = np.sort(self.env.train_start)
        self.env.train_start = np.unique(self.env.train_start, axis=0)

        self.env.train_seqs = np.array([self.env.x[i:i + self.env.seq_len] for i in self.env.train_start])  # 添加划分的数据
        self.env.clf.n_samples = len(self.env.train_seqs)

        self.env.clf.train_data = np.array([self.env.x[i:i + self.env.seq_len] for i in self.env.train_start])  # 添加划分的数据
        self.env.clf.train_loader = DataLoader(self.env.clf.train_data, batch_size=self.env.clf.batch_size, drop_last=False,
                                       shuffle=True, pin_memory=True)

        y_seqs = get_sub_seqs_label2(self.env.y, seq_starts=self.env.train_start,
                                     seq_len=self.env.seq_len) if self.env.y is not None else None
        self.env.train_label = y_seqs
        return