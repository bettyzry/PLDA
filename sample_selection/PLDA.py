from collections import namedtuple, deque
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os

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

from sklearn.preprocessing import normalize
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

Transition = namedtuple('Transition',
                        ('data', 'action', 'next_data', 'reward', 'state_t', 'next_state_t'))
# self.memory.push(data, torch.tensor([[action]], device=self.device), next_data, reward, state_t, next_state_t)


class PLDA():
    """
    PLDA agent that encapsulates the training and testing of the DQN
    """

    def __init__(self, env, rate=0.1, device='cpu',
                 n_episodes=4, steps_per_episode=500, max_memory=10000, eps_max=1, eps_min=0.1,
                 eps_decay=5000, hidden_size=10, learning_rate=0.25e-4, momentum=0.95,
                 min_squared_gradient=0.1, warmup_steps=1, gamma=0.1, batch_size=64,
                 target_update=2000, theta_update=100, weight_decay=1e-3, a=0.5):
        """
        Initialize the DPLAN agent
        :param env: the environment
        :param validation_set: the validation set
        :param test_set: the test set
        :param destination_path: the path where to save the model
        :param device: the device to use for training
        """
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
        # not sure if this works
        # self.policy_net._initialize_weights()
        self.target_net = DQN(self.n_feature, self.env.clf.seq_len, self.hidden_size, self.n_actions, device=self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())       # 加载存储好的网络
        # set target net weights to 0
        with torch.no_grad():
            for param in self.target_net.parameters():
                param.zero_()

        # setting up the environment's DQN
        self.env.DQN = self.policy_net
        # setting up the environment's intrinsic reward as function of netwo rk's theta_e (i.e. the hidden layer)
        # self.intrinsic_rewards = DQN_iforest(self.x_tensor, self.policy_net)            # 计算不同x的异常分数，即他们的孤立性
        # self.i = np.percentile(self.intrinsic_rewards, self.rate)

        # setting the rmsprop optimizer
        self.optimizer = optim.RMSprop(                                      # 优化器
            self.policy_net.parameters(),
            lr=self.LR,
            momentum=self.momentum,
            eps=self.min_squared_gradient,
            weight_decay=self.weight_decay
        )

    def select_action(self, data, steps_done):
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
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # .max(1)[1]：返回最大的索引
                # .view(1, 1): 变成1*1的格式
                # return self.policy_net(data).max(1)[1].view(1, 1)
                return self.policy_net(data).argmax().view(1, 1)
        else:
            return torch.tensor([[random.randint(0, self.env.action_space-1)]], device=self.device, dtype=torch.long)

    def init_model(self):
        batch_size = 16
        train_dataset = TensorDataset(torch.Tensor(self.env.train_seqs), torch.Tensor(self.reward))

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True)
        # optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=0.001)
        for e in range(20):
            losslist = []
            self.policy_net.zero_grad()
            for state_batch, y_batch in train_loader:
                state_action_values = self.policy_net(state_batch)
                # loss = My_loss()(state_action_values, y_batch)
                loss = nn.MSELoss(reduction='mean')(state_action_values, y_batch)
                # loss = nn.SmoothL1Loss(reduction='mean')(state_action_values, y_batch)
                # optimizer.zero_grad()
                loss.backward()
                # In-place gradient clipping
                # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
                self.optimizer.step()
                losslist.append(loss.cpu().detach().numpy())
            print(e+1, np.mean(losslist))
        policy_net_state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(policy_net_state_dict)

    def optimize_model(self):
        """
        Optimize the model using the replay memory
        """
        # self.memory.push(data, torch.tensor([[action]], device=self.device), next_data, reward, state_t, next_state_t)
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))  # 对应index的数据放到了一起，action，next_state等在一起

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_data)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_data
                                           if s is not None])
        state_batch = torch.cat(batch.data)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # 把一个batch的当前状态输入进去，得到预测的所有Q，并获得对应action的Q，即Q(st, at)
        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)                  # 实际的Q
        # gather:从原tensor中获取指定dim和指定index的数据

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # max(1): 返回A每一行最大值组成的一维数组
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]    # 预测的Q
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

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

    def OD_fit(self):
        self.env.clf.training_prepare(self.env.train_seqs, y=self.env.train_label)

        self.env.clf.net.train()
        for epoch in range(self.env.clf.epochs):
            t1 = time.time()
            loss = self.env.clf.training(epoch)
            self.SS_fit(epoch)
            self.sample_selection(epoch)

            print(f'epoch{epoch + 1:3d}, '
                  f'training loss: {loss:.6f}, '
                  f'time: {time.time() - t1:.1f}s')
        return

    def warmup_steps(self):
        """
        Implement the warmup steps to fill the replay memory using random actions
        """
        for _ in range(self.num_warmup_steps):
            state_a, state_t = self.env.reset_state()
            data = torch.tensor(self.env.train_seqs[state_t, :], dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in range(self.steps_per_episode):
                action = np.random.randint(0, self.n_actions)           # 随机挑选一个行动
                next_state_a, reward = self.env.step(action, state_a, state_t)
                reward = get_total_reward(action, reward, self.losses, state_t, a=self.a)
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                next_data = torch.tensor(self.env.x[next_state_a: next_state_a+self.env.clf.seq_len], dtype=torch.float32,
                                           device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(data, torch.tensor([[action]], device=self.device), next_data, reward, state_t, next_state_a)
                data = next_data
                state_a = next_state_a
                state_t = self.env.from_sa2st(next_state_a)

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
        self.warmup_steps()
        self.reset_counters()

        target_reward = pd.DataFrame()

        target_reward['0'] = self.a * self.losses + (1 - self.a) * (1 - self.env.param)
        target_reward['1'] = self.a * (1 - self.losses) + (1 - self.a) * (1 - self.env.param)
        target_reward['2'] = self.a * self.losses + (1 - self.a) * self.env.param

        self.reward = target_reward.values
        self.init_model()

        for i_episode in range(self.num_episodes):
            # self.policy_net.eval()
            # train_seqs = torch.tensor(self.env.train_seqs, dtype=torch.float32, device=self.device)
            # dis = self.policy_net(train_seqs).detach().cpu().numpy()
            self.policy_net.train()
            # Initialize the environment and get it's state
            reward_history = []

            state_a, state_t = self.env.reset_state()
            data = torch.tensor(self.env.train_seqs[state_t, :], dtype=torch.float32, device=self.device).unsqueeze(
                    0)
            for t in range(self.steps_per_episode):
                self.num_steps_done += 1

                # select_action encapsulates the epsilon-greedy policy
                action = self.select_action(data, self.num_steps_done)

                next_state_a, p_reward = self.env.step(action.item(), state_a, state_t)
                # states.append((self.env.x[observation,:],action.item()))

                # reward = get_total_reward(action, reward, self.intrinsic_rewards, state_a, e=self.env.e, i=self.i, a=self.a)
                reward = get_total_reward(action, p_reward, self.losses, state_t)

                reward_history.append(reward)
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                next_data = torch.tensor(self.env.x[next_state_a: next_state_a+self.env.clf.seq_len], dtype=torch.float32,
                                           device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(data, torch.tensor([[action]], device=self.device), next_data, reward, state_t, next_state_a)

                # Move to the next state
                data = next_data
                state_a = next_state_a
                state_t = self.env.from_sa2st(next_state_a)

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                    # update the target network
                if self.num_steps_done % self.target_update == 0:
                    policy_net_state_dict = self.policy_net.state_dict()
                    self.target_net.load_state_dict(policy_net_state_dict)

            # because the theta^e update is equal to the duration of the episode we can update the theta^e here
            self.episodes_total_reward.append(sum(reward_history))
            # print the results at the end of the episode
            avg_reward = np.average(reward_history)
            print('Episode: {} \t Steps: {} \t Average episode Reward: {}'.format(i_episode, t + 1, avg_reward))
        print('Complete')

    def sample_selection(self, epoch):
        self.policy_net.eval()
        train_seqs = torch.tensor(self.env.train_seqs, dtype=torch.float32, device=self.device)
        expected_future_total_reward = self.policy_net(train_seqs).detach().cpu().numpy()
        actions = np.argmax(expected_future_total_reward, axis=1)
        self.policy_net.train()

        add_index = np.where(actions == 0)[0]
        add_seq_starts = self.env.train_start[add_index]
        add_seq_starts = np.sort(add_seq_starts)

        delet_index = np.where(actions == 2)[0]
        delet_seq_starts = self.env.train_start[delet_index]
        delet_seq_starts = np.sort(delet_seq_starts)
        self.env.train_start = np.delete(self.env.train_start, delet_seq_starts, axis=0)

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

        y_seqs = get_sub_seqs_label2(self.env.y, seq_starts=self.env.train_start,
                                     seq_len=self.env.seq_len) if self.env.y is not None else None
        self.env.train_label = y_seqs
        self.env.clf.trainsets['seqstarts' + str(epoch)] = self.env.train_start
        if y_seqs is not None:
            self.env.clf.trainsets['yseq' + str(epoch)] = y_seqs
        self.env.clf.trainsets['dis' + str(epoch)] = expected_future_total_reward
        return
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        return 1 - cos_sim(x, y)


if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3], [2, 3, 4]])
    b = torch.tensor([[5, 7, 8], [5, 4, 6]])
    # a = torch.tensor([1, 2], dtype=float)
    # b = torch.tensor([5, 7], dtype=float)

    cos_sim = nn.CosineEmbeddingLoss(reduction='mean')
    sim = cos_sim(a, b, torch.ones([len(a)]))
    print(sim) # tensor(0.9878, dtype=torch.float64)
