import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from deepod.core.base_model import BaseDeepAD
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label, get_sub_seqs_label2
from deepod.metrics import ts_metrics, point_adjustment
import time
import warnings
from sklearn.ensemble import IsolationForest

torch.autograd.set_detect_anomaly = True
"""Adapted from https://github.com/KDD-OpenSource/DeepADoTS (MIT License)"""


class LSTMED(BaseDeepAD):
    def __init__(self, epochs=10, batch_size=20, lr=0.001,
                 hidden_size=5, seq_len=30, stride=1, train_val_percentage=0.25, epoch_steps=-1,
                 n_layers=(1, 1), use_bias=(True, True), dropout=(0, 0), prt_steps=10, verbose=2,
                 random_state=None, device='cuda', gpu=0, patience=5,
                 pca_comp=None, last_t_only=True, explained_var=None, set_hid_eq_pca=False, a=0.5):
        """
        If set_hid_eq_pca is True and one of pca_comp or explained_var is true, then hidden_size is ignored.
        Hidden size is set equal to number of pca components obtained.
        """
        super(LSTMED, self).__init__(
            model_name='LSTMED', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state, a=a
        )

        if set_hid_eq_pca:
            if pca_comp is not None or explained_var is not None:
                warnings.warn(
                    "set_hid_eq_pca is True and pca params provided. So hidden_size argument will be ignored. "
                    "Hidden size will be set equal to number of pca components")
                self.hidden_size = None
            else:
                set_hid_eq_pca = False
                warnings.warn("set_hid_eq_pca is True but pca params not provided. "
                              "So hidden_size argument will be used.")
                if hidden_size is None:
                    hidden_size = 5
                    warnings.warn("Hidden size was specified as None. Will use default value 5")

        self.num_epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.hidden_size = hidden_size
        self.sequence_length = seq_len
        self.stride = stride
        self.train_val_percentage = train_val_percentage
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.model = None
        self.torch_save = True
        self.gpu = gpu
        # self.scaler = None
        # self.pca = None
        self.pca_comp = pca_comp
        self.explained_var = explained_var
        self.last_t_only = last_t_only
        self.set_hid_eq_pca = set_hid_eq_pca

    def fit(self, X, y=None, Xtest=None, Ytest=None, X_seqs=None, y_seqs=None):
        self.fit_RODA(X, y, Xtest, Ytest, X_seqs, y_seqs)
        return

    def training(self, epoch):
        total_loss = 0
        self.net.zero_grad()
        for batch_x in self.train_loader:
            batch_x = batch_x.float().to(self.device)
            loss = self.training_forward(batch_x, self.net, self.criterion)
            loss.backward()
            if self.sample_selection == 5:  # ICML21
                to_concat_g = []
                to_concat_v = []
                clip = 0.2
                for name, param in self.net.named_parameters():
                    to_concat_g.append(param.grad.data.view(-1))
                    to_concat_v.append(param.data.view(-1))
                all_g = torch.cat(to_concat_g)
                all_v = torch.cat(to_concat_v)
                metric = torch.abs(all_g * all_v)
                num_params = all_v.size(0)
                nz = int(clip * num_params)
                top_values, _ = torch.topk(metric, nz)
                thresh = top_values[-1]

                for name, param in self.net.named_parameters():
                    mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
                    mask = mask * clip
                    param.grad.data = mask * param.grad.data

            self.optimizer.step()
            self.net.zero_grad()
            total_loss += loss.item()

        self.epoch_update()
        return total_loss

    def training_prepare(self, X, y):
        self.net = LSTMEDModule(self.n_features, self.seq_len, self.hidden_size, self.n_layers, self.use_bias, self.dropout,
                                device=self.device).to(self.device)
        self.train_loader = DataLoader(X, batch_size=self.batch_size,
                                       shuffle=True, pin_memory=True, drop_last=False)
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def training_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        output = net(batch_x)
        loss = torch.nn.MSELoss(reduction="mean")(output[:, -1], batch_x[:, -1])
        # loss = self.criterion(torch.sum(output[0][:, -1]), torch.sum(batch_x[:, -1]))
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        output = net(batch_x)
        error = torch.nn.MSELoss(reduction='none')(output[:, -1], batch_x[:, -1])
        # error = torch.nn.MSELoss(reduction='none')(torch.sum(output[0][:, -1]), torch.sum(batch_x[:, -1]))
        error = error.mean(axis=1)
        return output, error

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def do_sample_selection(self, epoch):
        if self.sample_selection == 0:          # 无操作
            # self.net.eval()  # 使用完全的网络来计算
            train_loss_now = np.array([])
            for ii, batch_x in enumerate(self.train_loader):
                _, losses = self.inference_forward(batch_x, self.net, self.criterion)
                train_loss_now = np.concatenate([train_loss_now, losses.cpu().detach().numpy()])
            self.trainsets['loss' + str(epoch)] = train_loss_now
            self.net.train()  # 使用完全的网络来计算

        elif self.sample_selection == 5:        # ICLR21
            pass

        elif self.sample_selection == 6:        # arxiv22
            if len(self.train_data) <= int(self.n_samples*0.3):
                return

            # 计算损失值
            # self.net.eval()                     # 使用完全的网络来计算
            train_loss_now = np.array([])
            for batch_x in self.train_loader:
                _, error = self.inference_forward(batch_x, self.net, self.criterion)
                train_loss_now = np.concatenate([train_loss_now, error.cpu().detach().numpy()])
            self.net.train()  # 使用完全的网络来计算
            self.train_loss_now = train_loss_now

            if self.train_loss_past is None:    # 第一轮迭代，直接返回
                self.train_loss_past = self.train_loss_now
                return

            save_num = max(int(self.save_rate * len(self.train_data)), int(self.n_samples*0.3))
            if epoch == 0:
                index1 = np.array([])
            else:
                delta = abs(self.train_loss_now - self.train_loss_past)
                index1 = delta.argsort()[:save_num]
            index2 = train_loss_now.argsort()[:save_num]

            index = np.concatenate([index1, index2])
            index = np.sort(index)
            index = np.unique(index, axis=0)

            self.train_data = self.train_data[np.sort(index)]
            self.train_loss_past = self.train_loss_now[np.sort(index)]
            self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=False,
                                      shuffle=True, pin_memory=True)

            self.trainsets['dis' + str(epoch)] = train_loss_now
            if epoch > 0 and self.train_label is not None:
                self.trainsets['yseq' + str(epoch)] = self.trainsets['yseq' + str(epoch-1)][np.sort(index)]

        elif self.sample_selection == 7:        # 我的方法
            if epoch >= 10:
                return

            dis = np.zeros(len(self.train_data))
            importance = None
            metrics = np.array([])
            losses = np.array([])
            # self.net.eval()
            self.init_param()
            for ii, batch_x in enumerate(self.train_loader):
                # metric = self.get_importance_dL(batch_x)
                metric, loss = self.get_importance_ICLR21(batch_x)
                # metric = self.get_importance_ICML17(batch_x)       # 巨慢

                if epoch == 0:
                    if ii == 0:
                        importance = np.sum(metric, axis=0)
                    else:
                        importance = importance + np.sum(metric, axis=0)
                else:
                    if ii == 0:
                        metrics = metric
                        losses = loss
                    else:
                        metrics = np.concatenate((metrics, metric), axis=0)
                        losses = np.concatenate((losses, loss), axis=0)

                #
                # if epoch == 0:      # 只累计importance
                #     pass
                # else:
                #     dis[ii*self.batch_size: (ii+1)*self.batch_size] = np.linalg.norm(self.true_key_param-metric, axis=1) # L2范数
            self.net.train()

            if epoch == 0:
                self.param_musk = np.sort(importance.argsort()[::-1][:1000])     # 前10000个最重要的数据
                # self.true_key_param = importance[self.param_musk] / len(self.train_data)
            else:
                # self.iforest.fit(metric_torch)
                # dis = -self.iforest.decision_function(metric_torch)
                iforest = IsolationForest().fit(metrics)
                dis = -iforest.decision_function(metrics)
                _range = np.max(dis) - np.min(dis)
                dis = (dis - np.min(dis)) / _range

                _range = np.max(losses) - np.min(losses)
                losses = (losses - np.min(losses)) / _range

                # importance = np.sum(metrics, axis=0) / len(self.train_data)
                # self.true_key_param = importance
                # dis = np.linalg.norm(importance - metrics, axis=1, ord=np.Inf)

                # metrics = np.insert(metrics, 0, self.train_label, axis=1)
                # df = pd.DataFrame(metrics)
                # df.to_csv('@g_detail/TcnED-DASADS-ICLR21-ori/%s.csv' % str(epoch))
                reward = pd.DataFrame()
                d = np.percentile(dis, self.rate)
                l = np.percentile(losses, self.rate)
                reward['0'] = self.a * (2 * d - dis) + (1 - self.a) * losses
                reward['1'] = self.a * (2 * d - dis) + (1 - self.a) * (
                            2 * l - losses)
                reward['2'] = self.a * dis + (1 - self.a) * l

                actions = np.argmax(reward.values, axis=1)

                add_index = np.where(actions == 0)[0]
                add_seq_starts = self.seq_starts[add_index]
                add_seq_starts = np.sort(add_seq_starts)

                delet_index = np.where(actions == 2)[0]
                # delet_seq_starts = self.seq_starts[delet_index]
                # delet_seq_starts = np.sort(delet_seq_starts)
                self.seq_starts = np.delete(self.seq_starts, delet_index, axis=0)
                #
                # add_num = min(int(self.add_rate * len(self.train_data)), int(self.n_samples * 0.4))  # 每次添加的数据量
                # index = dis.argsort()[:add_num]  # 扩展距离最小的40%
                # index = np.sort(index)
                # add_seq_starts = self.seq_starts[index]
                #
                # delet_num = min(int(self.del_rate * len(self.train_data)), int(self.n_samples * 0.2))  # 每次添加的数据量
                # index = dis.argsort()[::-1][:delet_num]  # 删除距离最大的20%
                # index = np.sort(index)
                # self.seq_starts = np.delete(self.seq_starts, index, axis=0)

                for add_seq_start in add_seq_starts:
                    if add_seq_start - self.split[0] >= 0:
                        self.seq_starts = np.append(self.seq_starts, add_seq_start - self.split[0])
                    if add_seq_start + self.split[1] < len(self.ori_data) - self.seq_len + 1:
                        self.seq_starts = np.append(self.seq_starts, add_seq_start + self.split[1])
                    if add_seq_start - self.split[1] > 0:
                        self.seq_starts = np.append(self.seq_starts, add_seq_start - self.split[1])
                    if add_seq_start + self.split[0] <= len(self.ori_data) - self.seq_len + 1:
                        self.seq_starts = np.append(self.seq_starts, add_seq_start + self.split[0])

                self.seq_starts = np.sort(self.seq_starts)
                self.seq_starts = np.unique(self.seq_starts, axis=0)
                self.train_data = np.array([self.ori_data[i:i + self.seq_len] for i in self.seq_starts])  # 添加划分的数据
                self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=False,
                                               shuffle=True, pin_memory=True)
                self.n_samples = len(self.train_data)

                y_seqs = get_sub_seqs_label2(self.ori_label, seq_starts=self.seq_starts,
                                             seq_len=self.seq_len) if self.ori_label is not None else None
                self.train_label = y_seqs
                self.trainsets['seqstarts' + str(epoch)] = self.seq_starts
                if y_seqs is not None:
                    self.trainsets['yseq' + str(epoch)] = y_seqs
                self.trainsets['dis' + str(epoch)] = dis
        else:
            print('ERROR')


class LSTMEDModule(nn.Module):
    def __init__(self, n_features: int, seq_len, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 device):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.device = device
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0]).to(self.device)
        # self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
        #                        num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1]).to(self.device)
        decoder_layers = []
        for i in range(self.seq_len):
            decoder_layers += [nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1]).to(self.device)]
        # self.decoder = torch.nn.Sequential(*decoder_layers)
        self.decoder = decoder_layers
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features).to(self.device)

    def _init_hidden(self, batch_size):
        return (torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(self.device),
                torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(self.device))

    def forward(self, ts_batch):
        batch_size = ts_batch.shape[0]

        # 1. Encode the time-series to make use of the last hidden state.
        # enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(),
                                     (torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(
                                         self.device),
                                      torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(
                                          self.device)))  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        # dec_hidden = enc_hidden
        # return dec_hidden[0]
        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = torch.Tensor(ts_batch.size()).zero_().to(self.device)
        # dec_hidden = [(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(
        #                                  self.device),
        #                               torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(
        #                                   self.device)) for i in range(ts_batch.shape[1]+1)]
        # dec_hidden[0] = enc_hidden
        dec_hidden = enc_hidden
        j = 0
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])
            _, dec_hidden = self.decoder[j](ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            j += 1
        return output

# def main():
#     from backup.datasets import Skab
#     seed = 0
#     print("Running main")
#     ds = Skab(seed=seed)
#     x_train, y_train, x_test, y_test = ds.data()
#     x_train = x_train[:1000]
#     y_train = y_train[:1000]
#     x_test = x_test[:1000]
#     y_test = y_test[:1000]
#     algo = LSTMED(num_epochs=1, seed=seed, gpu=0, batch_size=64, hidden_size=None,
#                   stride=10, train_val_percentage=0.25, explained_var=0.9, set_hid_eq_pca=True,)
#     algo.fit(x_train)
#     results = algo.predict(x_test)
#     print(results["error_tc"].shape)
#     print(results["error_tc"][:10])
#
#
# if __name__ == "__main__":
#     main()
