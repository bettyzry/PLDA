"""
TCN is adapted from https://github.com/locuslab/TCN
"""
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.ts_network_tcn import TcnAE
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from deepod.metrics import ts_metrics, point_adjustment


class TcnED(BaseDeepAD):
    def __init__(self, seq_len=100, stride=1, epochs=10, batch_size=32, lr=1e-4,
                 rep_dim=32, hidden_dims=32, kernel_size=3, act='LeakyReLU', bias=False, dropout=0.2,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42, a=0.5):
        super(TcnED, self).__init__(
            model_name='TcnED', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state, a=a
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        self.bias = bias

        return

    def fit(self, X, y=None, Xtest=None, Ytest=None, X_seqs=None, y_seqs=None):
        self.fit_RODA(X, y, Xtest, Ytest, X_seqs, y_seqs)
        return

    def training(self, epoch):
        total_loss = 0
        cnt = 0
        self.net.zero_grad()
        for batch_x in self.train_loader:
            batch_x = batch_x.float().to(self.device)
            loss = self.training_forward(batch_x, self.net, self.criterion)
            loss.backward()
            if self.sample_selection == 5:      # ICML21
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
            cnt += 1

            # terminate this epoch when reaching assigned maximum steps per epoch
            if cnt > self.epoch_steps != -1:
                break

        self.epoch_update()
        loss = total_loss / cnt
        return loss

    def training_prepare(self, X, y):
        self.train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.net = TcnAE(
            n_features=self.n_features,
            n_hidden=self.hidden_dims,
            n_emb=self.rep_dim,
            activation=self.act,
            bias=self.bias,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

        self.criterion = torch.nn.MSELoss(reduction="mean")

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr,
                                          eps=1e-6)

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        output, _ = net(batch_x)
        # loss = torch.nn.MSELoss(reduction='mean')(output[:, -1], batch_x[:, -1])
        loss = torch.nn.MSELoss(reduction='none')(output[:, -1], batch_x[:, -1])
        loss = loss.mean(axis=1)
        loss = loss.mean()
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        output, _ = net(batch_x)
        error = torch.nn.MSELoss(reduction='none')(output[:, -1], batch_x[:, -1])
        error = error.mean(axis=1)
        return output, error
