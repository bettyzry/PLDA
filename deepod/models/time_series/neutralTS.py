# -*- coding: utf-8 -*-
"""
Neural Transformation Learning-based Anomaly Detection
this script is partially adapted from https://github.com/boschresearch/NeuTraL-AD (AGPL-3.0 license)
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import MLPnet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
from deepod.core.networks.ts_network_tcn import TcnResidualBlock
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from numpy.random import RandomState
from torch.utils.data import Dataset
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from deepod.metrics import ts_metrics, point_adjustment
import time


class NeuTraLTS(BaseDeepAD):
    def __init__(self, epochs=100, batch_size=64, lr=0.001, seq_len=30, stride=1,
                 n_trans=11, trans_type='residual', temp=0.1,
                 hidden_dims='100,50', trans_hidden_dims=50,
                 act='LeakyReLU', bias=False, train_val_pc=0.25, dropout=0.0,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=1, random_state=42, a=0.5, rate=0.1):
        super(NeuTraLTS, self).__init__(
            model_name='NeuTraL', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state, seq_len=seq_len, stride=stride, a=a, rate=rate
        )

        self.n_trans = n_trans
        self.trans_type = trans_type
        self.temp = temp

        self.trans_hidden_dims = trans_hidden_dims
        self.hidden_dims = hidden_dims
        self.train_val_pc = train_val_pc
        self.dropout = dropout
        self.act = act
        self.bias = bias
        return

    def fit(self, X, y=None, Xtest=None, Ytest=None, X_seqs=None, y_seqs=None):
        self.fit_RODA(X, y, Xtest, Ytest, X_seqs, y_seqs)
        return

    def decision_function(self, X, return_rep=False):
        """
        Predict raw anomaly score of X using the fitted detector.
        For consistency, outliers are assigned with larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        test_sub_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        dataloader = DataLoader(dataset=test_sub_seqs, batch_size=self.batch_size, drop_last=False, shuffle=False)
        criterion = DCL(reduction='none')
        self.net.eval()
        with torch.no_grad():
            score_lst = []
            for x in dataloader:
                x = x.float().to(self.device)
                x_output = self.net(x)
                s = criterion(x_output)
                score_lst.append(s)

        scores = torch.cat(score_lst).data.cpu().numpy()
        scores_pad = np.hstack([0 * np.ones(self.seq_len - 1), scores])

        return scores_pad

    def training(self, epoch):
        self.net.train()
        loss_lst = []
        self.net.zero_grad()
        for ii, x0 in enumerate(self.train_loader):
            x0 = x0.float().to(self.device)

            x0_output = self.net(x0)
            loss = self.criterion(x0_output)

            loss.backward()

            if self.sample_selection == 5:  # ICML21
                to_concat_g = []
                to_concat_v = []
                clip = 0.2
                for name, param in self.net.named_parameters():
                    if param.grad is None:
                        continue
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
                    if param.grad is None:
                        continue
                    mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
                    mask = mask * clip
                    param.grad.data = mask * param.grad.data
            self.optimizer.step()
            self.net.zero_grad()

            loss_lst.append(loss)

        epoch_loss = torch.mean(torch.stack(loss_lst)).data.cpu().item()
        return epoch_loss

    def training_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        x0_output = self.net(batch_x)
        loss = self.criterion(x0_output)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        output = net(batch_x)
        error = DCL(temperature=self.temp, reduction='none')(output)
        return output, error

    def training_prepare(self, X, y):
        self.train_loader = DataLoader(dataset=X,
                                  batch_size=self.batch_size,
                                  drop_last=False, pin_memory=True, shuffle=True)

        self.net = TabNeutralADNet(
            n_features=self.n_features,
            n_trans=self.n_trans,
            trans_type=self.trans_type,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            trans_hidden_dims=self.trans_hidden_dims,
            activation=self.act,
            bias=self.bias,
            device=self.device
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.criterion = DCL(temperature=self.temp, reduction='mean')

    def inference_prepare(self, X):
        """define test_loader"""
        return


class TabNeutralADNet(torch.nn.Module):
    """
    network class of NeuTraL for tabular data

    Parameters
    ----------
    n_features: int
        dimensionality of input data

    n_trans: int
        the number of transformation times

    trans_type: str, default='residual'
        transformation type

    enc_hidden_dims: list or str or int
        the number of neural units of hidden layers in encoder net

    trans_hidden_dims: list or str or int
        the number of neural units of hidden layers in transformation net

    rep_dim: int
        representation dimensionality

    activation: str
        activation layer name

    device: str
        device
    """
    def __init__(self, n_features, hidden_dims, kernel_size=2, n_trans=11, dropout=0.0, trans_type='residual',
                 trans_hidden_dims=24,
                 activation='ReLU',
                 bias=False,
                 device='cuda'):
        super(TabNeutralADNet, self).__init__()

        self.layers = []

        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        elif type(hidden_dims) == str:
            hidden_dims = hidden_dims.split(',')
            hidden_dims = [int(a) for a in hidden_dims]

        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size - 1) * dilation_size
            in_channels = n_features if i == 0 else hidden_dims[i - 1]
            out_channels = hidden_dims[i]
            self.layers += [TcnResidualBlock(in_channels, out_channels, kernel_size,
                                             stride=1, dilation=dilation_size,
                                             padding=padding_size, dropout=dropout,
                                             bias=bias)]
        self.enc = torch.nn.Sequential(*self.layers)

        self.trans = torch.nn.ModuleList(
            [MLPnet(n_features=n_features,
                    n_hidden=trans_hidden_dims,
                    n_output=n_features,
                    activation=activation,
                    bias=bias,
                    batch_norm=False) for _ in range(n_trans)]
        )

        self.trans.to(device)
        self.enc.to(device)

        self.n_trans = n_trans
        self.trans_type = trans_type
        self.z_dim = hidden_dims[-1]

    def forward(self, x):
        x_transform = torch.empty(x.shape[0], self.n_trans, x.shape[1], x.shape[2]).to(x)

        for i in range(self.n_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_transform[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_transform[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_transform[:, i] = mask + x

        x_cat = torch.cat([x.unsqueeze(1), x_transform], 1)
        zs = self.enc(x_cat.reshape(-1, x.shape[2], x.shape[1])).transpose(2, 1)[:, -1]
        zs = zs.reshape(-1, self.n_trans+1, self.z_dim)
        return zs


class DCL(torch.nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super(DCL, self).__init__()
        self.temp = temperature
        self.reduction = reduction

    def forward(self, z):
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1, z
        batch_size, n_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(n_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, n_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1

        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp) # n,k-1
        K = n_trans - 1
        scale = 1 / np.abs(K*np.log(1.0 / K))

        loss = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale
        loss = loss.sum(1)

        reduction = self.reduction
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss

        return loss


class SubseqData(Dataset):
    def __init__(self, x, y=None, w1=None, w2=None):
        self.sub_seqs = x
        self.label = y
        self.sample_weight1 = w1
        self.sample_weight2 = w2

    def __len__(self):
        return len(self.sub_seqs)

    def __getitem__(self, idx):
        if self.label is not None and self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.label[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        if self.label is not None:
            return self.sub_seqs[idx], self.label[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is None:
            return self.sub_seqs[idx], self.sample_weight[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        return self.sub_seqs[idx]