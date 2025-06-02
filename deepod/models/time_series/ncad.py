# -*- coding: utf-8 -*-
"""

@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.ts_network_tcn import TcnResidualBlock
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from numpy.random import RandomState

import torch.nn.functional as F
import time

from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from testbed.utils import import_ts_data_unsupervised
from deepod.metrics import ts_metrics, point_adjustment


class NCAD(BaseDeepAD):
    """ Neural Contextual Anomaly Detection for Time Series (NCAD)
    'Neural Contextual Anomaly Detection for Time Series'. in IJCAI. 2022.

    Parameters
    ----------
    data_type: str, optional (default='tabular')
        Data type

    epochs: int, optional (default=100)
        Number of training epochs

    batch_size: int, optional (default=64)
        Number of samples in a mini-batch

    lr: float, optional (default=1e-3)
        Learning rate

    network: str, optional (default='MLP')
        network structure for different data structures

    rep_dim: int, optional (default=128)
        Dimensionality of the representation space

    hidden_dims: list, str or int, optional (default='100,50')
        Number of neural units in hidden layers
            - If list, each item is a layer
            - If str, neural units of hidden layers are split by comma
            - If int, number of neural units of single hidden layer

    act: str, optional (default='ReLU')
        activation layer name
        choice = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']

    bias: bool, optional (default=False)
        Additive bias in linear layer

    n_heads: int, optional(default=8):
        number of head in multi-head attention
        used when network='transformer', deprecated in other networks

    d_model: int, optional (default=64)
        number of dimensions in Transformer
        used when network='transformer', deprecated in other networks

    pos_encoding: str, optional (default='fixed')
        manner of positional encoding, deprecated in other networks
        choice = ['fixed', 'learnable']

    norm: str, optional (default='BatchNorm')
        manner of norm in Transformer, deprecated in other networks
        choice = ['LayerNorm', 'BatchNorm']

    epoch_steps: int, optional (default=-1)
        Maximum steps in an epoch
            - If -1, all the batches will be processed

    prt_steps: int, optional (default=10)
        Number of epoch intervals per printing

    device: str, optional (default='cuda')
        torch device,

    verbose: int, optional (default=1)
        Verbosity mode

    random_state： int, optional (default=42)
        the seed used by the random

    """

    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 seq_len=100, stride=1, s_length=5, coe_rate=1.5, mixup_rate=0.5,
                 hidden_dims='150,150,150', act='ReLU', bias=False,
                 kernel_size=7, train_val_pc=0.25, dropout=0.0,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42, a=0.5):
        super(NCAD, self).__init__(
            model_name='NCAD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state, a=a
        )

        self.s_length = s_length
        self.coe_rate = coe_rate
        self.mixup_rate = mixup_rate

        self.hidden_dims = hidden_dims
        self.act = act
        self.bias = bias
        self.dropout = dropout

        self.train_val_pc = train_val_pc

        self.kernel_size = kernel_size

        self.drop_last = True
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
        criterion = NCADLoss(reduction='none')
        self.net.eval()
        with torch.no_grad():
            score_lst = []
            for x in dataloader:
                x = x.float().to(self.device)
                x_c = x[:, :-self.s_length]
                x_output, xc_output = self.net(x, x_c)

                s = criterion(x_output, xc_output)
                s = torch.sigmoid(s)
                score_lst.append(s)

        scores = torch.cat(score_lst).data.cpu().numpy()
        scores_pad = np.hstack([0 * np.ones(self.seq_len - 1), scores])

        return scores_pad

    def training(self, epoch):
        self.net.train()
        loss_lst = []
        for ii, x0 in enumerate(self.train_loader):
            y0 = np.zeros(x0.shape[0])
            y0 = torch.tensor(y0)
            if self.coe_rate > 0:
                x_oe, y_oe = coe_batch(
                    x=x0.transpose(2, 1),
                    y=y0,
                    coe_rate=self.coe_rate,
                    suspect_window_length=self.s_length,
                    random_start_end=True,
                )
                # Add COE to training batch
                x0 = torch.cat((x0, x_oe.transpose(2, 1)), dim=0)
                y0 = torch.cat((y0, y_oe), dim=0)

            if self.mixup_rate > 0.0:
                x_mixup, y_mixup = mixup_batch(
                    x=x0.transpose(2, 1),
                    y=y0,
                    mixup_rate=self.mixup_rate,
                )
                # Add Mixup to training batch
                x0 = torch.cat((x0, x_mixup.transpose(2, 1)), dim=0)
                y0 = torch.cat((y0, y_mixup), dim=0)

            x0 = x0.float().to(self.device)
            y0 = y0.float().to(self.device)
            x_c = x0[:, :-self.s_length]

            x0_output, xc_output = self.net(x0, x_c)

            x0_output, xc_output = F.normalize(x0_output), F.normalize(xc_output)

            logits_anomaly = self.criterion(x0_output, xc_output)  # .squeeze()

            probs_anomaly = torch.sigmoid(logits_anomaly)
            # Calculate Loss
            loss = torch.nn.BCELoss()(probs_anomaly, y0)
            self.net.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_lst.append(loss)

        epoch_loss = torch.mean(torch.stack(loss_lst)).data.cpu().item()
        return epoch_loss

    def training_forward(self, x0, net, criterion):
        y0 = np.zeros(x0.shape[0])
        y0 = torch.tensor(y0).to(self.device)
        if self.coe_rate > 0:
            x_oe, y_oe = coe_batch(
                x=x0.transpose(2, 1),
                y=y0,
                coe_rate=self.coe_rate,
                suspect_window_length=self.s_length,
                random_start_end=True,
            )
            # Add COE to training batch
            x0 = torch.cat((x0, x_oe.transpose(2, 1)), dim=0)
            y0 = torch.cat((y0, y_oe), dim=0)

        if self.mixup_rate > 0.0:
            x_mixup, y_mixup = mixup_batch(
                x=x0.transpose(2, 1),
                y=y0,
                mixup_rate=self.mixup_rate,
            )
            # Add Mixup to training batch
            x0 = torch.cat((x0, x_mixup.transpose(2, 1)), dim=0)
            y0 = torch.cat((y0, y_mixup), dim=0)

        x0 = x0.float().to(self.device)
        y0 = y0.float().to(self.device)
        x_c = x0[:, :-self.s_length]

        x0_output, xc_output = self.net(x0, x_c)

        x0_output, xc_output = F.normalize(x0_output), F.normalize(xc_output)

        logits_anomaly = self.criterion(x0_output, xc_output)  # .squeeze()

        probs_anomaly = torch.sigmoid(logits_anomaly)
        # Calculate Loss
        loss = torch.nn.BCELoss()(probs_anomaly, y0)
        return loss

    def inference_forward(self, x0, net, criterion):
        criterion = NCADLoss(reduction='none')
        x = x0.float().to(self.device)
        x_c = x[:, :-self.s_length]
        x_output, xc_output = self.net(x, x_c)

        s = criterion(x_output, xc_output)
        loss = torch.sigmoid(s)
        return x_output, loss

    def training_prepare(self, X, y):
        self.train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.net = NCADNet(
            input_dim=self.n_features,
            hidden_dims=self.hidden_dims,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            bias=self.bias
        ).to(self.device)

        self.criterion = NCADLoss(reduction='none')

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        return

    def inference_prepare(self, X):
        """define test_loader"""
        return


def coe_batch(x: torch.Tensor, y: torch.Tensor, coe_rate: float, suspect_window_length: int,
              random_start_end: bool = True):
    """Contextual Outlier Exposure.

    Args:
        x : Tensor of shape (batch, ts channels, time)
        y : Tensor of shape (batch, )
        coe_rate : Number of generated anomalies as proportion of the batch size.
        random_start_end : If True, a random subset within the suspect segment is permuted between time series;
            if False, the whole suspect segment is randomly permuted.
    """

    if coe_rate == 0:       # 1.5
        raise ValueError(f"coe_rate must be > 0.")
    batch_size = x.shape[0]
    ts_channels = x.shape[1]
    oe_size = int(batch_size * coe_rate)

    # Select indices
    idx_1 = torch.arange(oe_size)
    idx_2 = torch.arange(oe_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()
        idx_2 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()

    if ts_channels > 3:
        numb_dim_to_swap = np.random.randint(low=3, high=ts_channels, size=(oe_size))
        # print(numb_dim_to_swap)
    else:
        numb_dim_to_swap = np.ones(oe_size) * ts_channels

    x_oe = x[idx_1].clone()  # .detach()
    oe_time_start_end = np.random.randint(
        low=x.shape[-1] - suspect_window_length, high=x.shape[-1] + 1, size=(oe_size, 2)
    )
    oe_time_start_end.sort(axis=1)
    # for start, end in oe_time_start_end:
    for i in range(len(idx_2)):
        # obtain the dimensons to swap
        numb_dim_to_swap_here = int(numb_dim_to_swap[i])
        dims_to_swap_here = np.random.choice(
            range(ts_channels), size=numb_dim_to_swap_here, replace=False
        )

        # obtain start and end of swap
        start, end = oe_time_start_end[i]

        # swap
        x_oe[i, dims_to_swap_here, start:end] = x[idx_2[i], dims_to_swap_here, start:end]

    # Label as positive anomalies
    y_oe = torch.ones(oe_size).type_as(y)

    return x_oe, y_oe


def mixup_batch(x: torch.Tensor, y: torch.Tensor, mixup_rate: float):
    """
    Args:
        x : Tensor of shape (batch, ts channels, time)
        y : Tensor of shape (batch, )
        mixup_rate : Number of generated anomalies as proportion of the batch size.
    """

    if mixup_rate == 0:
        raise ValueError(f"mixup_rate must be > 0.")
    batch_size = x.shape[0]
    mixup_size = int(batch_size * mixup_rate)  #

    # Select indices
    idx_1 = torch.arange(mixup_size)
    idx_2 = torch.arange(mixup_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()
        idx_2 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()

    # sample mixing weights:
    beta_param = float(0.05)
    beta_distr = torch.distributions.beta.Beta(
        torch.tensor([beta_param]), torch.tensor([beta_param])
    )
    weights = torch.from_numpy(np.random.beta(beta_param, beta_param, (mixup_size,))).type_as(x)
    oppose_weights = 1.0 - weights

    # Create contamination
    x_mix_1 = x[idx_1].clone()
    x_mix_2 = x[idx_1].clone()
    x_mixup = (
        x_mix_1 * weights[:, None, None] + x_mix_2 * oppose_weights[:, None, None]
    )  # .detach()

    # Label as positive anomalies
    y_mixup = y[idx_1].clone() * weights + y[idx_2].clone() * oppose_weights

    return x_mixup, y_mixup


class NCADNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=32, kernel_size=2, dropout=0.2,
                 bias=True):
        super(NCADNet, self).__init__()

        self.layers = []

        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        elif type(hidden_dims) == str:
            hidden_dims = hidden_dims.split(',')
            hidden_dims = [int(a) for a in hidden_dims]

        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            self.layers += [TcnResidualBlock(in_channels, out_channels, kernel_size,
                                             stride=1, dilation=dilation_size,
                                             padding=padding_size, dropout=dropout,
                                             bias=bias)]
        self.network = torch.nn.Sequential(*self.layers)


    def forward(self, x, x_c):
        x_out, xc_out = self.network(x.transpose(2, 1)).transpose(2, 1), self.network(x_c.transpose(2, 1)).transpose(2, 1)
        x_out, xc_out = x_out[:, -1], xc_out[:, -1]

        return x_out, xc_out


class NCADLoss(torch.nn.Module):
    """

    Parameters
    ----------
    reduction: str, optional (default='mean')
        choice = [``'none'`` | ``'mean'`` | ``'sum'``]
            - If ``'none'``: no reduction will be applied;
            - If ``'mean'``: the sum of the output will be divided by the number of
            elements in the output;
            - If ``'sum'``: the output will be summed

    """

    def __init__(self, reduction='mean'):
        super(NCADLoss, self).__init__()
        self.reduction = reduction
        self.eps = 1e-10

    def forward(self, rep, repc):
        dist = torch.sum((rep - repc) ** 2, dim=1)


        log_prob_equal = -dist
        # copied from torch lighting implementation
        prob_different = torch.clamp(1 - torch.exp(-dist), self.eps, 1)
        log_prob_different = torch.log(prob_different)

        loss = log_prob_different - log_prob_equal
        reduction = self.reduction
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss


if __name__ == '__main__':
    data_dir = '/home/xuhz/dataset/5-TSdata/_processed_data/'
    # data = 'ASD'
    # entities = 'omi-2,omi-5,omi-6,omi-11'   #ASD
    # data = 'SMD'
    # entities = 'machine-1-2,machine-1-5,machine-2-3,machine-2-6,machine-3-1,machine-3-5,machine-3-6,machine-3-9,machine-3-11'  #SMD
    # entities = 'machine-1-2,machine-1-5,machine-2-3,machine-2-6,machine-3-1,machine-3-5,machine-3-9'  # SMD
    data = 'MSL'
    # entities = 'P-15,S-2'  #MSL
    # data = 'SMAP'
    # entities = 'A-1,A-4,B-1,D-5,D-6,D-11,D-13,F-1,G-1,G-2,G-3'  # SMAP
    entities = 'T-9'   #SMAP
    # entities = 'FULL'
    # data = 'SWaT_cut'
    # entities = 'FULL'
    model_name = 'TSAD'
    train_df_lst, test_df_lst, label_lst, name_lst = import_ts_data_unsupervised(data_dir, data, entities=entities)
    print(name_lst)
    num_runs = 5

    f1_lst = []
    aupr_lst = []
    for train, test, label, name in zip(train_df_lst, test_df_lst, label_lst, name_lst):
        entries = []

        for i in range(num_runs):
            print(f'\nRunning [{i + 1}/{num_runs}] of [{model_name}] on Dataset [{name}]')

            clf = NCAD(seq_len=400, stride=1, hidden_dims=32, epochs=50, batch_size=64, lr=0.00002,
                       random_state=50 + i)


            clf.fit(train)

            scores = clf.decision_function(test)
            adj_eval_scores = ts_metrics(label, point_adjustment(label, scores))
            entries.append(adj_eval_scores)

        avg_entry = np.average(np.array(entries), axis=0)
        std_entry = np.std(np.array(entries), axis=0)
        print(f'{avg_entry[2]:.4f}, {std_entry[2]:.4f}, {avg_entry[1]:.4f}, {std_entry[1]:.4f}')
        f1_lst.append(avg_entry[2])
        aupr_lst.append(avg_entry[1])

    avg_ap, avg_f1 = np.average(aupr_lst), np.average(f1_lst)
    std_ap, std_f1 = np.std(aupr_lst), np.std(f1_lst)
    print(f'{avg_f1:.4f}, {std_f1:.4f}, {avg_ap:.4f}, {std_ap:.4f}')
