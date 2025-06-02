#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py
import os
from time import time
from copy import deepcopy
import random
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from torch.nn import MSELoss, BCELoss
import numpy as np
from sklearn.metrics import mean_squared_error
from deepod.core.base_model import BaseDeepAD
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from torch.utils.data import DataLoader
from deepod.metrics import ts_metrics, point_adjustment


def seed_all(seed=2020):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_all(2021)


class RNNEncoder(nn.Module):
    """
    An implementation of Encoder based on Recurrent neural networks.
    """
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):
        """
        args:
            inp_dim: dimension of input value
            z_dim: dimension of latent code
            hidden_dim: dimension of fully connection layers
            rnn_hidden_dim: dimension of rnn cell hidden states
            num_layers: number of layers of rnn cell
            bidirectional: whether use BiRNN cell
            cell: one of ['lstm', 'gru', 'rnn']
        """
        super(RNNEncoder, self).__init__()

        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.linear1 = nn.Linear(inp_dim, hidden_dim)
        if bidirectional:
            self.linear2 = nn.Linear(self.rnn_hidden_dim * 2, z_dim)
        else:
            self.linear2 = nn.Linear(self.rnn_hidden_dim, z_dim)

        if cell == 'lstm':
            self.rnn = nn.LSTM(hidden_dim,
                               rnn_hidden_dim,
                               num_layers=num_layers,
                               bidirectional=bidirectional)
        elif cell == 'gru':
            self.rnn = nn.GRU(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional)

    def forward(self, inp):
        # inp shape: [bsz, seq_len, inp_dim]
        self.rnn.flatten_parameters()
        inp = inp.permute(1, 0, 2)
        rnn_inp = torch.tanh(self.linear1(inp))
        rnn_out, _ = self.rnn(rnn_inp)
        z = self.linear2(rnn_out).permute(1, 0, 2)
        return z


class RNNDecoder(nn.Module):
    """
    An implementation of Decoder based on Recurrent neural networks.
    """
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):
        """
        args:
            Reference argument annotations of RNNEncoder.
        """
        super(RNNDecoder, self).__init__()

        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.linear1 = nn.Linear(z_dim, hidden_dim)
        if bidirectional:
            self.linear2 = nn.Linear(self.rnn_hidden_dim * 2, inp_dim)
        else:
            self.linear2 = nn.Linear(self.rnn_hidden_dim, inp_dim)

        if cell == 'lstm':
            self.rnn = nn.LSTM(hidden_dim,
                               rnn_hidden_dim,
                               num_layers=num_layers,
                               bidirectional=bidirectional)
        elif cell == 'gru':
            self.rnn = nn.GRU(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional)

    def forward(self, z):
        # z shape: [bsz, seq_len, z_dim]
        self.rnn.flatten_parameters()
        z = z.permute(1, 0, 2)
        rnn_inp = torch.tanh(self.linear1(z))
        rnn_out, _ = self.rnn(rnn_inp)
        re_x = self.linear2(rnn_out).permute(1, 0, 2)
        return re_x


class RNNAutoEncoder(nn.Module):
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):

        super(RNNAutoEncoder, self).__init__()

        self.encoder = RNNEncoder(inp_dim, z_dim, hidden_dim, rnn_hidden_dim,
                                  num_layers, bidirectional=bidirectional, cell=cell)
        self.decoder = RNNDecoder(inp_dim, z_dim, hidden_dim, rnn_hidden_dim,
                                  num_layers, bidirectional=bidirectional, cell=cell)

    def forward(self, inp):
        # inp shape: [bsz, seq_len, inp_dim]
        z = self.encoder(inp)
        re_inp = self.decoder(z)
        return re_inp, z


class MLPDiscriminator(nn.Module):
    def __init__(self, inp_dim, hidden_dim):
        super(MLPDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        seq, df = inp.shape
        c = self.dis(inp)
        return c.view(seq)


class AdaWeightedLoss(nn.Module):
    def __init__(self, strategy='linear'):
        super(AdaWeightedLoss, self).__init__()
        self.strategy = strategy

    def forward(self, input, target, global_step):
        """
        The reconstruction error will be calculated between x and x', where
        x is a vector of x_dim.

        args:
            input: original values, [bsz,seq,x_dim]
            target: reconstructed values
            global_step: training global step
            strategy: how fast the coefficient w2 shrink to 1.0
        return:
        """
        bsz, seq, x_dim = target.size()

        with torch.no_grad():
            # errors: [bsz,seq]
            # w1: [bsz,seq]
            errors = torch.sqrt(torch.sum((input - target) ** 2, dim=-1))
            error_mean = torch.mean(errors, dim=-1)[:, None]
            error_std = torch.std(errors, dim=-1)[:, None] + 1e-6
            z_score = (errors - error_mean) / error_std
            neg_z_score = -z_score
            w1 = torch.softmax(neg_z_score, dim=-1)

            # exp_z_score: [bsz,seq] -> [bsz,1,seq] -> [bsz,seq,seq] -> [bsz,seq]
            exp_z_score = torch.exp(neg_z_score)
            exp_z_score = exp_z_score[:, None, :].repeat(1, seq, 1)
            step_coeff = torch.ones(size=(seq, seq), dtype=target.dtype, device=target.device)

            for i in range(seq):
                if self.strategy == 'log':
                    step_coeff[i][i] *= np.log(global_step + np.e - 1)
                elif self.strategy == 'linear':
                    step_coeff[i][i] *= global_step
                elif self.strategy == 'nlog':
                    step_coeff[i][i] *= global_step * np.log(global_step + np.e - 1)
                elif self.strategy == 'quadratic':
                    step_coeff[i][i] *= (global_step ** 2)
                else:
                    raise KeyError('Decay function must be one of [\'log\',\'linear\',\'nlog\',\'quadratic\']')

            exp_z_score = exp_z_score * step_coeff
            w2 = torch.sum(exp_z_score, dim=-1) / exp_z_score[:, torch.arange(0, seq), torch.arange(0, seq)]
            w = w1 * w2
            # normalization
            w = (w / torch.sum(w, dim=-1)[:, None])[:, :, None]

        error_matrix = (target - input) ** 2
        return torch.sum(error_matrix * w) / (bsz * x_dim)


class FGANomaly(BaseDeepAD):
    def __init__(self, seq_len=100, stride=1, epochs=100, batch_size=128, lr=1e-3,
                 epoch_steps=-1, prt_steps=10, device='cuda', verbose=2, random_state=42,
                 if_scheduler=True, adv_rate=0.01, dis_ar_iter=1,
                 weighted_loss=True, strategy='linear', scheduler_step_size=5, scheduler_gamma=0.5,
                 z_dim=10, hidden_dim=50, rnn_hidden_dim=50, num_layers=1, bidirectional=True, cell='lstm'):
        super(FGANomaly, self).__init__(
            model_name='FGANomaly', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.ada_mse = None
        self.bce = None
        self.mse = None
        self.ar_scheduler = None
        self.ar_optimizer = None
        self.ae_scheduler = None
        self.ae_optimizer = None
        self.dataloader = None
        self.dis_ar = None
        self.ae = None
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell = cell

        self.if_scheduler = if_scheduler

        self.adv_rate = adv_rate
        self.dis_ar_iter = dis_ar_iter

        self.weighted_loss = weighted_loss
        self.strategy = strategy

        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        self.cur_step = 0
        self.cur_epoch = 0
        self.best_dis_ar = None
        self.best_val_loss = np.inf
        self.early_stop_count = 0
        self.re_loss = None
        self.adv_dis_loss = None
        self.time_per_epoch = None

    def fit(self, X, y=None, Xtest=None, Ytest=None, X_seqs=None, y_seqs=None):
        self.fit_RODA(X, y, Xtest, Ytest, X_seqs, y_seqs)

    def training(self, epoch):
        start_time = time()
        for ii, batch_x in enumerate(self.dataloader):
            self.cur_step += 1
            batch_x = batch_x.float().to(self.device)

            for _ in range(self.dis_ar_iter):
                self.dis_ar_train(batch_x)
            self.ae_train(batch_x)
        end_time = time()
        self.time_per_epoch = end_time - start_time
        if self.if_scheduler:
            self.ar_scheduler.step()
            self.ae_scheduler.step()
        return self.re_loss

    def training_forward(self, batch_x, net, criterion):
        """define forward step in training"""
        pass

    def inference_forward(self, batch_x, net, criterion):
        """define forward step in inference"""
        pass

    def training_prepare(self, X, y):
        self.ae = RNNAutoEncoder(inp_dim=self.n_features,
                                 z_dim=self.z_dim,
                                 hidden_dim=self.hidden_dim,
                                 rnn_hidden_dim=self.rnn_hidden_dim,
                                 num_layers=self.num_layers,
                                 bidirectional=self.bidirectional,
                                 cell=self.cell).to(self.device)
        self.dis_ar = MLPDiscriminator(inp_dim=self.n_features,
                                       hidden_dim=self.hidden_dim).to(self.device)

        self.dataloader = DataLoader(dataset=X, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False)

        self.ae_optimizer = Adam(params=self.ae.parameters(), lr=self.lr)
        self.ae_scheduler = lr_scheduler.StepLR(optimizer=self.ae_optimizer,
                                                step_size=self.scheduler_step_size,
                                                gamma=self.scheduler_gamma)
        self.ar_optimizer = Adam(params=self.dis_ar.parameters(), lr=self.lr)
        self.ar_scheduler = lr_scheduler.StepLR(optimizer=self.ar_optimizer,
                                                step_size=self.scheduler_step_size,
                                                gamma=self.scheduler_gamma)
        self.mse = MSELoss()
        self.bce = BCELoss()
        self.ada_mse = AdaWeightedLoss(self.strategy)

    def inference_prepare(self, X):
        """define test_loader"""
        pass

    def dis_ar_train(self, x):
        self.ar_optimizer.zero_grad()

        re_x, z = self.ae(x)
        soft_label, hard_label = self.value_to_label(x, re_x)

        actual_normal = x[torch.where(hard_label == 0)]
        re_normal = re_x[torch.where(hard_label == 0)]
        actual_target = torch.ones(size=(actual_normal.shape[0],), dtype=torch.float, device=self.device)
        re_target = torch.zeros(size=(actual_normal.shape[0],), dtype=torch.float, device=self.device)

        re_logits = self.dis_ar(re_normal)
        actual_logits = self.dis_ar(actual_normal)

        re_dis_loss = self.bce(input=re_logits, target=re_target)
        actual_dis_loss = self.bce(input=actual_logits, target=actual_target)

        dis_loss = re_dis_loss + actual_dis_loss
        dis_loss.backward()
        self.ar_optimizer.step()

    def ae_train(self, x):
        bsz, seq, fd = x.shape
        self.ae_optimizer.zero_grad()

        re_x, z = self.ae(x)

        # reconstruction loss
        if self.weighted_loss:
            self.re_loss = self.ada_mse(re_x, x, self.cur_step)
        else:
            self.re_loss = self.mse(re_x, x)

        # adversarial loss
        ar_inp = re_x.contiguous().view(bsz*seq, fd)
        actual_target = torch.ones(size=(ar_inp.shape[0],), dtype=torch.float, device=self.device)
        re_logits = self.dis_ar(ar_inp)
        self.adv_dis_loss = self.bce(input=re_logits, target=actual_target)

        loss = self.re_loss + self.adv_dis_loss * self.adv_rate
        loss.backward()
        self.ae_optimizer.step()

    def decision_function(self, X, return_rep=False):
        self.ae.eval()
        re_values = self.value_reconstruction_val(X, self.seq_len)
        scores = []
        for v1, v2 in zip(X, re_values):
            scores.append(np.sqrt(np.sum((v1 - v2) ** 2)))
        scores = np.array(scores)
        return scores

    def value_reconstruction_val(self, values, seq_len):
        piece_num = len(values) // seq_len
        reconstructed_values = []
        for i in range(piece_num):
            raw_values = values[i * seq_len:(i + 1) * seq_len, :]
            raw_values = torch.tensor([raw_values], dtype=torch.float).to(self.device)
            reconstructed_value_, z = self.ae(raw_values)

            reconstructed_value_ = reconstructed_value_.squeeze().detach().cpu().tolist()
            reconstructed_values.extend(reconstructed_value_)
        return np.array(reconstructed_values)

    def value_to_label(self, values, re_values):
        with torch.no_grad():
            errors = torch.sqrt(torch.sum((values - re_values) ** 2, dim=-1))
            error_mean = torch.mean(errors, dim=-1)[:, None]
            error_std = torch.std(errors, dim=-1)[:, None] + 1e-6
            z_score = (errors - error_mean) / error_std
            z_score = z_score * (1 - 1 / self.cur_epoch)

            soft_label = torch.sigmoid(z_score)
            rand = torch.rand_like(soft_label)
            hard_label = (soft_label > rand).float()
            return soft_label, hard_label
