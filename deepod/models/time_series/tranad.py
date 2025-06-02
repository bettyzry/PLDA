import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder
import time
import numpy as np
from torch.utils.data import DataLoader
import math
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from deepod.core.base_model import BaseDeepAD
from deepod.metrics import ts_metrics, point_adjustment


class TranAD(BaseDeepAD):
    def __init__(self, seq_len=100, stride=1, lr=0.001, epochs=5, batch_size=128,
                 epoch_steps=20, prt_steps=1, device='cuda',
                 verbose=2, random_state=42, a=0.5):
        super(TranAD, self).__init__(
            model_name='TranAD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state, a=a
        )
        self.net = None
        self.optimizer = None
        self.scheduler = None

        self.w_size = None
        self.n_features = None
        return

    def fit(self, X, y=None, Xtest=None, Ytest=None, X_seqs=None, y_seqs=None):
        self.fit_RODA(X, y, Xtest, Ytest, X_seqs, y_seqs)
        return

    def training_prepare(self, X, y):
        self.net = TranADNet(
            feats=self.n_features,
            n_window=self.seq_len
        ).to(self.device)

        self.train_loader = DataLoader(X, batch_size=self.batch_size,
                                shuffle=True, pin_memory=True)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.criterion = nn.MSELoss(reduction='none')

    def decision_function(self, X, return_rep=False):
        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.batch_size,
                                shuffle=False, drop_last=False)

        self.net.eval()
        loss, _ = self.inference(dataloader)  # (n,d)
        loss_final = np.mean(loss, axis=1)  # (n,)

        padding_list = np.zeros([X.shape[0]-loss.shape[0], loss.shape[1]])
        loss_pad = np.concatenate([padding_list, loss], axis=0)
        loss_final_pad = np.hstack([0 * np.ones(X.shape[0] - loss_final.shape[0]), loss_final])  # (8640,)

        return loss_final_pad

    def training(self, epoch):
        n = epoch + 1
        l1s, l2s = [], []

        self.optimizer.zero_grad()
        for ii, batch_x in enumerate(self.train_loader):
            local_bs = batch_x.shape[0]  #(128，30，19)
            window = batch_x.permute(1, 0, 2)  # (30, 128, 19)
            elem = window[-1, :, :].view(1, local_bs, self.n_features)  #(1, 128, 19)

            window = window.float().to(self.device)
            elem = elem.float().to(self.device)

            z = self.net(window, elem)
            l1 = (1/n) * self.criterion(z[0], elem) + (1-1/n) * self.criterion(z[1], elem)  #(1, 128, 19)

            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)

            loss.backward(retain_graph=True)

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
            self.optimizer.zero_grad()

            if self.epoch_steps != -1:
                if ii > self.epoch_steps:
                    break

        self.scheduler.step()
        return np.mean(l1s)

    def inference(self, dataloader):
        l1s = []
        preds = []
        for d in dataloader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, self.n_features)
            window = window.float().to(self.device)
            elem = elem.float().to(self.device)
            z = self.net(window, elem)
            if isinstance(z, tuple):
                z = z[1]
            l1 = self.criterion(z, elem)[0]
            l1 = l1.data.cpu()
            l1s.append(l1)

        l1s = torch.cat(l1s)
        l1s = l1s.numpy()
        return l1s, preds

    def training_forward(self, batch_x, net, criterion):
        n = self.epoch + 1
        l1s, l2s = [], []

        local_bs = batch_x.shape[0]  # (128，30，19)
        window = batch_x.permute(1, 0, 2)  # (30, 128, 19)
        elem = window[-1, :, :].view(1, local_bs, self.n_features)  # (1, 128, 19)

        window = window.float().to(self.device)
        elem = elem.float().to(self.device)

        z = self.net(window, elem)
        l1 = (1 / n) * self.criterion(z[0], elem) + (1 - 1 / n) * self.criterion(z[1], elem)  # (1, 128, 19)

        l1s.append(torch.mean(l1).item())
        loss = torch.mean(l1)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        local_bs = batch_x.shape[0]
        window = batch_x.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, self.n_features)
        window = window.float().to(self.device)
        elem = elem.float().to(self.device)
        z = self.net(window, elem)
        if isinstance(z, tuple):
            z = z[1]
        l1 = self.criterion(z, elem)[0]
        l1 = l1.mean(axis=1)
        return z, l1

    def inference_prepare(self, X):
        """define test_loader"""
        return


# Proposed Model + Self Conditioning + Adversarial + MAML
class TranADNet(nn.Module):
    def __init__(self, feats, n_window=10):
        super(TranADNet, self).__init__()
        # self.name = 'TranAD'
        # self.lr = lr
        # self.batch = batch_size

        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.4)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.4)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.4)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

