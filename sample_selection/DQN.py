import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class DQN(nn.Module):
    """
    Deep Q Network
    """

    def __init__(self, n_feature, seq_len, hidden_size, n_actions, device='gpu'):
        super(DQN, self).__init__()
        self.n_feature = n_feature
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.device = device
        self.latent = nn.Sequential(
          nn.Linear(n_feature, hidden_size)
        )
        # self.output_layer = nn.Linear(self.seq_len*self.hidden_size, n_actions)
        self.time_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=seq_len,  # input height
                out_channels=10,  # n_filters
                kernel_size=5,  # filter size
            ),
            nn.Linear(seq_len, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.cnn = nn.Conv1d(in_channels=seq_len, out_channels=10, kernel_size=1)
        self.l1 = nn.Linear(10, 10)
        self.l2 = nn.Linear(10, 1)
        self.output_layer = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.latent(x))
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        x = x.squeeze()
        return x

    def get_latent(self, x):
        """
        Get the latent representation of the input using the latent layer
        """
        self.eval()  # 关闭dropout层
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            latent_embs = F.relu(self.latent(x))
        self.train()
        return latent_embs
    #
    # def predict_label(self, x):
    #     self.eval()
    #     """
    #     Predict the label of the input as the argmax of the output layer
    #     """
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    #
    #     with torch.no_grad():
    #         ret = torch.argmax(self.forward(x), axis=1)
    #         self.train()
    #         return ret
    #
    # def _initialize_weights(self, ):
    #     with torch.no_grad():
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear):
    #                 nn.init.normal_(m.weight, 0.0, 0.01)
    #                 nn.init.constant_(m.bias, 0.0)
    #
    # def forward_latent(self, x):
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    #     latent = F.relu(self.latent(x))
    #     out = self.output_layer(latent)
    #     return out, latent
    #
    # def get_latent_grad(self, x):
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    #     latent_embs = F.relu(self.latent(x))
    #     return latent_embs


if __name__ == '__main__':
    batch_size = 64
    seq_len = 30
    n_feature = 55
    input = torch.randn(batch_size, seq_len, n_feature)
    d = DQN(n_feature, seq_len, 10, 3, device='gpu')
    out = d(input)
    print(out.size())
    # conv1 = nn.Conv1d(in_channels=256, out_channels = 100, kernel_size = 2)
    # input = torch.randn(32, 35, 256)
    # # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
    # input = input.permute(0, 2, 1)
    # out = conv1(input)
    # print(out.size())