import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    def __init__(self, input_dim=18, output_dim=1, layers_num=4, hidden_dim=50, dropout=0.2):
        super(MLP, self).__init__()
        assert layers_num >= 2

        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(Sin())
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:  # 隐藏层
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(Sin())
                layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):

    def __init__(self, input_dim=32):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 1),
            Sin(),
            # nn.Linear(input_dim//2, 1)
        )

    def forward(self, x):
        return self.net(x)


class Solution_u(nn.Module):
    def __init__(self,args):
        super(Solution_u, self).__init__()
        self.args = args
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=args.c_out, out_channels=args.d_model//2, kernel_size=3, padding='same'),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, padding=1),  # output: (batch_size, d_model, (seq_len-1)-1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=args.d_model//2, out_channels=args.d_model, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        encoder = []
        for i in range(1):
            encoder += [EncoderLayer(configs = args)]
        self.Encoder = nn.Sequential(*encoder)

        self.MLP = MLP(input_dim=args.d_model, output_dim=args.d_model, layers_num=args.e_layers,
                           hidden_dim=args.d_model//2, dropout=0.2)
        input_dim = args.d_model * (args.seq_len+1)
        self.predictor = Predictor(input_dim=input_dim)

    def forward(self, x):
        # x = x.permute(0, 2, 1)   #(1,15)
        x = self.conv1(x)   #(2048,15)
        x = self.conv2(x)   #(2048,15)
        x = self.Encoder(x)
        x = x.contiguous().view(x.size(0), -1)  # Shape: (batch_size, d_model * 2 * (seq_len-4))
        x = self.predictor(x)
        x = x.view(x.shape[0], -1, self.args.pred_len)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, configs):
        super(EncoderLayer, self).__init__()
        self.bn = nn.BatchNorm1d(configs.d_model)
        self.Attention = DualDomainAttention(configs)
        self.feedforward = nn.Sequential(nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                                         nn.ReLU(),
                                         nn.Dropout(configs.dropout),
                                         nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),)
        self.norm = nn.LayerNorm(configs.d_model)
    def forward(self, x):
        out = self.Attention(x)
        out = out.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        # out = self.norm(x + out)
        # out = self.norm(out + self.feedforward(out))
        out = x + out
        out = out + self.feedforward(out)
        # out = x + out
        return out

class DualDomainAttention(nn.Module):
    def __init__(self, args, reduction=16):
        super().__init__()
        channels = args.d_model
        features = args.seq_len+1
        # Channel Attention
        self.ca_fc1 = nn.Linear(channels, channels // reduction)
        self.ca_fc2 = nn.Linear(channels // reduction, channels)

        # Feature Attention
        self.fa_fc = nn.Linear(features, features)

    def forward(self, H):
        # H: (B, C, F)

        # ----- Channel Attention -----
        Hc = H.mean(dim=2)  # (B, C)
        ac = torch.sigmoid(
            self.ca_fc2(F.relu(self.ca_fc1(Hc)))
        )                   # (B, C)
        C_out = H * ac.unsqueeze(-1)

        # ----- Feature Attention -----
        Hf = H.mean(dim=1)  # (B, F)
        af = F.softmax(self.fa_fc(Hf), dim=-1)
        F_out = H * af.unsqueeze(1)

        out = C_out+F_out
        return out

class PINN(nn.Module):
    def __init__(self, configs):
        super(PINN, self).__init__()
        self.solution_u = Solution_u(configs).to(device)
        self.dynamical_F1 = MLP(
            input_dim=configs.seq_len+2, output_dim=1,
            layers_num=configs.e_layers, hidden_dim=configs.d_model, dropout=0.2
        ).to(device)
        self.dynamical_F2 = MLP(
            input_dim=2*configs.seq_len+3, output_dim=1,
            layers_num=configs.e_layers, hidden_dim=configs.d_model, dropout=0.2
        ).to(device)

        self.downsampling = self.Linear1 = nn.Linear(configs.window_size, 1)

        self.alpha = configs.alpha
        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()


    def forward(self, xt):
        xt.requires_grad = True

        delta_x, t = self.time_weighted_average(xt)
        u = self.solution_u(torch.cat((t, delta_x), dim=2))

        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True, allow_unused=True)[0]
        u_x = grad(u.sum(), delta_x, create_graph=True, only_inputs=True, allow_unused=True)[0]

        F1 = self.dynamical_F1(torch.cat([delta_x, u, u_t], dim=2))

        f1 = u_t - F1

        return u, f1

    def time_weighted_average(self, xt, alpha=0.1):
        B, D, L = xt.shape

        k = torch.arange(L, 0, -1, device=xt.device, dtype=xt.dtype)
        unnorm_weights = alpha ** (k - 1)
        weights = unnorm_weights / unnorm_weights.sum()
        weights = weights.view(1, 1, L)
        x = xt[:,1:,:]
        z = x * weights
        z = z.sum(dim=-1).view(B, 1, D-1)
        t = xt[:,:1,-1:]+1
        return z,t

    def delta(self, xt):
        diff_c = xt[:, -1:, -1:]
        t = xt[:, :1, -1:]
        xt_first = xt[:, 1:, -1:]
        xt_last = xt[:, 1:, :1]
        xt = xt_first - xt_last
        xt[:, -1:, :] = diff_c
        xt = torch.cat((t, xt), dim=1)
        xt = xt.permute(0, 2, 1)
        delta_x = xt[:, :,1:]
        t = xt[:,:, :1] + 1
        return delta_x, t
