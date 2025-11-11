# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:32:54 2025

@author: Administrator
"""

import torch
import torch.nn as nn

class LeanableAdjGNNLSTMRegressor(nn.Module):
    """Graph + Temporal model: per-timestep GCN over feature graph, followed by LSTM across time."""
    def __init__(self, num_nodes: int, node_feat_dim: int = 32,
                 gcn_layers: int = 2, lstm_hidden: int = 64, lstm_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.gcn_layers = gcn_layers
        # GCN weights (feature dimension 1 -> node_feat_dim, then node_feat_dim -> node_feat_dim)
        self.gcn_w1 = nn.Linear(1, node_feat_dim)
        self.gcn_w2 = nn.Linear(node_feat_dim, node_feat_dim)
        self.lstm = nn.LSTM(node_feat_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)
        self.A_learnable = nn.Parameter(torch.randn(num_nodes, num_nodes)) # 可学习参数

    def gcn_forward_bt(self, x_bt_n: torch.Tensor) -> torch.Tensor:
        """x_bt_n: [B*T, N] -> returns [B*T, node_feat_dim] after GCN + mean over nodes."""
        # project per node
        h = self.gcn_w1(x_bt_n.unsqueeze(-1))  # [BT, N, H]
        # graph aggregation
        A = self.A_learnable
        h = torch.relu(torch.einsum('ij,bnh->bih', A, h))  # [BT, N, H]
        if self.gcn_layers > 1:
            h = self.gcn_w2(h)  # [BT, N, H]
            h = torch.relu(torch.einsum('ij,bnh->bih', A, h))
        # pool nodes
        h_mean = h.mean(dim=1)  # [BT, H]
        return h_mean

    def forward(self, x):
        # x: [B, T, N]
        B, T, N = x.shape
        x_bt_n = x.reshape(B * T, N)
        h_bt = self.gcn_forward_bt(x_bt_n)  # [BT, H]
        h = h_bt.view(B, T, self.node_feat_dim)
        out, _ = self.lstm(h)
        last = out[:, -1, :]
        return self.fc(last)

if __name__=="__main__":
    A = np.random.randn(8, 30, 15) #这就是一个输入[B, T, N]
    A = torch.from_numpy(A).float()
    
    lagl = LeanableAdjGNNLSTMRegressor(15)
    
    print(lagl(A))

