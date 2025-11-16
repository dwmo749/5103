# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:00:29 2025

@author: Administrator
"""

from Olinear import OrthoTrans, get_q_matrix

import math
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]


class OTransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, q_mat, embed_size=4, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim*embed_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, 1)
        
        self.otrans = OrthoTrans(seq_len=seq_len, enc_in=input_dim, embed_size=embed_size, Q_chan_indep=False, q_mat=q_mat)

    def forward(self, x):
        x = self.otrans(x) # 正交变换[B, T, N]->[B, T, N*embed_size]
        
        # 之后正常处理
        h = self.input_proj(x)
        h = self.posenc(h)
        out = self.encoder(h)
        last = out[:, -1, :]
        return self.fc(last)
    
    
if __name__=='__main__':
    # 第一步
    A = np.random.randn(100, 30, 15) #这就是训练集[B, T, N]
    q_mat = get_q_matrix(A) # 1.这个预先求好，在整个训练集里只求一次
    print(q_mat.shape)
    
    # 第二步
    A = np.random.randn(8, 30, 15) #这就是一个输入[B, T, N]
    A = torch.from_numpy(A).float()
    
    otf = OTransformerRegressor(input_dim=15, seq_len=30, q_mat=q_mat) # 2.用q_mat初始化tf层
    
    print(otf(A))
    

    
    
