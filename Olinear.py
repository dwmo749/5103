# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:12:10 2025

@author: Administrator
"""


import numpy as np
import pandas as pd
from numpy.linalg import eigh
import os

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mask = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        # x [b,l,n]
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        self.mask = mask
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            if mask is None:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            else:
                assert isinstance(mask, torch.Tensor)
                # print(type(mask))
                x = x.masked_fill(mask, 0)  # in case other values are filled
                self.mean = (torch.sum(x, dim=1) / torch.sum(~mask, dim=1)).unsqueeze(1).detach()
                # self.mean could be nan or inf
                self.mean = torch.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is None:
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        else:
            self.stdev = (torch.sqrt(torch.sum((x - self.mean) ** 2, dim=1) / torch.sum(~mask, dim=1) + self.eps)
                          .unsqueeze(1).detach())
            self.stdev = torch.nan_to_num(self.stdev, nan=0.0, posinf=None, neginf=None)

    def _normalize(self, x, mask=None):
        self.mask = mask
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        x = x / self.stdev

        # x should be zero, if the values are masked
        if mask is not None:
            # forward fill
            # x, mask2 = forward_fill(x, mask)
            # x = x.masked_fill(mask2, 0)

            # mean imputation
            x = x.masked_fill(mask, 0)

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


def get_q_matrix(A:np.ndarray, time_lag=30):
    # for time_lag in [10]: #[256,468,512,300,400,500,620,820,920]
    A = A.astype(np.float32) # [B, T, N]
    B, T, N = A.shape
    total_T = B * T
    A = A.reshape(total_T, N)  # [Total_T, N]
    
    # Initialize a list to store covariance matrices for all features
    Sigma_list = []
    
    # Loop through all features
    for feature_idx in range(int(A.shape[1]/1)):
        # Construct the lagged matrix for the current feature
        lagged_matrix = np.array([
            A[i:A.shape[0]-time_lag+i+1, feature_idx]
            for i in range(time_lag)
        ])  #, dtype=np.float64
        
        if np.isnan(lagged_matrix).any():
            lagged_matrix = np.nan_to_num(lagged_matrix)
            print('nan in lagged_matrix')
            
        # Compute the covariance matrix for the lagged matrix
        cov_matrix = np.cov(lagged_matrix)
        diag_vec = np.diag(cov_matrix)
    
        if (diag_vec < 1e-4).any():
            continue
        
        cov_matrix = cov_matrix / diag_vec
    
        Sigma_list.append(np.array(cov_matrix, dtype=np.float32))
    
    # Average over all features to get the final Sigma
    Sigma = np.mean(Sigma_list, axis=0)
    
    # Compute eigenvalues and eigenvectors of Sigma
    eigenvalues, eigenvectors = eigh(Sigma)

    q_mat = np.flip(eigenvectors.T, axis=0)
    
    return q_mat # 衡量整个数据集分布情况的Q_T


class OrthoTrans(nn.Module):
    def __init__(self, seq_len, enc_in, embed_size, Q_chan_indep, q_mat):
        """
        初始化正交变换模块
        参数：
            seq_len: 输入时间步长度（T）
            enc_in: 变量数（N）
            embed_size: 嵌入维度（d）
            Q_chan_indep: 是否按变量独立使用Q矩阵
            precomputed_QT: 预计算的Q^⊤矩阵（numpy数组或torch.Tensor）
                            - 若Q_chan_indep=True：形状为 [N, T, T]（每个变量一个Q^⊤）
                            - 若Q_chan_indep=False：形状为 [T, T]（共享Q^⊤）
        """
        super(OrthoTrans, self).__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.embed_size = embed_size
        self.Q_chan_indep = Q_chan_indep
        
        # ---------------------- 处理预计算的Q^⊤ ----------------------
        # 转换为Tensor并移动到设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_mat = q_mat.copy()  # 关键：通过复制生成新数组，步长变为正
        if isinstance(q_mat, np.ndarray):
            self.Q_mat = torch.from_numpy(q_mat).to(torch.float32).to(device)
        else:
            self.Q_mat = q_mat.to(device)  # 若已是Tensor直接移动设备
            
        # 冻结Q^⊤参数（不参与训练更新）
        self.Q_mat.requires_grad = False  # 核心：固定参数
        
        # 嵌入向量（可学习，但Q^⊤固定）
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        
        self.revin_layer = RevIN(self.enc_in, affine=True)
        
        # learnable delta
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.seq_len))

    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_Trans(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # orthogonal transformation
        # [B, N, D, T]
        
        if self.Q_chan_indep:
            x_trans = torch.einsum('bndt,ntv->bndv', x, self.Q_mat.transpose(-1, -2))
        else:
            x_trans = torch.einsum('bndt,tv->bndv', x, self.Q_mat.transpose(-1, -2)) + self.delta1
            # added on 25/1/30
            # x_trans = F.gelu(x_trans)
            # [B, N, D, T]
        assert x_trans.shape[-1] == self.seq_len
        
        return x_trans # [B, N, D, T]

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        # x = self.revin_layer(x, mode='norm')
        x_ori = x

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, tau, D]
        x = self.Fre_Trans(x) # [B, N, D, T]
        
        B, N, D, T = x.shape
        # 步骤1：调整维度顺序，将 T 移到第2位 → [B, T, N, D]
        x = x.transpose(1, 3).transpose(2, 3)  # 先交换 N 和 T，再交换 D 和 N
        # 步骤2：合并最后两个维度（N 和 D）→ [B, T, N*D]
        x = x.reshape(B, T, N*D)
        
        return x


if __name__=='__main__':
    # A = np.array([[1,2,3],[4,5,6],[1,3,4],[2,3,4]])
    A = np.random.randn(20, 30, 15)
    q_mat = get_q_matrix(A)
    print(q_mat.shape)
    
    model = OrthoTrans(seq_len=30, enc_in=15, embed_size=4, Q_chan_indep=False, q_mat=q_mat)
    
    A = np.random.randn(2, 30, 15)
    A = torch.from_numpy(A).float()
    
    print(model(A).shape)

