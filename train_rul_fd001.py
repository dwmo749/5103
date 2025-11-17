#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CMAPSS FD001 剩余寿命（RUL）预测脚本：支持 RNN / LSTM / Transformer

功能概览：
- 稳健数据读取（自动处理无表头、逗号或空白分隔）
- 预处理（去低方差传感器、MinMax归一化、滑动窗口序列化、按发动机划分训练/验证）
- 模型（RNN/LSTM/Transformer，PyTorch实现）
- 训练与验证（MSELoss + Adam，输出损失曲线）
- 测试评估（每台测试发动机的最后窗口预测，指标：MAE/RMSE）
- 可视化图表保存（EDA、训练曲线、预测散点与残差、指标对比）
- 模型与预测保存

用法示例：
  python train_rul_fd001.py --data_dir data/ --models all --epochs 20 --seq_len 30 --batch_size 128
  python train_rul_fd001.py --data_dir data/ --models transformer --epochs 15 --seq_len 50
  python train_rul_fd001.py --models all --epochs 20 --seq_len 30 --batch_size 128 --sensor_set paper --scaler zscore --rul_piecewise --rul_max 130

依赖：pandas numpy scikit-learn matplotlib seaborn torch tqdm
  pip install pandas numpy scikit-learn matplotlib seaborn torch tqdm

注意：本脚本默认数据目录结构为 NASA CMAPSS 的 FD001 版本：
- train_FD001.csv, test_FD001.csv, RUL_FD001.csv
"""

import os
import re
import math
import json
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Olinear import OrthoTrans, get_q_matrix
from Otransformer import OTransformerRegressor
from LearnableAdjGNNLSTMRegressor import LeanableAdjGNNLSTMRegressor


# -----------------------------
# Utils & Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
CMAPSS_COLS = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]


def read_cmaps_file(path: str, encodings: List[str] = None) -> pd.DataFrame:
    """Robust reader for CMAPSS-like CSV files with unknown delimiter and encoding.
    - Tries multiple encodings for comma-separated first.
    - Falls back to whitespace-separated (sep='\s+', engine='python').
    """
    if encodings is None:
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']

    # Try comma-separated
    for enc in encodings:
        try:
            df = pd.read_csv(path, header=None, names=CMAPSS_COLS, encoding=enc)
            if df.shape[1] == len(CMAPSS_COLS):
                return df
        except Exception:
            pass

    # Fallback: whitespace-separated
    for enc in encodings:
        try:
            df = pd.read_csv(path, header=None, names=CMAPSS_COLS, sep=r'\s+', engine='python', encoding=enc)
            if df.shape[1] == len(CMAPSS_COLS):
                return df
        except Exception:
            pass

    raise RuntimeError(f"Failed to read file '{path}' with supported encodings and separators.")


def build_train_rul(df: pd.DataFrame, piecewise: bool = False, rul_max: int = 130) -> pd.DataFrame:
    df = df.copy()
    # ensure no NA in unit/cycle before arithmetic
    df = df.dropna(subset=['unit', 'cycle'])
    # cast to int for groupby stability
    df['unit'] = df['unit'].astype('int64')
    df['cycle'] = df['cycle'].astype('int64')
    df['max_cycle'] = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = df['max_cycle'] - df['cycle']
    if piecewise and rul_max is not None and rul_max > 0:
        df['RUL'] = np.minimum(df['RUL'], rul_max)
    return df


def load_fd001(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, 'train_FD001.csv')
    test_path = os.path.join(data_dir, 'test_FD001.csv')
    rul_path = os.path.join(data_dir, 'RUL_FD001.csv')

    train_df = read_cmaps_file(train_path)
    test_df = read_cmaps_file(test_path)
    # RUL file is simple single-column; try multiple encodings too
    for enc in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
        try:
            rul_df = pd.read_csv(rul_path, header=None, names=['RUL'], encoding=enc)
            break
        except Exception:
            continue
    return train_df, test_df, rul_df


def enforce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # unit / cycle as integers
    df['unit'] = pd.to_numeric(df['unit'], errors='coerce').astype('Int64')
    df['cycle'] = pd.to_numeric(df['cycle'], errors='coerce').astype('Int64')
    # settings and sensors as float
    cols_float = ['setting1', 'setting2', 'setting3'] + [c for c in df.columns if c.startswith('s')]
    for c in cols_float:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Float64')
    return df


def select_sensors_by_std(train_df: pd.DataFrame, sensor_cols: List[str], thr: float = 1e-4) -> List[str]:
    stds = train_df[sensor_cols].std()
    selected = [s for s in sensor_cols if stds[s] > thr]
    return selected


def scale_feature_frames(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], scaler_type: str = 'zscore') -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    if scaler_type == 'zscore':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError('Unknown scaler_type: ' + scaler_type)
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, scaler


def build_adjacency_from_corr(train_df: pd.DataFrame, feature_cols: List[str], thr: float = 0.3) -> np.ndarray:
    """Build a feature graph adjacency (normalized) based on absolute Pearson correlation.
    Nodes are features in feature_cols; edges if |corr| >= thr; include self-loops; symmetric normalization.
    """
    corr = train_df[feature_cols].corr().values.astype(np.float32)
    adj = (np.abs(corr) >= float(thr)).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    deg = adj.sum(axis=1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D_inv_sqrt = np.diag(inv_sqrt)
    A_hat = D_inv_sqrt @ adj @ D_inv_sqrt
    return A_hat


def split_units(all_units: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    units = list(sorted(all_units))
    rng.shuffle(units)
    val_size = max(1, int(len(units) * val_ratio))
    val_units = units[:val_size]
    train_units = units[val_size:]
    return train_units, val_units


def generate_sequences(df_scaled: pd.DataFrame, feature_cols: List[str], units: List[int], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for u in units:
        udf = df_scaled[df_scaled['unit'] == u].sort_values('cycle')
        feats = udf[feature_cols].values
        rul = udf['RUL'].values
        for i in range(len(udf) - seq_len + 1):
            X.append(feats[i:i + seq_len])
            y.append(rul[i + seq_len - 1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def build_test_last_windows(test_df_scaled: pd.DataFrame, feature_cols: List[str], seq_len: int) -> Tuple[np.ndarray, List[int]]:
    X_last, order = [], []
    for u in sorted(test_df_scaled['unit'].dropna().astype('int64').unique()):
        udf = test_df_scaled[test_df_scaled['unit'] == u].sort_values('cycle')
        feats = udf[feature_cols].values
        if len(feats) < seq_len:
            continue
        X_last.append(feats[-seq_len:])
        order.append(u)
    X_last = np.array(X_last, dtype=np.float32)
    return X_last, order


def map_test_units_to_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> Dict[int, int]:
    units = sorted(test_df['unit'].dropna().astype('int64').unique().tolist())
    if len(units) != len(rul_df):
        print("[WARN] Test units and RUL rows count mismatch. Assuming order-aligned mapping.")
    return {u: int(rul_df.iloc[i]['RUL']) for i, u in enumerate(units)}


# -----------------------------
# Dataset & Dataloader
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# Models
# -----------------------------
class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, nonlinearity='tanh'):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, nonlinearity=nonlinearity)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


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


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.posenc(h)
        out = self.encoder(h)
        last = out[:, -1, :]
        return self.fc(last)


class GNNLSTMRegressor(nn.Module):
    """Graph + Temporal model: per-timestep GCN over feature graph, followed by LSTM across time."""
    def __init__(self, num_nodes: int, adj_matrix: np.ndarray, node_feat_dim: int = 32,
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
        self.register_buffer('A_hat', torch.tensor(adj_matrix, dtype=torch.float32))

    def gcn_forward_bt(self, x_bt_n: torch.Tensor) -> torch.Tensor:
        """x_bt_n: [B*T, N] -> returns [B*T, node_feat_dim] after GCN + mean over nodes."""
        # project per node
        h = self.gcn_w1(x_bt_n.unsqueeze(-1))  # [BT, N, H]
        # graph aggregation
        A = self.A_hat.to(x_bt_n.device)
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


# -----------------------------
# Training & Evaluation
# -----------------------------
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                epochs: int = 20, lr: float = 1e-3) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {'train_loss': [], 'val_loss': []}

    for ep in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        hist['train_loss'].append(float(np.mean(train_losses)))
        hist['val_loss'].append(float(np.mean(val_losses)))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs} - train {hist['train_loss'][-1]:.4f} - val {hist['val_loss'][-1]:.4f}")
    return model, hist


def evaluate_test_last(model: nn.Module, X_test_last_tensor: torch.Tensor, device: torch.device,
                       test_units_order: List[int], unit_to_final_rul: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    model.eval()
    with torch.no_grad():
        preds = model(X_test_last_tensor.to(device)).cpu().numpy().squeeze()
    y_true = np.array([unit_to_final_rul[u] for u in test_units_order], dtype=np.float32)
    mae = mean_absolute_error(y_true, preds)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    return preds, y_true, {'MAE': float(mae), 'RMSE': rmse}


def plot_and_save_training_curves(hist: Dict[str, List[float]], title: str, out_path: str):
    plt.figure(figsize=(6, 4))
    plt.plot(hist['train_loss'], label='train')
    plt.plot(hist['val_loss'], label='val')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_and_save_test_comparison(y_true: np.ndarray, preds_dict: Dict[str, np.ndarray], out_path_scatter: str, out_path_residuals: str, out_path_bar: str):
    # Scatter
    n = len(preds_dict)
    plt.figure(figsize=(4 * n, 4))
    for i, (name, preds) in enumerate(preds_dict.items(), start=1):
        plt.subplot(1, n, i)
        plt.scatter(y_true, preds, alpha=0.7)
        mn, mx = float(np.min(y_true)), float(np.max(y_true))
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('True RUL')
        plt.ylabel('Pred RUL')
        plt.title(name)
    plt.tight_layout()
    plt.savefig(out_path_scatter, dpi=150)
    plt.close()

    # Residuals
    plt.figure(figsize=(4 * n, 4))
    for i, (name, preds) in enumerate(preds_dict.items(), start=1):
        plt.subplot(1, n, i)
        sns.histplot(preds - y_true, bins=20, kde=True)
        plt.title(f"{name} residuals")
    plt.tight_layout()
    plt.savefig(out_path_residuals, dpi=150)
    plt.close()

    # Metrics bar
    import pandas as pd
    rows = []
    for name, preds in preds_dict.items():
        mae = mean_absolute_error(y_true, preds)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        rows.append({'Model': name, 'MAE': mae, 'RMSE': rmse})
    dfm = pd.DataFrame(rows)
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=dfm.melt(id_vars='Model'), x='Model', y='value', hue='variable')
    ax.set_title('Test metrics comparison')
    plt.tight_layout()
    plt.savefig(out_path_bar, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='CMAPSS FD001 RUL training script (RNN/LSTM/Transformer)')
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory containing FD001 CSV files')
    parser.add_argument('--seq_len', type=int, default=30, help='Sliding window length')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio by unit')
    parser.add_argument('--models', type=str, default='all', choices=['all', 'rnn', 'lstm', 'transformer','otransformer', 'gnn', 'learnable_gnn'], help='Which model(s) to train')
    parser.add_argument('--save_dir', type=str, default='artifacts', help='Directory to save model weights')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Directory to save outputs and figures')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Computation device')
    # Paper-replication preprocessing flags
    parser.add_argument('--sensor_set', type=str, default='paper', choices=['auto', 'paper', 'all'], help='Sensor selection strategy')
    parser.add_argument('--scaler', type=str, default='zscore', choices=['zscore', 'minmax'], help='Feature scaling method')
    parser.add_argument('--rul_piecewise', action='store_true', help='Use piecewise RUL with cap')
    parser.add_argument('--rul_max', type=int, default=130, help='RUL cap value when piecewise')
    parser.add_argument('--use_setting3', action='store_true', help='Include setting3 feature (FD001 often excludes)')
    # Time decoupling / time feature
    parser.add_argument('--time_feature', type=str, default='none', choices=['none', 'frac'], help='Add time feature channel (e.g., cycle/max_cycle)')
    # GNN-specific
    parser.add_argument('--gnn_corr_thr', type=float, default=0.3, help='Correlation threshold to build feature graph')
    parser.add_argument('--gnn_node_dim', type=int, default=32, help='Node feature dim in GCN')
    parser.add_argument('--gnn_lstm_hidden', type=int, default=64, help='Hidden dim of LSTM after GCN')
    parser.add_argument('--gnn_lstm_layers', type=int, default=2, help='Number of LSTM layers after GCN')

    args = parser.parse_args()
    # Defaults aligned with common paper settings: piecewise cap, exclude setting3
    if not args.rul_piecewise:
        # Allow overriding via flag, but set default True if user didn't specify
        # We infer whether the flag is present via argparse; since action='store_true', absence means False.
        # To honor paper replication request, enable when not explicitly disabled.
        args.rul_piecewise = True
    # Default: exclude setting3 unless explicitly requested
    if args.use_setting3 is None:
        args.use_setting3 = False

    set_seed(args.seed)
    ensure_dir(args.save_dir)
    ensure_dir(args.out_dir)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print('[INFO] Using device:', device)

    # Load data
    train_df, test_df, rul_df = load_fd001(args.data_dir)
    # Enforce numeric types to avoid string arithmetic errors
    train_df = enforce_numeric_types(train_df)
    test_df = enforce_numeric_types(test_df)
    # RUL column numeric
    rul_df['RUL'] = pd.to_numeric(rul_df['RUL'], errors='coerce').astype('Int64')
    train_df = build_train_rul(train_df, piecewise=args.rul_piecewise, rul_max=args.rul_max)

    # Test pseudo RUL (for visualization) & unit mapping
    test_df = test_df.copy()
    test_df['max_cycle'] = test_df.groupby('unit')['cycle'].transform('max')
    test_df['pseudo_RUL'] = test_df['max_cycle'] - test_df['cycle']
    if args.rul_piecewise and args.rul_max is not None and args.rul_max > 0:
        test_df['pseudo_RUL'] = np.minimum(test_df['pseudo_RUL'], args.rul_max)

    unit_to_final_rul = map_test_units_to_rul(test_df, rul_df)

    # Sensor selection & scaling
    sensor_cols_all = [c for c in train_df.columns if re.fullmatch(r's\d+', c)]
    paper_sensors = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    if args.sensor_set == 'paper':
        selected_sensors = paper_sensors
    elif args.sensor_set == 'all':
        selected_sensors = sensor_cols_all
    else:
        selected_sensors = select_sensors_by_std(train_df, sensor_cols_all, thr=1e-4)
    settings = ['setting1', 'setting2'] + (['setting3'] if args.use_setting3 else [])
    feature_cols = settings + selected_sensors
    print('[INFO] Selected sensors:', selected_sensors)
    print('[INFO] Using settings:', settings)

    train_scaled, test_scaled, scaler = scale_feature_frames(train_df, test_df, feature_cols, scaler_type=args.scaler)
    # Precompute feature graph for GNN
    A_hat = build_adjacency_from_corr(train_scaled, feature_cols, thr=args.gnn_corr_thr)

    # Optional: add normalized time feature (cycle/max_cycle) to decouple temporal phase
    if args.time_feature == 'frac':
        # Guard against division by zero
        train_scaled['max_cycle'] = train_scaled.groupby('unit')['cycle'].transform('max')
        test_scaled['max_cycle'] = test_scaled.groupby('unit')['cycle'].transform('max')
        eps = 1e-6
        train_scaled['time_frac'] = (train_scaled['cycle'].astype('float32') / (train_scaled['max_cycle'].astype('float32') + eps)).astype('float32')
        test_scaled['time_frac'] = (test_scaled['cycle'].astype('float32') / (test_scaled['max_cycle'].astype('float32') + eps)).astype('float32')
        feature_cols = feature_cols + ['time_frac']
        print('[INFO] Time feature enabled: time_frac added to features')

    # EDA: optional summary/plots
    try:
        plt.figure(figsize=(10, 8))
        corr = train_scaled[feature_cols].corr()
        sns.heatmap(corr, cmap='coolwarm', center=0)
        plt.title('Sensor & setting correlation (Train, scaled)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'corr_heatmap.png'), dpi=150)
        plt.close()
    except Exception as e:
        print('[WARN] EDA heatmap failed:', e)

    # Split units
    all_units = sorted(train_scaled['unit'].dropna().astype('int64').unique().tolist())
    train_units, val_units = split_units(all_units, args.val_ratio, args.seed)
    print('[INFO] Train units:', len(train_units), 'Val units:', len(val_units))

    # Build sequences
    X_train, y_train = generate_sequences(train_scaled, feature_cols, train_units, args.seq_len)
    X_val, y_val = generate_sequences(train_scaled, feature_cols, val_units, args.seq_len)
    X_test_last, test_units_order = build_test_last_windows(test_scaled, feature_cols, args.seq_len)
    X_dim = X_train.shape[-1]
    print('[INFO] X_train:', X_train.shape, 'X_val:', X_val.shape, 'X_test_last:', X_test_last.shape, 'X_dim:', X_dim)

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    X_test_last_tensor = torch.tensor(X_test_last, dtype=torch.float32)

    # Train models
    # X_train_whole = train_scaled[feature_cols].values  # 实际上你需要按unit和cycle排好序形成 [B, T, N] tensor
    # X_train_whole = X_train_whole.reshape(-1, args.seq_len, len(feature_cols)).astype(np.float32)
    q_mat = get_q_matrix(X_train)  # 输出形状为 [T, T]
    trained = {}
    histories = {}
    to_run = []
    if args.models == 'all':
        to_run = ['rnn', 'lstm', 'transformer', 'otransformer', 'gnn', 'learnable_gnn']
    else:
        to_run = [args.models]

    for name in to_run:
        if name == 'rnn':
            model = RNNRegressor(input_dim=X_dim, hidden_dim=64, num_layers=2)
        elif name == 'lstm':
            model = LSTMRegressor(input_dim=X_dim, hidden_dim=64, num_layers=2)
        elif name == 'transformer':
            model = TransformerRegressor(input_dim=X_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1)
        elif name == 'otransformer':
            model = OTransformerRegressor(input_dim=X_dim, seq_len=args.seq_len, q_mat=q_mat,
                                      embed_size=4, d_model=128, nhead=4, num_layers=2)
        elif name == 'gnn':
            model = GNNLSTMRegressor(num_nodes=X_dim, adj_matrix=A_hat, node_feat_dim=args.gnn_node_dim,
                                     gcn_layers=2, lstm_hidden=args.gnn_lstm_hidden, lstm_layers=args.gnn_lstm_layers)
        elif name == 'learnable_gnn':
            model = LeanableAdjGNNLSTMRegressor(num_nodes=X_dim,
                                           node_feat_dim=args.gnn_node_dim,
                                           gcn_layers=2,
                                           lstm_hidden=args.gnn_lstm_hidden,
                                           lstm_layers=args.gnn_lstm_layers)
        else:
            raise ValueError('Unknown model: ' + name)

        print(f'[INFO] Training {name} ...')
        model, hist = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
        trained[name] = model
        histories[name] = hist

        # Save model
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'{name}_fd001.pt'))

        # Save training curve
        plot_and_save_training_curves(hist, title=f'{name.upper()} loss', out_path=os.path.join(args.out_dir, f'{name}_loss.png'))
        with open(os.path.join(args.out_dir, f'{name}_loss.json'), 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)

    # Evaluate on test last window
    preds_dict = {}
    metrics_dict = {}
    for name, model in trained.items():
        preds, y_true, metrics = evaluate_test_last(model, X_test_last_tensor, device, test_units_order, unit_to_final_rul)
        preds_dict[name.upper()] = preds
        metrics_dict[name.upper()] = metrics
        # Save predictions per-unit
        out_pred_csv = os.path.join(args.out_dir, f'{name}_test_preds.csv')
        pd.DataFrame({'unit': test_units_order, 'true_RUL': y_true, 'pred_RUL': preds}).to_csv(out_pred_csv, index=False)

    # Save metrics and plots
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

    print('[INFO] Test metrics:')
    for name, m in metrics_dict.items():
        print(name, m)

    if len(preds_dict) > 0:
        plot_and_save_test_comparison(
            y_true=y_true,
            preds_dict=preds_dict,
            out_path_scatter=os.path.join(args.out_dir, 'test_scatter.png'),
            out_path_residuals=os.path.join(args.out_dir, 'test_residuals.png'),
            out_path_bar=os.path.join(args.out_dir, 'test_metrics_bar.png')
        )

    print('[INFO] All done. Models saved to:', args.save_dir, 'Outputs saved to:', args.out_dir)


if __name__ == '__main__':
    main()