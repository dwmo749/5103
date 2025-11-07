# CMAPSS FD001 RUL 预测脚本说明

本项目提供一个用于 NASA CMAPSS FD001 数据集的剩余寿命（RUL）预测训练与评估脚本，支持 `RNN` / `LSTM` / `Transformer` / `GNN+LSTM` 四类模型，包含稳健的数据读取、可配置的预处理、序列构建、训练、验证、测试评估与可视化输出。

## 目录结构与数据
- `train_rul_fd001.py`：主训练脚本。
- `data/`：FD001 数据目录（需包含 `train_FD001.csv`、`test_FD001.csv`、`RUL_FD001.csv`）。
- `artifacts/`：训练后保存的模型权重（`rnn_fd001.pt` 等）。
- `outputs/`：训练曲线、预测结果与评估图表输出。

FD001 数据文件格式：每行包括 `unit, cycle, setting1, setting2, setting3, s1..s21`。脚本自动适配无表头、逗号或空白分隔、常见编码。

## 依赖安装
请使用 Python 3.9+，并安装如下依赖：

```
pip install pandas numpy scikit-learn matplotlib seaborn torch tqdm
```

## 快速开始（Windows PowerShell）
- 训练所有模型并对测试集最后窗口评估：
```
python train_rul_fd001.py --data_dir data/ --models all --epochs 20 --seq_len 30 --batch_size 128
```

- 单独训练 Transformer：
```
python train_rul_fd001.py --data_dir data --models transformer --epochs 15 --seq_len 50
```

- 使用相关性图（GNN）：
```
python train_rul_fd001.py --data_dir data/ --models gnn --epochs 20 --seq_len 30 --gnn_corr_thr 0.3 --gnn_node_dim 32 --gnn_lstm_hidden 64 --gnn_lstm_layers 2
```

训练完成后：
- 模型权重保存在 `artifacts/`（如 `rnn_fd001.pt`）。
- 训练损失曲线保存在 `outputs/*_loss.png` 与对应 JSON。
- 测试每台发动机的最后窗口预测保存在 `outputs/*_test_preds.csv`。
- 综合评估图保存在 `outputs/test_scatter.png`, `outputs/test_residuals.png`, `outputs/test_metrics_bar.png`，指标汇总 `outputs/metrics.json`。

## 参数说明（CLI）
- `--data_dir`：数据目录，默认 `data`。
- `--seq_len`：滑动窗口长度（序列长度）。
- `--batch_size`：训练批大小。
- `--epochs`：训练轮数。
- `--lr`：学习率，默认 `1e-3`。
- `--val_ratio`：按发动机划分的验证比例，默认 `0.2`。
- `--models`：选择训练模型，`all | rnn | lstm | transformer | gnn`。
- `--save_dir`：保存模型目录，默认 `artifacts`。
- `--out_dir`：输出图表与预测目录，默认 `outputs`。
- `--seed`：随机种子，默认 `42`。
- `--device`：设备选择，`auto | cpu | cuda`，默认自动。

预处理相关：
- `--sensor_set`：传感器选择策略，`paper | all | auto`，默认 `paper`（使用论文常见的 15 个传感器）。
- `--scaler`：特征缩放方法，`zscore | minmax`，默认 `zscore`。
- `--rul_piecewise`：使用分段 RUL（上限裁剪），缺省行为为启用（如需禁用请移除此标志）。
- `--rul_max`：分段 RUL 上限值，默认 `130`。
- `--use_setting3`：是否包含 `setting3`（FD001 常见做法是不包含）。

GNN 特定参数：
- `--gnn_corr_thr`：基于绝对皮尔逊相关构图的阈值，默认 `0.3`。
- `--gnn_node_dim`：GCN 节点特征维度。
- `--gnn_lstm_hidden`：GCN 后 LSTM 隐藏维度。
- `--gnn_lstm_layers`：GCN 后 LSTM 层数。

## 训练与评估流程概览
1. 读取并规范化数据类型；为训练集构建 `RUL` 列（可分段裁剪）。
2. 选择传感器与工况设定列；对训练/测试帧进行缩放（`zscore` 或 `minmax`）。
3. 基于训练帧相关矩阵构建特征图（用于 GNN）。
4. 按发动机划分训练/验证集，滑动窗口序列化，标签为窗口末端的 RUL。
5. 训练所选模型，记录训练/验证损失曲线。
6. 在测试集上以“每台发动机的最后窗口”评估，计算 MAE/RMSE，输出预测与图表。

## 函数与类的原理和说明

以下描述均位于 `train_rul_fd001.py`：

- `set_seed(seed: int)`
  - 原理：统一 Python/NumPy/PyTorch 随机状态，确保复现实验。
  - 说明：设置 CPU 与 CUDA 的随机种子。

- `ensure_dir(path: str)`
  - 原理：幂等创建目录。
  - 说明：不存在时创建输出或模型保存目录。

- `CMAPSS_COLS`
  - 原理：定义标准列名顺序，便于无表头文件读取。
  - 说明：`['unit','cycle','setting1','setting2','setting3','s1'..'s21']`。

- `read_cmaps_file(path: str, encodings: List[str] = None) -> pd.DataFrame`
  - 原理：鲁棒读取；优先尝试逗号分隔，多编码；失败回退到空白分隔。
  - 说明：返回包含标准列的 DataFrame，失败抛错。

- `build_train_rul(df: pd.DataFrame, piecewise: bool = False, rul_max: int = 130) -> pd.DataFrame`
  - 原理：按发动机内最大周期 `max_cycle` 计算剩余寿命 `RUL = max_cycle - cycle`；可按上限裁剪（分段 RUL）。
  - 说明：返回含 `RUL` 与 `max_cycle` 的训练帧。

- `load_fd001(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - 原理：读取训练、测试与 RUL 文件；RUL 文件为单列，尝试多编码。
  - 说明：返回 `(train_df, test_df, rul_df)`。

- `enforce_numeric_types(df: pd.DataFrame) -> pd.DataFrame`
  - 原理：将 `unit/cycle` 转为整数、设定与传感器列转为浮点，避免字符串算术问题。
  - 说明：返回类型规范后的数据帧。

- `select_sensors_by_std(train_df: pd.DataFrame, sensor_cols: List[str], thr: float = 1e-4) -> List[str]`
  - 原理：依据标准差阈值过滤低方差传感器，保留信息量更高的特征。
  - 说明：返回选中的传感器列名列表。

- `scale_feature_frames(train_df, test_df, feature_cols, scaler_type='zscore') -> (train_scaled, test_scaled, scaler)`
  - 原理：对选定特征进行标准化或归一化；在训练集拟合，在测试集变换。
  - 说明：返回缩放后的训练/测试帧与缩放器对象。

- `build_adjacency_from_corr(train_df: pd.DataFrame, feature_cols: List[str], thr: float = 0.3) -> np.ndarray`
  - 原理：基于训练帧特征的绝对皮尔逊相关构图；相关系数绝对值≥阈值建立边；含自环；对称度归一化 `D^{-1/2} A D^{-1/2}`。
  - 说明：返回规范化邻接矩阵 `A_hat`，用于 GCN 聚合。

- `split_units(all_units: List[int], val_ratio: float, seed: int) -> (train_units, val_units)`
  - 原理：按随机种子打乱发动机编号；按比例切分训练/验证集合，保证发动机级划分。
  - 说明：返回训练与验证的发动机列表。

- `generate_sequences(df_scaled: pd.DataFrame, feature_cols: List[str], units: List[int], seq_len: int) -> (X, y)`
  - 原理：对每台发动机按时间排序，滑动窗口构建序列特征 `X`，标签为窗口末端的 `RUL`。
  - 说明：返回形状 `[num_windows, seq_len, feat_dim]` 的 `X` 和 `[num_windows]` 的 `y`。

- `build_test_last_windows(test_df_scaled: pd.DataFrame, feature_cols: List[str], seq_len: int) -> (X_last, order)`
  - 原理：对测试集每台发动机取最后 `seq_len` 个周期作为评估窗口。
  - 说明：返回堆叠的最后窗口与对应发动机编号顺序。

- `map_test_units_to_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> Dict[int, int]`
  - 原理：将测试发动机顺序与 `RUL_FD001.csv` 中的最终 RUL 行对齐映射。
  - 说明：若数量不匹配会警告但仍按顺序映射。

- `class SeqDataset(Dataset)`
  - 原理：简单序列数据集封装，返回张量形式的 `(X, y)`。
  - 说明：用于 `DataLoader` 迭代训练与验证。

- `class RNNRegressor(nn.Module)`
  - 原理：标准 RNN（tanh）按时间编码，取最后时间步特征经全连接输出 RUL。
  - 说明：适合较短序列与较简单时序关系。

- `class LSTMRegressor(nn.Module)`
  - 原理：LSTM 跨时间聚合，取最后时间步特征经全连接输出 RUL。
  - 说明：在长序列与非线性时序依赖下表现稳定。

- `class PositionalEncoding(nn.Module)`
  - 原理：正余弦位置编码，为 Transformer 提供时间位置信息。
  - 说明：与线性投影后相加到特征序列。

- `class TransformerRegressor(nn.Module)`
  - 原理：线性投影到 `d_model`，叠加位置编码，Transformer 编码器堆叠，取最后时间步回归。
  - 说明：通过注意力机制建模跨时间与跨特征的关系。

- `class GNNLSTMRegressor(nn.Module)`
  - 原理：每个时间步上对特征图执行 GCN 聚合（两层线性+邻接归一化传播），得到节点嵌入后再经 LSTM 跨时间建模，最后回归 RUL。
  - 说明：先聚合“特征间关系”，再学习“时间动态”；图由相关性构建，避免外部依赖。

- `train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3) -> (model, hist)`
  - 原理：MSE 损失 + Adam 优化；记录训练/验证损失曲线。
  - 说明：返回训练完的模型与损失历史字典。

- `evaluate_test_last(model, X_test_last_tensor, device, test_units_order, unit_to_final_rul) -> (preds, y_true, metrics)`
  - 原理：对每台测试发动机的最后窗口做前向，计算 MAE/RMSE。
  - 说明：返回预测数组、真值数组与指标字典。

- `plot_and_save_training_curves(hist, title, out_path)`
  - 原理：绘制并保存训练/验证损失曲线。
  - 说明：PNG 输出便于对比不同模型收敛情况。

- `plot_and_save_test_comparison(y_true, preds_dict, out_path_scatter, out_path_residuals, out_path_bar)`
  - 原理：绘制预测散点图、残差分布与指标柱状图，对比多模型表现。
  - 说明：并保存到 `outputs/` 便于复盘。

- `main()`
  - 原理：解析命令行参数，串联数据处理、训练与评估全流程。
  - 说明：脚本入口；在 `__main__` 中调用。

## 结果与文件输出说明
- `artifacts/*.pt`：各模型的权重文件。
- `outputs/*_loss.(png|json)`：训练/验证损失曲线与数值。
- `outputs/*_test_preds.csv`：测试每台发动机的最后窗口预测与真值。
- `outputs/test_scatter.png`：真值 vs 预测 散点。
- `outputs/test_residuals.png`：各模型残差分布。
- `outputs/test_metrics_bar.png`：各模型 MAE/RMSE 对比图。
- `outputs/metrics.json`：各模型测试指标汇总。

## 常见问题与建议
- 若出现“预测值趋同”或不收敛：
  - 增加 `epochs`（如 20–50）、调优 `lr`（如 1e-3→2e-3）。
  - 检查 `sensor_set`（尝试 `all`）与 `scaler`（尝试 `minmax`）。
  - 适度调整 `seq_len` 或验证比例。
- GNN 构图：
  - 可尝试不同 `--gnn_corr_thr`（如 0.2/0.4）或改为加权边（需要自定义）。
- 复现论文设置：
  - 按需开启/关闭 `--rul_piecewise` 与 `--use_setting3`，严格对齐传感器子集与预处理策略。

## 版权与许可
本仓库未附加额外许可证条款；如需开源许可，请自行添加相应 LICENSE 文件与头注释。