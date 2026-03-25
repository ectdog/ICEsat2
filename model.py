# -*- coding: utf-8 -*-
"""
model.py — 时序Transformer网络结构及训练辅助组件

功能说明：
    定义多输出时序Transformer（TemporalTransformerMO）及其训练所需的
    辅助类与工具函数，包括数据集封装、EMA影子权重、傅里叶位置编码等。

主要内容：
    TemporalTransformerMO — 多年多输出时序Transformer，输入各年遥感特征序列，
                            同时预测各年的潮间带高程（height_1985）。
    EMA                   — 指数滑动平均，用于验证阶段提供更稳定的预测。
    PackedDS              — PyTorch Dataset封装，供DataLoader使用。
    fourier_features      — 2D坐标傅里叶位置编码。
    train_val_split       — 可复现的随机训练/验证划分。
    make_loader           — DataLoader函数，使用config中的稳定性参数。
    set_seed              — 全局随机种子设置。
    r2_score_np           — NumPy实现的R²评价指标。

作者：张仟龙
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config as cfg


# ─────────────────────────────────────────────
# 数据集封装
# ─────────────────────────────────────────────

class PackedDS(Dataset):
    """
    轻量级PyTorch Dataset，封装已组装好的输入张量与目标张量。

    参数
    ----
    Xin : float32 [N, T, C_ext] — 扩展特征序列（特征+掩膜+距离+空间编码）
    Y   : float32 [N, T]        — 目标高程值
    Ym  : float32 [N, T]        — 目标可用性掩膜
    """
    def __init__(self, Xin: np.ndarray, Y: np.ndarray, Ym: np.ndarray):
        self.Xin = Xin.astype(np.float32)
        self.Y   = Y.astype(np.float32)
        self.Ym  = Ym.astype(np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.Xin[i]),
            torch.from_numpy(self.Y[i]),
            torch.from_numpy(self.Ym[i]),
        )


# ─────────────────────────────────────────────
# 模型定义
# ─────────────────────────────────────────────

class TemporalTransformerMO(nn.Module):
    """
    多输出时序Transformer，用于潮间带高程反演。

    网络结构
    --------
    1. 线性投影：将扩展特征向量（特征 + 有效掩膜 + 归一化距离 + 傅里叶空间编码）
       映射到 d_model 维隐空间。
    2. Token Dropout：训练时以 p=0.05 随机将整个年份Token置零，起正则化作用。
    3. 可学习年份嵌入：按年份位置加上可学习的加性嵌入向量（Xavier初始化）。
    4. Transformer编码器：n_layers 层 Pre-LN Transformer层，GELU激活，batch_first=True。
    5. 逐Token回归头：LayerNorm → Linear → GELU → Dropout → Linear → 标量，
       对每个年份的输出Token独立计算预测值。
    6. 年份偏置：每个年份位置附加一个可学习偏置项。

    参数
    ----
    in_dim  : int — 输入特征维度 (C_ext = C + 1 + 1 + 2 + 4*FOURIER_K)
    n_years : int — 年份序列长度（通常为5）
    d_model, n_head, n_layers, d_ff, dropout : Transformer标准结构参数
    """

    def __init__(
        self,
        in_dim: int,
        n_years: int,
        d_model: int   = cfg.D_MODEL,
        n_head:  int   = cfg.N_HEAD,
        n_layers: int  = cfg.N_LAYERS,
        d_ff:    int   = cfg.D_FF,
        dropout: float = cfg.DROPOUT,
    ):
        super().__init__()
        self.n_years = n_years

        self.in_proj    = nn.Linear(in_dim, d_model, bias=True)
        self.token_drop = nn.Dropout(0.05)

        self.year_emb = nn.Parameter(torch.zeros(1, n_years, d_model))
        nn.init.xavier_uniform_(self.year_emb)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head_per_token = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.year_bias = nn.Parameter(torch.zeros(1, n_years))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : float32 Tensor [B, T, C_ext]

        返回
        ----
        pred : float32 Tensor [B, T] — 各年份预测高程（原始单位，米）
        """
        B, T, _ = x.shape
        h = self.in_proj(x)
        h = self.token_drop(h)
        h = h + self.year_emb[:, :T, :]
        z = self.encoder(h)
        pred = self.head_per_token(z).squeeze(-1)
        pred = pred + self.year_bias[:, :T]
        return pred


# ─────────────────────────────────────────────
# EMA（指数滑动平均）
# ─────────────────────────────────────────────

class EMA:
    """
    维护模型参数的指数滑动平均影子权重。

    验证时通过 load_shadow / restore 临时将影子权重换入模型，
    使验证指标更稳定，不影响正常训练参数更新。

    参数
    ----
    model : nn.Module
    decay : float — 滑动平均衰减系数（通常取 0.99～0.999）
    """

    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.decay = decay
        self.shadow: dict = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup: dict = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        """每个训练step后调用，更新影子权重。"""
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def load_shadow(self, model: nn.Module):
        """将影子权重临时写入模型（验证前调用）。"""
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                key = n
                if key not in self.shadow and key.startswith("module."):
                    key = key[len("module."):]
                if key in self.shadow:
                    self.backup[n] = p.data.clone()
                    p.data.copy_(self.shadow[key])

    def restore(self, model: nn.Module):
        """验证结束后恢复训练参数（验证后调用）。"""
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def set_seed(seed: int = cfg.SEED):
    """设置全局随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算R²决定系数（NumPy实现）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def train_val_split(N: int, val_ratio: float = cfg.VAL_SPLIT, seed: int = cfg.SEED):
    """
    返回 (train_indices, val_indices)，通过可复现的随机打乱实现划分。
    前 val_ratio 比例的打乱索引构成验证集。
    """
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(N * val_ratio)
    return idx[n_val:], idx[:n_val]


def fourier_features(xy_norm: np.ndarray, K: int) -> np.ndarray:
    """
    2D坐标傅里叶位置编码。

    对 k = 0 … K-1，拼接 [sin(2^k π x), cos(2^k π x), sin(2^k π y), cos(2^k π y)]。
    输出形状：[N, 4*K]。
    """
    if K <= 0:
        return np.zeros((xy_norm.shape[0], 0), dtype=np.float32)
    xs, ys = xy_norm[:, 0], xy_norm[:, 1]
    feats = []
    for k in range(K):
        s = (2.0 ** k) * math.pi
        feats.extend([np.sin(s * xs), np.cos(s * xs), np.sin(s * ys), np.cos(s * ys)])
    return np.stack(feats, axis=1).astype(np.float32)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    """使用config中的稳定性配置创建DataLoader。"""
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        timeout=cfg.DATALOADER_TIMEOUT,
        drop_last=drop_last,
    )
    if cfg.NUM_WORKERS and cfg.NUM_WORKERS > 0:
        if cfg.PERSISTENT_WORKERS is not None:
            kwargs["persistent_workers"] = cfg.PERSISTENT_WORKERS
        if cfg.PREFETCH_FACTOR is not None:
            kwargs["prefetch_factor"] = cfg.PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)
