# -*- coding: utf-8 -*-
"""
config.py — 多年期时序Transformer训练超参数与路径配置

功能说明：
    集中管理所有可由用户修改的常量（数据路径、模型超参数、训练策略等），
    使 data_pipeline.py、model.py、trainer.py 保持与路径无关，
    便于在不同运行环境中复现实验。

使用方法：
    在运行 trainer.py 之前，将"数据路径"部分修改为本地实际路径，
    其余参数可根据需要调整。

作者：张仟龙
"""

from typing import List

# ============================================================
# 数据路径  ← 根据本地环境修改
# ============================================================
CSV_FILES: List[str] = [
    r"E:\zql\Data\Train_features\Train_csv\cleaned\normalized\ATL03_with_features_2019_calibrated_normalized.csv",
    r"E:\zql\Data\Train_features\Train_csv\cleaned\normalized\ATL03_with_features_2020_calibrated_normalized.csv",
    r"E:\zql\Data\Train_features\Train_csv\cleaned\normalized\ATL03_with_features_2021_calibrated_normalized.csv",
    r"E:\zql\Data\Train_features\Train_csv\cleaned\normalized\ATL03_with_features_2022_calibrated_normalized.csv",
    r"E:\zql\Data\Train_features\Train_csv\cleaned\normalized\ATL03_with_features_2023_calibrated_normalized.csv",
]
OUTDIR: str = r"E:\CrossYear_Diagnostics\runs_r2_0p8_plus_1231"   # 训练结果输出目录
WARM_START_CKPT: str = r""  # 热启动检查点路径，为空则冷启动

# ============================================================
# 目标变量与特征列配置
# ============================================================
TARGET_COL: str = "height_1985"
CLIP_RANGE = (-10.0, 10.0)          # 目标值硬截断范围（单位：米）

FEATURE_PREFIXES: List[str] = [
    "Inundation", "S1", "CCI", "NDWI", "GCC", "GNDVI",
    "IBI", "NDVIre", "NIRv", "PRI", "VARI"
]
MANUAL_FEATURES: List[str] = []     # 若非空，则直接使用该列表，忽略前缀匹配
PHENO_EXCLUDE_PREFIXES: List[str] = []


# ============================================================
# KNN 空间软聚合参数
# ============================================================
TOL_M:    float = 60.0    # 最大搜索半径（米）
KNN_K:    int   = 3       # 最近邻数量
KNN_SIGMA: float = 25.0   # 距离加权高斯核标准差（米）
MIN_STEPS: int  = 2       # 每个锚点至少需要有效年份数
IMPUTE_STRATEGY: str = "median"   # 缺失特征填补策略："median" 或 "mean"
UTM_EPSG: int   = 32651   # UTM投影坐标系（辽河口区域使用51N带）

# ============================================================
# 训练超参数
# ============================================================
SEED:                int   = 42
BATCH_SIZE:          int   = 4096
NUM_EPOCHS:          int   = 2000
PHASE1_FREEZE_EPOCHS: int  = 10    # 阶段1（仅训练头部）的轮数
BASE_LR_HEAD:        float = 1e-3
BASE_LR_FULL:        float = 3e-4
WEIGHT_DECAY:        float = 1e-4
VAL_SPLIT:           float = 0.10
PATIENCE:            int   = 120   # 早停耐心轮数
USE_AMP:             bool  = True  # 是否使用自动混合精度
GRAD_CLIP_NORM:      float = 2.0   # 梯度裁剪范数上限
TOPK_TO_SAVE:        int   = 3     # 保留验证RMSE最优的K个检查点
SAVE_EVERY:          int   = 5     # 每隔多少轮保存 last.pt

# DataLoader 稳定性配置
NUM_WORKERS:        int   = 0
PIN_MEMORY:         bool  = False
PERSISTENT_WORKERS: bool  = False
PREFETCH_FACTOR     = None
DATALOADER_TIMEOUT: int   = 0

# ============================================================
# 模型结构参数
# ============================================================
D_MODEL:   int   = 192
N_HEAD:    int   = 6
N_LAYERS:  int   = 8
D_FF:      int   = 768
DROPOUT:   float = 0.15
FOURIER_K: int   = 6       # 傅里叶位置编码频率数
FOURIER_NORM: bool = True  # 是否将 (x,y) 归一化至 [0,1] 后再做傅里叶编码

# ============================================================
# 正则化参数
# ============================================================
LAMBDA_TV: float = 0.03    # 时序总变差正则化权重（推荐范围 0.01～0.1）
