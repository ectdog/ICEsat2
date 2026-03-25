# -*- coding: utf-8 -*-
"""
infer_config.py — DEM逐像元推理的路径与参数配置

功能说明：
    集中管理 infer_pipeline.py 所需的全部可配置项，包括模型检查点路径、
    特征数据根目录、输出目录及推理质量控制参数。
    修改本文件后无需改动核心推理代码即可适配不同实验或路径环境。

与训练的关联：
    CKPT_PATH 指向训练阶段保存的最优检查点（如 best_ep1624.pt）。
    run_dir（由 CKPT_PATH 的父目录自动确定）须含以下训练产出文件：
        meta.json            — 特征名称与坐标归一化参数
        feat_mu_per_year.npy — 各年份特征均值
        feat_sd_per_year.npy — 各年份特征标准差

使用方法：
    在运行 infer_pipeline.py 之前，将"路径配置"部分修改为本地实际路径。

作者：张仟龙
"""

# ============================================================
# 路径配置  ← 根据本地环境修改
# ============================================================

# 训练最优检查点路径（其所在目录须包含 meta.json 等训练产出文件）
CKPT_PATH: str = r"E:\CrossYear_Diagnostics\runs_r2_0p8_plus_1231\best_ep1624.pt"

# 所有遥感特征栅格的根目录（与训练时保持一致）
BASE_DIR: str = r"E:\zql\Data\Train_features"

# DEM预测结果输出目录（每年输出一个 DEM_pred_{year}.tif）
OUT_DIR: str = r"E:\zql\Data\Train_features\DEM"

# ============================================================
# 推理参数
# ============================================================

# 分块推理的瓦片大小（像素），影响峰值内存使用量
TILE_SIZE: int = 512

# 输出栅格的nodata值
NODATA: float = -9999.0

# 是否将潮间带掩膜以外的像素设为nodata
MASK_OUTSIDE_TIDAL: bool = True

# 若需仅保留特定 gridcode 类别的潮间带，填入列表（如 [1, 2]）；
# None 表示保留所有 gridcode 类别
GRIDCODE_KEEP = None

# 用于确定推理栅格空间参考与分辨率的模板特征列名
# 须与训练时 meta.json 中 features 列表的某一条目完全一致
TEMPLATE_FEATURE: str = "Inundation_band1"

# 检查点参数覆盖率阈值：低于该值时推理将被中止（防止模型定义与检查点不匹配）
MIN_CKPT_COVERAGE: float = 0.98

# 特征标准化安全参数
STD_EPS: float = 1e-6   # 标准差下限（防止除零）
CLIP_Z: float  = 10.0   # z-score截断范围；设为 None 则不截断

# ============================================================
# 傅里叶位置编码参数（须与训练时保持一致）
# ============================================================
FOURIER_K: int = 6
