# -*- coding: utf-8 -*-
"""
infer_pipeline.py — 基于训练检查点的多年DEM逐像元推理流水线

功能说明：
    加载训练阶段保存的最优检查点，对2019—2023年全域栅格逐瓦片进行
    多年份联合DEM预测，并将各年份预测结果分别输出为GeoTIFF文件。

推理流程：
    1. 加载检查点与训练元数据（meta.json、特征归一化统计量）
    2. 打开模板栅格，确定输出空间参考与像素网格
    3. 按 TILE_SIZE × TILE_SIZE 像素分块遍历全图：
       a. 计算各像素UTM坐标，生成傅里叶空间编码
       b. 按年份读取特征栅格（自动重投影到模板网格）
       c. 光照掩膜（潮间带范围矢量栅格化）+ 特征标准化
       d. 组装扩展输入向量并批量推理
    4. 将各年份预测高程写入 DEM_pred_{year}.tif

容错机制：
    · 检查点参数覆盖率低于阈值时自动终止（防止静默错误推理）
    · GPU批量前向传播失败时自动缩减批大小并重试，最终回退到CPU

输出文件（保存到 infer_config.OUT_DIR）：
    DEM_pred_{year}.tif — 各年份DEM预测结果（float32，LZW压缩，分块存储）

使用方法：
    python infer_pipeline.py

作者：张仟龙
"""

import json
import math
import os
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn

import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from pyproj import Transformer as PJTransformer

import infer_config as icfg


# ============================================================
# 模型定义（须与训练时完全一致）
# ============================================================

class TemporalTransformerMO(nn.Module):
    """
    多输出时序Transformer（推理版本）。

    网络结构与超参数须与 model.py 中的训练版本完全一致；
    任何不一致都会导致检查点参数覆盖率下降并触发安全终止。
    """

    def __init__(self, in_dim, n_years, d_model=192, n_head=6, n_layers=8, d_ff=768, dropout=0.15):
        super().__init__()
        self.n_years = n_years
        self.in_proj = nn.Linear(in_dim, d_model, bias=True)
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

    def forward(self, x):
        h = self.in_proj(x)
        h = self.token_drop(h)
        h = h + self.year_emb[:, :h.shape[1], :]
        z = self.encoder(h)
        pred = self.head_per_token(z).squeeze(-1)
        pred = pred + self.year_bias[:, :pred.shape[1]]
        return pred


# ============================================================
# 检查点加载（含参数覆盖率校验）
# ============================================================

def _extract_state_dict(ckpt_obj):
    """
    从检查点对象中提取 state_dict。

    支持以下格式：
    · 直接为 state_dict（字典，值为张量）
    · {"model": state_dict, ...}
    · {"state_dict": state_dict, ...}
    """
    if isinstance(ckpt_obj, dict):
        for k in ["model", "state_dict", "net", "ema", "model_state_dict"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if any(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("无法从检查点中提取 state_dict。")


def load_model_weights(model, ckpt_path: str, min_coverage: float = 0.98) -> float:
    """
    加载检查点权重，并校验参数覆盖率（形状完全匹配的参数占总参数数的比例）。

    覆盖率低于 min_coverage 时抛出 RuntimeError，防止模型定义与检查点不匹配
    导致的静默错误推理。

    参数
    ----
    model        : TemporalTransformerMO 实例
    ckpt_path    : 检查点文件路径
    min_coverage : 最低可接受的参数覆盖率（默认 0.98）

    返回
    ----
    实际覆盖率（float）
    """
    ckpt   = torch.load(ckpt_path, map_location="cpu")
    sd_raw = _extract_state_dict(ckpt)

    # 仅剥离 "module." 前缀（DDP训练产生的），不做其他字符串操作
    sd = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in sd_raw.items()
    }

    msd      = model.state_dict()
    loadable = {}
    skipped  = []

    for k, v in sd.items():
        if k not in msd:
            skipped.append((k, "不在模型中", tuple(v.shape) if torch.is_tensor(v) else None, None))
            continue
        if not torch.is_tensor(v):
            skipped.append((k, "非张量", None, tuple(msd[k].shape)))
            continue
        if tuple(v.shape) != tuple(msd[k].shape):
            skipped.append((k, "形状不匹配", tuple(v.shape), tuple(msd[k].shape)))
            continue
        loadable[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(loadable, strict=False)

    total    = len(msd)
    loaded   = len(loadable)
    coverage = loaded / max(total, 1)

    print(f"[检查点] 总参数={total}, 成功加载={loaded}, 覆盖率={coverage:.3%}")
    print(f"[检查点] 缺失={len(missing_keys)}, 多余={len(unexpected_keys)}")
    if skipped:
        print("[检查点] 跳过示例（最多显示20条）:")
        for i, (k, reason, s_ckpt, s_model) in enumerate(skipped[:20], 1):
            print(f"  {i:02d}. {k} | {reason} | 检查点={s_ckpt} 模型={s_model}")

    if coverage < min_coverage:
        raise RuntimeError(
            f"检查点参数覆盖率过低: {coverage:.3%} < {min_coverage:.1%}。\n"
            "推理已终止，请确认推理代码的模型定义与训练时完全一致。"
        )

    return coverage


# ============================================================
# 特征标准化（数值安全版本）
# ============================================================

def safe_standardize_tile(
    X_chw: np.ndarray,
    mu_c:  np.ndarray,
    sd_c:  np.ndarray,
    valid_mask = None,
    eps:   float = 1e-6,
    clip_z: float = 10.0,
) -> np.ndarray:
    """
    对单个瓦片的特征数据进行逐通道标准化，并进行数值安全处理。

    处理步骤：
    1. 将非有限值（NaN/Inf）替换为 NaN
    2. 在 valid_mask 为 False 的位置置 NaN
    3. 标准化：Z = (X − μ) / σ
    4. 将 NaN/Inf 结果替换为 0（模型视为"缺失"）
    5. 将 Z 截断至 [-clip_z, clip_z]

    参数
    ----
    X_chw      : float32 [C, H, W] — 原始特征瓦片
    mu_c, sd_c : float32 [C]       — 各通道均值与标准差
    valid_mask : bool [H, W]        — 有效像素掩膜
    eps        : 标准差下限（防止除零）
    clip_z     : z-score截断上限；None 表示不截断
    """
    X  = X_chw.astype(np.float32, copy=False)
    mu = mu_c.astype(np.float32,  copy=False)
    sd = sd_c.astype(np.float32,  copy=False)

    sd = np.where(np.isfinite(sd), sd, 1.0)
    sd = np.maximum(sd, eps)
    X  = np.where(np.isfinite(X), X, np.nan)

    if valid_mask is not None:
        inv = ~valid_mask
        if inv.any():
            X[:, inv] = np.nan

    Z = (X - mu[:, None, None]) / sd[:, None, None]
    Z = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32, copy=False)

    if clip_z is not None:
        Z = np.clip(Z, -float(clip_z), float(clip_z), out=Z)

    return Z


# ============================================================
# 辅助函数：元数据、路径、IO
# ============================================================

def ensure_dir(p: str):
    """创建目录（含父级），已存在时不报错。"""
    os.makedirs(p, exist_ok=True)


def load_meta_and_norm(run_dir: str) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    从训练输出目录加载元数据与特征归一化统计量。

    参数
    ----
    run_dir : 训练输出目录（含 meta.json、feat_mu_per_year.npy、feat_sd_per_year.npy）

    返回
    ----
    (meta_dict, feat_mu [T, C], feat_sd [T, C])
    """
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json 不存在: {meta_path}")
    meta = json.load(open(meta_path, "r", encoding="utf-8"))

    mu_path = os.path.join(run_dir, "feat_mu_per_year.npy")
    sd_path = os.path.join(run_dir, "feat_sd_per_year.npy")
    if not (os.path.exists(mu_path) and os.path.exists(sd_path)):
        raise FileNotFoundError("feat_mu_per_year.npy / feat_sd_per_year.npy 不存在于训练目录。")
    mu = np.load(mu_path).astype(np.float32)
    sd = np.load(sd_path).astype(np.float32)
    return meta, mu, sd


def fourier_features(xy_norm: np.ndarray, K: int) -> np.ndarray:
    """
    2D坐标傅里叶位置编码（须与训练时保持完全一致）。

    对 k = 0 … K-1，拼接 [sin(2^k π x), cos(2^k π x), sin(2^k π y), cos(2^k π y)]。
    输出形状：[N, 4*K]。
    """
    if K <= 0:
        return np.zeros((xy_norm.shape[0], 0), dtype=np.float32)
    xs, ys = xy_norm[:, 0], xy_norm[:, 1]
    feats  = []
    for k in range(K):
        s = (2.0 ** k) * math.pi
        feats.extend([np.sin(s * xs), np.cos(s * xs), np.sin(s * ys), np.cos(s * ys)])
    return np.stack(feats, axis=1).astype(np.float32)


def build_feature_path(base_dir: str, year: int, feat_name: str) -> Tuple[str, int]:
    """
    根据特征名称（来自 meta.json）推断对应栅格文件路径与波段序号。

    命名规则（与训练时 atl03_pipeline.py 的输出列名对应）：
    · S1_band{i}      → S1_VV_texture_{year}.tif 的第 i 波段
    · Inundation_band1 → archive_inundation_frequency{year}.tif 第1波段
    · {VI}_band1      → {year}VIs/{year}{VI}.tif 第1波段
    """
    if feat_name.startswith("S1_band"):
        band = int(feat_name.replace("S1_band", ""))
        p = os.path.join(base_dir, "S1_texture", f"S1_VV_texture_{year}.tif")
        return p, band

    if feat_name == "Inundation_band1":
        p = os.path.join(base_dir, "Inundation", f"archive_inundation_frequency{year}.tif")
        return p, 1

    if feat_name.endswith("_band1"):
        vi = feat_name.replace("_band1", "")
        p = os.path.join(base_dir, "VIS", f"{year}VIs", f"{year}{vi}.tif")
        return p, 1

    raise ValueError(f"无法识别的特征名（来自meta.json）: {feat_name}")


def open_template_ds(year: int, features: List[str]) -> rasterio.io.DatasetReader:
    """
    打开模板特征栅格，用于确定推理的空间参考、分辨率和像素网格。

    模板特征由 infer_config.TEMPLATE_FEATURE 指定，须存在于 features 列表中。
    """
    tpl_feat = icfg.TEMPLATE_FEATURE
    if tpl_feat not in features:
        raise RuntimeError(f"模板特征 '{tpl_feat}' 不在 meta.json 的特征列表中。")
    p, _ = build_feature_path(icfg.BASE_DIR, year, tpl_feat)
    if not os.path.exists(p):
        raise FileNotFoundError(f"模板栅格不存在: {p}")
    return rasterio.open(p)


def rasterize_tidal_mask(
    shp_path: str, template_ds, window: Window, gridcode_keep=None
) -> np.ndarray:
    """
    将潮间带范围矢量栅格化为与模板瓦片一致的二值掩膜（uint8）。

    参数
    ----
    shp_path      : 潮间带Shapefile路径
    template_ds   : 模板栅格DatasetReader（提供空间参考和仿射变换）
    window        : 当前瓦片的 rasterio.Window
    gridcode_keep : 若非None，则仅保留指定gridcode值的多边形

    返回
    ----
    uint8数组 [H, W]，潮间带内为1，范围外为0
    """
    gdf = gpd.read_file(shp_path)
    if gridcode_keep is not None and "gridcode" in gdf.columns:
        gdf = gdf[gdf["gridcode"].isin(gridcode_keep)].copy()

    h, w = int(window.height), int(window.width)
    if gdf.empty:
        return np.zeros((h, w), dtype=np.uint8)

    if template_ds.crs is None:
        raise ValueError("模板栅格无CRS，无法安全栅格化潮间带掩膜。")

    gdf    = gdf.to_crs(template_ds.crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((h, w), dtype=np.uint8)

    win_transform = rasterio.windows.transform(window, template_ds.transform)
    return rasterize(
        shapes=shapes, out_shape=(h, w),
        transform=win_transform, fill=0, dtype="uint8", all_touched=False,
    )


def pixel_center_xy(ds, window: Window) -> Tuple[np.ndarray, np.ndarray]:
    """计算瓦片内各像素中心点的投影坐标（与模板栅格坐标系一致）。"""
    row_off, col_off = int(window.row_off), int(window.col_off)
    h, w = int(window.height), int(window.width)
    rows = np.arange(row_off, row_off + h)
    cols = np.arange(col_off, col_off + w)
    cc, rr = np.meshgrid(cols, rows)
    xs, ys = rasterio.transform.xy(ds.transform, rr, cc, offset="center")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def read_feature_window_as_template(
    feat_path: str,
    feat_band: int,
    template_ds,
    window: Window,
    resampling=Resampling.bilinear,
) -> np.ndarray:
    """
    读取特征栅格的指定瓦片区域，自动重投影对齐至模板栅格网格。

    若特征栅格与模板网格完全一致（CRS、仿射变换、尺寸均相同），
    则直接读取以避免不必要的重投影开销。

    返回
    ----
    float32数组 [H, W]
    """
    h, w = int(window.height), int(window.width)
    with rasterio.open(feat_path) as src:
        same_grid = (
            (src.crs == template_ds.crs) and
            (src.transform == template_ds.transform) and
            (src.width  == template_ds.width) and
            (src.height == template_ds.height)
        )
        if same_grid:
            return src.read(feat_band, window=window).astype(np.float32)

        dst = np.full((h, w), np.nan, dtype=np.float32)
        dst_transform = rasterio.windows.transform(window, template_ds.transform)
        reproject(
            source=rasterio.band(src, feat_band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=template_ds.crs,
            resampling=resampling,
            num_threads=2,
        )
        return dst


# ============================================================
# 容错前向推理（GPU自动缩减批大小 + CPU回退）
# ============================================================

@torch.no_grad()
def safe_forward_predict(
    model, xb: torch.Tensor, prefer_device: str = "cuda", min_bs: int = 2000
) -> np.ndarray:
    """
    带容错的Transformer前向推理。

    策略：
    · 优先在GPU上以完整批大小推理
    · 遇到CUDA内核配置错误时，将批大小减半并重试
    · 批大小缩减至 min_bs 仍失败时，回退到CPU推理

    参数
    ----
    model         : eval状态的 TemporalTransformerMO
    xb            : float Tensor [B, T, C_ext]，可在GPU或CPU上
    prefer_device : 首选推理设备
    min_bs        : 最小允许批大小

    返回
    ----
    float32 NumPy数组 [B, T]
    """
    assert xb.ndim == 3, f"xb 须为 [B, T, C]，实际形状: {xb.shape}"

    if xb.device.type != "cuda":
        return model(xb).detach().float().cpu().numpy()

    b   = xb.shape[0]
    cur = b

    while True:
        try:
            out = model(xb[:cur]).detach().float().cpu().numpy()
            if cur == b:
                return out
            outs  = [out]
            start = cur
            while start < b:
                end = min(start + cur, b)
                outs.append(model(xb[start:end]).detach().float().cpu().numpy())
                start = end
            return np.concatenate(outs, axis=0)

        except RuntimeError as e:
            msg = str(e).lower()
            is_cuda_issue = (
                "cuda error" in msg or
                "invalid configuration argument" in msg or
                "cublas" in msg
            )
            if not is_cuda_issue:
                raise

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            if cur <= min_bs:
                # 最终回退到CPU
                xb_cpu    = xb.detach().float().cpu()
                model_cpu = model.to("cpu"); model_cpu.eval()
                out_cpu   = model_cpu(xb_cpu).detach().float().numpy()
                model.to("cuda"); model.eval()
                return out_cpu

            cur = max(min_bs, cur // 2)


# ============================================================
# 训练验证指标汇总（可选）
# ============================================================

def summarize_training_metrics(run_dir: str):
    """
    若训练输出目录包含 OOF 验证结果文件，则在推理前打印训练验证指标摘要，
    便于确认当前检查点的性能水平。
    """
    import pandas as pd
    oof = os.path.join(run_dir, "oof_val_best.csv")
    byy = os.path.join(run_dir, "val_metrics_by_year_best.csv")

    if not (os.path.exists(oof) or os.path.exists(byy)):
        return

    if os.path.exists(byy):
        df = pd.read_csv(byy)
        print("\n[验证指标] 各年份:")
        print(df.to_string(index=False))

    if os.path.exists(oof):
        df = pd.read_csv(oof)
        y  = df["y"].to_numpy(np.float32)
        p  = df["p"].to_numpy(np.float32)
        m  = np.isfinite(y) & np.isfinite(p)
        y  = y[m]; p = p[m]
        if y.size:
            mae  = float(np.mean(np.abs(p - y)))
            rmse = float(np.sqrt(np.mean((p - y) ** 2)))
            r2   = float(1.0 - np.sum((p - y) ** 2) /
                         (np.sum((y - y.mean()) ** 2) + 1e-12))
            print(f"\n[验证指标] 全局OOF: n={y.size} | MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}")


# ============================================================
# 主推理函数
# ============================================================

def infer_dem_multiyear():
    """
    多年份DEM逐像元推理主函数。

    执行流程：
    1. 加载元数据与特征归一化统计量
    2. 构建模型并加载检查点（含覆盖率校验）
    3. 打开模板栅格，初始化各年份输出GeoTIFF写入器
    4. 按瓦片逐块推理并写入结果
    5. 关闭所有文件句柄
    """
    ensure_dir(icfg.OUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[信息] 推理设备: {device}")

    run_dir = os.path.dirname(icfg.CKPT_PATH)
    meta, feat_mu, feat_sd = load_meta_and_norm(run_dir)

    years    = [int(y) for y in meta["years"]]
    features = list(meta["features"])
    meta_xy  = meta["meta_xy"]
    x_min    = float(meta_xy["x_min"])
    y_min    = float(meta_xy["y_min"])
    xy_range = np.array(meta_xy["xy_range"], dtype=np.float32)

    T = len(years)
    C = len(features)
    in_dim = C + 1 + 1 + (2 + 4 * icfg.FOURIER_K)  # 特征 + 掩膜 + 距离 + 空间编码

    # 初始化模型
    model = TemporalTransformerMO(
        in_dim=in_dim, n_years=T,
        d_model=192, n_head=6, n_layers=8, d_ff=768, dropout=0.15,
    ).to(device)
    model.eval()

    coverage = load_model_weights(model, icfg.CKPT_PATH, min_coverage=icfg.MIN_CKPT_COVERAGE)
    print(f"[信息] 检查点覆盖率: {coverage:.3%}")

    # 打开模板栅格
    tpl_ds           = open_template_ds(years[0], features)
    H, W             = tpl_ds.height, tpl_ds.width
    template_profile = tpl_ds.profile.copy()
    template_crs     = tpl_ds.crs

    # 坐标转换（若模板栅格不是UTM 51N则需额外变换）
    need_xy_transform = True
    if template_crs is not None and "32651" in str(template_crs):
        need_xy_transform = False

    xy_transformer = None
    if need_xy_transform:
        if template_crs is None:
            raise ValueError("模板栅格无CRS，无法将像素坐标转换为 EPSG:32651。")
        xy_transformer = PJTransformer.from_crs(template_crs, "EPSG:32651", always_xy=True)
        print(f"[警告] 模板栅格CRS={template_crs}，将进行坐标转换至 EPSG:32651。")

    # 创建各年份输出GeoTIFF写入器
    writers: Dict[int, rasterio.io.DatasetWriter] = {}
    for y in years:
        out_path = os.path.join(icfg.OUT_DIR, f"DEM_pred_{y}.tif")
        prof = template_profile.copy()
        prof.update(
            count=1, dtype="float32", nodata=float(icfg.NODATA),
            compress="LZW", predictor=2, tiled=True,
            blockxsize=icfg.TILE_SIZE, blockysize=icfg.TILE_SIZE,
        )
        writers[y] = rasterio.open(out_path, "w", **prof)

    # 各年份潮间带掩膜Shapefile路径
    shp_paths = {
        y: os.path.join(icfg.BASE_DIR, "潮间带范围", f"Tidal_{y}.shp")
        for y in years
    }
    for y, p in shp_paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"潮间带矢量文件缺失: {p}")

    # 分块推理
    tile        = int(icfg.TILE_SIZE)
    n_tiles_y   = (H + tile - 1) // tile
    n_tiles_x   = (W + tile - 1) // tile
    total       = n_tiles_y * n_tiles_x
    done        = 0

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            row0   = ty * tile
            col0   = tx * tile
            h      = min(tile, H - row0)
            w      = min(tile, W - col0)
            window = Window(col0, row0, w, h)

            # 计算像素中心坐标（UTM）
            xs, ys = pixel_center_xy(tpl_ds, window)
            if need_xy_transform:
                x2, y2 = xy_transformer.transform(xs.reshape(-1), ys.reshape(-1))
                x_utm  = np.asarray(x2, dtype=np.float32).reshape(h, w)
                y_utm  = np.asarray(y2, dtype=np.float32).reshape(h, w)
            else:
                x_utm = xs.astype(np.float32)
                y_utm = ys.astype(np.float32)

            # 空间编码
            xy      = np.stack([x_utm.reshape(-1), y_utm.reshape(-1)], axis=1)
            xy_norm = (xy - np.array([x_min, y_min], dtype=np.float32)) / xy_range
            xy_norm = np.clip(xy_norm, 0.0, 1.0)
            F       = fourier_features(xy_norm, icfg.FOURIER_K)
            XYF     = np.concatenate([xy_norm.astype(np.float32), F], axis=1)
            XYF_rep = np.repeat(XYF[:, None, :], T, axis=1)

            # 按年份读取特征并标准化
            X_years     = np.zeros((T, C, h, w), dtype=np.float32)
            valid_years = np.ones((T, h, w), dtype=bool)

            for ti, y in enumerate(years):
                tidal = rasterize_tidal_mask(
                    shp_paths[y], tpl_ds, window, gridcode_keep=icfg.GRIDCODE_KEEP
                ).astype(bool)

                for ci, feat in enumerate(features):
                    p, band = build_feature_path(icfg.BASE_DIR, y, feat)
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"特征栅格缺失: feat={feat}, year={y}, 路径={p}")

                    rs  = Resampling.nearest if feat.startswith("S1_band") else Resampling.bilinear
                    arr = read_feature_window_as_template(p, band, tpl_ds, window, resampling=rs)
                    X_years[ti, ci] = arr
                    valid_years[ti] &= np.isfinite(arr)

                if icfg.MASK_OUTSIDE_TIDAL:
                    valid_years[ti] &= tidal

                X_years[ti] = safe_standardize_tile(
                    X_years[ti], feat_mu[ti], feat_sd[ti],
                    valid_mask=valid_years[ti],
                    eps=icfg.STD_EPS,
                    clip_z=icfg.CLIP_Z,
                )

            # 组装扩展输入
            N       = h * w
            X_seq   = X_years.reshape(T, C, N).transpose(2, 0, 1).copy()  # [N,T,C]
            M_seq   = valid_years.reshape(T, N).transpose(1, 0).astype(np.float32)  # [N,T]
            dist_n  = (1.0 - M_seq)[..., None].astype(np.float32)
            mask_in = M_seq[..., None].astype(np.float32)
            Xin     = np.concatenate([X_seq, mask_in, dist_n, XYF_rep], axis=-1).astype(np.float32)

            any_valid = (M_seq.sum(axis=1) > 0.5)
            if not np.any(any_valid):
                for y in years:
                    writers[y].write(
                        np.full((h, w), icfg.NODATA, dtype=np.float32), 1, window=window
                    )
                done += 1
                if done % 20 == 0:
                    print(f"[进度] {done}/{total} 瓦片")
                continue

            # 批量推理
            idx       = np.where(any_valid)[0]
            Xin_valid = Xin[idx]
            bs        = 60000 if device == "cuda" else 20000
            pred_all  = np.full((N, T), np.nan, dtype=np.float32)

            for s in range(0, Xin_valid.shape[0], bs):
                e  = min(s + bs, Xin_valid.shape[0])
                xb = torch.from_numpy(Xin_valid[s:e]).to(device)
                pb = safe_forward_predict(model, xb, prefer_device=device, min_bs=2000)
                pred_all[idx[s:e]] = pb

            # 写入各年份输出
            for ti, y in enumerate(years):
                out = pred_all[:, ti].reshape(h, w).astype(np.float32)
                out[~valid_years[ti]] = icfg.NODATA
                writers[y].write(out, 1, window=window)

            done += 1
            if done % 20 == 0:
                print(f"[进度] {done}/{total} 瓦片")

    tpl_ds.close()
    for y in years:
        writers[y].close()
    print(f"[完成] DEM预测结果已保存至: {icfg.OUT_DIR}")


# ============================================================
# 主程序入口
# ============================================================

def main():
    ensure_dir(icfg.OUT_DIR)
    run_dir = os.path.dirname(icfg.CKPT_PATH)
    summarize_training_metrics(run_dir)
    infer_dem_multiyear()


if __name__ == "__main__":
    main()
