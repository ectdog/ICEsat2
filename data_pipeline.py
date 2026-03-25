# -*- coding: utf-8 -*-
"""
data_pipeline.py — ICESat-2多年特征序列构建（KNN空间软聚合）

功能说明：
    读取各年份CSV特征表，以所有年份光子点的并集作为空间锚点网格，
    通过KNN距离加权高斯核聚合每个锚点在各年份的特征观测值，
    最终返回打包好的NumPy数组，供标准化和模型输入拼接使用。

主要接口：
    build_sequences(csv_files)
        返回值（按顺序）：
            X     : float32 [N, T, C]  — 多年特征序列
            M     : float32 [N, T]     — 有效性掩膜（1=该锚点在该年有数据）
            Dst   : float32 [N, T]     — 各年KNN有效最近邻距离（米）
            XY    : float32 [N, 2]     — 锚点UTM坐标
            Y     : float32 [N, T]     — 目标高程（height_1985）
            Ym    : float32 [N, T]     — 目标可用性掩膜
            keys  : object  [N]        — "lon|lat"字符串锚点标识
            years_order : list[str]    — 年份列表（已排序）
            features    : list[str]    — 特征列名列表（有序）
            meta_xy     : dict         — {x_min, y_min, xy_range}，供空间归一化使用

作者：张仟龙
"""

import os
import numpy as np
import pandas as pd
from pyproj import Transformer as PJTransformer
from sklearn.neighbors import KDTree

import config as cfg


# ─────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────

def _xy_from_lonlat(transformer, lon_arr: np.ndarray, lat_arr: np.ndarray):
    """将WGS84经纬度数组投影到UTM坐标，使用预构建的pyproj Transformer。"""
    lon_arr = np.asarray(lon_arr).reshape(-1)
    lat_arr = np.asarray(lat_arr).reshape(-1)
    x, y = transformer.transform(lon_arr, lat_arr)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape[0] != lon_arr.shape[0] or y.shape[0] != lat_arr.shape[0]:
        raise ValueError("坐标转换长度不一致")
    return x, y


def _year_feature_list(df: pd.DataFrame) -> list:
    """
    返回 df 中的特征列列表。

    若 cfg.MANUAL_FEATURES 非空则直接使用；否则根据 cfg.FEATURE_PREFIXES
    自动匹配列名，并排除黑名单列和 cfg.PHENO_EXCLUDE_PREFIXES 中的前缀。
    """
    if cfg.MANUAL_FEATURES:
        return list(cfg.MANUAL_FEATURES)

    ban = {
        "lon", "lat", "lon_k", "lat_k", "key", "file_name", "beam",
        "signal_conf", cfg.TARGET_COL, "height_base",
        "height_icesat2_ellipsoid", "landcover", "year", "X", "Y",
    }
    cols: list = []
    if cfg.FEATURE_PREFIXES:
        for p in cfg.FEATURE_PREFIXES:
            cols.extend([c for c in df.columns if c.startswith(p + "_")])
    else:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    cols = [c for c in cols if c not in ban]

    if cfg.PHENO_EXCLUDE_PREFIXES:
        cols = [
            c for c in cols
            if not any(c.startswith(px) for px in cfg.PHENO_EXCLUDE_PREFIXES)
        ]

    return sorted(list(dict.fromkeys(cols)))


# ─────────────────────────────────────────────
# 主接口
# ─────────────────────────────────────────────

def build_sequences(csv_files):
    """
    从各年份CSV文件构建多年KNN软聚合特征序列。

    将所有年份中出现的唯一（lon, lat）位置作为锚点。对每个锚点的每个年份，
    在半径 cfg.TOL_M 米内查找最多 cfg.KNN_K 个最近邻观测，
    用高斯核（sigma = cfg.KNN_SIGMA）加权聚合特征值。
    有效年份数不足 cfg.MIN_STEPS 的锚点将被丢弃。

    参数
    ----
    csv_files : list[str]
        各年份CSV特征文件路径列表，文件名中须包含四位年份字符串（如 "…2021…"）。

    返回值见模块文档字符串。
    """
    per_year: dict = {}
    years_found: list = []

    # ── 步骤1：加载各年份CSV ──────────────────────────────────────────────────
    for p in csv_files:
        year = None
        base = os.path.basename(p)
        for y in ["2019", "2020", "2021", "2022", "2023"]:
            if y in base:
                year = y
                break
        if year is None:
            raise ValueError(f"无法从文件名推断年份: {p}")

        df = pd.read_csv(p)
        if not {"lon", "lat"}.issubset(df.columns):
            raise ValueError(f"缺少 lon/lat 列: {p}")

        if cfg.TARGET_COL in df.columns:
            n0 = len(df)
            df = df[df[cfg.TARGET_COL].between(*cfg.CLIP_RANGE)].copy()
            print(f"[{year}] 保留 {len(df)}/{n0} 行（目标值在 {cfg.CLIP_RANGE} 范围内）")

        df["year"] = str(year)
        per_year[year] = df
        years_found.append(year)

    years_order: list = sorted(list(dict.fromkeys(years_found)))

    # ── 步骤2：确定各年公共特征集 ───────────────────────────────────────────
    feat_all = None
    for y in years_order:
        feats = _year_feature_list(per_year[y])
        feat_all = set(feats) if feat_all is None else (feat_all & set(feats))
    features: list = sorted(list(feat_all))
    if not features:
        raise RuntimeError(
            "各年份间无公共特征列，请检查 config.py 中的 FEATURE_PREFIXES / PHENO_EXCLUDE_PREFIXES。"
        )

    # ── 步骤3：构建锚点网格（所有年份唯一经纬度的并集）─────────────────────
    all_pts = pd.concat(
        [per_year[y].dropna(subset=["lon", "lat"])[["lon", "lat"]]
         for y in years_order],
        ignore_index=True,
    )
    all_pts["lon_r"] = all_pts["lon"].round(6)
    all_pts["lat_r"] = all_pts["lat"].round(6)
    anchor = (
        all_pts.drop_duplicates(subset=["lon_r", "lat_r"])[["lon_r", "lat_r"]]
        .rename(columns={"lon_r": "lon", "lat_r": "lat"})
        .reset_index(drop=True)
    )

    # ── 步骤4：投影到UTM坐标 ────────────────────────────────────────────────
    t_xy = PJTransformer.from_crs("EPSG:4326", f"EPSG:{cfg.UTM_EPSG}", always_xy=True)

    all_xy = np.vstack([
        np.c_[_xy_from_lonlat(t_xy,
                                per_year[y].dropna(subset=["lon", "lat"])["lon"].to_numpy(),
                                per_year[y].dropna(subset=["lon", "lat"])["lat"].to_numpy())]
        for y in years_order
    ])
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    xy_range = np.array([x_max - x_min, y_max - y_min], dtype=np.float64) + 1e-6

    ax, ay = _xy_from_lonlat(t_xy, anchor["lon"].to_numpy(), anchor["lat"].to_numpy())
    anchor["X"] = ax
    anchor["Y"] = ay
    anchor["key"] = (anchor["lon"].round(6).astype(str) + "|"
                     + anchor["lat"].round(6).astype(str))

    # ── 步骤5：各年份KDTree与缺失值填补基准 ─────────────────────────────────
    trees: dict = {}
    aux: dict   = {}
    year_fill: dict = {}

    for y in years_order:
        dfy = per_year[y].dropna(subset=["lon", "lat"]).copy()
        dx, dy = _xy_from_lonlat(t_xy, dfy["lon"].to_numpy(), dfy["lat"].to_numpy())
        dfy["X"] = dx
        dfy["Y"] = dy
        aux[y] = dfy.reset_index(drop=True)
        trees[y] = KDTree(np.c_[dx, dy])

        ser = aux[y][features]
        if cfg.IMPUTE_STRATEGY == "median":
            year_fill[y] = ser.median(skipna=True).values.astype(np.float32)
        elif cfg.IMPUTE_STRATEGY == "mean":
            year_fill[y] = ser.mean(skipna=True).values.astype(np.float32)
        else:
            year_fill[y] = None

    # ── 步骤6：遍历锚点进行KNN软聚合 ────────────────────────────────────────
    T = len(years_order)
    C = len(features)
    coords       = anchor[["X", "Y"]].to_numpy()
    keys_anchor  = anchor["key"].to_numpy()

    rows_X, rows_M, rows_Dst = [], [], []
    rows_XY, rows_Y, rows_Ym, rows_key = [], [], [], []
    kept, tried = 0, len(anchor)

    for i in range(len(anchor)):
        xi, yi = coords[i]
        key = keys_anchor[i]

        feat_seq = np.full((T, C), np.nan, dtype=np.float32)
        mask_seq = np.zeros(T,        dtype=np.float32)
        dist_seq = np.full(T, cfg.TOL_M, dtype=np.float32)
        Y_seq    = np.full(T, np.nan, dtype=np.float32)
        Ym_seq   = np.zeros(T,        dtype=np.float32)
        hit = 0

        for t, y in enumerate(years_order):
            k = max(1, int(cfg.KNN_K))
            dist, nn = trees[y].query(np.array([[xi, yi]]), k=k, return_distance=True)
            di = dist[0]; ji = nn[0]
            mask_in = di <= cfg.TOL_M
            if not np.any(mask_in):
                continue
            di = di[mask_in]; ji = ji[mask_in]
            rows = aux[y].iloc[ji]

            vals = rows[features].to_numpy(dtype=np.float32)
            if vals.ndim == 1:
                vals = vals[None, :]

            if len(di) > 1:
                w = np.exp(-0.5 * (di / max(1e-6, cfg.KNN_SIGMA)) ** 2)
                w /= w.sum() + 1e-8
                vagg = (vals * w[:, None]).sum(axis=0)
                d_eff = float(di.min())
            else:
                vagg = vals[0]
                d_eff = float(di[0])

            if np.any(np.isfinite(vagg)):
                feat_seq[t, :] = vagg
                mask_seq[t]    = 1.0
                dist_seq[t]    = d_eff
                hit += 1

            if cfg.TARGET_COL in rows.columns:
                yk = rows[cfg.TARGET_COL].to_numpy(dtype=np.float32)
                yk = yk[np.isfinite(yk)]
                if yk.size > 0:
                    if len(di) > 1 and yk.size == len(di):
                        w = np.exp(-0.5 * (di / max(1e-6, cfg.KNN_SIGMA)) ** 2)
                        w /= w.sum() + 1e-8
                        Y_seq[t] = float((yk * w).sum())
                    else:
                        Y_seq[t] = float(yk[0])
                    Ym_seq[t] = 1.0

        if hit < cfg.MIN_STEPS:
            continue

        # 用年份级别填补值修复NaN特征
        for t, y in enumerate(years_order):
            if np.isnan(feat_seq[t, :]).any():
                fill = year_fill[y]
                if fill is None:
                    feat_seq = None
                    break
                m = ~np.isfinite(feat_seq[t, :])
                feat_seq[t, m] = fill[m]
        if feat_seq is None:
            continue
        if Ym_seq.sum() == 0:
            continue

        rows_X.append(feat_seq); rows_M.append(mask_seq); rows_Dst.append(dist_seq)
        rows_XY.append([xi, yi]); rows_Y.append(Y_seq); rows_Ym.append(Ym_seq)
        rows_key.append(key)
        kept += 1

    X   = np.asarray(rows_X,   dtype=np.float32)
    M   = np.asarray(rows_M,   dtype=np.float32)
    Dst = np.asarray(rows_Dst, dtype=np.float32)
    XY  = np.asarray(rows_XY,  dtype=np.float32)
    Y   = np.asarray(rows_Y,   dtype=np.float32)
    Ym  = np.asarray(rows_Ym,  dtype=np.float32)
    keys = np.asarray(rows_key, dtype=object)

    print(
        f"[KNN={cfg.KNN_K}] 锚点总数={tried}，保留={kept} | "
        f"N={len(Y)}, T={T}, C={X.shape[2]}"
    )
    meta_xy = {
        "x_min": float(x_min), "y_min": float(y_min),
        "xy_range": xy_range.astype(float).tolist(),
    }
    return X, M, Dst, XY, Y, Ym, keys, years_order, features, meta_xy
