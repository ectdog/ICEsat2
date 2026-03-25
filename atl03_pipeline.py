# -*- coding: utf-8 -*-
"""
atl03_pipeline.py — ICESat-2 ATL03光子高程标定与多源遥感特征提取核心流水线

功能说明：
    针对辽河口潮间带2019—2023年ICESat-2 ATL03数据，实现以下五个处理阶段：
    1. 读取ATL03 HDF5文件 → 信号置信度过滤 → DBSCAN空间去噪
    2. 基于潮间带范围矢量（Shapefile）的空间裁剪
    3. 高程标定：
       a. EGM2008大地水准面校正（椭球高 → 正高）
       b. RTK-GNSS平面拟合（RANSAC）对准1985国家高程基准
    4. 多源遥感特征栅格采样（rasterio点采样，多线程批量处理）
    5. 质检可视化与统计报告输出

    所有路径与参数配置均位于 atl03_config.py，核心处理逻辑与路径解耦。

使用方法：
    python atl03_pipeline.py

作者：张仟龙
"""

import gc
import glob
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.warp import transform as rio_transform
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm

import atl03_config as acfg

warnings.filterwarnings("ignore")


# ============================================================
# 工具函数
# ============================================================

def check_memory(threshold: float = acfg.Config.MAX_MEMORY_PERCENT, wait: int = 5):
    """若当前进程内存占用超过阈值，则暂停并等待内存释放。"""
    mem_pct = psutil.Process(os.getpid()).memory_percent()
    while mem_pct > threshold:
        print(f"⚠️  内存使用 {mem_pct:.1f}%，等待 {wait} s …")
        time.sleep(wait)
        gc.collect()
        mem_pct = psutil.Process(os.getpid()).memory_percent()


def cleanup_temps(pattern: str = "temp_*.parquet"):
    """清除上次运行遗留的临时parquet中间文件。"""
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except OSError:
            pass


def lonlat_to_utm51(lon, lat):
    """WGS84地理坐标 → UTM51N（EPSG:32651）投影坐标。"""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    x, y = t.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def utm51_to_lonlat(x, y):
    """UTM51N（EPSG:32651）投影坐标 → WGS84地理坐标。"""
    t = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x, y)
    return np.asarray(lon), np.asarray(lat)


# ============================================================
# 阶段1 — ATL03读取与去噪
# ============================================================

def read_atl03_file(file_path: str) -> pd.DataFrame:
    """
    读取单个ATL03 HDF5文件的全部六个光束轨道。

    仅保留信号置信度 signal_conf_ph > 1（中等及高置信度）的光子。
    文件读取失败时返回空DataFrame。
    """
    beams = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
    all_dfs = []
    try:
        with h5py.File(file_path, "r") as f:
            for beam in beams:
                if beam not in f:
                    continue
                try:
                    g    = f[beam]["heights"]
                    lat  = g["lat_ph"][:]
                    lon  = g["lon_ph"][:]
                    h    = g["h_ph"][:]
                    conf = g["signal_conf_ph"][:, 0]
                    mask = conf > 1
                    if mask.sum() == 0:
                        continue
                    all_dfs.append(pd.DataFrame({
                        "lat":                      lat[mask],
                        "lon":                      lon[mask],
                        "height_icesat2_ellipsoid": h[mask],
                        "signal_conf":              conf[mask],
                        "beam":                     beam,
                    }))
                except Exception:
                    continue
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def dbscan_denoise(
    df: pd.DataFrame,
    eps: float = acfg.Config.DBSCAN_EPS,
    min_samples: int = acfg.Config.DBSCAN_MIN_SAMPLES,
) -> pd.DataFrame:
    """
    使用DBSCAN空间聚类去除孤立噪声光子。

    被标记为 -1（噪声）的光子将被丢弃。聚类在地理坐标（lon, lat）空间进行；
    eps ≈ 0.0005° ≈ 中纬度50m。
    """
    if len(df) < min_samples * 2:
        return pd.DataFrame()
    try:
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(
            df[["lon", "lat"]]
        )
        return df[labels != -1].reset_index(drop=True)
    except Exception:
        return df


def process_single_atl03(args) -> pd.DataFrame:
    """
    多进程Worker：读取单个ATL03文件 → DBSCAN去噪 → 潮间带范围空间裁剪。

    任意步骤失败或无有效光子时返回空DataFrame。
    """
    file_path, landcover_gdf = args
    df = read_atl03_file(file_path)
    if df.empty:
        return pd.DataFrame()
    df = dbscan_denoise(df)
    if df.empty:
        return pd.DataFrame()
    try:
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
        )
        if landcover_gdf.crs != gdf.crs:
            landcover_gdf = landcover_gdf.to_crs(gdf.crs)
        result = gpd.sjoin(
            gdf, landcover_gdf[["geometry", "gridcode"]], how="inner", predicate="within"
        )
        return pd.DataFrame(
            result.rename(columns={"gridcode": "landcover"}).drop(columns="geometry")
        )
    except Exception:
        return pd.DataFrame()


def read_atl03_folder(
    folder_path: str, landcover_shp: str, max_workers: int = None
) -> pd.DataFrame:
    """
    并行批量处理指定文件夹中的所有ATL03 HDF5文件。

    文件按每批10个分块处理以控制峰值内存；中间结果溢写为snappy压缩的
    parquet临时文件，处理结束后合并并删除临时文件。

    参数
    ----
    folder_path   : 包含 *.h5 ATL03粒子文件的目录
    landcover_shp : 潮间带范围掩膜矢量文件（须含'gridcode'或可识别的分类字段）
    max_workers   : 并行进程数（默认为 CPU核数 × Config.CPU_USAGE_RATIO）
    """
    files = glob.glob(os.path.join(folder_path, "*.h5"))
    if not files:
        print(f"❌  未找到H5文件: {folder_path}")
        return pd.DataFrame()
    print(f"📂  找到 {len(files)} 个H5文件")

    if max_workers is None:
        max_workers = max(1, int(multiprocessing.cpu_count() * acfg.Config.CPU_USAGE_RATIO))

    # 加载潮间带掩膜矢量
    try:
        landcover = gpd.read_file(landcover_shp)
        if "gridcode" not in landcover.columns:
            candidates = [
                c for c in landcover.columns
                if any(k in c.lower() for k in ["code", "class", "type", "value"])
            ]
            if candidates:
                landcover = landcover.rename(columns={candidates[0]: "gridcode"})
            else:
                numeric = landcover.select_dtypes(include=["number"]).columns
                if len(numeric) > 0:
                    landcover = landcover.rename(columns={numeric[0]: "gridcode"})
        print(f"✓  掩膜: {len(landcover)} 个多边形, CRS={landcover.crs}")
    except Exception as e:
        print(f"❌  掩膜加载失败: {e}")
        return pd.DataFrame()

    # 分块并行处理
    chunk_size = 10
    temp_files = []

    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        tasks = [(f, landcover) for f in chunk]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(
                executor.map(process_single_atl03, tasks),
                total=len(tasks), desc=f"分块 {i // chunk_size + 1}",
            ):
                if not result.empty:
                    tmp = f"temp_chunk{len(temp_files)}.parquet"
                    result.to_parquet(tmp, index=False, compression="snappy")
                    temp_files.append(tmp)
        gc.collect()
        check_memory()

    if not temp_files:
        return pd.DataFrame()

    print(f"📦  合并 {len(temp_files)} 个分块 …")
    final_df = pd.concat(
        [pd.read_parquet(f) for f in tqdm(temp_files)], ignore_index=True
    )
    for f in temp_files:
        try:
            os.remove(f)
        except OSError:
            pass
    return final_df


# ============================================================
# 阶段2 — 栅格特征提取
# ============================================================

def _extract_raster_batch(coords, raster_path: str, raster_crs, nodata):
    """在给定(lon, lat)坐标列表上采样栅格（多线程Worker）。"""
    try:
        lon, lat = zip(*coords)
        with rasterio.open(raster_path) as src:
            if raster_crs != CRS.from_epsg(4326):
                xs, ys = rio_transform(CRS.from_epsg(4326), raster_crs, lon, lat)
            else:
                xs, ys = lon, lat
            values = np.array(list(src.sample(list(zip(xs, ys)))))
            if nodata is not None:
                values = np.where(values == nodata, np.nan, values)
            elif src.nodata is not None:
                values = np.where(values == src.nodata, np.nan, values)
        return values
    except Exception:
        return np.full((len(coords), 1), np.nan)


def extract_raster_features(
    df: pd.DataFrame,
    raster_paths,
    prefixes,
    batch_size: int = acfg.Config.BATCH_SIZE_RASTER,
) -> pd.DataFrame:
    """
    在每个光子位置批量采样多个栅格，并将采样值以新列追加到 df。

    列命名规则：单波段栅格为 ``{prefix}_band1``，多波段为 ``{prefix}_band{i}``。
    缺失值（nodata）存储为 NaN。

    参数
    ----
    df            : 含 'lon' 和 'lat' 列的DataFrame
    raster_paths  : 栅格文件路径序列
    prefixes      : 与路径一一对应的列名前缀序列
    batch_size    : 每次采样的坐标点数
    """
    df_new = df.copy()
    coords  = list(zip(df["lon"], df["lat"]))
    batches = [coords[i:i + batch_size] for i in range(0, len(coords), batch_size)]

    for raster_path, prefix in zip(raster_paths, prefixes):
        print(f"  {prefix}: {os.path.basename(raster_path)}")
        try:
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
                n_bands    = src.count
                nodata     = src.nodata

            all_values = np.full((len(df), n_bands), np.nan)
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                futures = {
                    executor.submit(_extract_raster_batch, b, raster_path, raster_crs, nodata): idx
                    for idx, b in enumerate(batches)
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=prefix):
                    idx = futures[future]
                    try:
                        vals  = future.result(timeout=30)
                        start = idx * batch_size
                        all_values[start:start + len(batches[idx])] = vals
                    except Exception:
                        pass

            for i in range(n_bands):
                df_new[f"{prefix}_band{i + 1}"] = all_values[:, i]
            del all_values
            gc.collect()
        except Exception as e:
            print(f"    ⚠️  提取失败: {e}")

    return df_new


# ============================================================
# 阶段3 — RTK-GNSS高程标定
# ============================================================

def read_rtk(rtk_path: str) -> pd.DataFrame:
    """
    从CSV或Excel文件加载RTK-GNSS控制点数据。

    支持两种坐标格式：
    · UTM投影坐标（X, Y, Z）：自动识别（含 'X' 和 'Y' 列时）。
    · 地理坐标（lon, lat, height）：否则使用此格式。

    两种格式均返回包含以下列的DataFrame：
    ['lon', 'lat', 'x_utm', 'y_utm', 'height_1985']
    """
    if not os.path.exists(rtk_path):
        raise FileNotFoundError(f"RTK文件不存在: {rtk_path}")

    ext = os.path.splitext(rtk_path)[1].lower()
    df  = pd.read_excel(rtk_path) if ext in [".xls", ".xlsx"] else pd.read_csv(rtk_path)
    print(f"✓  RTK: {len(df)} 点, 字段: {df.columns.tolist()}")

    colmap = {c.lower().strip(): c for c in df.columns}
    has_xy     = "x" in colmap and "y" in colmap
    has_lonlat = (any(k in colmap for k in ["lon", "longitude"])
                  and any(k in colmap for k in ["lat", "latitude"]))

    if has_xy:
        print("  检测到：UTM投影坐标 (X, Y, Z)")
        x_col = colmap["x"]; y_col = colmap["y"]
        z_candidates = [k for k in ["z", "height", "elev", "elevation", "height_1985", "高程"]
                        if k in colmap]
        if not z_candidates:
            raise ValueError("未找到高程字段")
        z_col = colmap[z_candidates[0]]
        rtk = pd.DataFrame({
            "x_utm":       df[x_col].astype(float),
            "y_utm":       df[y_col].astype(float),
            "height_1985": df[z_col].astype(float),
        })
        rtk["lon"], rtk["lat"] = utm51_to_lonlat(rtk["x_utm"].values, rtk["y_utm"].values)

    elif has_lonlat:
        print("  检测到：地理坐标 (lon, lat, height)")
        lon_col = next(colmap[k] for k in ["lon", "longitude", "经度"] if k in colmap)
        lat_col = next(colmap[k] for k in ["lat", "latitude",  "纬度"] if k in colmap)
        h_candidates = [k for k in ["height_1985", "height", "z", "elev", "高程"] if k in colmap]
        if not h_candidates:
            raise ValueError("未找到高程字段")
        h_col = colmap[h_candidates[0]]
        rtk = pd.DataFrame({
            "lon":         df[lon_col].astype(float),
            "lat":         df[lat_col].astype(float),
            "height_1985": df[h_col].astype(float),
        })
        rtk["x_utm"], rtk["y_utm"] = lonlat_to_utm51(rtk["lon"].values, rtk["lat"].values)
    else:
        raise ValueError(f"无法识别坐标格式，字段: {df.columns.tolist()}")

    rtk = rtk.dropna().reset_index(drop=True)
    print(f"  清理后: {len(rtk)} 点, "
          f"高程=[{rtk['height_1985'].min():.3f}, {rtk['height_1985'].max():.3f}] m")
    return rtk


def pair_rtk_icesat(df_all: pd.DataFrame, rtk_df: pd.DataFrame):
    """
    使用KD树将每个RTK控制点匹配到最近的ICESat-2光子。

    按 Config.RTK_RADIUS_SEQUENCE 中的半径依次尝试，直至找到至少
    Config.RTK_MIN_PAIRS 个有效配对为止。

    返回
    ----
    (pairs_df, 使用的搜索半径)  或  (None, None)（配对失败时）
    """
    print("\n🔍  RTK–ICESat配对")
    xs, ys = lonlat_to_utm51(df_all["lon"].values, df_all["lat"].values)
    xr, yr = rtk_df["x_utm"].values, rtk_df["y_utm"].values

    tree = KDTree(np.c_[xs, ys])
    dist_all, idx_all = tree.query(np.c_[xr, yr], k=1)
    d = dist_all[:, 0]
    print(f"  最近邻距离: min={d.min():.1f}, median={np.median(d):.1f}, max={d.max():.1f} m")

    for rad in acfg.Config.RTK_RADIUS_SEQUENCE:
        ok = d <= rad
        if ok.sum() >= acfg.Config.RTK_MIN_PAIRS:
            pairs = pd.DataFrame({
                "x":           xs[idx_all[ok, 0]],
                "y":           ys[idx_all[ok, 0]],
                "height_base": df_all.iloc[idx_all[ok, 0]]["height_base"].values,
                "height_1985": rtk_df.iloc[ok]["height_1985"].values,
            }).drop_duplicates(subset=["x", "y"]).dropna()

            if len(pairs) >= acfg.Config.RTK_MIN_PAIRS:
                print(f"  ✅  半径 {rad} m: {len(pairs)} 对")
                return pairs, rad
        print(f"  ⏸️  半径 {rad} m: {ok.sum()} 对（不足）")

    print("  ❌  配对失败")
    return None, None


def fit_plane_model(pairs: pd.DataFrame) -> dict:
    """
    对 ICESat-2 – RTK 残差拟合空间平面 δ(x,y) = a·x + b·y + c。

    ≥3对使用RANSAC，2对使用OLS，1对使用常数偏移。
    返回的平面参数后续用于 apply_calibration() 中的系统偏差改正。

    返回
    ----
    包含 a, b, c, mae, rmse, mode 键的字典
    """
    X = pairs[["x", "y"]].values
    y = (pairs["height_base"] - pairs["height_1985"]).values
    n = len(pairs)

    if n >= 3:
        try:
            ransac = RANSACRegressor(
                estimator=LinearRegression(), residual_threshold=0.2, random_state=42
            )
        except TypeError:
            ransac = RANSACRegressor(
                base_estimator=LinearRegression(), residual_threshold=0.2, random_state=42
            )
        ransac.fit(X, y)
        a, b = ransac.estimator_.coef_
        c    = ransac.estimator_.intercept_
        resid = y - ransac.predict(X)
        mode = "RANSAC"
    elif n == 2:
        lr = LinearRegression().fit(X, y)
        a, b = lr.coef_
        c    = lr.intercept_
        resid = y - lr.predict(X)
        mode = "OLS"
    else:
        a, b = 0.0, 0.0
        c    = float(y[0])
        resid = y - c
        mode = "CONST"

    mae  = np.mean(np.abs(resid))
    rmse = np.sqrt(np.mean(resid ** 2))
    print(f"  模型: {mode} | n={n} | a={a:.3e}, b={b:.3e}, c={c:.3f}")
    print(f"  误差: MAE={mae:.3f} m, RMSE={rmse:.3f} m")
    return {"a": float(a), "b": float(b), "c": float(c),
            "mae": mae, "rmse": rmse, "mode": mode}


def apply_calibration(df: pd.DataFrame, plane: dict, msl_offset_mm: float = 0) -> pd.DataFrame:
    """
    将标定平面应用于ICESat-2高程，生成 height_1985 列：

        height_1985 = height_base − δ(x,y) − msl_offset_mm / 1000

    其中 δ(x,y) = a·x + b·y + c 为 fit_plane_model() 拟合的空间偏差面。
    msl_offset_mm 为附加的平均海平面垂直偏差（毫米），通常设为0。
    """
    xs, ys = lonlat_to_utm51(df["lon"].values, df["lat"].values)
    delta = plane["a"] * xs + plane["b"] * ys + plane["c"]
    df["height_1985"] = df["height_base"] - delta - (msl_offset_mm / 1000.0)
    return df


# ============================================================
# 阶段4 — 质检可视化
# ============================================================

def quality_check(df: pd.DataFrame, feature_cols: list, output_dir: str):
    """
    生成单年份标定数据集的质检图表与统计报告。

    输出至 output_dir 的文件：
    feature_stats.csv   — 所有特征列的描述性统计
    height_analysis.png — 四宫格图（高程直方图 + 散点图）
    height_summary.csv  — 标定前后高程差的数值汇总
    """
    os.makedirs(output_dir, exist_ok=True)

    if feature_cols:
        df[feature_cols].describe().T.to_csv(os.path.join(output_dir, "feature_stats.csv"))

    if "height_base" in df.columns and "height_1985" in df.columns:
        diff = df["height_base"] - df["height_1985"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].hist(df["height_base"].dropna(), bins=50, alpha=0.7, edgecolor="k")
        axes[0, 0].set_title("大地水准面高（m）")

        axes[0, 1].hist(df["height_1985"].dropna(), bins=50, alpha=0.7,
                        color="green", edgecolor="k")
        axes[0, 1].set_title("1985国家高程（m）")

        axes[1, 0].hist(diff.dropna(), bins=50, alpha=0.7, color="orange", edgecolor="k")
        axes[1, 0].set_title("差值：大地水准面高 − 1985高程（m）")

        sample = min(5000, len(df))
        idx = np.random.choice(len(df), sample, replace=False)
        axes[1, 1].scatter(df.iloc[idx]["height_base"], df.iloc[idx]["height_1985"],
                           alpha=0.3, s=1, c="blue")
        mn, mx = df["height_base"].min(), df["height_base"].max()
        axes[1, 1].plot([mn, mx], [mn, mx], "r--", lw=2)
        axes[1, 1].set_xlabel("大地水准面高（m）"); axes[1, 1].set_ylabel("1985高程（m）")
        axes[1, 1].set_title("相关性")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "height_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()

        pd.DataFrame([{
            "n_points":   len(df),
            "base_mean":  df["height_base"].mean(),
            "base_std":   df["height_base"].std(),
            "1985_mean":  df["height_1985"].mean(),
            "1985_std":   df["height_1985"].std(),
            "diff_mean":  diff.mean(),
            "diff_std":   diff.std(),
            "diff_min":   diff.min(),
            "diff_max":   diff.max(),
        }]).to_csv(os.path.join(output_dir, "height_summary.csv"), index=False)

    print(f"✅  质检完成: {output_dir}")


# ============================================================
# 阶段5 — 单年份流水线调度
# ============================================================

def process_year(
    year: int,
    folder_path: str,
    landcover_shp: str,
    output_csv: str,
    output_dir: str,
    geoid_tif: str,
    rtk_path: str,
    msl_offset: float,
    workers: int,
) -> bool:
    """
    对单个年份执行完整的四阶段处理流程。

    参数
    ----
    year          : 处理年份（如 2021）
    folder_path   : ATL03 HDF5粒子文件所在目录
    landcover_shp : 潮间带范围掩膜矢量文件
    output_csv    : 输出特征CSV文件路径
    output_dir    : 质检图表与统计报告输出目录
    geoid_tif     : EGM2008大地水准面栅格（为空字符串则跳过Geoid校正）
    rtk_path      : RTK控制点CSV/Excel文件路径
    msl_offset    : 附加垂直偏差（毫米），通常为0
    workers       : 并行进程数

    返回
    ----
    True（成功） 或 False（失败）
    """
    print("\n" + "=" * 70)
    print(f"▶▶  年份 {year}")
    print("=" * 70)

    if not os.path.exists(folder_path):
        print(f"❌  ATL03文件夹不存在: {folder_path}"); return False
    if not os.path.exists(landcover_shp):
        print(f"❌  掩膜矢量不存在: {landcover_shp}"); return False

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # 步骤1：读取ATL03
    print("\n[1/4] 读取ATL03 + DBSCAN去噪 + 潮间带裁剪")
    t1 = time.time()
    df = read_atl03_folder(folder_path, landcover_shp, workers)
    if df.empty:
        print("⚠️  无数据"); return False
    print(f"✓  {len(df):,} 个光子, 耗时 {time.time() - t1:.1f} s")

    # 步骤2：高程标定
    print("\n[2/4] 高程标定")
    t2 = time.time()

    if geoid_tif and os.path.exists(geoid_tif):
        print("  应用EGM2008大地水准面校正 …")
        df = extract_raster_features(df, [geoid_tif], ["GEOID"], batch_size=10000)
        df["geoid_m"]     = df["GEOID_band1"]
        df["height_base"] = df["height_icesat2_ellipsoid"] - df["geoid_m"]
        df = df.drop(columns=["GEOID_band1"])
    else:
        print("  跳过Geoid校正")
        df["height_base"] = df["height_icesat2_ellipsoid"]

    calibrated = False
    if os.path.exists(rtk_path):
        try:
            rtk   = read_rtk(rtk_path)
            pairs, rad = pair_rtk_icesat(df, rtk)
            if pairs is not None:
                plane = fit_plane_model(pairs)
                df    = apply_calibration(df, plane, msl_offset)
                calibrated = True
            else:
                df["height_1985"] = df["height_base"]
        except Exception as e:
            print(f"  ⚠️  RTK标定失败: {e}")
            df["height_1985"] = df["height_base"]
    else:
        df["height_1985"] = df["height_base"]

    print(f"✓  标定{'完成' if calibrated else '跳过'}, 耗时 {time.time() - t2:.1f} s")

    # 步骤3：特征提取
    print("\n[3/4] 遥感特征提取")
    t3 = time.time()
    valid_rasters = acfg.get_raster_list(year)
    print(f"  找到 {len(valid_rasters)} 个有效栅格")
    if valid_rasters:
        paths, prefixes = zip(*valid_rasters)
        df = extract_raster_features(df, paths, prefixes)
    print(f"✓  耗时 {time.time() - t3:.1f} s")

    # 保存输出CSV
    print(f"\n💾  保存: {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"  {len(df):,} 行 × {len(df.columns)} 列")

    # 步骤4：质检
    print("\n[4/4] 质检")
    feature_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in prefixes)] \
        if valid_rasters else []
    quality_check(df, feature_cols, output_dir)

    elapsed = time.time() - t_start
    print(f"\n✅  年份 {year} 完成，总耗时 {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    return True


# ============================================================
# 主程序入口
# ============================================================

def main():
    print("=" * 70)
    print(" ICESat-2 多年份高程标定与遥感特征提取 ")
    print("=" * 70)

    cleanup_temps()
    workers = max(1, int(multiprocessing.cpu_count() * acfg.Config.CPU_USAGE_RATIO))
    print(f"🖥️   CPU: 共 {multiprocessing.cpu_count()} 核，使用 {workers} 个进程")
    print(f"💾  内存: {psutil.virtual_memory().total / 1024**3:.1f} GB\n")

    results = {}
    t_total = time.time()

    for cfg_entry in acfg.YEARS_CONFIG:
        success = process_year(
            year          = cfg_entry["year"],
            folder_path   = cfg_entry["folder"],
            landcover_shp = cfg_entry["landcover"],
            output_csv    = cfg_entry["output_csv"],
            output_dir    = cfg_entry["output_dir"],
            geoid_tif     = acfg.GEOID_TIF,
            rtk_path      = acfg.RTK_PATH,
            msl_offset    = acfg.MSL_OFFSET.get(cfg_entry["year"], 0),
            workers       = workers,
        )
        results[cfg_entry["year"]] = success
        gc.collect()
        check_memory(threshold=60)

    print("\n" + "=" * 70)
    print(" 处理完成 ")
    print("=" * 70)
    for year, success in results.items():
        print(f"  {year}: {'✅' if success else '❌'}")
    elapsed = time.time() - t_total
    print(f"\n总耗时: {elapsed:.1f} s ({elapsed / 60:.1f} min)")

    cleanup_temps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        cleanup_temps()
    except Exception as e:
        import traceback
        print(f"\n❌  错误: {e}")
        traceback.print_exc()
        cleanup_temps()
    finally:
        plt.close("all")
        gc.collect()
