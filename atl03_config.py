# -*- coding: utf-8 -*-
"""
atl03_config.py — ICESat-2 ATL03高程标定与特征提取流水线的路径与参数配置

功能说明：
    集中管理 atl03_pipeline.py 所需的全部可配置项，包括处理参数、
    输入数据路径、各年份输入/输出配置，以及遥感特征栅格文件模板。
    修改本文件即可适配不同的本地环境，无需改动核心处理代码。

使用方法：
    在运行 atl03_pipeline.py 之前，将"数据路径"和"逐年处理配置"部分
    修改为本地实际路径。

作者：张仟龙
"""

import os

# ============================================================
# 处理参数
# ============================================================

class Config:
    # 内存管理
    MAX_MEMORY_PERCENT: float = 70.0   # 进程内存占用超过该百分比时暂停处理
    CPU_USAGE_RATIO:    float = 0.75   # 使用CPU核心数的比例

    # 栅格特征提取
    BATCH_SIZE_RASTER: int = 5000      # 每批处理的光子点数量

    # DBSCAN光子去噪参数
    DBSCAN_EPS:         float = 0.0005  # 邻域半径（单位：度，约50m）
    DBSCAN_MIN_SAMPLES: int   = 10      # 最小聚类样本数

    # RTK-ICESat配对参数
    RTK_RADIUS_SEQUENCE = [30.0, 60.0, 100.0, 200.0]  # 依次尝试的搜索半径（米）
    RTK_MIN_PAIRS: int = 3             # 拟合平面模型所需的最少有效配对数


# ============================================================
# 全局输入路径  ← 根据本地环境修改
# ============================================================

# EGM2008大地水准面栅格（用于将椭球高转换为正高）
GEOID_TIF: str = r"E:\zql\Data\geoid\egm2008_liaohe_2p5min.tif"

# RTK-GNSS控制点文件（CSV或Excel格式，字段说明见 atl03_pipeline.py 中的 read_rtk()）
RTK_PATH: str = r"C:\Users\DELL\Desktop\RTK1.csv"

# 各年份平均海平面垂直偏差（毫米），RTK标定后叠加；无需调整时设为0
MSL_OFFSET: dict = {2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 0}

# 所有遥感特征栅格的根目录
FEATURE_BASE: str = r"E:\zql\Data\Train_features"


# ============================================================
# 逐年处理配置  ← 根据本地环境修改
# ============================================================
# 每条记录对应一个处理年份，包含输入ATL03文件夹、潮间带掩膜矢量、
# 输出特征CSV路径及质检输出目录。
# 注释掉不需要（重新）处理的年份即可跳过。

YEARS_CONFIG = [
    {
        "year":        2019,
        "folder":      r"D:\Icesat2_ATL03_clipped_h5\ATL03_2019_clipped",
        "landcover":   r"E:\zql\Data\Train_features\潮间带范围\Tidal_2019.shp",
        "output_csv":  r"E:\zql\Data\Train_data_all_11\ATL03_with_features_2019_calibrated.csv",
        "output_dir":  r"E:\zql\Data\Train_data_all_11\feature_check_2019_85",
    },
    {
        "year":        2020,
        "folder":      r"D:\Icesat2_ATL03_clipped_h5\ATL03_2020_clipped",
        "landcover":   r"E:\zql\Data\Train_features\潮间带范围\Tidal_2020.shp",
        "output_csv":  r"E:\zql\Data\Train_data_all_11\ATL03_with_features_2020_calibrated.csv",
        "output_dir":  r"E:\zql\Data\Train_data_all_11\feature_check_2020_85",
    },
    {
        "year":        2021,
        "folder":      r"D:\Icesat2_ATL03_clipped_h5\ATL03_2021_clipped",
        "landcover":   r"E:\zql\Data\Train_features\潮间带范围\Tidal_2021.shp",
        "output_csv":  r"E:\zql\Data\Train_data_all_11\ATL03_with_features_2021_calibrated.csv",
        "output_dir":  r"E:\zql\Data\Train_data_all_11\feature_check_2021_85",
    },
    {
        "year":        2022,
        "folder":      r"D:\Icesat2_ATL03_clipped_h5\ATL03_2022_clipped",
        "landcover":   r"E:\zql\Data\Train_features\潮间带范围\Tidal_2022.shp",
        "output_csv":  r"E:\zql\Data\Train_data_all_11\ATL03_with_features_2022_calibrated.csv",
        "output_dir":  r"E:\zql\Data\Train_data_all_11\feature_check_2022_85",
    },
    {
        "year":        2023,
        "folder":      r"D:\Icesat2_ATL03_clipped_h5\ATL03_2023_clipped",
        "landcover":   r"E:\zql\Data\Train_features\潮间带范围\Tidal_2023.shp",
        "output_csv":  r"E:\zql\Data\Train_data_all_11\ATL03_with_features_2023_calibrated.csv",
        "output_dir":  r"E:\zql\Data\Train_data_all_11\feature_check_2023_85",
    },
]


# ============================================================
# 遥感特征栅格文件模板
# ============================================================
# 每条记录为 (相对路径模板, 列名前缀)。
# 模板中的 {year} 在运行时被替换为实际年份。
# 路径基于 FEATURE_BASE 拼接；不存在的文件会被自动跳过。

RASTER_TEMPLATES = [
    (os.path.join("Inundation", "archive_inundation_frequency{year}.tif"), "Inundation"),
    (os.path.join("S1_texture",  "S1_VV_texture_{year}.tif"),               "S1"),
    (os.path.join("VIS", "{year}VIs", "{year}CCI.tif"),                     "CCI"),
    (os.path.join("VIS", "{year}VIs", "{year}NDWI.tif"),                    "NDWI"),
    (os.path.join("VIS", "{year}VIs", "{year}GCC.tif"),                     "GCC"),
    (os.path.join("VIS", "{year}VIs", "{year}GNDVI.tif"),                   "GNDVI"),
    (os.path.join("VIS", "{year}VIs", "{year}IBI.tif"),                     "IBI"),
    (os.path.join("VIS", "{year}VIs", "{year}NDVIre.tif"),                  "NDVIre"),
    (os.path.join("VIS", "{year}VIs", "{year}NIRv.tif"),                    "NIRv"),
    (os.path.join("VIS", "{year}VIs", "{year}PRI.tif"),                     "PRI"),
    (os.path.join("VIS", "{year}VIs", "{year}VARI.tif"),                    "VARI")
]


def get_raster_list(year: int):
    """
    将栅格模板解析为指定年份的绝对路径列表，仅返回磁盘上实际存在的文件。

    返回
    ----
    list of (绝对路径, 列名前缀) 元组
    """
    resolved = [
        (os.path.join(FEATURE_BASE, tmpl.format(year=year)), prefix)
        for tmpl, prefix in RASTER_TEMPLATES
    ]
    return [(p, prefix) for p, prefix in resolved if os.path.exists(p)]
