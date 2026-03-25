# -*- coding: utf-8 -*-
"""
trainer.py — 两阶段训练循环、验证评估与检查点管理

功能说明：
    实现完整的训练流程，包括：
    · 阶段1（PHASE1_FREEZE_EPOCHS轮）：冻结Transformer编码器，仅训练回归头，
      使网络快速收敛到合理的输出范围；
    · 阶段2（剩余轮次）：解冻全部参数，使用Cosine Annealing with Warm Restarts
      学习率调度，并在阶段2的1/3处开始累积SWA权重；
    · EMA影子权重用于验证集推理，提供更稳定的指标；
    · 按验证RMSE维护Top-K检查点，支持早停。

主要接口：
    train() — 读取 config.py 中的所有设置，执行完整训练流程。

输出文件（保存到 config.OUTDIR）：
    train.log                    — 训练日志
    meta.json                    — 特征与坐标元信息
    feat_mu_per_year.npy         — 各年特征均值（归一化参数）
    feat_sd_per_year.npy         — 各年特征标准差（归一化参数）
    y_mu_sd_per_year.npy         — 各年目标z-score参数（仅用于损失空间）
    epoch_metrics.csv            — 每轮训练/验证指标
    best_ep{N}.pt                — 验证RMSE最优的Top-K检查点
    last.pt                      — 最新检查点（每 SAVE_EVERY 轮保存）
    oof_val_best.csv             — 最优epoch时验证集逐点预测结果
    val_metrics_by_year_best.csv — 最优epoch时各年份验证指标
    residual_hist_best.csv       — 最优epoch时残差直方图
    residual_vs_dist_best.csv    — 最优epoch时残差-距离关系
    summary.json                 — 训练结束时的汇总配置与最优结果

作者：张仟龙
"""

import gc
import json
import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import amp as torch_amp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, update_bn

import config as cfg
from data_pipeline import build_sequences
from model import (
    EMA, PackedDS, TemporalTransformerMO,
    fourier_features, make_loader, r2_score_np, set_seed, train_val_split,
)


# ─────────────────────────────────────────────
# 日志工具
# ─────────────────────────────────────────────

def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(os.path.join(cfg.OUTDIR, "train.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────
# 单轮训练/验证
# ─────────────────────────────────────────────

def run_epoch(
    model, loader, device, scaler,
    optimizer=None,
    y_mu: np.ndarray = None,
    y_sd: np.ndarray = None,
    return_mask: bool = False,
):
    """
    对 loader 执行一轮完整的前向（及可选的反向）传播。

    损失函数：带掩膜的逐样本 Huber 损失（在目标z-score空间计算），
    加上时序总变差正则化项（cfg.LAMBDA_TV）。

    参数
    ----
    model      : TemporalTransformerMO（或SWA封装版本）
    loader     : DataLoader，每批返回 (Xin, Y, Ym)
    device     : "cuda" 或 "cpu"
    scaler     : torch.amp.GradScaler
    optimizer  : 若为 None 则以评估模式运行（不计算梯度）
    y_mu, y_sd : float32 数组 [T]，各年目标归一化统计量
    return_mask: 若为 True，同时返回逐样本有效性掩膜

    返回
    ----
    (loss, mae, rmse, r2, preds, ys)
    或（当 return_mask=True 时）
    (loss, mae, rmse, r2, preds, ys, mask_np)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    preds_all: List[np.ndarray] = []
    ys_all:    List[np.ndarray] = []
    masks_all: List[np.ndarray] = []

    T = len(y_mu) if y_mu is not None else 5
    y_mu_t = torch.tensor(y_mu, dtype=torch.float32, device=device) if y_mu is not None else None
    y_sd_t = torch.tensor(y_sd, dtype=torch.float32, device=device) if y_sd is not None else None

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for x, Y, Ym in loader:
            x  = x.to(device,  non_blocking=True)
            Y  = Y.to(device,  non_blocking=True)
            Ym = Ym.to(device, non_blocking=True)

            with torch_amp.autocast("cuda", enabled=cfg.USE_AMP and device == "cuda"):
                predT = model(x)
                valid = (Ym > 0.5) & torch.isfinite(Y)

                if y_mu_t is not None and y_sd_t is not None:
                    Yz    = (Y - y_mu_t) / y_sd_t
                    predZ = (predT - y_mu_t) / y_sd_t
                else:
                    Yz    = Y
                    predZ = predT

                if valid.any():
                    abs_err = (predZ - Yz).abs()
                    huber   = torch.where(abs_err < 1.0,
                                          0.5 * abs_err ** 2,
                                          abs_err - 0.5)
                    elem    = huber[valid]
                    B       = x.size(0)
                    b_idx   = valid.nonzero(as_tuple=True)[0]
                    loss_per_sample  = torch.zeros(B, device=device)
                    count_per_sample = torch.zeros(B, device=device)
                    loss_per_sample.index_add_(0, b_idx, elem)
                    count_per_sample.index_add_(0, b_idx, torch.ones_like(elem))
                    loss_core = (loss_per_sample / count_per_sample.clamp_min(1.0)).mean()
                else:
                    loss_core = (predZ - Yz).abs().mean() * 0.0

                # 时序总变差正则化
                if cfg.LAMBDA_TV > 0:
                    valid_pairs = (Ym[:, 1:] > 0.5) & (Ym[:, :-1] > 0.5)
                    if valid_pairs.any():
                        tv = (predT[:, 1:] - predT[:, :-1]).abs()[valid_pairs]
                        loss_core = loss_core + cfg.LAMBDA_TV * tv.mean()

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_core).backward()
                if cfg.GRAD_CLIP_NORM is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss_core.detach()) * x.size(0)
            preds_all.append(predT.detach().float().cpu().numpy())
            ys_all.append(Y.detach().float().cpu().numpy())
            if return_mask:
                masks_all.append(valid.detach().cpu().numpy())

    preds = np.concatenate(preds_all, 0) if preds_all else np.zeros((0, T), dtype=np.float32)
    ys    = np.concatenate(ys_all,   0) if ys_all   else np.zeros((0, T), dtype=np.float32)

    mask_np = np.isfinite(ys)
    if return_mask and masks_all:
        mask_np = mask_np & np.concatenate(masks_all, 0).astype(bool)

    if mask_np.any():
        p1 = preds[mask_np]; y1 = ys[mask_np]
        mae  = float(np.mean(np.abs(p1 - y1)))
        rmse = float(np.sqrt(np.mean((p1 - y1) ** 2)))
        r2   = r2_score_np(y1, p1)
    else:
        mae = rmse = r2 = float("nan")

    denom = max(1, len(loader.dataset))
    if return_mask:
        return total_loss / denom, mae, rmse, r2, preds, ys, mask_np
    return total_loss / denom, mae, rmse, r2, preds, ys


# ─────────────────────────────────────────────
# 最优epoch诊断CSV导出
# ─────────────────────────────────────────────

def export_artifacts(
    va_pred, va_true, va_mask,
    Dva, Mva, XYva, keys_va, years_order,
):
    """
    当验证RMSE刷新最优时，将四个诊断CSV文件写入 cfg.OUTDIR：

    oof_val_best.csv             — 验证集逐点、逐年预测值与真实值
    val_metrics_by_year_best.csv — 各年份 MAE / RMSE / R²
    residual_hist_best.csv       — 残差直方图（50个区间，1~99百分位范围）
    residual_vs_dist_best.csv    — 残差与KNN距离的对应关系
    """
    T = len(years_order)
    n_val = va_true.shape[0]
    rows = []

    for i in range(n_val):
        for t in range(T):
            if va_mask[i, t]:
                rows.append({
                    "key":      str(keys_va[i]),
                    "year":     str(years_order[t]),
                    "y":        float(va_true[i, t]),
                    "p":        float(va_pred[i, t]),
                    "dist_m":   float(Dva[i, t]),
                    "mask_hit": int(Mva[i, t] > 0.5),
                    "X_utm":    float(XYva[i, 0]),
                    "Y_utm":    float(XYva[i, 1]),
                })

    if not rows:
        return

    df_oof = pd.DataFrame(rows)
    df_oof.to_csv(os.path.join(cfg.OUTDIR, "oof_val_best.csv"), index=False)

    by_year = []
    for t, y in enumerate(years_order):
        sub = df_oof[df_oof["year"] == str(y)]
        if len(sub):
            yv = sub["y"].to_numpy(np.float32)
            pv = sub["p"].to_numpy(np.float32)
            by_year.append({
                "year": y,
                "MAE":  float(np.mean(np.abs(pv - yv))),
                "RMSE": float(np.sqrt(np.mean((pv - yv) ** 2))),
                "R2":   float(1.0 - np.sum((pv - yv) ** 2) /
                              (np.sum((yv - yv.mean()) ** 2) + 1e-12)),
                "n":    int(len(sub)),
            })
    if by_year:
        pd.DataFrame(by_year).to_csv(
            os.path.join(cfg.OUTDIR, "val_metrics_by_year_best.csv"), index=False)

    res = df_oof["p"].to_numpy(np.float32) - df_oof["y"].to_numpy(np.float32)
    lo, hi = np.percentile(res, 1), np.percentile(res, 99)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        lo, hi = res.min(), res.max()
    hist, edges = np.histogram(res, bins=50, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    pd.DataFrame({"residual_bin": centers, "count": hist}).to_csv(
        os.path.join(cfg.OUTDIR, "residual_hist_best.csv"), index=False)

    df_rvd = df_oof[["dist_m"]].copy()
    df_rvd["residual"] = res
    df_rvd["year"] = df_oof["year"].values
    df_rvd.to_csv(os.path.join(cfg.OUTDIR, "residual_vs_dist_best.csv"), index=False)


# ─────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────

def train():
    """
    完整训练入口，所有配置均从 config.py 读取。

    执行顺序：
    1. 构建多年KNN序列（build_sequences）
    2. 按训练/验证比例划分样本
    3. 按年份标准化特征（使用训练集统计量）
    4. 拼接扩展输入（特征 + 掩膜 + 归一化距离 + 傅里叶空间编码）
    5. 计算各年目标z-score参数（仅用于损失空间归一化）
    6. 构建DataLoader
    7. 初始化模型（支持热启动）
    8. 两阶段训练（Phase1冻结编码器 → Phase2全参数 + SWA）
    9. SWA最终评估
    10. 保存 summary.json
    """
    os.makedirs(cfg.OUTDIR, exist_ok=True)
    set_seed(cfg.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"运行设备: {device}")

    # 1. 构建序列
    X, M, Dst, XY, Y, Ym, keys, years_order, features, meta_xy = build_sequences(cfg.CSV_FILES)
    if len(Y) == 0:
        _log("样本为空，退出。")
        return
    T, C = X.shape[1], X.shape[2]

    json.dump(
        {"years": years_order, "features": features, "meta_xy": meta_xy,
         "excluded_prefixes": cfg.PHENO_EXCLUDE_PREFIXES,
         "knn_k": cfg.KNN_K, "tol_m": cfg.TOL_M, "knn_sigma": cfg.KNN_SIGMA},
        open(os.path.join(cfg.OUTDIR, "meta.json"), "w"), indent=2,
    )
    _log(f"C={C} 特征列示例: {features[:min(20, len(features))]}")

    # 2. 划分训练/验证集
    tr_idx, va_idx = train_val_split(len(Y))
    Xtr, Mtr, Dtr, XYtr, Ytr, Ymtr = (
        X[tr_idx], M[tr_idx], Dst[tr_idx], XY[tr_idx], Y[tr_idx], Ym[tr_idx])
    Xva, Mva, Dva, XYva, Yva, Ymva = (
        X[va_idx], M[va_idx], Dst[va_idx], XY[va_idx], Y[va_idx], Ym[va_idx])
    keys_va = keys[va_idx]

    # 3. 按年份标准化特征
    mu_list, sd_list = [], []
    for t in range(T):
        mu_t = np.nanmean(Xtr[:, t, :], axis=0)
        sd_t = np.nanstd(Xtr[:, t, :],  axis=0)
        sd_t[sd_t == 0] = 1.0
        mu_t[np.isnan(mu_t)] = 0.0
        sd_t[np.isnan(sd_t)] = 1.0
        Xtr[:, t, :] = (Xtr[:, t, :] - mu_t) / sd_t
        Xva[:, t, :] = (Xva[:, t, :] - mu_t) / sd_t
        mu_list.append(mu_t.astype(np.float32))
        sd_list.append(sd_t.astype(np.float32))
    np.save(os.path.join(cfg.OUTDIR, "feat_mu_per_year.npy"), np.stack(mu_list))
    np.save(os.path.join(cfg.OUTDIR, "feat_sd_per_year.npy"), np.stack(sd_list))

    # 4. 拼接扩展输入
    x_min, y_min = meta_xy["x_min"], meta_xy["y_min"]
    xy_range = np.array(meta_xy["xy_range"], dtype=np.float32)

    def _xyf(XY_):
        xy_norm = (XY_ - np.array([x_min, y_min], dtype=np.float32)) / xy_range
        if cfg.FOURIER_NORM:
            xy_norm = np.clip(xy_norm, 0.0, 1.0)
        F = fourier_features(xy_norm, cfg.FOURIER_K)
        return np.concatenate([xy_norm.astype(np.float32), F], axis=1)

    XYF_tr = _xyf(XYtr); XYF_va = _xyf(XYva)
    XYF_tr_rep = np.repeat(XYF_tr[:, None, :], T, axis=1)
    XYF_va_rep = np.repeat(XYF_va[:, None, :], T, axis=1)
    dist_n_tr  = np.clip(Dtr / max(1e-6, cfg.TOL_M), 0.0, 1.0)[..., None]
    dist_n_va  = np.clip(Dva / max(1e-6, cfg.TOL_M), 0.0, 1.0)[..., None]
    Xin_tr = np.concatenate([Xtr, Mtr[..., None], dist_n_tr, XYF_tr_rep], axis=-1).astype(np.float32)
    Xin_va = np.concatenate([Xva, Mva[..., None], dist_n_va, XYF_va_rep], axis=-1).astype(np.float32)

    # 5. 各年目标z-score参数
    y_mu_list, y_sd_list = [], []
    for t in range(T):
        y_tr_t = Ytr[:, t]
        mu_t = float(np.nanmean(y_tr_t))
        sd_t = float(np.nanstd(y_tr_t))
        if not np.isfinite(sd_t) or sd_t < 1e-6:
            sd_t = 1.0
        y_mu_list.append(mu_t); y_sd_list.append(sd_t)
    y_mu = np.array(y_mu_list, dtype=np.float32)
    y_sd = np.array(y_sd_list, dtype=np.float32)
    np.save(os.path.join(cfg.OUTDIR, "y_mu_sd_per_year.npy"), np.stack([y_mu, y_sd]))
    _log(f"y_mu={np.round(y_mu, 4).tolist()}, y_sd={np.round(y_sd, 4).tolist()}")

    # 6. DataLoader
    loader_tr = make_loader(PackedDS(Xin_tr, Ytr, Ymtr), cfg.BATCH_SIZE, shuffle=True,  drop_last=False)
    loader_va = make_loader(PackedDS(Xin_va, Yva, Ymva), cfg.BATCH_SIZE, shuffle=False, drop_last=False)

    # 7. 模型初始化
    C_ext = Xin_tr.shape[-1]
    model = TemporalTransformerMO(in_dim=C_ext, n_years=T).to(device)

    def _strip_module(sd):
        return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    if cfg.WARM_START_CKPT and os.path.exists(cfg.WARM_START_CKPT):
        ckpt = torch.load(cfg.WARM_START_CKPT, map_location="cpu")
        sd = _strip_module(ckpt.get("model", ckpt))
        msd = model.state_dict()
        loadable = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        model.load_state_dict(loadable, strict=False)
        _log(f"热启动：加载 {len(loadable)}/{len(msd)} 个参数张量。")
    else:
        _log("冷启动（随机初始化）。")

    # 8. 优化器 / 调度器 / EMA
    amp_scaler = torch_amp.GradScaler("cuda", enabled=cfg.USE_AMP and device == "cuda", init_scale=512.0)
    ema = EMA(model, decay=0.998)

    def _set_encoder_grad(flag: bool):
        for n, p in model.named_parameters():
            if "head_per_token" in n or "year_bias" in n:
                p.requires_grad = True
            else:
                p.requires_grad = flag

    _set_encoder_grad(False)  # 阶段1：仅训练头部
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.BASE_LR_HEAD, weight_decay=cfg.WEIGHT_DECAY,
    )
    sch1 = CosineAnnealingWarmRestarts(opt1, T_0=max(1, cfg.PHASE1_FREEZE_EPOCHS),
                                        T_mult=1, eta_min=cfg.BASE_LR_HEAD * 0.1)

    best_rmse = float("inf")
    patience  = cfg.PATIENCE
    topk_list: List[Tuple[float, str]] = []

    epoch_metrics_path = os.path.join(cfg.OUTDIR, "epoch_metrics.csv")
    if not os.path.exists(epoch_metrics_path):
        pd.DataFrame(columns=[
            "epoch", "tr_loss", "tr_mae", "tr_rmse", "tr_r2",
            "va_loss", "va_mae", "va_rmse", "va_r2", "lr",
        ]).to_csv(epoch_metrics_path, index=False)

    def _save_ckpt(obj, path):
        torch.save(obj, path, _use_new_zipfile_serialization=False)

    def _update_topk(rmse_val, ep, optimizer):
        nonlocal best_rmse, patience, topk_list
        if rmse_val < best_rmse - 1e-5:
            best_rmse = rmse_val
            patience  = cfg.PATIENCE
            path = os.path.join(cfg.OUTDIR, f"best_ep{ep:03d}.pt")
            _save_ckpt(
                {"epoch": ep, "model": model.state_dict(),
                 "optim": optimizer.state_dict(),
                 "scaler": amp_scaler.state_dict(),
                 "best_rmse": best_rmse},
                path,
            )
            topk_list.append((rmse_val, path))
            topk_list = sorted(topk_list, key=lambda x: x[0])[:cfg.TOPK_TO_SAVE]
            _log(f"✔ 最优RMSE={best_rmse:.4f}  ({path})")
        else:
            patience -= 1

    def _one_epoch(ep, optimizer, scheduler):
        t0 = time.time()
        model.train()
        tr_loss, tr_mae, tr_rmse, tr_r2, _, _ = run_epoch(
            model, loader_tr, device, amp_scaler, optimizer=optimizer, y_mu=y_mu, y_sd=y_sd)
        ema.update(model)

        model.eval(); ema.load_shadow(model)
        with torch.no_grad():
            va_loss, va_mae, va_rmse, va_r2, va_pred, va_true, va_mask = run_epoch(
                model, loader_va, device, amp_scaler,
                optimizer=None, y_mu=y_mu, y_sd=y_sd, return_mask=True)
        ema.restore(model)
        scheduler.step(ep)

        _log(
            f"[Ep{ep:03d}] 训练: loss={tr_loss:.4f} MAE={tr_mae:.4f} RMSE={tr_rmse:.4f} R2={tr_r2:.4f} | "
            f"验证: loss={va_loss:.4f} MAE={va_mae:.4f} RMSE={va_rmse:.4f} R2={va_r2:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | 耗时={time.time()-t0:.1f}s"
        )
        pd.DataFrame([{
            "epoch": ep,
            "tr_loss": tr_loss, "tr_mae": tr_mae, "tr_rmse": tr_rmse, "tr_r2": tr_r2,
            "va_loss": va_loss, "va_mae": va_mae, "va_rmse": va_rmse, "va_r2": va_r2,
            "lr": optimizer.param_groups[0]["lr"],
        }]).to_csv(epoch_metrics_path, mode="a", header=False, index=False)

        if ep % cfg.SAVE_EVERY == 0:
            _save_ckpt(
                {"epoch": ep, "model": model.state_dict(),
                 "optim": optimizer.state_dict(),
                 "scaler": amp_scaler.state_dict(), "best_rmse": best_rmse},
                os.path.join(cfg.OUTDIR, "last.pt"),
            )
        return va_rmse, va_pred, va_true, va_mask

    # 阶段1：冻结编码器，仅训练头部
    for ep in range(1, cfg.PHASE1_FREEZE_EPOCHS + 1):
        va_rmse, va_pred, va_true, va_mask = _one_epoch(ep, opt1, sch1)
        _update_topk(va_rmse, ep, opt1)
        if patience == 0:
            _log("✋ 早停触发（阶段1）。"); break

    # 阶段2：解冻全部参数 + SWA
    _set_encoder_grad(True)
    opt2 = torch.optim.AdamW(model.parameters(), lr=cfg.BASE_LR_FULL, weight_decay=cfg.WEIGHT_DECAY)
    sch2 = CosineAnnealingWarmRestarts(opt2, T_0=50, T_mult=2, eta_min=cfg.BASE_LR_FULL * 0.1)

    start_ep  = cfg.PHASE1_FREEZE_EPOCHS + 1
    end_ep    = cfg.NUM_EPOCHS
    swa_start = max(start_ep + 50, start_ep + (end_ep - start_ep) // 3)
    swa_model = AveragedModel(model)

    for ep in range(start_ep, end_ep + 1):
        prev_best = best_rmse
        va_rmse, va_pred, va_true, va_mask = _one_epoch(ep, opt2, sch2)
        _update_topk(va_rmse, ep, opt2)

        if ep >= swa_start:
            swa_model.update_parameters(model)

        if best_rmse < prev_best + 1e-12:
            export_artifacts(va_pred, va_true, va_mask, Dva, Mva, XYva, keys_va, years_order)

        if patience == 0:
            _log("✋ 早停触发（阶段2）。"); break

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # SWA最终评估
    if best_rmse < float("inf"):
        _log(">>> SWA: 更新BN统计量并评估")
        update_bn(loader_tr, swa_model, device=device)
        swa_model.eval()
        with torch.no_grad():
            _, _, va_rmse_swa, va_r2_swa, _, _ = run_epoch(
                swa_model, loader_va, device, amp_scaler, optimizer=None, y_mu=y_mu, y_sd=y_sd)
        _log(f">>> SWA 验证: RMSE={va_rmse_swa:.4f} R2={va_r2_swa:.4f}")

    # 保存汇总
    with open(os.path.join(cfg.OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "years": years_order, "features": features,
            "excluded_prefixes": cfg.PHENO_EXCLUDE_PREFIXES,
            "best_rmse": best_rmse,
            "topk": [p for _, p in sorted(topk_list, key=lambda x: x[0])],
            "config": {
                "BATCH_SIZE": cfg.BATCH_SIZE, "EPOCHS": cfg.NUM_EPOCHS,
                "LR_HEAD": cfg.BASE_LR_HEAD, "LR_FULL": cfg.BASE_LR_FULL,
                "WEIGHT_DECAY": cfg.WEIGHT_DECAY,
                "D_MODEL": cfg.D_MODEL, "N_HEAD": cfg.N_HEAD,
                "N_LAYERS": cfg.N_LAYERS, "D_FF": cfg.D_FF,
                "DROPOUT": cfg.DROPOUT, "FOURIER_K": cfg.FOURIER_K,
                "TOL_M": cfg.TOL_M, "MIN_STEPS": cfg.MIN_STEPS,
                "KNN_K": cfg.KNN_K, "KNN_SIGMA": cfg.KNN_SIGMA,
                "LAMBDA_TV": cfg.LAMBDA_TV,
            },
        }, f, indent=2)
    _log("训练完成。")


if __name__ == "__main__":
    train()
