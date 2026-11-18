#!/usr/bin/env python3
"""客户端上行 DP 加噪器（第4阶段 · DP-效用扫描）。

威胁模型（与 `dp.py` / `inversion.py` 同一口径）：
    客户端把共享参数（trunk + normalizer + modalities_tokens）上传给服务器，
    攻击者/服务器观察上行内容。要提供 (ε, δ) 级隐私，需在**上行前**对更新
    做 L2 裁剪 + 高斯加噪（DP-FedAvg 的标准形式）：

        Δ_c ← Δ_c · min(1, C/‖Δ_c‖₂) + N(0, (σ·C)²I)

    隐私语义：**ε 只依赖 σ**（噪声标准差 σC / 裁剪阈值 C = σ），
    C 只影响效用，不影响隐私保证 —— 所以 C 可以自由按数据标定。

与隐私侧 `dp.py` 的口径一致性：
    · 上行判定用 `_is_shared_param` 同口径（排除 `stems.*` / `heads.*`）；
    · 裁剪+加噪的数学形式用 `dp._clip_and_noise_group` 的等价实现
      （整组 L2 clip + 各向同性高斯噪声）；
    · σ 为**统一预算**（所有模态通路共用一个 σ），对应 `allocate_by_sigma` 的
      `modality_aware=False` 口径 ⇒ 本实验的 ε 与隐私侧 uniform 方案**同 ε**。

用法：
    # 正常加噪（原地改写 client 的 model.pth）
    python dp_noise_client.py --client-dir <dir> --prev-global <dir> \
        --sigma 0.1 --clip-norm 12.3 --round 7 --client 2

    # 只测量更新范量，不改文件（用于标定 C）
    python dp_noise_client.py --client-dir <dir> --prev-global <dir> --calibrate
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# 上行共享参数判定：与 aggregate.py 的 _is_shared_param 同口径。
# stems.* / heads.* 是个性化参数，留在客户端本地，不走链路。
def is_shared_uplink_key(key: str) -> bool:
    return not (key.startswith("stems.") or key.startswith("heads."))


def _load(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到权重：{path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _shared_float_keys(sd: dict):
    return [k for k, v in sd.items()
            if torch.is_tensor(v) and v.is_floating_point() and is_shared_uplink_key(k)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-dir", required=True, help="客户端输出目录（含 model.pth）")
    ap.add_argument("--prev-global", default=None, help="上一轮全局模型目录（基线）")
    ap.add_argument("--sigma", type=float, default=0.0, help="噪声倍率 σ（0 = 不加噪）")
    ap.add_argument("--clip-norm", type=float, default=1.0, help="L2 裁剪阈值 C")
    ap.add_argument("--round", type=int, default=-1)
    ap.add_argument("--client", type=int, default=-1)
    ap.add_argument("--calibrate", action="store_true",
                    help="只测量并打印 ‖Δ‖₂，不修改任何文件")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stat-out", default=None, help="把范数统计写到该 JSON")
    args = ap.parse_args()

    cpath = os.path.join(args.client_dir, "model.pth")
    cur = _load(cpath)

    if not args.prev_global:
        print(f"[dp] 无 prev-global，跳过（round={args.round} client={args.client}）")
        return 0
    ppath = os.path.join(args.prev_global, "model.pth")
    if not os.path.exists(ppath):
        print(f"[dp] 找不到基线 {ppath}，跳过")
        return 0
    prev = _load(ppath)

    keys = [k for k in _shared_float_keys(cur) if k in prev and prev[k].shape == cur[k].shape]
    if not keys:
        print("[dp] 没有可加噪的共享参数，跳过")
        return 0

    # ---- 计算更新量 Δ = cur - prev 的整体 L2 范数（只统计上行共享部分）----
    sq = 0.0
    for k in keys:
        d = (cur[k].float() - prev[k].float())
        sq += float(torch.sum(d * d))
    norm = float(sq) ** 0.5

    n_shared = sum(int(cur[k].numel()) for k in keys)
    stat = {
        "round": args.round, "client": args.client,
        "delta_l2": norm, "n_shared_params": n_shared,
        "sigma": args.sigma, "clip_norm": args.clip_norm,
    }
    print(f"[dp] client={args.client} round={args.round} "
          f"‖Δ‖₂={norm:.4f}  上行参数量={n_shared:,}")

    if args.calibrate:
        if args.stat_out:
            os.makedirs(os.path.dirname(args.stat_out) or ".", exist_ok=True)
            with open(args.stat_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(stat, ensure_ascii=False) + "\n")
        return 0

    if args.sigma <= 0:
        return 0

    # ---- 裁剪 + 加噪（原地写回 client 的 model.pth）----
    C = float(args.clip_norm)
    scale = 1.0 if norm <= 0 else min(1.0, C / (norm + 1e-12))

    # 确定性种子：同一个 (round, client) 的噪声可复现
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed) * 1000003 + int(args.round) * 101 + int(args.client) + 7)

    noise_l2 = 0.0
    for k in keys:
        d = (cur[k].float() - prev[k].float()) * scale
        noise = torch.randn(cur[k].shape, generator=gen, dtype=torch.float32) * (args.sigma * C)
        noise_l2 += float(torch.sum(noise * noise))
        cur[k] = (prev[k].float() + d + noise).to(cur[k].dtype)
    noise_l2 = float(noise_l2) ** 0.5

    torch.save(cur, cpath)
    print(f"[dp] 已加噪写回 {cpath}：scale={scale:.4f} "
          f"噪声L2={noise_l2:.4f} (目标 σ·C={args.sigma * C:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
