#!/usr/bin/env python3
"""汇总 DP 效用扫描的 C 标定结果。

输入：dp_noise_client.py 在 --calibrate 模式下写出的 JSONL
      每行一条 {"round":r, "client":c, "delta_l2":‖Δ‖₂, ...}
输出：calib.json —— 取所有有效样本的**中位数**作为 L2 裁剪阈值 C。

为什么取中位数：
    · 均值会被个别异常轮次（数据量/任务不同导致更新量差很多）拉偏；
    · 中位数是 DP-FL 文献里"典型更新范数"的标准取法；
    · 标定后 σ 直接读作 **噪声/信号比**（噪声 L2 = σ·C ≈ σ·‖Δ‖_median）。
"""
from __future__ import annotations

import argparse
import json
import statistics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", required=True, help="JSONL 统计文件")
    ap.add_argument("--out", required=True, help="输出 calib.json")
    args = ap.parse_args()

    rows = []
    with open(args.stat, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            # 只用 r>=1 的样本：r=0 的客户端是从 hf 预训练开始训的，
            # 它的 Δ 相对的是预训练权重，不是"联邦更新"，量级不可比。
            if int(d.get("round", -1)) < 1:
                continue
            v = float(d.get("delta_l2", 0.0))
            if v > 0:
                rows.append(v)

    if not rows:
        print("[!!] 没有有效样本（r>=1）")
        return 1

    rows_sorted = sorted(rows)
    c = statistics.median(rows_sorted)
    out = {
        "n_samples": len(rows),
        "delta_l2_median": c,
        "delta_l2_mean": statistics.fmean(rows_sorted),
        "delta_l2_min": rows_sorted[0],
        "delta_l2_max": rows_sorted[-1],
        "clip_norm_C": c,
        "n_shared_params": 0,
        "note": ("C = ‖Δ‖₂ 中位数。σ 直接读作噪声/信号比：噪声 L2 ≈ σ·C，"
                 "信号 L2 ≈ ‖Δ‖₂ ≈ C。ε 只依赖 σ，与 C 无关。"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[标定] 样本数={len(rows)}  ‖Δ‖₂ 中位数={c:.4f}  "
          f"min={rows_sorted[0]:.4f}  max={rows_sorted[-1]:.4f}")
    print(f"[标定] → C = {c:.4f}（写入 {args.out}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
