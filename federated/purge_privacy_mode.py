#!/usr/bin/env python3
"""从 privacy_results.json 里剔除指定 mode 的全部记录（用于作废参数跑错的扫描）。

用法：
    python federated/purge_privacy_mode.py --json <path> --mode modality_rev

为什么需要（2026-11-08 实测事故）：
    手动补跑 `modality_rev` 时**漏了 `--dp-steps 50`**，用了脚本默认值 25。
    DP 的 ε 与步数近似成正比 ⇒ 同一 σ_ref 下总 ε 只有主扫描的一半
    （实测 2.5e9 vs 5e9）⇒ 两套曲线的横轴语义不同，**放在一起比较是错的**。
    而隐私脚本的续跑逻辑会跳过 (样本, 模式, σ, 目标) 已存在的组合，
    不清掉就会把错的记录永久留在结果里。
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--mode", required=True, help="要剔除的 mode 名，如 modality_rev")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs", [])
    keep = [r for r in runs if r.get("mode") != args.mode]
    drop = len(runs) - len(keep)
    print(f"总记录 {len(runs)} → 保留 {len(keep)}，剔除 {drop} 条 mode={args.mode}")
    if args.dry_run:
        return 0
    if drop == 0:
        print("无需改动")
        return 0
    data["runs"] = keep
    tmp = args.json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.json)
    print(f"已重写 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
