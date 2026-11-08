#!/usr/bin/env python3
"""FedVLA 第 4 阶段：通信损伤扫描的结果提取。

输入（云端）：
  /root/HPT/output/eval_c_<配置>_mujoco_metaworld_task3.log   3 个 seed 的成功率
  /root/HPT/output/c_<配置>/comm_report_r*.json               每轮的上行字节数
输出：
  /root/HPT/output/comm_raw.json     机器可读
  stdout                                可读表格

统计口径：3 seeds × 20 rollouts/seed，报均值 ± 95% CI（t 分布，df=2）。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

OUT = "/root/HPT/output"

# (tag, 丢包率, 量化方案, 人类可读标签)
CONFIGS = [
    ("c_drop00", 0.00, "fp32",   "无损基线 (fp32)"),
    ("c_drop25", 0.25, "fp32",   "丢包 25%"),
    ("c_drop50", 0.50, "fp32",   "丢包 50%"),
    ("c_drop75", 0.75, "fp32",   "丢包 75%"),
    ("c_int8",   0.00, "int8_g", "int8 per-group(64)"),
    ("c_int4",   0.00, "int4_g", "int4 per-group(64)"),
    ("c_bit1",   0.00, "bit1",   "1-bit 符号化"),
]

T_CRIT_DF2 = 4.302652729911275      # t(0.975, df=2)


def parse_seed_successes(path: str):
    """从评估日志里解析每个 seed 的成功率。

    日志格式（每个 seed 一段）：
        | task    |   reward |
        | reach-v2|     0.95 |
        | avg     |     0.95 |
    """
    if not os.path.exists(path):
        return []
    txt = open(path, encoding="utf-8", errors="ignore").read()
    # 抓 "| avg | <数值> |" 这一行，每个 seed 出现一次
    vals = []
    for m in re.finditer(r"\|\s*avg\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", txt):
        vals.append(float(m.group(1)))
    return vals


def mean_ci(vals):
    """均值 ± 95% CI（t 分布）。"""
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return {"mean": m, "ci": 0.0, "n": 1, "seeds": vals}
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    half = T_CRIT_DF2 * sd / math.sqrt(n) if n == 2 else \
        (T_CRIT_DF2 * sd / math.sqrt(n) if n == 3 else 1.96 * sd / math.sqrt(n))
    return {"mean": m, "ci": half, "n": n, "seeds": vals,
            "sd": sd, "lo": m - half, "hi": m + half}


def avg_comm_bytes(root: str):
    """该配置所有轮次的上行字节数均值（总上行/轮，含所有客户端）。"""
    files = sorted(glob.glob(os.path.join(root, "comm_report_r*.json")))
    if not files:
        return None
    tot = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        tb = d.get("total_bytes")
        if tb:
            tot.append(tb)
    if not tot:
        return None
    return sum(tot) / len(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "comm_raw.json"))
    ap.add_argument("--domain", default="mujoco_metaworld_task3")
    args = ap.parse_args()

    rows = []
    for tag, drop, quant, label in CONFIGS:
        logp = os.path.join(OUT, f"eval_{tag}_{args.domain}.log")
        vals = parse_seed_successes(logp)
        stat = mean_ci(vals)
        root = os.path.join(OUT, tag)
        b = avg_comm_bytes(root)
        rows.append({
            "tag": tag, "label": label, "drop_rate": drop, "quant": quant,
            "domain": args.domain,
            "seeds": vals,
            "n": len(vals),
            "mean": stat["mean"] if stat else None,
            "ci": stat["ci"] if stat else None,
            "lo": stat.get("lo") if stat else None,
            "hi": stat.get("hi") if stat else None,
            "up_bytes_per_round": b,
            "up_mib_per_round": (b / 1048576) if b else None,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"domain": args.domain, "rows": rows}, f,
                  ensure_ascii=False, indent=2)

    print("=" * 108)
    print(f"第 4 阶段：通信损伤扫描结果（{args.domain}，3 seeds × 20 rollouts，均值 ± 95% CI）")
    print("=" * 108)
    print(f"{'配置':<26}{'丢包':>7}{'量化':>10}{'上行MiB/轮':>12}"
          f"{'成功率':>18}{'seeds':>22}")
    print("-" * 108)
    for r in rows:
        if r["mean"] is None:
            print(f"{r['label']:<26}{r['drop_rate']:>7.0%}{r['quant']:>10}"
                  f"{'—':>12}{'（无评估日志）':>18}{'':>22}")
            continue
        s = f"{r['mean']:.3f} ± {r['ci']:.3f}"
        seeds = ",".join(f"{v:.2f}" for v in r["seeds"])
        mib = f"{r['up_mib_per_round']:.1f}" if r["up_mib_per_round"] else "—"
        print(f"{r['label']:<26}{r['drop_rate']:>7.0%}{r['quant']:>10}"
              f"{mib:>12}{s:>18}{seeds:>22}")

    base = next((r for r in rows if r["tag"] == "c_drop00"), None)
    if base and base["mean"] is not None:
        print()
        print("相对无损基线的下降（百分点）：")
        for r in rows:
            if r["mean"] is None or r["tag"] == "c_drop00":
                continue
            d = (base["mean"] - r["mean"]) * 100
            print(f"  {r['label']:<26} {d:+6.1f} pp"
                  f"   （{r['mean']:.3f} vs {base['mean']:.3f}）")
    print()
    print(f"[输出] {args.out}")
    print("COMM_EXTRACT_DONE")


if __name__ == "__main__":
    main()
