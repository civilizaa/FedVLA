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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:                                    # pragma: no cover
    plt = None

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

# 量化配置的显示位宽（用于横轴；fp32=32bit, int8=8bit, int4=4bit, bit1=1bit）
QUANT_BITS = {"fp32": 32, "int8_g": 8, "int4_g": 4, "bit1": 1}
QUANT_LABEL = {"fp32": "FP32", "int8_g": "INT8", "int4_g": "INT4", "bit1": "1-bit"}

T_CRIT_DF2 = 4.302652729911275      # t(0.975, df=2)  ← 3 seeds 时用这个
T_CRIT_DF1 = 12.706204736432095     # t(0.975, df=1)  ← 2 seeds 时用这个
T_CRIT_DF0 = 0.0                    # n=1 没法算 CI（只有 1 个观测）


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
    """均值 ± 95% CI（t 分布）。

    ⚠️ 2026-11-09 修正：临界值必须按 **df = n−1** 取，不能一律用 df=2。
    n=3 → df=2 → t=4.303；n=2 → df=1 → t=12.706。
    原实现对 n=2 也用 4.303，会把 CI **系统性低估约 66%**，
    进而让"CI 重叠 ⇒ 不显著"的判据失效（2026-11-08 在 3-seed CI 上踩过同类坑）。
    """
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        # 单次观测无法估计方差 ⇒ CI 无意义，明确标 0 而不是编一个数
        return {"mean": m, "ci": 0.0, "n": 1, "seeds": vals, "df": 0}
    var = sum((v - m) ** 2 for v in vals) / (n - 1)      # 样本方差，df = n-1
    sd = math.sqrt(var)
    tcrit = {2: T_CRIT_DF1, 3: T_CRIT_DF2}.get(n, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return {"mean": m, "ci": half, "n": n, "seeds": vals, "df": n - 1,
            "sd": sd, "lo": m - half, "hi": m + half}


def avg_comm_bytes(root: str):
    """该配置所有轮次的上行字节数均值（总上行/轮，含所有客户端）。

    口径优先级（2026-11-18 补）：
      1) `offered_bytes_per_round` —— 供给量（所有客户端 × 当前位宽），
         **fp32 基线与丢包配置只有这个值**（旧代码把字节统计写在量化分支里，
         导致这些配置的 total_bytes 是 null ⇒ 曲线缺基线点）。
      2) `total_bytes` —— 实际送达量（丢包后只剩 kept 客户端）。
      3) 两者都没有 ⇒ None。
    """
    files = sorted(glob.glob(os.path.join(root, "comm_report_r*.json")))
    if not files:
        return None
    tot = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        tb = d.get("offered_bytes_per_round") or d.get("total_bytes")
        if tb:
            tot.append(tb)
    if not tot:
        return None
    return sum(tot) / len(tot)


def _ok(row):
    """该配置是否有可用结果。"""
    return row.get("mean") is not None


def plot_results(rows, outdir, domain):
    """出 3 张论文图（英文标注，避免云端 matplotlib 缺中文字形）。

    fig7_comm_drop   : 丢包率 vs 成功率（含 95% CI 误差棒）
    fig7_comm_quant  : 量化位宽 vs 成功率（含 95% CI 误差棒）
    fig7_comm_tradeoff : 上行带宽 vs 成功率（通信-性能权衡）
    """
    if plt is None:
        print("[warn] matplotlib 不可用，跳过绘图")
        return []
    made = []
    os.makedirs(outdir, exist_ok=True)

    def _ser(pred):
        """挑出满足条件的配置，按 x 排序。"""
        out = [r for r in rows if _ok(r) and pred(r)]
        return out

    # ---------- 图 1：丢包率 vs 成功率 ----------
    drops = _ser(lambda r: r["quant"] == "fp32")
    if drops:
        drops.sort(key=lambda r: r["drop_rate"])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        xs = [r["drop_rate"] * 100 for r in drops]
        ys = [r["mean"] for r in drops]
        es = [r["ci"] for r in drops]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#2f6fbf", capsize=4,
                    lw=1.8, ms=6, label="MoE + cluster-aware EDA")
        for r, x, y in zip(drops, xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        ax.set_xlabel("Packet drop rate (%)")
        ax.set_ylabel("Task success rate")
        ax.set_title(f"Robustness to packet loss ({domain.replace('mujoco_metaworld_','')})",
                     fontsize=10)
        ax.set_xticks(xs)
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=8)
        p = os.path.join(outdir, "fig7_comm_drop.png")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    # ---------- 图 2：量化位宽 vs 成功率 ----------
    quants = [r for r in rows if _ok(r) and r["quant"] != "fp32" and r["drop_rate"] == 0.0]
    base = next((r for r in rows if r["tag"] == "c_drop00" and _ok(r)), None)
    if quants:
        quants.sort(key=lambda r: QUANT_BITS.get(r["quant"], 99))
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        if base is not None:
            ax.axhline(base["mean"], color="#8a8a8a", ls=":", lw=1.5,
                       label=f"FP32 baseline ({base['mean']:.2f})")
            ax.fill_between([0.5, 32], base["mean"] - base["ci"], base["mean"] + base["ci"],
                            color="#bbbbbb", alpha=0.3)
        xs = [QUANT_BITS.get(r["quant"], 0) for r in quants]
        ys = [r["mean"] for r in quants]
        es = [r["ci"] for r in quants]
        ax.errorbar(xs, ys, yerr=es, fmt="s-", color="#c1544a", capsize=4,
                    lw=1.8, ms=6, label="Quantized uplink")
        for r, x, y in zip(quants, xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([QUANT_LABEL.get(r["quant"], r["quant"]) for r in quants])
        ax.set_xlabel("Uplink quantization bit-width")
        ax.set_ylabel("Task success rate")
        ax.set_title(f"Robustness to quantization ({domain.replace('mujoco_metaworld_','')})",
                     fontsize=10)
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=8)
        p = os.path.join(outdir, "fig7_comm_quant.png")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    # ---------- 图 3：上行带宽 vs 成功率（权衡） ----------
    both = [r for r in rows if _ok(r) and r.get("up_mib_per_round")]
    if len(both) >= 2:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for r in both:
            ax.scatter(r["up_mib_per_round"], r["mean"], s=70,
                       marker="o" if r["quant"] == "fp32" else "s",
                       color="#2f6fbf" if r["quant"] == "fp32" else "#c1544a")
            ax.annotate(r["tag"].replace("c_", ""),
                        (r["up_mib_per_round"], r["mean"]),
                        textcoords="offset points", xytext=(7, 4), fontsize=7)
        if base is not None:
            ax.axhline(base["mean"], color="#8a8a8a", ls=":", lw=1.5)
        ax.set_xlabel("Uplink payload per round (MiB)")
        ax.set_ylabel("Task success rate")
        ax.set_title("Communication-performance trade-off", fontsize=10)
        ax.grid(alpha=0.3, ls="--")
        p = os.path.join(outdir, "fig7_comm_tradeoff.png")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    for p in made:
        print(f"[图] {p}")
    return made


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
        print("相对无损基线的成功率变化（pp，正=变差，负=变好）：")
        for r in rows:
            if r["mean"] is None or r["tag"] == "c_drop00":
                continue
            d = (base["mean"] - r["mean"]) * 100
            verdict = "变差" if d > 0 else ("变好" if d < 0 else "持平")
            print(f"  {r['label']:<26} {d:+6.1f} pp ({verdict})"
                  f"   （{r['mean']:.3f} vs {base['mean']:.3f}）")

    # ===== 基线自检（关键：曲线能不能用全看这一条）=====
    # c_drop00 的实验台与第3阶段的 P3-C 完全相同（异构 + MoE + 簇感知），
    # 只是轮数也对齐到 50 ⇒ 它的 task3 成功率**必须**落在 P3-C 的观测附近。
    #
    # ⚠️ 2026-11-18 修正：P3-C 的 task3 是 **3 个 seed = 0.90 / 0.60 / 0.50**
    #   （均值 0.667）。早先写的 "0.60 / 0.50，均值 0.55" 是**漏掉 seed 0** 的
    #   2-seed 观测，与第3阶段报告里的 0.667 自相矛盾 ⇒ 参考值偏低 0.117。
    #   容差仍取 ±0.20：P3-C 三个 seed 自身跨度就有 0.40（0.90 vs 0.50），
    #   说明该任务的 seed 方差本来就大，容差过窄会把正常基线误杀。
    P3C_REF = 0.667
    P3C_TOL = 0.20
    P3C_SEEDS = "0.90 / 0.60 / 0.50（3 seeds，均值 0.667）"
    print()
    if base is None or base["mean"] is None:
        print("[!!] 基线自检：**c_drop00 没有评估结果** —— 无法确认基线有效，整组曲线不可用")
        print("     处置：先修好 c_drop00 的训练/评估，再重跑提取。")
    elif base["mean"] < P3C_REF - P3C_TOL:
        print(f"[!!] 基线自检**未通过**：c_drop00 = {base['mean']:.3f}，"
              f"低于 P3-C 参考值 {P3C_REF:.3f}（容差 ±{P3C_TOL:.2f}）")
        print("     这说明本次扫描的基线本身有问题（不是通信损伤导致的），")
        print("     其余配置的相对下降值**不可解释** → 本组曲线作废，不得写进论文。")
        print(f"     参考：P3-C task3 实测 {P3C_SEEDS}；"
              f"c_drop00 的 seeds = {base['seeds']}")
    else:
        print(f"[OK] 基线自检通过：c_drop00 = {base['mean']:.3f} ± {base['ci']:.3f}"
              f"（P3-C 参考 {P3C_REF:.3f}，容差 ±{P3C_TOL:.2f}）")
        print("     ⇒ 相对下降值可解释，曲线可用。")

    print()
    print(f"[输出] {args.out}")

    # ---- 出图（英文标注）----
    try:
        plot_results(rows, os.path.dirname(os.path.abspath(args.out)), args.domain)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 绘图失败：{type(e).__name__}: {e}")

    print("COMM_EXTRACT_DONE")


if __name__ == "__main__":
    main()
