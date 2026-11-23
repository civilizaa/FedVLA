#!/usr/bin/env python3
"""从 phase5_raw.json（第3阶段 raw 也可）生成主表 + 配对 t 检验（论文口径）。

用法：
    python tools/phase5_table.py results/phase5/phase5_raw.json \
        --phase3 results/phase3_raw.json > results/phase5/phase5_table.md

统计口径（与第 3/4 阶段逐字一致，勿改）：
  · 每个 (配置, domain) = 3 seeds × 20 rollouts 的成功率
  · 均值 ± 95% CI，t 分布 **df = n − 1 = 2 ⇒ t_crit = 4.303**
  · 配对比较用**同 seed 配对 t 检验**（同 domain、同 seed 索引）

🔴 两条铁律：
  1. **df = n−1**（不是 n）。云端曾有一份陈旧的 extract_phase3_results.py 用 `get(n)`
     ⇒ CI 偏窄（3.182 而非 4.303）⇒ 本脚本**独立复算** CI 与 p，不信任上游的 stats。
  2. **MoE 生效门槛**：每个 domain 的 params_M 必须 ≥ 18.0（完整 MoE = 22.38M，
     残缺 = 13.94M）⇒ 低于门槛的行**显式作废**，不参与任何比较。

df=2 的双尾 p 有闭式解（并已用 F(1,2) 关系独立复核）：
    p = 1 − |t| / sqrt(t² + 2)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

# t 分布 97.5 分位（df → t_crit），df = n − 1
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

DOMAINS = ["mujoco_metaworld_task1", "mujoco_metaworld_task2", "mujoco_metaworld_task3"]
DOMAIN_SHORT = {"mujoco_metaworld_task1": "reach-v2 (task1)",
                "mujoco_metaworld_task2": "push-v2 (task2)",
                "mujoco_metaworld_task3": "button-press (task3)"}
PARAMS_MIN_M = 18.0

# ---- 论文关心的配对比较：(A, B, 说明)----
PAIRS = [
    ("p3_moe_eda", "p3_moe_fedavg", "H1 均衡臂 EDA(中心性) vs FedAvg"),
    ("p3_moe_eda_cluster", "p3_moe_fedavg", "H2 均衡臂 簇感知 vs FedAvg"),
    ("p3_moe_eda_cluster", "p3_moe_eda", "H2 均衡臂 簇感知 vs 中心性 EDA"),
    ("p3_moe_eda_cluster", "g2b_hetero", "H2 均衡臂 簇感知 vs 异构基线(无MoE)"),
    ("p5_bal_fedprox", "p3_moe_fedavg", "均衡臂 FedProx vs FedAvg"),
    ("p5_bal_scaffold", "p3_moe_fedavg", "均衡臂 SCAFFOLD vs FedAvg"),
    ("p5_bal_scaffold", "p5_bal_fedprox", "均衡臂 SCAFFOLD vs FedProx"),
    ("p5_iso_eda", "p5_iso_fedavg", "孤立臂 EDA(中心性) vs FedAvg"),
    ("p5_iso_edaclu", "p5_iso_fedavg", "H2 孤立臂 簇感知 vs FedAvg"),
    ("p5_iso_edaclu", "p5_iso_eda", "H2 孤立臂 簇感知 vs 中心性 EDA"),
    ("p5_iso_fedprox", "p5_iso_fedavg", "孤立臂 FedProx vs FedAvg"),
    ("p5_iso_scaffold", "p5_iso_fedavg", "孤立臂 SCAFFOLD vs FedAvg"),
    ("p5_iso_fedprox", "p5_iso_scaffold", "孤立臂 FedProx vs SCAFFOLD"),
]

# 参考行（不参与配对检验，只作上下界）：(tag, 中文名, 说明)
REF_ROWS = [
    ("g2b_central_eq", "集中式等预算", "性能上界（无联邦）"),
    ("g2b_hetero", "异构联邦基线（无 MoE）", "联邦下界"),
]


def seeds_of(entry):
    """兼容两种 raw 结构：phase5 用 seeds，phase3 用 per_seed。"""
    if not entry:
        return []
    for k in ("seeds", "per_seed"):
        v = entry.get(k)
        if v:
            return [float(x) for x in v]
    return []


def params_of(entry):
    if not entry:
        return None
    return entry.get("params_M")


def mean_ci(vals):
    """独立复算均值与 95% CI（df = n − 1）。"""
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n < 2:
        return {"mean": m, "ci": 0.0, "lo": m, "hi": m, "n": n}
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    t = T_CRIT_95.get(n - 1, 1.96)
    half = t * sd / math.sqrt(n)
    return {"mean": m, "ci": half, "lo": m - half, "hi": m + half, "n": n, "sd": sd}


def paired_t(a, b):
    """同 seed 配对 t 检验，返回 (t, df, 差值均值, 差值sd)。"""
    n = min(len(a), len(b))
    if n < 2:
        return None
    d = [a[i] - b[i] for i in range(n)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    if sd == 0:
        return (float("inf") if m > 0 else (0.0 if m == 0 else float("-inf")), n - 1, m, sd)
    return (m / (sd / math.sqrt(n)), n - 1, m, sd)


def p_from_t(t, df):
    """双尾 p。df=2 有闭式解：p = 1 − |t|/√(t²+2)（已用 F(1,2) 独立复核）。"""
    if t is None:
        return None
    if df == 2:
        return 1.0 - abs(t) / math.sqrt(t * t + 2.0)
    return None


def load_stages(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    stages = data.get("stages") or data.get("rows") or []
    if isinstance(stages, dict):
        stages = list(stages.values())
    return data, {s.get("tag"): s for s in stages}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="phase5_raw.json")
    ap.add_argument("--phase3", default=None, help="phase3_raw.json（提供参考行与异构基线）")
    args = ap.parse_args()

    data5, by_tag5 = load_stages(args.raw)
    by_tag = dict(by_tag5)
    if args.phase3 and os.path.exists(args.phase3):
        _, by_tag3 = load_stages(args.phase3)
        for k, v in by_tag3.items():
            by_tag.setdefault(k, v)

    out = []
    out.append("# 第 5 阶段主实验矩阵（成功率，均值 ± 95% CI，t df=n−1=2 ⇒ t_crit=4.303）")
    out.append("")
    out.append("统计口径：3 seeds × 20 rollouts/seed；CI 与 p 均由本脚本独立复算（不采信上游 stats）。")
    out.append("")

    # ---- 参考上下界 ----
    if any(t in by_tag for t, _, _ in REF_ROWS):
        out.append("## 参考上下界（第 2 阶段门2实验，不重跑）")
        out.append("")
        out.append("| 配置 | 说明 | " + " | ".join(DOMAIN_SHORT[d] for d in DOMAINS) + " |")
        out.append("|" + "---|" * (2 + len(DOMAINS)))
        for tag, name, desc in REF_ROWS:
            s = by_tag.get(tag)
            if not s:
                continue
            cells = []
            for d in DOMAINS:
                e = (s.get("domains") or {}).get(d)
                sd = seeds_of(e)
                if not sd:
                    cells.append("—")
                    continue
                st = mean_ci(sd)
                cells.append(f"{st['mean']:.3f} ± {st['ci']:.3f}")
            out.append(f"| {name} | {desc} | " + " | ".join(cells) + " |")
        out.append("")

    # ---- 主表 ----
    out.append("## 主表（第 5 阶段主实验矩阵）")
    out.append("")
    hdr = "| 配置 | 划分 | 来源 | " + " | ".join(DOMAIN_SHORT[d] for d in DOMAINS) + \
          " | 三任务均值 | MoE 参数量(M) |"
    out.append(hdr)
    out.append("|" + "---|" * (3 + len(DOMAINS) + 2))

    for s in (data5.get("stages") or []):
        cells, means, pms = [], [], []
        for d in DOMAINS:
            e = (s.get("domains") or {}).get(d)
            sd = seeds_of(e)
            if not sd:
                cells.append("—")
                continue
            pm = params_of(e)
            if pm is not None and pm < PARAMS_MIN_M:
                cells.append(f"**作废**({pm}M)")
                continue
            st = mean_ci(sd)
            cells.append(f"{st['mean']:.3f} ± {st['ci']:.3f}")
            means.append(st["mean"])
            if pm is not None:
                pms.append(pm)
        avg = f"{sum(means)/len(means):.3f}" if means else "—"
        pmstr = f"{min(pms):.2f}" if pms else "—"
        name = s.get("label") or s.get("name") or s.get("tag")
        out.append(f"| {name} | {s.get('split','')} | {s.get('source','')} | " +
                   " | ".join(cells) + f" | {avg} | {pmstr} |")

    # ---- 配对 t 检验 ----
    out.append("")
    out.append("## 配对 t 检验（同 seed 配对，df = n−1 = 2）")
    out.append("")
    out.append("| 比较 | domain | Δ均值 | t | p(双尾) | 结论 |")
    out.append("|---|---|---|---|---|---|")
    for a_tag, b_tag, desc in PAIRS:
        a, b = by_tag.get(a_tag), by_tag.get(b_tag)
        if not a or not b:
            continue
        for d in DOMAINS:
            ae = (a.get("domains") or {}).get(d)
            be = (b.get("domains") or {}).get(d)
            sa, sb = seeds_of(ae), seeds_of(be)
            if not sa or not sb:
                continue
            # MoE 门槛：任一侧残缺模型 ⇒ 该格作废，不做检验
            bad = False
            for e in (ae, be):
                pm = params_of(e)
                if pm is not None and pm < PARAMS_MIN_M:
                    bad = True
            if bad:
                continue
            t, df, dm, dsd = paired_t(sa, sb)
            if t is None:
                continue
            p = p_from_t(t, df)
            if t == float("inf") or t == float("-inf"):
                verdict = "A 恒优于 B" if dm > 0 else "A 恒劣于 B"
                pstr = "<0.001"
                tstr = "∞"
            elif t == 0.0:
                verdict = "无差异"
                pstr = "1.000"
                tstr = "0.000"
            else:
                tstr = f"{t:.3f}"
                pstr = f"{p:.3f}" if p is not None else "—"
                verdict = "**显著**" if (p is not None and p < 0.05) else "不显著"
            out.append(f"| {desc} | {DOMAIN_SHORT[d]} | {dm:+.3f} | {tstr} | {pstr} | {verdict} |")

    out.append("")
    out.append("> p 用 df=2 的闭式解 p = 1 − |t|/√(t²+2)（双尾，已用 F(1,2) 关系独立复核）；显著性门槛 p < 0.05。")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
