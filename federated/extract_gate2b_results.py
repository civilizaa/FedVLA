#!/usr/bin/env python3
"""提取门2' 消融的【成功率】结果，输出 JSON + Markdown 表。

数据来源：各变体的评估日志 `/root/HPT/output/eval_{LOG_TAG}_{DOMAIN}.log`
（由 run_gate2b_eval.sh 用 eval_model.sh 产生）。
每个 seed 一张 `| avg |` 表 -> 取 avg 作为该 seed 的成功率，
报均值 ± 95% CI（t 分布，3 seeds）。

用法：
    python extract_gate2b_results.py --out-root /root/HPT/output \
        --out /root/HPT/output/gate2b_results.json \
        [--md /root/HPT/output/gate2b_table.md]
"""
import argparse
import json
import math
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_eval import AVG_RE, TASK_RE, mean_ci  # noqa: E402

T1 = "mujoco_metaworld_task1"
T2 = "mujoco_metaworld_task2"
T3 = "mujoco_metaworld_task3"

# (阶段, LOG_TAG, {domain: env}, 说明)
VARIANTS = [
    ("V1", "g2b_iid_v1_r9", {T1: "reach-v2"},
     "IID 联邦 10 轮 · 只开全局 lr"),
    ("V2", "g2b_iid_v2_r9", {T1: "reach-v2"},
     "IID 联邦 10 轮 · 全局 lr + Adam 持久化"),
    ("E3", "g2b_central_eq", {T1: "reach-v2"},
     "集中式等预算 2250 步 · 全局 lr（调度口径上界）"),
    ("E4", "g2b_iid_e4_r49", {T1: "reach-v2"},
     "IID 联邦 50 轮 · 全局 lr + Adam 持久化（修复后联邦基线）"),
    ("E5", "g2b_hetero_r49", {T1: "reach-v2", T2: "push-v2", T3: "button-press-topdown-v2"},
     "异构联邦 50 轮 · 全局 lr + Adam 持久化（第 4 章正式结论）"),
]

# 门2（旧调度）的既有基线，用于对照
OLD_BASELINE = {
    "集中式 6000 步（旧）": {T1: 0.900, T2: 0.983, T3: 0.967},
    "等预算集中式 2250 步（旧）": {T1: 0.950},
    "IID 联邦 50 轮（旧调度）": {T1: 0.633},
    "异构联邦 50 轮（旧调度）": {T1: 0.017, T2: 0.000, T3: 0.183},
}


def parse_log(path):
    try:
        txt = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception:
        return [], {}
    avgs = [float(x) for x in AVG_RE.findall(txt)]
    per_task = {}
    for task, val in TASK_RE.findall(txt):
        per_task.setdefault(task, []).append(float(val))
    return avgs, per_task


def collect(out_root, stage, tag, domains):
    res = {"stage": stage, "log_tag": tag, "domains": {}}
    for domain in domains:
        logf = os.path.join(out_root, f"eval_{tag}_{domain}.log")
        if not os.path.isfile(logf):
            res["domains"][domain] = {"log": logf, "status": "missing"}
            continue
        avgs, per_task = parse_log(logf)
        entry = {
            "log": logf,
            "status": "ok" if avgs else "no_table",
            "per_seed": avgs,
            "per_task": per_task,
        }
        if avgs:
            m, half, lo, hi = mean_ci(avgs)
            entry.update({"mean": m, "ci": half, "lo": lo, "hi": hi, "n": len(avgs)})
        else:
            entry.update({"mean": None, "ci": None, "lo": None, "hi": None, "n": 0})
        res["domains"][domain] = entry
    return res


def fmt(entry):
    if not entry or entry.get("mean") is None:
        return "—"
    return f"{entry['mean']:.3f} ± {entry['ci']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/root/HPT/output")
    ap.add_argument("--out", default="/root/HPT/output/gate2b_results.json")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    results = []
    for stage, tag, domains, desc in VARIANTS:
        r = collect(args.out_root, stage, tag, list(domains))
        r["desc"] = desc
        results.append(r)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "statistic": "3 seeds x 20 rollouts, mean +- 95% CI (t-distribution)",
        "variants": results,
        "old_baseline_gate2": OLD_BASELINE,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ── Markdown 表 ────────────────────────────────────────────────────────
    lines = []
    lines.append("# 门2' 消融：成功率结果（3 seeds × 20 rollouts）\n")
    lines.append(f"生成时间：{payload['generated_at']}　统计口径：{payload['statistic']}\n")
    lines.append("\n## 新方案（全局 lr 调度）\n")
    lines.append("| 阶段 | 方案 | task1 (reach-v2) | task2 (push-v2) | task3 (button) |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        d = r["domains"]
        cells = [fmt(d.get(t)) for t in (T1, T2, T3)]
        lines.append(f"| {r['stage']} | {r['desc']} | " + " | ".join(cells) + " |")

    lines.append("\n## 门2 基线（旧调度，仅作对照）\n")
    lines.append("| 方案 | task1 | task2 | task3 |")
    lines.append("|---|---|---|---|")
    for name, d in OLD_BASELINE.items():
        cells = [f"{d[t]:.3f}" if t in d else "—" for t in (T1, T2, T3)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    # ── 关键判据 ───────────────────────────────────────────────────────────
    lines.append("\n## 关键判据\n")
    e3 = next((r for r in results if r["stage"] == "E3"), None)
    e4 = next((r for r in results if r["stage"] == "E4"), None)
    e5 = next((r for r in results if r["stage"] == "E5"), None)
    v1 = next((r for r in results if r["stage"] == "V1"), None)
    v2 = next((r for r in results if r["stage"] == "V2"), None)

    def m(r, t=T1):
        if not r:
            return None
        e = r["domains"].get(t) or {}
        return e.get("mean")

    pairs = [
        ("V1 -> V2：Adam 持久化的净贡献（IID 10 轮）", m(v1), m(v2)),
        ("E3 -> E4：联邦管线损耗（同调度口径）", m(e3), m(e4)),
        ("E4 -> E5：异构专有损害", m(e4), m(e5)),
    ]
    lines.append("| 对比 | 前者 | 后者 | 差值 |")
    lines.append("|---|---|---|---|")
    for name, a, b in pairs:
        if a is None or b is None:
            lines.append(f"| {name} | — | — | 待评估 |")
        else:
            lines.append(f"| {name} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")

    lines.append("\n判读规则：")
    lines.append("- **E4 → E5 的差值** = 异构专有损害；若接近 0，说明旧调度下的巨大差距"
                 "其实来自调度缺陷而非异构性，H2（按簇聚合更优）的前提需要重估。")
    lines.append("- **E3 → E4 的差值** = 同调度口径下联邦管线损耗；若已很小，"
                 "则第 4 章可改写为「异构是主因」。")
    lines.append("- **V1 → V2 的差值** = Adam 状态跨轮持久化的净贡献。")

    md = "\n".join(lines) + "\n"
    md_path = args.md or os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\n[OK] JSON -> {args.out}")
    print(f"[OK] Markdown -> {md_path}")
    print("GATE2B_RESULTS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
