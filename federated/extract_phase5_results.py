#!/usr/bin/env python3
"""FedVLA 第 5 阶段结果提取器（**在云端跑**，产出 JSON + Markdown 表）。

把主实验矩阵各配置的评估日志解析成结构化结果，供本地生成论文主表。

═══════════════════════════════════════════════════════════════════════════════
口径一致性（关键）
═══════════════════════════════════════════════════════════════════════════════
本文件**直接复用** extract_phase3_results.py 的解析与统计函数
（parse_log / mean_ci / T_CRIT_95），而不是自己重写一遍：
  * 成功率口径（3 seeds × 20 rollouts，取每张表的 avg 行）
  * 95% CI 口径（t 分布，**df = n − 1**；3 seeds → df=2 → t=4.303）
必须与第 3 阶段完全一致，否则两张表的数字没法并列。

⚠️ df 陷阱（第 3 阶段已踩过）：t 临界值要按 **df = n − 1** 取，
   3 个样本 → df=2 → 4.303。写成按 n 取会把 CI 系统性低估。

═══════════════════════════════════════════════════════════════════════════════
数字不在脚本里硬编码 —— 全部来自真实评估日志
═══════════════════════════════════════════════════════════════════════════════
用法（云端）：
    python extract_phase5_results.py --out-root /root/HPT/output/phase5 \
        --out /root/HPT/output/phase5/phase5_raw.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 复用第 3 阶段的解析/统计实现（口径必须一致，不重写）
from extract_phase3_results import (T_CRIT_95, collect_convergence,  # noqa: E402
                                    collect_eda, mean_ci, parse_log)

DOMAINS = ["mujoco_metaworld_task1", "mujoco_metaworld_task2",
           "mujoco_metaworld_task3"]

# 主实验矩阵：(tag, 中文名, 划分)
STAGES = [
    ("p5_bal_fedprox",    "均衡划分 + FedProx",        "balanced"),
    ("p5_bal_scaffold",   "均衡划分 + SCAFFOLD",       "balanced"),
    ("p5_iso_fedavg",     "孤立客户端 + FedAvg",       "isolated"),
    ("p5_iso_fedprox",    "孤立客户端 + FedProx",      "isolated"),
    ("p5_iso_scaffold",   "孤立客户端 + SCAFFOLD",     "isolated"),
    ("p5_iso_eda",        "孤立客户端 + EDA 中心性",    "isolated"),
    ("p5_iso_edaclu",     "孤立客户端 + 簇感知（本文）", "isolated"),
]


def find_eval_log(out_root: str, tag: str, domain: str):
    """定位评估日志（命名与 eval_model.sh 的 LOG_TAG 一致）。"""
    cands = [
        os.path.join(out_root, f"eval_{tag}_{domain}.log"),
        os.path.join(out_root, f"eval_{tag}_{domain}.out"),
    ]
    for pat in (f"eval_{tag}_r*_{domain}.log", f"eval_{tag}_*_{domain}.log"):
        cands += sorted(glob.glob(os.path.join(out_root, pat)))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def collect_stage(out_root: str, tag: str):
    """收集一个配置的全部结果。"""
    st = {"tag": tag, "domains": {}, "eval_logs": {}}
    for d in DOMAINS:
        p = find_eval_log(out_root, tag, d)
        if not p:
            st["domains"][d] = None
            continue
        avgs, per_task = parse_log(p)
        st["eval_logs"][d] = p
        st["domains"][d] = {
            "seeds": avgs,
            "per_task": per_task,
            "stats": mean_ci(avgs),
            "n_tables": len(avgs),
        }
        # 参数量（MoE 是否生效的判据：> 18M）
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "number of params (M):" in line:
                        v = line.split("number of params (M):")[1].strip().split()[0]
                        st["domains"][d]["params_M"] = float(v)
                        break
        except Exception:
            pass
    return st


def collect_baseline(out_root: str, tag: str):
    """复用第 3 阶段的集中式基线（不重跑！）。"""
    p = os.path.join(out_root, f"..", "phase3_raw.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            blob = json.load(f)
        for s in blob.get("stages", []):
            if s.get("tag") == tag:
                return s
    except Exception:
        return None
    return None


def to_markdown(result: dict) -> str:
    """生成可直接贴进论文的主表（Markdown）。"""
    lines = []
    lines.append("| 配置 | 划分 | " + " | ".join(
        ["reach-v2", "push-v2", "button-press"]))
    lines.append("|---|---|---|---|---|")
    for st in result["stages"]:
        cells = []
        for d in DOMAINS:
            e = st["domains"].get(d)
            if not e or not e.get("stats") or e["stats"].get("n", 0) < 1:
                cells.append("—")
            else:
                s = e["stats"]
                cells.append(f"{s['mean']:.3f} ± {s['ci']:.3f}")
        lines.append(f"| {st['label']} | {st['split']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/root/HPT/output/phase5")
    ap.add_argument("--out", default="/root/HPT/output/phase5/phase5_raw.json")
    args = ap.parse_args()

    out_root = args.out_root
    result = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "out_root": out_root,
        "t_crit_note": "95% CI 用 t 分布，df = n - 1（3 seeds -> df=2 -> t=4.303）",
        "stages": [],
    }

    for tag, label, split in STAGES:
        st = collect_stage(out_root, tag)
        st["label"] = label
        st["split"] = split
        st["convergence"] = collect_convergence(out_root, tag)
        result["stages"].append(st)
        got = [d for d in DOMAINS
               if st["domains"].get(d) and st["domains"][d].get("n_tables", 0) > 0]
        print(f"[{tag:<20}] 有结果的任务: {len(got)}/{len(DOMAINS)}  {got}")

    # 复用第 3 阶段的集中式上界（不重跑）
    result["centralized_upper_bound"] = collect_baseline(
        os.path.dirname(out_root.rstrip("/")), "g2b_central_eq")

    result["eda"] = collect_eda(out_root, ["p5_iso_eda", "p5_iso_edaclu"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 第 5 阶段主实验矩阵（task3 成功率，均值 ± 95% CI）\n\n")
        f.write(to_markdown(result) + "\n")
    print(f"[OK] 已写出 {args.out}")
    print(f"[OK] 已写出 {md_path}")
    print("EXTRACT_DONE")


if __name__ == "__main__":
    sys.exit(main())
