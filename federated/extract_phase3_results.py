#!/usr/bin/env python3
"""FedVLA 第 3 阶段结果提取器（**在云端跑**，产出 JSON）。

把各阶段的评估日志解析成结构化结果，供本地 `build_phase3_report.py` 生成
论文表格与图表。**数字不在脚本里硬编码** —— 全部来自真实评估日志，
避免"论文里的数和实验里的数对不上"。

评估日志格式（HPT 的 tabulate 输出）：
    | task     |   reward |
    |----------+----------|
    | reach-v2 |     1.00 |
    | avg      |     1.00 |
每个 seed 一张表 -> 取 avg 行作为该 seed 的成功率。

统计口径：3 seeds × 20 rollouts/seed，报均值 ± 95% CI（t 分布）。

用法（云端）：
    python extract_phase3_results.py --out /root/HPT/output/phase3_raw.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

AVG_RE = re.compile(r"^\|\s*avg\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", re.M | re.I)
TASK_RE = re.compile(r"^\|\s*([A-Za-z0-9_\-]+-v\d)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", re.M)

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

# 阶段定义：(tag, 中文名, 是否异构)
STAGES = [
    ("g2b_central_eq", "集中式等预算", False),
    ("g2b_iid_e4",     "IID 联邦（修复调度）", False),
    ("g2b_hetero",     "异构联邦（修复调度，基线）", True),
    ("p3_moe_fedavg",  "异构 + MoE + FedAvg", True),
    ("p3_moe_eda",     "异构 + MoE + EDA 中心性", True),
    ("p3_moe_eda_cluster", "异构 + MoE + EDA 簇感知", True),
    ("p3_moe_iid",     "IID + MoE + FedAvg", False),
]

DOMAINS = ["mujoco_metaworld_task1", "mujoco_metaworld_task2",
           "mujoco_metaworld_task3"]


def mean_ci(vals):
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n < 2:
        return {"mean": m, "ci": 0.0, "lo": m, "hi": m, "n": n}
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    t = T_CRIT_95.get(n, 1.96)
    half = t * sd / math.sqrt(n)
    return {"mean": m, "ci": half, "lo": m - half, "hi": m + half, "n": n}


def parse_log(path):
    """返回 (每张表的 avg 列表, {task: [values]})"""
    try:
        txt = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception:
        return [], {}
    avgs = [float(x) for x in AVG_RE.findall(txt)]
    per_task = {}
    for task, val in TASK_RE.findall(txt):
        per_task.setdefault(task, []).append(float(val))
    return avgs, per_task


def find_eval_log(out_root, tag, domain):
    """评估日志名带 LOG_TAG 与 domain（见 eval_model.sh）。

    ⚠️ tag 可能带轮次后缀，必须容错匹配：
       门2' 的评估守护用的是 `g2b_hetero_r49` / `g2b_iid_e4_r49` 这类 LOG_TAG，
       而本提取器的 STAGES 用的是不带后缀的 `g2b_hetero` / `g2b_iid_e4`。
       若只做精确匹配，基线三列会全空（表格只剩第 3 阶段的行）。
       所以这里按「精确 → 带轮次后缀 → 任意后缀」逐级放宽。

    多个候选同时存在时，选**表数最多**的那个（= 评估最完整），
    而不是最新改动的那个（可能只跑了 1 个 seed 就中断）。
    """
    cands = [
        os.path.join(out_root, f"eval_{tag}_{domain}.log"),
        os.path.join(out_root, f"eval_{tag}_{domain}.out"),
    ]
    # 带轮次后缀：eval_g2b_hetero_r49_<domain>.log
    for pat in (f"eval_{tag}_r*_{domain}.log", f"eval_{tag}_*_{domain}.log"):
        cands += sorted(glob.glob(os.path.join(out_root, pat)))

    best, best_n = None, -1
    for c in cands:
        if not os.path.isfile(c):
            continue
        try:
            txt = open(c, "r", encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        n = len(AVG_RE.findall(txt))
        if n > best_n:
            best, best_n = c, n
    return best


def collect_stage(out_root, tag):
    """收集一个阶段在三个任务上的评估结果。"""
    res = {"tag": tag, "domains": {}}
    for d in DOMAINS:
        p = find_eval_log(out_root, tag, d)
        if not p:
            res["domains"][d] = None
            continue
        avgs, per_task = parse_log(p)
        entry = {
            "log": p,
            "per_seed": avgs,
            "per_task": {k: v for k, v in per_task.items()},
        }
        if avgs:
            entry.update(mean_ci(avgs))
        else:
            entry.update({"mean": None, "ci": None, "lo": None, "hi": None, "n": 0})
        res["domains"][d] = entry
    return res


def collect_eda(out_root, tags):
    """把 eda_weights.pt 里的权重导成 JSON（论文表格直接用）。"""
    out = {}
    try:
        import torch
    except Exception:
        return out
    for tag in tags:
        pats = glob.glob(os.path.join(out_root, tag, "global_r*", "eda_weights.pt"))
        if not pats:
            continue
        p = sorted(pats)[-1]
        try:
            blob = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        w = blob.get("weights")
        if w is None:
            continue
        out[tag] = {
            "file": p,
            "weights": w.tolist() if hasattr(w, "tolist") else w,
            "block_map": blob.get("block_map"),
            "clusters": blob.get("clusters"),
            "mean_weights": (w.mean(dim=0).tolist()
                             if hasattr(w, "mean") else None),
        }
    return out


def collect_convergence(out_root, tag):
    """收敛曲线：各轮客户端 loss 的均值。"""
    root = os.path.join(out_root, tag)
    if not os.path.isdir(root):
        return {}
    conv = {}
    for f in sorted(glob.glob(os.path.join(root, "log_client_*_r*.txt"))):
        m = re.search(r"log_client_(\d+)_r(\d+)\.txt", os.path.basename(f))
        if not m:
            continue
        c, r = int(m.group(1)), int(m.group(2))
        try:
            txt = open(f, "r", encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        losses = [float(x) for x in re.findall(r"Train loss:\s*([0-9]*\.?[0-9]+)", txt)]
        if not losses:
            continue
        conv.setdefault(r, {})[c] = losses[-1]
    return {str(r): (sum(v.values()) / len(v) if v else None)
            for r, v in sorted(conv.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/HPT/output/phase3_raw.json")
    ap.add_argument("--out-root", default="/root/HPT/output")
    args = ap.parse_args()

    out_root = args.out_root
    result = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "out_root": out_root,
        "stages": [],
    }

    for tag, label, hetero in STAGES:
        st = collect_stage(out_root, tag)
        st["label"] = label
        st["hetero"] = hetero
        st["convergence"] = collect_convergence(out_root, tag)
        result["stages"].append(st)
        got = [d for d, e in st["domains"].items() if e and e.get("n")]
        print(f"[{tag:<22}] 有结果的任务: {len(got)}/{len(DOMAINS)}  {got}")

    result["eda"] = collect_eda(out_root, ["p3_moe_eda", "p3_moe_eda_cluster"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已写出 {args.out}")
    print("EXTRACT_DONE")


if __name__ == "__main__":
    sys.exit(main())
