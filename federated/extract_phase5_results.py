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

# 🔴 2026-11-23：MoE 生效判据（参数量下限）。低于此值 = 评估的是残缺模型，
#    该 domain 的结果一律作废并显式报错（修 fail-open：原来只记录不校验）。
PARAMS_MIN_M = 18.0

DOMAINS = ["mujoco_metaworld_task1", "mujoco_metaworld_task2",
           "mujoco_metaworld_task3"]

# 主实验矩阵：(tag, 中文名, 划分, 来源)
#
# 🔴 2026-11-23 补：balanced 臂的 FedAvg / EDA / EDA-Cluster **复用第 3 阶段产物**
#    （划分 t1/t2/t3/t1、超参、轮数逐字同构 ⇒ 铁律 D 成立），
#    否则主表 balanced 侧会缺 3 个格子。
#    第 3 阶段的评估日志在 /root/HPT/output/（根），第 5 阶段在 /root/HPT/output/phase5/
#    ⇒ find_eval_log / collect_convergence / collect_eda 都必须**跨两个根**找。
STAGES = [
    ("p3_moe_fedavg",      "均衡划分 + FedAvg",         "balanced", "第3阶段复用"),
    ("p3_moe_eda",         "均衡划分 + EDA 中心性",       "balanced", "第3阶段复用"),
    ("p3_moe_eda_cluster", "均衡划分 + 簇感知（本文）",     "balanced", "第3阶段复用"),
    ("p5_bal_fedprox",     "均衡划分 + FedProx",         "balanced", "第5阶段新跑"),
    ("p5_bal_scaffold",    "均衡划分 + SCAFFOLD",        "balanced", "第5阶段新跑"),
    ("p5_iso_fedavg",      "孤立客户端 + FedAvg",        "isolated", "第5阶段新跑"),
    ("p5_iso_fedprox",     "孤立客户端 + FedProx",       "isolated", "第5阶段新跑"),
    ("p5_iso_scaffold",    "孤立客户端 + SCAFFOLD",      "isolated", "第5阶段新跑"),
    ("p5_iso_eda",         "孤立客户端 + EDA 中心性",     "isolated", "第5阶段新跑"),
    ("p5_iso_edaclu",      "孤立客户端 + 簇感知（本文）",  "isolated", "第5阶段新跑"),
]
# 第 3 阶段的 IID 对照（只有 task1）—— 不进主表，仅供附录
EXTRA_TAGS = ["p3_moe_iid"]


def find_eval_log_multi(roots, tag: str, domain: str):
    """在多个根里找评估日志（第 3 阶段在根目录，第 5 阶段在 phase5/ 下）。"""
    for r in roots:
        p = find_eval_log(r, tag, domain)
        if p:
            return p
    return None


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


def collect_stage(out_root, tag: str):
    """收集一个配置的全部结果（out_root 可以是单个根或根列表）。"""
    roots = [out_root] if isinstance(out_root, str) else list(out_root)
    st = {"tag": tag, "domains": {}, "eval_logs": {}}
    for d in DOMAINS:
        p = find_eval_log_multi(roots, tag, d)
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
        # 🔴 门槛：参数量不足 ⇒ MoE 未生效 ⇒ 该 domain 作废（显式报错，不静默）
        pm = (st["domains"][d] or {}).get("params_M")
        if pm is not None and pm < PARAMS_MIN_M:
            print("[FATAL] %s/%s 参数量仅 %.2fM < %.1fM ⇒ MoE 未生效，"
                  "评的是残缺模型，本项作废" % (tag, d, pm, PARAMS_MIN_M))
            st["domains"][d]["invalid_reason"] = (
                "params_M=%.2f < %.1f ⇒ MoE 未生效（残缺模型），结果作废"
                % (pm, PARAMS_MIN_M))
            st["domains"][d]["stats"] = None
            st["domains"][d]["seeds"] = []
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
    lines.append("| 配置 | 划分 | 来源 | " + " | ".join(
        ["reach-v2", "push-v2", "button-press"]))
    lines.append("|---|---|---|---|---|---|")
    for st in result["stages"]:
        cells = []
        for d in DOMAINS:
            e = st["domains"].get(d)
            if not e or not e.get("stats") or e["stats"].get("n", 0) < 1:
                cells.append("—")
            else:
                s = e["stats"]
                cells.append(f"{s['mean']:.3f} ± {s['ci']:.3f}")
        lines.append(f"| {st['label']} | {st['split']} | {st.get('source','')} | "
                     + " | ".join(cells) + " |")
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

    # 🔴 两个根：第 5 阶段（phase5/）与第 3 阶段（根目录）
    eval_roots = [out_root, os.path.dirname(out_root.rstrip("/"))]
    cfg_roots = [out_root, os.path.dirname(out_root.rstrip("/"))]
    for tag, label, split, source in STAGES:
        st = collect_stage(eval_roots, tag)
        st["label"] = label
        st["split"] = split
        st["source"] = source
        conv = {}
        for r in cfg_roots:
            conv = collect_convergence(r, tag)
            if conv:
                break
        st["convergence"] = conv
        result["stages"].append(st)
        got = [d for d in DOMAINS
               if st["domains"].get(d) and st["domains"][d].get("n_tables", 0) > 0]
        print(f"[{tag:<20}] 有结果的任务: {len(got)}/{len(DOMAINS)}  {got}")

    # 复用第 3 阶段的集中式上界（不重跑）
    result["centralized_upper_bound"] = collect_baseline(
        os.path.dirname(out_root.rstrip("/")), "g2b_central_eq")

    eda = {}
    for r in cfg_roots:
        eda = collect_eda(r, ["p3_moe_eda", "p3_moe_eda_cluster",
                               "p5_iso_eda", "p5_iso_edaclu"])
        if eda:
            break
    result["eda"] = eda

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
