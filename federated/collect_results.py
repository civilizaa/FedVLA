#!/usr/bin/env python3
"""汇总第 3 阶段全部评估结果，输出对照表。

同时做三件事：
  1. 逐日志提取成功率（avg 行 + 任务行），报均值 ± 95% CI
  2. 抓每个日志的参数量（number of total params (M)）→ 判断 MoE 是否生效
  3. 抓每个变体最后一轮的客户端 loss → 判断训练是否收敛/崩坏

用法：python collect_results.py [输出目录，默认 /root/HPT/output]
"""
import math
import os
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/HPT/output"

AVG_RE = re.compile(r"^\|\s*avg\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", re.M | re.I)
TASK_RE = re.compile(r"^\|\s*([A-Za-z0-9_\-]+-v\d)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", re.M)
PARAM_RE = re.compile(r"number of total params \(M\):\s*([0-9.]+)")
TRUNK_RE = re.compile(r"trunk:\s*([0-9.]+)")

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def mean_ci(vals):
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return m, 0.0, m, m
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    half = T_CRIT_95.get(n - 1, 1.96) * sd / math.sqrt(n)   # 自由度 df = n - 1（2026-11-08 修正：原来错用 n）
    return m, half, m - half, m + half


# 变体 -> (显示名, 任务列表)
VARIANTS = [
    ("g2b_central_eq", "集中式(等预算)", ["mujoco_metaworld_task1"]),
    ("g2b_iid_e4_r49", "IID联邦50轮", ["mujoco_metaworld_task1"]),
    ("g2b_hetero_r49", "异构联邦50轮", ["mujoco_metaworld_task1",
                                       "mujoco_metaworld_task2",
                                       "mujoco_metaworld_task3"]),
    ("p3_moe_fedavg", "P3-A 异构+MoE+FedAvg", ["mujoco_metaworld_task1",
                                                "mujoco_metaworld_task2",
                                                "mujoco_metaworld_task3"]),
    ("p3_moe_eda", "P3-B 异构+MoE+EDA中心性", ["mujoco_metaworld_task1",
                                                "mujoco_metaworld_task2",
                                                "mujoco_metaworld_task3"]),
    ("p3_moe_eda_cluster", "P3-C 异构+MoE+EDA簇感知", ["mujoco_metaworld_task1",
                                                       "mujoco_metaworld_task2",
                                                       "mujoco_metaworld_task3"]),
    ("p3_moe_iid", "P3-D IID+MoE+FedAvg", ["mujoco_metaworld_task1"]),
]

TASK_SHORT = {"mujoco_metaworld_task1": "task1/reach",
               "mujoco_metaworld_task2": "task2/push",
               "mujoco_metaworld_task3": "task3/button"}


def read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def parse_eval(path):
    txt = read(path)
    avgs = [float(x) for x in AVG_RE.findall(txt)]
    tasks = {}
    for t, v in TASK_RE.findall(txt):
        tasks.setdefault(t, []).append(float(v))
    pm = PARAM_RE.search(txt)
    tm = TRUNK_RE.search(txt)
    return avgs, tasks, (float(pm.group(1)) if pm else None), \
           (float(tm.group(1)) if tm else None)


def last_loss(tag):
    """抓该变体最后一轮的客户端训练 loss。"""
    d = os.path.join(OUT, tag)
    if not os.path.isdir(d):
        return {}
    best = {}
    for name in os.listdir(d):
        if not name.startswith("client_"):
            continue
        logs = sorted(
            (os.path.join(d, name, f) for f in os.listdir(os.path.join(d, name))
             if f.endswith(".log") or f.endswith(".out")),
            key=lambda p: os.path.getmtime(p))
        if not logs:
            continue
        txt = read(logs[-1])
        losses = re.findall(r"loss[:= ]+\s*([0-9]+\.?[0-9]*(?:e-?\d+)?)", txt, re.I)
        if losses:
            best[name] = float(losses[-1])
    return best


def main():
    print("=" * 100)
    print("第 3 阶段评估结果汇总   （3 seeds × 20 rollouts，均值 ± 95% CI）")
    print("=" * 100)

    all_rows = []
    for tag, disp, domains in VARIANTS:
        print(f"\n### {disp}   [tag={tag}]")
        for dom in domains:
            path = os.path.join(OUT, f"eval_{tag}_{dom}.log")
            if not os.path.isfile(path):
                print(f"  {TASK_SHORT.get(dom, dom):<14} [无日志] {path}")
                all_rows.append((disp, TASK_SHORT.get(dom, dom), None, None, None, None))
                continue
            avgs, tasks, npm, trunk = parse_eval(path)
            n = len(avgs)
            if n == 0:
                print(f"  {TASK_SHORT.get(dom, dom):<14} [未完成] 表数=0")
                all_rows.append((disp, TASK_SHORT.get(dom, dom), None, npm, trunk, 0))
                continue
            r = mean_ci(avgs)
            m, half, lo, hi = r
            moe = "MoE✓" if (npm and npm > 18) else ("无MoE" if npm else "?")
            seeds = " ".join(f"{x:.2f}" for x in avgs)
            print(f"  {TASK_SHORT.get(dom, dom):<14} {m:.3f} ± {half:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]  seeds=[{seeds}]  n={n}  "
                  f"params={npm}M({moe})")
            all_rows.append((disp, TASK_SHORT.get(dom, dom), m, half, npm, n))

    # ---- 对照表 ----
    print("\n" + "=" * 100)
    print("对照表（task1 成功率）")
    print("=" * 100)
    print(f"{'变体':<26} {'task1':>16}  {'参数量':>10}  {'MoE':>6}")
    for disp, task, m, half, npm, n in all_rows:
        if task != "task1/reach":
            continue
        s = f"{m:.3f} ± {half:.3f}" if m is not None else "—"
        p = f"{npm:.2f}M" if npm else "—"
        flag = "✓" if (npm and npm > 18) else ("-" if npm else "?")
        print(f"{disp:<26} {s:>16}  {p:>10}  {flag:>6}")

    # ---- 训练末轮 loss ----
    print("\n" + "=" * 100)
    print("末轮客户端训练 loss（判断是否收敛）")
    print("=" * 100)
    for tag, disp, _ in VARIANTS:
        if not tag.startswith("p3_") and tag != "g2b_hetero_r49":
            continue
        ll = last_loss(tag)
        if not ll:
            print(f"  {disp:<26} [无客户端日志]")
            continue
        s = "  ".join(f"{k.replace('client_', 'c').split('_')[0]}={v:.4f}"
                      for k, v in sorted(ll.items()))
        print(f"  {disp:<26} {s}")

    # ---- 关键告警 ----
    print("\n" + "=" * 100)
    print("告警检查")
    print("=" * 100)
    bad = False
    for disp, task, m, half, npm, n in all_rows:
        if npm is not None and npm <= 18 and "MoE" in disp:
            print(f"  [!!] {disp} / {task} 参数量仅 {npm}M —— MoE 未生效，该项作废")
            bad = True
        if m is not None and m == 0.0:
            print(f"  [!!] {disp} / {task} 成功率 = 0.000 —— 需人工核查")
            bad = True
        if n is not None and 0 < n < 3:
            print(f"  [--] {disp} / {task} 只有 {n}/3 个 seed")
    if not bad:
        print("  无异常")
    print("\nCOLLECT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
