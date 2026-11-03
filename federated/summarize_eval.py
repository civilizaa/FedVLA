#!/usr/bin/env python3
"""从 HPT 评估日志里提取成功率，汇总成「均值 ± 95% 置信区间」。

用法：
    python summarize_eval.py <日志目录或文件> [...]
每个参数可以是：
    - 单个日志文件（含有多个 seed 的表格）
    - 目录（递归找 *.log 与 hydra 输出目录里的 run_eval.log）

HPT 的评估输出格式（utils.tabulate_print_state）：
    | task     |   reward |
    |----------+----------|
    | reach-v2 |     1.00 |
    | avg      |     1.00 |
每个 seed 产生一张表 -> 取每张表的 avg 行作为该 seed 的成功率。

统计口径（论文底线）：3 seeds × 20 rollouts，报均值 ± 95% CI
    CI = t 分布的半宽（seed 数少时 t 分布比正态更保守）
"""
import math
import os
import re
import sys

# 匹配 "| avg      |     0.95 |" 这类行
AVG_RE = re.compile(r"^\|\s*avg\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", re.M | re.I)
# 匹配 "| reach-v2 |     1.00 |" 这类行（任务名 -> 成功率）
TASK_RE = re.compile(r"^\|\s*([A-Za-z0-9_\-]+-v\d)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", re.M)

# t 分布 95% 双侧临界值（seed 数 -> t），常用几个
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def collect_logs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                for n in names:
                    if n.endswith(".log") or n.endswith(".out"):
                        files.append(os.path.join(root, n))
        elif os.path.isfile(p):
            files.append(p)
    return files


def parse_file(path):
    """返回 (每张表的 avg 列表, {task: [success...]})"""
    try:
        txt = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception:
        return [], {}
    avgs = [float(x) for x in AVG_RE.findall(txt)]
    per_task = {}
    for task, val in TASK_RE.findall(txt):
        per_task.setdefault(task, []).append(float(val))
    return avgs, per_task


def mean_ci(vals):
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return m, 0.0, m, m
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    t = T_CRIT_95.get(n, 1.96)
    half = t * sd / math.sqrt(n)
    return m, half, m - half, m + half


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    files = collect_logs(sys.argv[1:])
    all_avg = []
    all_task = {}
    per_file = []
    for f in files:
        avgs, tasks = parse_file(f)
        if not avgs and not tasks:
            continue
        all_avg.extend(avgs)
        for t, vs in tasks.items():
            all_task.setdefault(t, []).extend(vs)
        per_file.append((f, avgs))

    if not all_avg and not all_task:
        print("[!!] 没解析到任何评估结果。检查日志里是否有 '| avg |' 表格。")
        print("     扫过的文件数：", len(files))
        return 2

    print(f"解析到 {len(files)} 个日志文件，{len(all_avg)} 张结果表")
    print()
    print("===== 逐 seed =====")
    for f, avgs in per_file:
        if avgs:
            print(f"  {os.path.basename(os.path.dirname(f)) or f}: {avgs}")
    print()
    print("===== 逐任务汇总（均值 ± 95% CI）=====")
    for t, vs in sorted(all_task.items()):
        r = mean_ci(vs)
        if r:
            m, half, lo, hi = r
            print(f"  {t:<32} n={len(vs)}  {m:.3f} ± {half:.3f}   [{lo:.3f}, {hi:.3f}]")
    if all_avg:
        r = mean_ci(all_avg)
        if r:
            m, half, lo, hi = r
            print(f"  {'OVERALL(avg 行)':<32} n={len(all_avg)}  {m:.3f} ± {half:.3f}   [{lo:.3f}, {hi:.3f}]")
    print()
    print("SUMMARY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
