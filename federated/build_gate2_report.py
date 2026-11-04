#!/usr/bin/env python3
"""FedVLA 门2 结果表 + 论文图表生成（全部基于云端实测数据）。

数据来源（AutoDL 实例，2026-11-04 采集）：
  - 评估成功率：3 seeds × 20 rollouts/seed，均值 ± 95% CI (t 分布)
  - 收敛曲线：output/fedavg_{hetero,iid}/log_client_*_r*.txt
  - 通信开销：comm_cost.py / comm_analysis.py 实测参数量 + 真实张量量化模拟

用法：
    python build_gate2_report.py [--outdir <目录>]
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 字体
CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC",
]


def setup_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] 使用中文字体：{name}")
            return name
    for path in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] 注册系统字体：{path} -> {prop.get_name()}")
            return prop.get_name()
    print("[font] 未找到中文字体 -> 图表标签改用英文")
    return None


# ---------------------------------------------------------------- 统计
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean_ci(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, m, m
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    half = T_CRIT_95.get(n, 1.96) * sd / math.sqrt(n)
    return m, half, m - half, m + half


# ---------------------------------------------------------------- 数据
TASKS = {
    "reach-v2": "reach-v2 (推取至目标点)",
    "push-v2": "push-v2 (推动方块至目标点)",
    "button-press-topdown-v2": "button-press-topdown-v2 (按压按钮)",
}

# (键, 中文标签, 任务键, 逐 seed 成功率, 是否为有效对照)
EVAL_ROWS = [
    ("central_t1",   "集中式 (6000 步)",            "reach-v2", [0.90, 0.85, 0.95], True),
    ("central_t2",   "集中式 (6000 步)",            "push-v2",   [1.00, 0.95, 1.00], True),
    ("central_t3",   "集中式 (6000 步)",            "button-press-topdown-v2", [0.95, 0.95, 1.00], True),
    ("equal_t1",     "等预算集中式 (2250 步)",       "reach-v2", [1.00, 0.95, 0.90], True),
    ("iid_t1",       "IID 联邦 (50 轮, 全 task1)",   "reach-v2", [0.70, 0.70, 0.50], True),
    ("iid_t2",       "IID 联邦 (50 轮, 全 task1)",   "push-v2",   [0.00, 0.00, 0.00], False),
    ("iid_t3",       "IID 联邦 (50 轮, 全 task1)",   "button-press-topdown-v2", [0.00, 0.00, 0.00], False),
    ("hetero_t1",    "异构联邦 (50 轮)",             "reach-v2", [0.05, 0.00, 0.00], True),
    ("hetero_t2",    "异构联邦 (50 轮)",             "push-v2",   [0.00, 0.00, 0.00], True),
    ("hetero_t3",    "异构联邦 (50 轮)",             "button-press-topdown-v2", [0.30, 0.15, 0.10], True),
    ("client0_t1",   "异构 client_0 本地模型 (诊断)", "reach-v2", [0.35, 0.35, 0.25], True),
]

HETERO_CURVE = [
    # round, mean, c0, c1, c2, c3
    (0, 0.3828, 0.347, 0.519, 0.317, 0.347), (1, 0.3754, 0.323, 0.524, 0.331, 0.323),
    (2, 0.3187, 0.262, 0.461, 0.290, 0.262), (3, 0.2893, 0.238, 0.416, 0.264, 0.238),
    (4, 0.2713, 0.222, 0.394, 0.247, 0.222), (5, 0.2584, 0.210, 0.378, 0.236, 0.210),
    (6, 0.2488, 0.200, 0.366, 0.229, 0.200), (7, 0.2415, 0.192, 0.358, 0.224, 0.192),
    (8, 0.2358, 0.186, 0.351, 0.220, 0.186), (9, 0.2311, 0.181, 0.345, 0.217, 0.181),
    (10, 0.2271, 0.177, 0.340, 0.214, 0.177), (11, 0.2236, 0.174, 0.335, 0.211, 0.174),
    (12, 0.2205, 0.171, 0.332, 0.209, 0.171), (13, 0.2177, 0.168, 0.328, 0.207, 0.168),
    (14, 0.2151, 0.165, 0.325, 0.205, 0.165), (15, 0.2125, 0.162, 0.322, 0.203, 0.162),
    (16, 0.2101, 0.160, 0.320, 0.202, 0.160), (17, 0.2078, 0.157, 0.317, 0.200, 0.157),
    (18, 0.2057, 0.155, 0.314, 0.199, 0.155), (19, 0.2036, 0.153, 0.311, 0.198, 0.153),
    (20, 0.2016, 0.151, 0.308, 0.197, 0.151), (21, 0.1996, 0.149, 0.305, 0.196, 0.149),
    (22, 0.1976, 0.147, 0.303, 0.195, 0.147), (23, 0.1958, 0.145, 0.300, 0.194, 0.145),
    (24, 0.1941, 0.143, 0.298, 0.193, 0.143), (25, 0.1926, 0.142, 0.295, 0.192, 0.142),
    (26, 0.1913, 0.140, 0.293, 0.191, 0.140), (27, 0.1900, 0.139, 0.291, 0.191, 0.139),
    (28, 0.1889, 0.139, 0.288, 0.190, 0.139), (29, 0.1880, 0.138, 0.286, 0.189, 0.138),
    (30, 0.1871, 0.138, 0.284, 0.189, 0.138), (31, 0.1862, 0.137, 0.282, 0.188, 0.137),
    (32, 0.1853, 0.137, 0.281, 0.187, 0.137), (33, 0.1845, 0.136, 0.279, 0.186, 0.136),
    (34, 0.1837, 0.136, 0.277, 0.186, 0.136), (35, 0.1831, 0.136, 0.276, 0.185, 0.136),
    (36, 0.1826, 0.136, 0.274, 0.184, 0.136), (37, 0.1825, 0.137, 0.273, 0.184, 0.137),
    (38, 0.1826, 0.138, 0.272, 0.183, 0.138), (39, 0.1819, 0.137, 0.271, 0.182, 0.137),
    (40, 0.1811, 0.137, 0.269, 0.182, 0.137), (41, 0.1802, 0.136, 0.268, 0.181, 0.136),
    (42, 0.1794, 0.135, 0.267, 0.180, 0.135), (43, 0.1786, 0.134, 0.266, 0.180, 0.134),
    (44, 0.1777, 0.133, 0.265, 0.179, 0.133), (45, 0.1768, 0.133, 0.263, 0.179, 0.133),
    (46, 0.1758, 0.131, 0.262, 0.178, 0.131), (47, 0.1748, 0.130, 0.261, 0.178, 0.130),
    (48, 0.1737, 0.129, 0.260, 0.177, 0.129), (49, 0.1722, 0.127, 0.259, 0.177, 0.127),
]

IID_CURVE = [
    (0, 0.3474), (1, 0.3245), (2, 0.2495), (3, 0.2173), (4, 0.1982), (5, 0.1843),
    (6, 0.1746), (7, 0.1673), (8, 0.1616), (9, 0.1566), (10, 0.1522), (11, 0.1483),
    (12, 0.1448), (13, 0.1417), (14, 0.1390), (15, 0.1367), (16, 0.1349), (17, 0.1335),
    (18, 0.1318), (19, 0.1296), (20, 0.1275), (21, 0.1256), (22, 0.1238), (23, 0.1220),
    (24, 0.1203), (25, 0.1187), (26, 0.1172), (27, 0.1157), (28, 0.1143), (29, 0.1129),
    (30, 0.1117), (31, 0.1105), (32, 0.1094), (33, 0.1087), (34, 0.1073), (35, 0.1085),
    (36, 0.1072), (37, 0.1098), (38, 0.1085), (39, 0.1076), (40, 0.1069), (41, 0.1065),
    (42, 0.1060), (43, 0.1058), (44, 0.1052), (45, 0.1039), (46, 0.1036), (47, 0.1050),
    (48, 0.1065), (49, 0.1062),
]

# 客户端 -> 任务（异构划分：c0/c3 = reach-v2, c1 = push-v2, c2 = button-press-topdown-v2）
CLIENT_TASK = {0: "reach-v2", 1: "push-v2", 2: "button-press-topdown-v2", 3: "reach-v2"}

# 通信开销（云端 comm_cost.py / comm_analysis.py 实测）
COMM = {
    "trunk_params": 12_644_352,
    "personal_params": 3_892_838,
    "full_params": 16_537_190,
    "uplink_mib": 192.94,        # 4 客户端上行合计 / 轮
    "downlink_mib": 48.23,       # 服务器广播 / 轮
    "per_round_mib": 241.17,
    "quant": [
        # 方案, bit/param, 50轮通信量(GB, 十进制), 相对压缩%, 相对误差
        ("fp32 (不压缩)",       32, 12.64, 100.0, 0.0000),
        ("int8 per-tensor",     8,  3.16,  25.0, 0.0073),
        ("int8 per-group(64)",  8,  3.16,  25.0, 0.0050),
        ("int4 per-group(64)",  4,  1.58,  12.5, 0.0895),
        ("int2 per-group(64)",  2,  0.79,   6.2, 0.5842),
        ("1-bit 符号化",         1,  0.40,   3.1, 0.5571),
    ],
    "budget": [
        # 预算GB, fp32轮, int8轮, int4轮, 1bit轮
        (1, 4, 16, 32, 127),
        (5, 20, 79, 158, 633),
        (10, 40, 158, 316, 1265),
        (50, 198, 791, 1582, 6327),
        (100, 395, 1582, 3163, 12654),
    ],
}

C_PRIMARY = "#2F6FB5"
C_HETERO = "#C0504D"
C_IID = "#4E9F50"
C_EQ = "#7A6FA8"
C_DIAG = "#E08A1E"


# ---------------------------------------------------------------- 表格
def build_table():
    rows = []
    for key, label, task, seeds, valid in EVAL_ROWS:
        m, half, lo, hi = mean_ci(seeds)
        rows.append({
            "key": key, "label": label, "task": task,
            "n_seeds": len(seeds), "per_seed": seeds,
            "mean": m, "ci95_half": half, "ci95_lo": lo, "ci95_hi": hi,
            "valid_control": valid,
        })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["方法", "任务", "seed数", "逐seed成功率", "均值", "95%CI半宽",
                    "95%CI下限", "95%CI上限", "是否有效对照"])
        for r in rows:
            w.writerow([
                r["label"], TASKS[r["task"]], r["n_seeds"],
                " ".join(f"{v:.2f}" for v in r["per_seed"]),
                f"{r['mean']:.3f}", f"{r['ci95_half']:.3f}",
                f"{r['ci95_lo']:.3f}", f"{r['ci95_hi']:.3f}",
                "是" if r["valid_control"] else "否(该任务未在IID对照组中训练)",
            ])
    print(f"[ok] CSV -> {path}")


def write_md(rows, path):
    def cell(r):
        return f"{r['mean']:.3f} ± {r['ci95_half']:.3f}"

    def get(key):
        return next(r for r in rows if r["key"] == key)

    lines = []
    lines.append("# FedVLA 门2 结果表\n")
    lines.append("统计口径：**3 seeds × 20 rollouts/seed**，报均值 ± 95% 置信区间（t 分布，n=3 → t₀.₉₇₅=3.182）。")
    lines.append("评估环境：Meta-World (MuJoCo)，主干 HPT-Base 12.6M（冻结 stem，只训 trunk+head）。\n")
    lines.append("## 1. 主对比表（task1 = reach-v2）\n")
    lines.append("| 方法 | 任务 | 逐 seed | 均值 ± 95% CI |")
    lines.append("|---|---|---|---|")
    for key in ("central_t1", "equal_t1", "iid_t1", "hetero_t1", "client0_t1"):
        r = get(key)
        ps = " / ".join(f"{v:.2f}" for v in r["per_seed"])
        lines.append(f"| {r['label']} | {TASKS[r['task']]} | {ps} | **{cell(r)}** |")
    lines.append("")
    lines.append("## 2. 三任务完整结果\n")
    lines.append("| 任务 | 集中式(6000步) | 等预算集中式(2250步) | IID联邦(50轮) | 异构联邦(50轮) |")
    lines.append("|---|---|---|---|---|")
    for task in ("reach-v2", "push-v2", "button-press-topdown-v2"):
        cen = get({"reach-v2": "central_t1", "push-v2": "central_t2",
                   "button-press-topdown-v2": "central_t3"}[task])
        eq = get("equal_t1") if task == "reach-v2" else None
        iid = get({"reach-v2": "iid_t1", "push-v2": "iid_t2",
                   "button-press-topdown-v2": "iid_t3"}[task])
        het = get({"reach-v2": "hetero_t1", "push-v2": "hetero_t2",
                   "button-press-topdown-v2": "hetero_t3"}[task])
        eqs = cell(eq) if eq else "—"
        mark = "" if iid["valid_control"] else " ᵃ"
        lines.append(
            f"| {TASKS[task]} | {cell(cen)} | {eqs} | {cell(iid)}{mark} | **{cell(het)}** |")
    lines.append("")
    lines.append("ᵃ IID 对照组**只训练了 task1**，模型中不存在 push-v2 / button-press-topdown-v2 的 "
                 "stem 与 head，因此这两格是「未训练任务 = 0」的平凡结果，**不是「IID 更差」**。"
                 "IID 仅在 task1 上与异构联邦构成有效对照。\n")
    lines.append("## 3. 性能差距分解（task1 = reach-v2）\n")
    lines.append("| 对比 | 含义 | 差距 |")
    lines.append("|---|---|---|")
    cen, eq, iid, het, cli = (get(k) for k in
                              ("central_t1", "equal_t1", "iid_t1", "hetero_t1", "client0_t1"))
    lines.append(f"| 集中式(6000步) {cen['mean']:.3f} → 等预算集中式(2250步) {eq['mean']:.3f} "
                 f"| 排除「训练量不足」这一替代解释 | {eq['mean']-cen['mean']:+.3f} |")
    lines.append(f"| 等预算集中式 {eq['mean']:.3f} → IID联邦 {iid['mean']:.3f} "
                 f"| 联邦管线自身的损耗（每轮重启 warmup+cosine 等） | {iid['mean']-eq['mean']:+.3f} |")
    lines.append(f"| IID联邦 {iid['mean']:.3f} → 异构联邦 {het['mean']:.3f} "
                 f"| **异构专有的损害（门2 的判据）** | **{het['mean']-iid['mean']:+.3f}** |")
    lines.append(f"| IID联邦 {iid['mean']:.3f} → 异构 client_0 本地 {cli['mean']:.3f} "
                 f"| 异构在**训练过程中**就污染了 trunk，不只是最后平均那一下 | {cli['mean']-iid['mean']:+.3f} |")
    lines.append("")
    lines.append("## 4. 通信开销（实测）\n")
    lines.append(f"- 共享 trunk：**{COMM['trunk_params']:,}** 参数（{COMM['trunk_params']/1e6:.3f} M）")
    lines.append(f"- 个性化参数（stem/head/normalizer）：{COMM['personal_params']:,}")
    lines.append(f"- 完整模型：{COMM['full_params']:,}")
    lines.append(f"- 单轮通信量（4 客户端，fp32）：上行 {COMM['uplink_mib']:.2f} MiB + "
                 f"下行广播 {COMM['downlink_mib']:.2f} MiB = **{COMM['per_round_mib']:.2f} MiB/轮**\n")
    lines.append("| 量化方案 | bit/参数 | 50 轮通信量 | 相对压缩 | 相对量化误差 |")
    lines.append("|---|---|---|---|---|")
    for name, bits, gb, comp, err in COMM["quant"]:
        lines.append(f"| {name} | {bits} | {gb:.2f} GB | {comp:.1f}% | {err:.4f} |")
    lines.append("")
    lines.append("给定通信预算可训练的轮数：\n")
    lines.append("| 预算 | fp32 | int8 | int4 | 1-bit |")
    lines.append("|---|---|---|---|---|")
    for b, a, c_, d, e in COMM["budget"]:
        lines.append(f"| {b} GB | {a} | {c_} | {d} | {e} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ok] Markdown -> {path}")


# ---------------------------------------------------------------- 图
def fig_convergence(path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    hr = [c[0] for c in HETERO_CURVE]
    hm = [c[1] for c in HETERO_CURVE]
    ir = [c[0] for c in IID_CURVE]
    im = [c[1] for c in IID_CURVE]
    ax.plot(ir, im, "o-", ms=3.2, lw=1.8, color=C_IID, label="IID 联邦（4 客户端全 task1）")
    ax.plot(hr, hm, "s-", ms=3.2, lw=1.8, color=C_HETERO, label="异构联邦（task1×2 / task2 / task3）")
    ax.set_xlabel("通信轮次 (round)")
    ax.set_ylabel("联邦平均训练 loss")
    ax.set_title("图5-1  联邦训练收敛曲线（4 客户端平均）")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=9, frameon=False)
    ax.annotate(f"IID 末轮 {im[-1]:.3f}\n异构末轮 {hm[-1]:.3f}",
                xy=(49, hm[-1]), xytext=(30, 0.22),
                fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.9))

    ax = axes[1]
    cols = {0: C_PRIMARY, 1: C_HETERO, 2: "#E08A1E", 3: "#7FA8D4"}
    styles = {0: "-", 1: "-", 2: "--", 3: ":"}
    for ci in range(4):
        ys = [c[2 + ci] for c in HETERO_CURVE]
        ax.plot(hr, ys, styles[ci], lw=1.7 if ci != 3 else 2.6, color=cols[ci],
                label=f"client_{ci}（{CLIENT_TASK[ci]}）")
    ax.set_xlabel("通信轮次 (round)")
    ax.set_ylabel("本地训练 loss")
    ax.set_title("图5-2  异构联邦各客户端的本地 loss")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=9, frameon=False)
    ax.text(0.02, 0.06, "c0 与 c3 同为 reach-v2，曲线完全重合；\nc1(push-v2) 始终最高",
            transform=ax.transAxes, fontsize=8.5, color="#444")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[ok] 图 -> {path}")


def fig_comparison(rows, path):
    def get(k):
        return next(r for r in rows if r["key"] == k)

    items = [
        ("集中式\n(6000步)", get("central_t1"), C_PRIMARY),
        ("等预算集中式\n(2250步)", get("equal_t1"), C_EQ),
        ("IID联邦\n(50轮)", get("iid_t1"), C_IID),
        ("异构联邦\n(50轮)", get("hetero_t1"), C_HETERO),
        ("异构client_0\n本地模型", get("client0_t1"), C_DIAG),
    ]
    labels = [i[0] for i in items]
    means = [i[1]["mean"] for i in items]
    errs = [i[1]["ci95_half"] for i in items]
    cols = [i[2] for i in items]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    bars = ax.bar(labels, means, yerr=errs, capsize=4, color=cols,
                  edgecolor="white", linewidth=0.8,
                  error_kw={"elinewidth": 1.2, "ecolor": "#333"})
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.045, f"{m:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("任务成功率 (3 seeds × 20 rollouts)")
    ax.set_xlabel("训练范式")
    ax.set_title("图5-3  reach-v2 成功率对比（误差棒 = 95% CI）")
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_axisbelow(True)

    ax = axes[1]
    tasks = ["reach-v2", "push-v2", "button-press-topdown-v2"]
    cn = ["central_t1", "central_t2", "central_t3"]
    hd = ["hetero_t1", "hetero_t2", "hetero_t3"]
    x = range(len(tasks))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], [get(k)["mean"] for k in cn], w,
                label="集中式 (6000步)", color=C_PRIMARY, edgecolor="white")
    b2 = ax.bar([i + w / 2 for i in x], [get(k)["mean"] for k in hd], w,
                label="异构联邦 (50轮)", color=C_HETERO, edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["reach-v2", "push-v2", "button-press-\ntopdown-v2"], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("任务成功率")
    ax.set_xlabel("Meta-World 任务")
    ax.set_title("图5-4  三任务：集中式 vs 异构联邦")
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False)
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                    f"{b.get_height():.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[ok] 图 -> {path}")


def fig_comm(path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    names = [q[0] for q in COMM["quant"]]
    gbs = [q[2] for q in COMM["quant"]]
    errs = [q[4] for q in COMM["quant"]]
    ax = axes[0]
    bars = ax.bar(names, gbs, color="#3D6E9E", edgecolor="white", width=0.62)
    for b, g, e in zip(bars, gbs, errs):
        ax.text(b.get_x() + b.get_width() / 2, g + 0.85, f"{g:.2f} GB",
                ha="center", fontsize=9.5, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, g + 0.28, f"量化误差 {e:.4f}",
                ha="center", fontsize=8.5, color="#555")
    ax.set_ylim(0, 15.6)
    ax.set_ylabel("50 轮累计通信量 (GB)")
    ax.set_xlabel("量化方案")
    ax.set_title("图5-5  量化比特数 vs 通信量（下方为相对量化误差）")
    ax.tick_params(axis="x", labelsize=8.5, rotation=12)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_axisbelow(True)

    ax = axes[1]
    budgets = [b[0] for b in COMM["budget"]]
    for idx, (lbl, col) in enumerate([("fp32", C_PRIMARY), ("int8", C_IID),
                                      ("int4", C_DIAG), ("1-bit", C_HETERO)], start=1):
        ax.plot(budgets, [b[idx] for b in COMM["budget"]], "o-", lw=1.8,
                ms=4.5, color=col, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("通信预算 (GB, 对数轴)")
    ax.set_ylabel("可训练轮数 (对数轴)")
    ax.set_title("图5-6  给定通信预算下可训练轮数")
    ax.grid(alpha=0.3, ls="--", which="both")
    ax.legend(fontsize=9, frameon=False)
    ax.axhline(50, color="#888", ls=":", lw=1.2)
    ax.text(1.15, 54, "本次实验轮数 = 50", fontsize=8.5, color="#666")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[ok] 图 -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    setup_font()
    rows = build_table()
    write_csv(rows, os.path.join(outdir, "gate2_table.csv"))
    write_md(rows, os.path.join(outdir, "gate2_table.md"))
    fig_convergence(os.path.join(figdir, "fig5_convergence.png"))
    fig_comparison(rows, os.path.join(figdir, "fig5_comparison.png"))
    fig_comm(os.path.join(figdir, "fig5_comm_cost.png"))
    print("REPORT_DONE")


if __name__ == "__main__":
    main()
