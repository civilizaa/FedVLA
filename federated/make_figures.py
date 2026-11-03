#!/usr/bin/env python3
"""FedVLA：论文图表生成（从评估日志直接出图）。

生成三张图：
  fig5_convergence.png    —— 联邦训练轮数 vs 各任务成功率（需要每轮评估日志）
  fig6_comparison.png     —— 集中式 vs 联邦 vs 等预算 的分组柱状图（带 95% CI 误差棒）
  fig7_comm_cost.png      —— 通信量 vs 量化比特数的权衡曲线

用法：
    python make_figures.py --compare /root/HPT/output/compare_results.txt
    python make_figures.py --all
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_cjk_font():
    """设置中文字体。找不到时回退到英文标签，绝不让论文图表出现方块乱码。"""
    cjk_candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
        "PingFang SC", "Heiti SC", "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in cjk_candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] 使用中文字体：{name}")
            return name
    # Linux 常见路径（云端 matplotlib 找不到系统字体时直接注册）
    for path in (
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] 注册系统字体：{path} -> {prop.get_name()}")
            return prop.get_name()
    print("[warn] 找不到中文字体，图中的中文可能显示为方块")
    return None


HAS_CJK = setup_cjk_font() is not None

FIG_DIR = "/root/HPT/output/figures"

# 论文配色（浅色主题，打印友好）
C_CENTRAL = "#2563eb"   # 集中式 —— 蓝
C_FED = "#dc2626"        # 联邦 —— 红
C_EQUAL = "#f59e0b"     # 等预算 —— 橙
C_GRID = "#d4d4d8"


def L(zh, en):
    """有中文字体用中文，否则退回英文。"""
    return zh if HAS_CJK else en


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def parse_compare(path):
    """从 compare_results.txt 解析出 {task: (central, ci_c, fed, ci_f)}"""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            # 形如:  reach-v2     0.983 ± 0.053     0.900 ± 0.060
            m = re.match(
                r"^(\S+-v\d)\s+([\d.]+)\s*±\s*([\d.]+)\s+([\d.]+)\s*±\s*([\d.]+)", line.strip()
            )
            if m:
                task, cm, cc, fm, fc = m.groups()
                out[task] = (float(cm), float(cc), float(fm), float(fc))
    return out


def fig_comparison(compare, out_path):
    if not compare:
        print("[skip] 没有可用的对比数据")
        return False
    tasks = list(compare.keys())
    x = range(len(tasks))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    central = [compare[t][0] for t in tasks]
    central_err = [compare[t][1] for t in tasks]
    fed = [compare[t][2] for t in tasks]
    fed_err = [compare[t][3] for t in tasks]

    b1 = ax.bar([i - w/2 for i in x], central, w, yerr=central_err, capsize=4,
                label=L("集中式 (6000 步)", "Centralized (6000 steps)"), color=C_CENTRAL, edgecolor="white")
    b2 = ax.bar([i + w/2 for i in x], fed, w, yerr=fed_err, capsize=4,
                label=L("联邦 FedAvg (50 轮)", "Federated FedAvg (50 rounds)"), color=C_FED, edgecolor="white")

    # 标签放在"柱顶 + 误差棒 + 余量"处，避免被误差棒线穿过
    for bars, errs in ((b1, central_err), (b2, fed_err)):
        for bar, e in zip(bars, errs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + e + 0.03, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(L("成功率 (3 seeds × 20 rollouts)", "Success rate (3 seeds x 20 rollouts)"))
    # ylim 要留出图例 + 数值标签的空间，否则会与柱顶标签重叠
    ax.set_ylim(0, 1.32)
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("-v2", "") for t in tasks])
    ax.set_title(L("集中式 vs 联邦 FedAvg：Meta-World 任务成功率",
                   "Centralized vs Federated FedAvg: Meta-World success rate"))
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
              fontsize=9)
    ax.grid(axis="y", color=C_GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")
    return True


def fig_comm_cost(out_path, trunk_params=12_644_352, rounds=50, clients=4):
    """通信量 vs 量化比特数"""
    bpp_list = [32, 16, 8, 4, 2, 1]
    up = [clients * trunk_params * b / 8 for b in bpp_list]
    down = [trunk_params * b / 8 for b in bpp_list]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(bpp_list, [u/1e9 for u in up], "o-", color=C_FED, label=L("上行（4 客户端合计）", "Uplink (4 clients total)"))
    ax.plot(bpp_list, [d/1e9 for d in down], "s--", color=C_CENTRAL, label=L("下行（服务器广播）", "Downlink (server broadcast)"))
    ax.plot(bpp_list, [(u+d)/1e9 for u, d in zip(up, down)], "^-", color="#7c3aed",
            label=L("单轮合计", "Total per round)"))

    for b, tot in zip(bpp_list, [(u+d) for u, d in zip(up, down)]):
        ax.annotate(f"{tot*rounds/1e9:.2f} GB", (b, tot/1e9),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(bpp_list)
    ax.set_xticklabels([f"{b}bit" for b in bpp_list])
    ax.set_xlabel(L("量化精度（比特/参数）", "Quantization (bits/param)"))
    ax.set_ylabel(L("通信量 (GB)", "Communication (GB)"))
    ax.set_title(L(f"通信开销与量化精度的权衡（{rounds} 轮 × {clients} 客户端）",
                     f"Communication overhead vs quantization ({rounds} rounds x {clients} clients)"))
    ax.legend(frameon=False)
    ax.grid(color=C_GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")
    return True


def fig_trunk_share(out_path, trunk=12_644_352, personal=3_892_838):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    vals = [trunk, personal]
    labels = [L("共享 trunk\n(联邦聚合/传输)", "Shared trunk\n(federated, transmitted)"),
              L("个性化 stem/head\n(不上传)", "Personalized stem/head\n(not transmitted)")]
    colors = [C_FED, "#94a3b8"]
    bars = ax.bar(labels, [v/1e6 for v in vals], color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v/1e6 + 0.2,
                f"{v/1e6:.2f} M\n({v/(trunk+personal)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(L("参数量 (M)", "Parameters (M)"))
    ax.set_ylim(0, max(trunk, personal)/1e6 * 1.32)
    ax.set_title(L("联邦只传输共享 trunk，省下个性化参数",
                   "Federation transmits only the shared trunk"))
    ax.grid(axis="y", color=C_GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")
    return True


def fig_convergence(json_path, out_path):
    """联邦收敛曲线：各客户端 + 联邦平均"""
    if not os.path.exists(json_path):
        print(f"[skip] 找不到 {json_path}")
        return False
    import json
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    curve = data.get("curve") or []
    if not curve:
        print("[skip] 曲线为空")
        return False

    rounds = [c["round"] for c in curve]
    mean = [c["mean_train_loss"] for c in curve]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    # 联邦平均（粗线）
    ax.plot(rounds, mean, color="#111827", linewidth=2.2, marker="o", markersize=3.5,
            label=L("联邦平均", "Federated average"), zorder=5)

    # 各客户端（细线）
    colors = ["#2563eb", "#dc2626", "#16a34a", "#a855f7"]
    labels = {
        0: L("c0: reach-v2", "c0: reach-v2"),
        1: L("c1: push-v2", "c1: push-v2"),
        2: L("c2: button-press", "c2: button-press"),
        3: L("c3: reach-v2", "c3: reach-v2"),
    }
    all_keys = sorted({k for c in curve for k in c.get("per_client_train_loss", {})})
    for i, k in enumerate(all_keys):
        xs, ys = [], []
        for c in curve:
            v = c.get("per_client_train_loss", {}).get(k)
            if v is not None:
                xs.append(c["round"]); ys.append(v)
        if xs:
            ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=1.3, alpha=0.75,
                    marker="s", markersize=2.5, label=labels.get(int(k), f"c{k}"))

    ax.set_xlabel(L("联邦通信轮数", "Federated round"))
    ax.set_ylabel(L("训练损失", "Training loss"))
    ax.set_title(L("联邦训练收敛曲线（异构 4 客户端）",
                   "Federated convergence (4 heterogeneous clients)"))
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.grid(color=C_GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default="/root/HPT/output/compare_results.txt")
    ap.add_argument("--convergence", default="/root/HPT/output/convergence_hetero.json")
    ap.add_argument("--outdir", default=FIG_DIR)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    made = []

    if args.all:
        compare = parse_compare(args.compare)
        made.append(fig_comparison(compare, os.path.join(args.outdir, "fig6_comparison.png")))
        made.append(fig_comm_cost(os.path.join(args.outdir, "fig7_comm_cost.png")))
        made.append(fig_trunk_share(os.path.join(args.outdir, "fig5_param_share.png")))
        made.append(fig_convergence(args.convergence, os.path.join(args.outdir, "fig4_convergence.png")))
    else:
        compare = parse_compare(args.compare)
        made.append(fig_comparison(compare, os.path.join(args.outdir, "fig6_comparison.png")))

    if not any(made):
        print("[!!] 没有生成任何图。检查 --compare 文件是否有可解析的数据。")
        return 2
    print("FIGS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
