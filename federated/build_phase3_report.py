#!/usr/bin/env python3
"""FedVLA 第 3 阶段结果表 + 论文图表生成（**在本地跑**，读云端提取出的 JSON）。

输入：extract_phase3_results.py 在云端产出的 phase3_raw.json
输出：
  results/phase3_table.{md,csv}         主对比表
  results/figures/fig6_moe_effect.png   MoE/EDA 的贡献（分组柱状图）
  results/figures/fig6_eda_weights.png  EDA 权重热力图 + 簇划分

用法：
    python build_phase3_report.py --raw phase3_raw.json [--outdir <目录>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC",
]

DOMAIN_LABEL = {
    "mujoco_metaworld_task1": "reach-v2",
    "mujoco_metaworld_task2": "push-v2",
    "mujoco_metaworld_task3": "button-press",
}
DOMAIN_ORDER = list(DOMAIN_LABEL.keys())


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
            return prop.get_name()
    print("[font] 未找到中文字体 -> 图表标签改用英文")
    return None


# ---------------------------------------------------------------- 读数据
def load(raw_path):
    with open(raw_path, "r", encoding="utf-8") as f:
        return json.load(f)


def stage_map(data):
    return {s["tag"]: s for s in data["stages"]}


def fmt(entry):
    if not entry or entry.get("mean") is None:
        return "—"
    return f"{entry['mean']:.3f} ± {entry['ci']:.3f}"


# ---------------------------------------------------------------- 表格
def build_rows(data):
    """每个阶段一行，三个任务各一列 + 均值。"""
    rows = []
    for s in data["stages"]:
        r = {
            "tag": s["tag"],
            "label": s["label"],
            "hetero": s.get("hetero"),
            "n_seeds": 0,
        }
        dmean = {}          # {domain: mean} —— 用于 Δ 计算时按任务取交集
        means = []
        for d in DOMAIN_ORDER:
            e = s["domains"].get(d)
            r[d] = fmt(e)
            if e and e.get("mean") is not None:
                means.append(e["mean"])
                dmean[d] = e["mean"]
                r["n_seeds"] = max(r["n_seeds"], e.get("n", 0))
        r["avg_over_tasks"] = (f"{sum(means)/len(means):.3f}"
                               if means else "—")
        r["_means"] = means
        r["_dmean"] = dmean
        rows.append(r)
    return rows


def delta_vs(base_row, row):
    """Δ = 本阶段均值 − 基线均值，**只算两者都有结果的任务**。

    必须按任务取交集：若基线有 3 个任务而本阶段只有 2 个，
    直接拿各自的均值相减会把不同任务的集合混在一起（论文里这是硬伤）。
    """
    if not base_row or not row:
        return None, []
    common = [d for d in DOMAIN_ORDER
              if d in base_row["_dmean"] and d in row["_dmean"]]
    if not common:
        return None, []
    b = sum(base_row["_dmean"][d] for d in common) / len(common)
    v = sum(row["_dmean"][d] for d in common) / len(common)
    return v - b, common


def write_csv(rows, path):
    cols = ["label", "tag", "n_seeds"] + DOMAIN_ORDER + ["avg_over_tasks"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["阶段", "tag", "seeds"] +
                   [DOMAIN_LABEL[d] for d in DOMAIN_ORDER] + ["三任务均值"])
        for r in rows:
            w.writerow([r["label"], r["tag"], r["n_seeds"]] +
                       [r[d] for d in DOMAIN_ORDER] + [r["avg_over_tasks"]])


def write_md(rows, path, data):
    base = "g2b_hetero"
    bmap = {r["tag"]: r for r in rows}
    lines = []
    lines.append("# 第 3 阶段结果表（MoE + EDA）\n")
    lines.append(f"统计口径：3 seeds × 20 rollouts/seed，均值 ± 95% CI（t 分布）。"
                 f"提取于 {data.get('generated_at','?')}。\n")
    lines.append("## 主对比表\n")
    lines.append("| 阶段 | seeds | reach-v2 | push-v2 | button-press | 三任务均值 | Δ vs 异构基线 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["tag"] == base:
            delta = "—（基线）"
        else:
            d, common = delta_vs(bmap.get(base), r)
            if d is None:
                delta = "—"
            else:
                delta = f"{d:+.3f}"
                if len(common) < len(DOMAIN_ORDER):
                    delta += f"（{len(common)} 任务）"
        lines.append(f"| {r['label']} | {r['n_seeds']} | " +
                     " | ".join(r[d2] for d2 in DOMAIN_ORDER) +
                     f" | {r['avg_over_tasks']} | {delta} |")

    # EDA 权重
    eda = data.get("eda") or {}
    if eda:
        lines.append("\n## EDA 聚合权重\n")
        for tag, blob in eda.items():
            w = blob.get("weights") or []
            clusters = blob.get("clusters")
            lines.append(f"### {tag}\n")
            if clusters:
                lines.append(f"簇划分：`{clusters}`（客户端 c0~c3）\n")
            if w:
                names = [f"c{i}" for i in range(len(w[0]))]
                lines.append("| MoE 层 | " + " | ".join(names) + " |")
                lines.append("|---" * (len(names) + 1) + "|")
                for i, row in enumerate(w):
                    bm = (blob.get("block_map") or [f"L{i}"])[i] \
                        if blob.get("block_map") else f"L{i}"
                    lines.append(f"| block {bm} | " +
                                 " | ".join(f"{v:.4f}" for v in row) + " |")
                mw = blob.get("mean_weights")
                if mw:
                    lines.append("| **平均** | " +
                                 " | ".join(f"**{v:.4f}**" for v in mw) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] 已写出 {path}")


# ---------------------------------------------------------------- 图
def fig_moe_effect(rows, path):
    """异构场景下各变体的三任务成功率对比（分组柱状图）。"""
    sel = [r for r in rows if r["hetero"] and r["_means"]]
    if not sel:
        print("[skip] 没有可画的异构阶段结果")
        return
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    n = len(sel)
    width = 0.8 / max(1, len(DOMAIN_ORDER))
    x = range(n)
    for j, d in enumerate(DOMAIN_ORDER):
        # ⚠️ 必须按 domain 名查，不能用 _means 的下标 ——
        #    某个任务缺结果时 _means 会被压缩，下标会串任务（画出错位的柱子）
        vals = [r["_dmean"].get(d, 0.0) for r in sel]
        offs = [i + (j - 1) * width for i in x]
        bars = ax.bar(offs, vals, width,
                      label=DOMAIN_LABEL[d],
                      color=["#4C78A8", "#F58518", "#54A24B"][j % 3])
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in sel], fontsize=9)
    ax.set_ylabel("成功率（3 seeds × 20 rollouts）")
    ax.set_ylim(0, 1.08)
    ax.set_title("异构联邦下 MoE 与 EDA 的贡献")
    ax.legend(title="任务", ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[OK] 已写出 {path}")


def fig_eda_weights(data, path):
    """EDA 权重热力图 + 簇划分标注。"""
    eda = data.get("eda") or {}
    if not eda:
        print("[skip] 没有 EDA 权重数据")
        return
    tags = list(eda.keys())
    fig, axes = plt.subplots(1, len(tags), figsize=(6.2 * len(tags), 4.6),
                             squeeze=False)
    for ax, tag in zip(axes[0], tags):
        blob = eda[tag]
        w = blob.get("weights") or []
        if not w:
            continue
        bm = blob.get("block_map") or list(range(len(w)))
        im = ax.imshow(w, cmap="YlOrRd", aspect="auto", vmin=0.0)
        ax.set_xticks(range(len(w[0])))
        ax.set_xticklabels([f"c{i}" for i in range(len(w[0]))])
        ax.set_yticks(range(len(w)))
        ax.set_yticklabels([f"block {b}" for b in bm])
        for i, row in enumerate(w):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v > max(max(r) for r in w) * 0.6 else "black")
        clusters = blob.get("clusters")
        title = tag
        if clusters:
            title += f"\n簇划分: {clusters}"
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label="聚合权重")
    fig.suptitle("EDA 逐层聚合权重（行 = MoE 层，列 = 客户端）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[OK] 已写出 {path}")


def fig_convergence(data, path):
    """各阶段的联邦平均 loss 收敛曲线。"""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    any_curve = False
    for s in data["stages"]:
        conv = s.get("convergence") or {}
        pts = sorted(((int(k), v) for k, v in conv.items() if v is not None))
        if len(pts) < 2:
            continue
        any_curve = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", ms=3, label=s["label"])
    if not any_curve:
        print("[skip] 没有收敛曲线数据")
        plt.close(fig)
        return
    ax.set_xlabel("联邦轮次")
    ax.set_ylabel("各客户端训练 loss 均值")
    ax.set_title("收敛曲线对比")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[OK] 已写出 {path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(here, "phase3_raw.json"))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "results"))
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        raise SystemExit(f"[!!] 找不到 {args.raw}（先在云端跑 extract_phase3_results.py）")

    outdir = os.path.abspath(args.outdir)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    setup_font()
    data = load(args.raw)
    rows = build_rows(data)
    write_csv(rows, os.path.join(outdir, "phase3_table.csv"))
    write_md(rows, os.path.join(outdir, "phase3_table.md"), data)
    fig_moe_effect(rows, os.path.join(figdir, "fig6_moe_effect.png"))
    fig_eda_weights(data, os.path.join(figdir, "fig6_eda_weights.png"))
    fig_convergence(data, os.path.join(figdir, "fig6_convergence.png"))
    print("REPORT_DONE")


if __name__ == "__main__":
    main()
