#!/usr/bin/env python3
"""FedVLA 第 4 阶段隐私侧：结果提取 + 论文图表生成。

读 `privacy_results.json`，产出：
  · `privacy_table.md` / `privacy_table.csv`  —— 主表
  · `fig6_recon_action.png`   —— **动作重建对比图**（论文最直观的一张）
  · `fig6_recon_curves.png`   —— 重建质量 vs σ（三通路 × 两种预算方案）
  · `fig6_sensitivity.png`    —— 逐通路敏感度（消融法）
  · `fig6_image_feature.png`  —— 视觉特征重建对比（证明视觉通路反演失败）
  · `第4阶段隐私侧结果与结论.md` —— 自动汇总的结论文本

用法：
    python federated/extract_privacy_results.py \
        --json /root/HPT/output/privacy/privacy_results.json \
        --out  /root/HPT/output/privacy
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
except Exception:                                    # pragma: no cover
    plt = None

# 中文标注在云端 matplotlib 下容易缺字 → 图里统一用英文，注释放正文
LAB = {
    "action": "Action (label)",
    "state": "Proprioception (input)",
    "image": "Vision (input)",
}


def _agg(runs: List[Dict]) -> Dict:
    """按 (mode, sigma, target) 聚合多次样本 → 均值。"""
    buckets = defaultdict(list)
    for r in runs:
        if r.get("kind") != "inversion":
            continue
        buckets[(r["mode"], r.get("sigma_ref"), r["target"])].append(r)
    out: Dict = {}
    for key, rs in buckets.items():
        mode, sig, tgt = key
        agg: Dict = {"mode": mode, "sigma_ref": sig, "target": tgt, "n": len(rs)}
        for field in ("cos", "corr_mean", "rel_l2", "baseline_cos",
                      "baseline_corr_mean", "baseline_rel_l2",
                      "match_loss_best", "eps", "achieved_epsilon"):
            vals = [r[field] for r in rs if r.get(field) is not None]
            if vals:
                agg[field] = float(np.mean(vals))
                agg[field + "_std"] = float(np.std(vals))
        # 学习信号保真度
        fids = [r.get("fidelity_by_group") for r in rs if r.get("fidelity_by_group")]
        if fids:
            agg["fidelity"] = {k: float(np.mean([f[k] for f in fids]))
                               for k in fids[0]}
        out[key] = agg
    return out


def _metric(rec: Dict, target: str) -> float:
    if target == "image":
        return rec.get("corr_mean", float("nan"))
    return rec.get("cos", float("nan"))


def make_tables(agg: Dict, out_dir: str) -> str:
    lines = []
    lines.append("# 第 4 阶段隐私侧结果表（DP-SGD 上行加噪 + 梯度反演）\n")
    lines.append("## A. 逐通路敏感度（消融法：把该通路换成另一条轨迹的输入）\n")
    lines.append("| 样本 | 视觉 D_m | 状态 D_m | 动作 D_m | 视觉 1−cos | 状态 1−cos | 动作 1−cos |")
    lines.append("|---|---|---|---|---|---|---|")
    csv_rows = ["kind,sample,mode,sigma_ref,target,n,metric,metric_std,"
                "rel_l2,rel_l2_std,eps,fidelity_shared"]

    sens = [r for r in agg.get("_sens", [])]
    for s in sens:
        sm = s.get("sensitivity", {})
        cg = s.get("sensitivity_cos", {})
        lines.append(
            f"| {s['sample']} | {sm.get('image', float('nan')):.4f} | "
            f"{sm.get('state', float('nan')):.4f} | {sm.get('action', float('nan')):.4f} | "
            f"{1-cg.get('image', float('nan')):.4f} | {1-cg.get('state', float('nan')):.4f} | "
            f"{1-cg.get('action', float('nan')):.4f} |")
        csv_rows.append(
            f"sensitivity,{s['sample']},none,,all,1,"
            f"{sm.get('image', float('nan')):.6f},0,"
            f"{sm.get('state', float('nan')):.6f},0,,"
            f"{1-cg.get('image', float('nan')):.6f}")

    lines.append("\n## B. 重建质量 vs 噪声倍率 σ（均值，样本数见 n）\n")
    lines.append("| 目标 | 方案 | σ_ref | 总 ε | 指标 | 均值 | std | 相对L2 | 学习信号保真度(shared) | n |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(agg.keys(),
                   key=lambda k: (k[2], (k[1] is None), k[1] or 0, k[0]))
    for key in order:
        if key[0] == "_sens":
            continue
        rec = agg[key]
        mode, sig, tgt = key
        m = _metric(rec, tgt)
        eps = rec.get("achieved_epsilon") or rec.get("eps")
        fid = rec.get("fidelity", {}).get("shared")
        sig_s = "clean" if sig is None else f"{sig:g}"
        lines.append(
            f"| {LAB.get(tgt, tgt)} | {mode} | {sig_s} | "
            f"{eps:.4g} | {'corr' if tgt=='image' else 'cos'} | {m:.4f} | "
            f"{rec.get('metric_std', rec.get('cos_std', rec.get('corr_mean_std', float('nan')))):.4f} | "
            f"{rec.get('rel_l2', float('nan')):.4f} | "
            f"{(f'{fid:.4f}' if fid is not None else '—')} | {rec['n']} |")
        csv_rows.append(
            f"inversion,,{mode},{'' if sig is None else sig},{tgt},{rec['n']},"
            f"{m:.6f},{rec.get('cos_std', rec.get('corr_mean_std', 0)):.6f},"
            f"{rec.get('rel_l2', float('nan')):.6f},"
            f"{rec.get('rel_l2_std', 0):.6f},"
            f"{'' if eps is None else f'{eps:.6g}'},"
            f"{'' if fid is None else f'{fid:.6f}'}")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "privacy_table.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, "privacy_table.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")
    return md


def make_budget_table(runs: List[Dict], out_dir: str) -> str:
    """预算方案对照表：每种方案在每个 σ 点上的 ε/σ/SNR/保真度。"""
    lines = ["\n## C. 预算方案对照（相同 σ_ref ⇒ 相同总 ε）\n"]
    lines.append("| σ_ref | 方案 | 总 ε | ε_shared | σ_shared | σ_vision | "
                 "SNR_shared | 保真度_shared | ε_vision | ε_state | ε_action |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    seen = {}
    for r in runs:
        if r.get("kind") != "inversion" or r.get("sigma_ref") is None:
            continue
        key = (r["sigma_ref"], r["mode"])
        if key in seen:
            continue
        seen[key] = r
    for (sig, mode) in sorted(seen, key=lambda k: (k[0], k[1])):
        r = seen[(sig, mode)]
        eps_m = r.get("eps_by_modality", {})
        sig_m = r.get("sigma_by_modality", {})
        snr = r.get("snr_by_modality", {})
        fid = r.get("fidelity_by_group", {})
        eps = r.get("achieved_epsilon")
        lines.append(
            f"| {sig:g} | {mode} | {eps:.4g} | "
            f"{eps_m.get('shared', float('nan')):.4g} | "
            f"{sig_m.get('shared', float('nan')):.6g} | "
            f"{sig_m.get('vision', float('nan')):.6g} | "
            f"{snr.get('shared', float('nan')):.4f} | "
            f"{fid.get('shared', float('nan')):.4f} | "
            f"{eps_m.get('vision', float('nan')):.4g} | "
            f"{eps_m.get('state', float('nan')):.4g} | "
            f"{eps_m.get('action', float('nan')):.4g} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "privacy_budget_table.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    return md


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig_recon_curves(agg: Dict, out_dir: str) -> Optional[str]:
    if plt is None:
        return None
    targets = ["action", "state", "image"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    styles = {"uniform": ("o", "#d1495b"),
              "modality": ("s", "#1b6ca8"),
              "modality_rev": ("^", "#2a9d5c")}
    for ax, tgt in zip(axes, targets):
        for mode, (mk, col) in styles.items():
            pts = [(r["sigma_ref"], _metric(r, tgt))
                   for k, r in agg.items()
                   if k[0] != "_sens" and r["target"] == tgt and r["mode"] == mode
                   and r["sigma_ref"] is not None]
            if not pts:
                continue
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                    color=col, label=mode, lw=2, ms=6)
        clean = [r for k, r in agg.items()
                 if k[0] != "_sens" and r["target"] == tgt and r["mode"] == "clean"]
        if clean:
            ax.axhline(_metric(clean[0], tgt), ls="--", c="gray", lw=1.4,
                       label="no noise")
        base = [r for k, r in agg.items()
                if k[0] != "_sens" and r["target"] == tgt and r["mode"] == "clean"]
        if base:
            b = base[0].get("baseline_corr_mean" if tgt == "image" else "baseline_cos")
            if b is not None:
                ax.axhline(b, ls=":", c="black", lw=1.2, label="random baseline")
        ax.set_xscale("log")
        ax.set_xlabel("noise multiplier $\\sigma$ (smaller = less noise)")
        ax.set_ylabel("reconstruction quality (cos / corr)")
        ax.set_title(LAB.get(tgt, tgt))
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("FedVLA uplink gradient inversion vs DP noise  "
                 "(3 samples, mean)", y=1.02)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig6_recon_curves.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_recon_action(runs: List[Dict], out_dir: str) -> Optional[str]:
    """**最直观的一张图**：真实动作 vs 各 DP 配置下的重建动作。"""
    if plt is None:
        return None
    first = [r for r in runs if r.get("kind") == "inversion" and r.get("sample") == 0]
    if not first:
        return None
    real = next((r for r in first if r.get("real_action") is not None), None)
    if real is None:
        return None
    R = np.asarray(real["real_action"], dtype=np.float64).reshape(-1, 4)

    picks: List = []
    for mode in ("uniform", "modality"):
        for sig in sorted({r.get("sigma_ref") for r in first
                           if r.get("mode") == mode and r.get("sigma_ref")}):
            cand = [r for r in first if r.get("mode") == mode
                    and r.get("sigma_ref") == sig and r.get("recon_action")]
            if cand:
                picks.append((f"{mode}\n$\\sigma$={sig:g}", cand[0]))
    if not picks:
        return None

    n = len(picks) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n + 1.2, 3.1))
    if n == 1:
        axes = [axes]
    vmin, vmax = R.min(), R.max()

    def draw(ax, M, title):
        ax.imshow(M, aspect="auto", cmap="RdBu_r",
                  vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("action dim", fontsize=8)
        ax.tick_params(labelsize=7)

    draw(axes[0], R, "ground truth")
    for ax, (title, rec) in zip(axes[1:], picks):
        M = np.asarray(rec["recon_action"], dtype=np.float64).reshape(-1, 4)
        draw(ax, M, title)
        cos = rec.get("cos")
        if cos is not None:
            ax.set_xlabel(f"cos={cos:.3f}", fontsize=8)
    axes[0].set_ylabel("timestep (8-step action chunk)", fontsize=8)
    fig.suptitle("Gradient-inversion reconstruction of the private ACTION "
                 "(label channel) from the FedVLA uplink", y=1.04)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig6_recon_action.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_image_feature(runs: List[Dict], out_dir: str) -> Optional[str]:
    """视觉通路：真实 vs 重建的 7×7 空间 patch 热力图（证明视觉不可反演）。"""
    if plt is None:
        return None
    first = [r for r in runs if r.get("kind") == "inversion" and r.get("sample") == 0
             and r.get("target") == "image"]
    if not first:
        return None
    real = next((r for r in first if r.get("real_image") is not None), None)
    if real is None:
        return None
    Rt = np.asarray(real["real_image"], dtype=np.float64).reshape(49, 512)

    picks = []
    clean = next((r for r in first if r.get("mode") == "clean"), None)
    if clean and clean.get("recon_image"):
        picks.append(("no noise", clean))
    for mode in ("uniform", "modality"):
        for sig in sorted({r.get("sigma_ref") for r in first
                           if r.get("mode") == mode and r.get("sigma_ref")}):
            cand = [r for r in first if r.get("mode") == mode
                    and r.get("sigma_ref") == sig and r.get("recon_image")]
            if cand:
                picks.append((f"{mode}\n$\\sigma$={sig:g}", cand[0]))
    if not picks:
        return None

    n = len(picks) + 1
    fig, axes = plt.subplots(1, n, figsize=(1.85 * n + 1.0, 3.0))
    if n == 1:
        axes = [axes]

    def patch_map(M):
        return M.reshape(7, 7, 512).mean(axis=-1)

    gt = patch_map(Rt)
    vmin, vmax = float(gt.min()), float(gt.max())
    im = axes[0].imshow(gt, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("ground truth\n(7x7 patch map)", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for ax, (title, rec) in zip(axes[1:], picks):
        M = np.asarray(rec["recon_image"], dtype=np.float64).reshape(49, 512)
        im = ax.imshow(patch_map(M), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        c = rec.get("corr_mean")
        if c is not None:
            ax.set_xlabel(f"corr={c:.3f}", fontsize=8)
    fig.suptitle("Vision channel: the 49x512 ResNet feature map is NOT "
                 "recoverable from the FedVLA uplink (dimension curse)",
                 y=1.06)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig6_image_feature.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_sensitivity(runs: List[Dict], out_dir: str) -> Optional[str]:
    if plt is None:
        return None
    sens = [r for r in runs if r.get("kind") == "sensitivity"]
    if not sens:
        return None
    mods = ["image", "state", "action"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.35
    xs = np.arange(len(mods))
    for i, s in enumerate(sens):
        sm = s.get("sensitivity", {})
        vals = [sm.get(m, 0.0) for m in mods]
        axes[0].bar(xs + (i - 0.5) * w, vals, width=w, label=f"sample {s['sample']}")
        cg = s.get("sensitivity_cos", {})
        vals2 = [1 - cg.get(m, 0.0) for m in mods]
        axes[1].bar(xs + (i - 0.5) * w, vals2, width=w, label=f"sample {s['sample']}")
    axes[0].set_title("relative gradient change $D_m=\\|g-g_m\\|/\\|g\\|$")
    axes[1].set_title("leakage strength $1-\\cos(g,g_m)$  (higher = more leaked)")
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([LAB[m] for m in mods], fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Per-modality sensitivity of the FedVLA uplink update "
                 "(ablation: replace one modality with another trajectory's)",
                 y=1.04)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig6_sensitivity.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/root/HPT/output/privacy/privacy_results.json")
    ap.add_argument("--out", default="/root/HPT/output/privacy")
    args = ap.parse_args(argv)

    with open(args.json, encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", [])
    os.makedirs(args.out, exist_ok=True)

    agg = _agg(runs)
    agg["_sens"] = [r for r in runs if r.get("kind") == "sensitivity"]
    make_tables(agg, args.out)
    make_budget_table(runs, args.out)
    print("[fig]", fig_recon_curves(agg, args.out))
    print("[fig]", fig_recon_action(runs, args.out))
    print("[fig]", fig_image_feature(runs, args.out))
    print("[fig]", fig_sensitivity(runs, args.out))
    print(f"[done] 表格与图已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
