#!/usr/bin/env python3
"""FedVLA 第 4 阶段：DP 效用扫描的结果提取（真实成功率 × 代理指标交叉验证）。

输入（云端）：
  /root/HPT/output/eval_dp_s*_mujoco_metaworld_task3.log   各 σ 的 3 seed 成功率
  /root/HPT/output/eval_c_drop00_mujoco_metaworld_task3.log  无噪声基线
  /root/HPT/output/dp_calib/calib.json                      C 标定结果
  /root/HPT/output/privacy/privacy_results.json             隐私侧代理指标
输出：
  /root/HPT/output/dp_utility_raw.json
  stdout 可读表格
  fig7_dp_utility.png   真实成功率 vs σ（双轴叠加代理指标）

统计口径：3 seeds × 20 rollouts，报均值 ± 95% CI（t 分布，df = n−1 = 2，t=4.303）。
ε 口径：δ=1e-5，q=1/4（4 客户端各被采样），steps=50（50 轮联邦复合）
        —— 与隐私侧 uniform 方案**完全一致**，所以两边的 σ 可以直接对表。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:                                    # pragma: no cover
    plt = None

OUT = "/root/HPT/output"

# (tag, σ, 人类可读标签)
# ⚠️ σ 网格必须与隐私侧 uniform 方案的网格**完全一致**（1e-4 / 1e-3 / 1e-2），
#    才能在同一个 σ 上把「真实任务成功率」和「代理指标（保真度 / 动作 cos）」对表。
#    语义：per-coordinate 噪声 std = σ·C ⇒ 噪声/信号比 = σ·√d（d=17.1M ⇒ √d≈4134）。
DP_CONFIGS = [
    ("dp_s0001", 1e-4, "σ=1e-4 (噪声/信号 0.41)"),
    ("dp_s0003", 1e-3, "σ=1e-3 (噪声/信号 4.1)"),
    ("dp_s0012", 1e-2, "σ=1e-2 (噪声/信号 41)"),
]

T_CRIT = {1: 12.706204736432095, 2: 4.302652729911275}
DP_STEPS = 50
DP_DELTA = 1e-5
DP_Q = 0.25          # q = 1 / num_clients = 1/4


def parse_seed_successes(path: str):
    """从评估日志里解析每个 seed 的成功率（每张表一行 '| avg | x |'）。"""
    if not os.path.exists(path):
        return []
    txt = open(path, encoding="utf-8", errors="ignore").read()
    return [float(m.group(1)) for m in
            re.finditer(r"\|\s*avg\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", txt)]


def mean_ci(vals):
    """均值 ± 95% CI（t 分布，df = n−1）。

    ⚠️ 临界值必须按 df=n−1 取：n=3 → df=2 → t=4.303。
    原实现对 n=2 也用 4.303，会把 CI 系统性低估约 66%。
    """
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return {"mean": m, "ci": 0.0, "n": 1, "seeds": vals, "df": 0}
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    tcrit = T_CRIT.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return {"mean": m, "ci": half, "n": n, "seeds": vals, "df": n - 1,
            "sd": sd, "lo": m - half, "hi": m + half}


def load_privacy_proxy(path: str):
    """从隐私侧结果里取 uniform 方案的代理指标：{sigma: {fidelity, action_cos}}。

    口径：mode='uniform'、kind='inversion' 的记录。
      · fidelity = fidelity_by_group['shared']（共享主干保真度）
      · action_cos = target='action' 的 cos（标签泄漏强度）
    """
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}
    recs = data.get("records", data) if isinstance(data, dict) else data
    if not isinstance(recs, list):
        return {}

    acc = {}
    for r in recs:
        if not isinstance(r, dict):
            continue
        if r.get("kind") != "inversion":
            continue
        if r.get("mode") != "uniform":
            continue
        sig = r.get("sigma_ref")
        if sig is None:
            continue
        d = acc.setdefault(float(sig), {"fidelity": [], "action_cos": []})
        fid = (r.get("fidelity_by_group") or {}).get("shared")
        if fid is not None:
            d["fidelity"].append(float(fid))
        if r.get("target") == "action" and r.get("cos") is not None:
            d["action_cos"].append(float(r["cos"]))

    out = {}
    for sig, d in acc.items():
        out[sig] = {
            "fidelity": (sum(d["fidelity"]) / len(d["fidelity"])) if d["fidelity"] else None,
            "action_cos": (sum(d["action_cos"]) / len(d["action_cos"])) if d["action_cos"] else None,
        }
    return out


def eps_for_sigma(sigma: float):
    """用隐私侧同一个 RDP accountant 反算 ε（口径完全一致）。"""
    try:
        from dp import epsilon_from_sigma
        e, order = epsilon_from_sigma(DP_Q, sigma, DP_STEPS, DP_DELTA)
        return float(e), int(order)
    except Exception as e:                              # noqa: BLE001
        print(f"[warn] ε 计算失败（sigma={sigma}）：{type(e).__name__}: {e}")
        return None, None


def plot_results(rows, outdir, base, proxy):
    """双轴图：左轴 = 真实任务成功率，右轴 = 代理指标（共享主干保真度）。"""
    if plt is None:
        print("[warn] matplotlib 不可用，跳过绘图")
        return []
    os.makedirs(outdir, exist_ok=True)
    pts = [r for r in rows if r["mean"] is not None]
    if len(pts) < 2:
        print("[warn] 可用数据点不足 2 个，跳过绘图")
        return []

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    xs = [r["sigma"] for r in pts]
    ys = [r["mean"] for r in pts]
    es = [r["ci"] for r in pts]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#2f6fbf", capsize=4,
                lw=1.9, ms=7, label="Real task success rate")
    for r, x, y in zip(pts, xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#2f6fbf")
    if base and base["mean"] is not None:
        ax.axhline(base["mean"], color="#8a8a8a", ls=":", lw=1.5,
                   label=f"No-DP baseline ({base['mean']:.2f})")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("DP noise multiplier $\\sigma$  (noise / signal ratio)")
    ax.set_ylabel("Task success rate", color="#2f6fbf")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, ls="--")

    # 右轴：代理指标（共享主干保真度）
    fids = [(r["sigma"], r["proxy_fidelity"]) for r in pts if r.get("proxy_fidelity") is not None]
    if fids:
        ax2 = ax.twinx()
        ax2.plot([f[0] for f in fids], [f[1] for f in fids], "s--",
                 color="#c1544a", lw=1.6, ms=6, label="Proxy: shared-trunk fidelity")
        ax2.set_ylabel("Shared-trunk fidelity (proxy)", color="#c1544a")
        ax2.tick_params(axis="y", labelcolor="#c1544a")
        ax2.set_ylim(-0.05, 1.05)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")

    ax.set_title("DP noise vs real task success (task3)", fontsize=10)
    p = os.path.join(outdir, "fig7_dp_utility.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {p}")
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "dp_utility_raw.json"))
    ap.add_argument("--domain", default="mujoco_metaworld_task3")
    ap.add_argument("--privacy-json", default=os.path.join(OUT, "privacy", "privacy_results.json"))
    ap.add_argument("--calib", default=os.path.join(OUT, "dp_calib", "calib.json"))
    args = ap.parse_args()

    proxy = load_privacy_proxy(args.privacy_json)
    calib = {}
    if os.path.exists(args.calib):
        calib = json.load(open(args.calib, encoding="utf-8"))

    # ---- 基线（无 DP）----
    base_vals = parse_seed_successes(
        os.path.join(OUT, f"eval_c_drop00_{args.domain}.log"))
    base = mean_ci(base_vals)

    rows = []
    for tag, sigma, label in DP_CONFIGS:
        vals = parse_seed_successes(os.path.join(OUT, f"eval_{tag}_{args.domain}.log"))
        st = mean_ci(vals)
        eps, order = eps_for_sigma(sigma)
        px = proxy.get(sigma, {})
        rows.append({
            "tag": tag, "label": label, "sigma": sigma,
            "epsilon": eps, "best_order": order,
            "seeds": vals, "n": len(vals),
            "mean": st["mean"] if st else None,
            "ci": st["ci"] if st else None,
            "lo": st.get("lo") if st else None,
            "hi": st.get("hi") if st else None,
            "proxy_fidelity": px.get("fidelity"),
            "proxy_action_cos": px.get("action_cos"),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"domain": args.domain, "calib": calib,
                   "baseline": base, "rows": rows}, f,
                  ensure_ascii=False, indent=2)

    C = calib.get("clip_norm_C")

    print("=" * 112)
    print(f"第 4 阶段：DP 噪声 vs 真实任务成功率（{args.domain}，3 seeds × 20 rollouts）")
    print("=" * 112)
    print(f"C（‖Δ‖₂ 中位数，用于标定 σ）= {C if C is not None else '—'}")
    print(f"ε 口径：δ={DP_DELTA}, q={DP_Q}, steps={DP_STEPS}（与隐私侧 uniform 完全一致）")
    print()
    hdr = (f"{'配置':<24}{'σ':>8}{'ε':>14}{'成功率':>16}"
           f"{'代理保真度':>13}{'动作cos':>10}{'seeds':>20}")
    print(hdr)
    print("-" * 112)
    if base and base["mean"] is not None:
        bs = f"{base['mean']:.3f} ± {base['ci']:.3f}"
        bseeds = ",".join(f"{v:.2f}" for v in base["seeds"])
        print(f"{'无 DP 基线 (c_drop00)':<24}{'0':>8}{'∞':>14}{bs:>16}"
              f"{'1.000':>13}{'—':>10}{bseeds:>20}")
    for r in rows:
        if r["mean"] is None:
            eps_s = "—" if r["epsilon"] is None else f"{r['epsilon']:.4g}"
            print(f"{r['label']:<24}{r['sigma']:>8g}{eps_s:>14}"
                  f"{'（无评估日志）':>16}{'':>13}{'':>10}{'':>20}")
            continue
        s = f"{r['mean']:.3f} ± {r['ci']:.3f}"
        e = f"{r['epsilon']:.4g}" if r["epsilon"] is not None else "—"
        fid = f"{r['proxy_fidelity']:.4f}" if r.get("proxy_fidelity") is not None else "—"
        cos = f"{r['proxy_action_cos']:.3f}" if r.get("proxy_action_cos") is not None else "—"
        seeds = ",".join(f"{v:.2f}" for v in r["seeds"])
        print(f"{r['label']:<24}{r['sigma']:>8g}{e:>14}{s:>16}{fid:>13}{cos:>10}{seeds:>20}")

    # ---- 相对基线的变化 ----
    if base and base["mean"] is not None:
        print()
        print("相对无 DP 基线的成功率变化（pp，正=变差）：")
        for r in rows:
            if r["mean"] is None:
                continue
            d = (base["mean"] - r["mean"]) * 100
            verdict = "变差" if d > 0 else ("变好" if d < 0 else "持平")
            print(f"  {r['label']:<24} {d:+6.1f} pp ({verdict})"
                  f"   （{r['mean']:.3f} vs {base['mean']:.3f}）")

    print()
    print(f"[输出] {args.out}")

    try:
        plot_results(rows, os.path.dirname(os.path.abspath(args.out)), base, proxy)
    except Exception as e:                              # noqa: BLE001
        print(f"[warn] 绘图失败：{type(e).__name__}: {e}")

    print("DP_UTILITY_EXTRACT_DONE")


if __name__ == "__main__":
    main()
