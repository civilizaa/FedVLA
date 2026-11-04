#!/usr/bin/env python3
"""FedVLA：联邦学习率调度诊断（逐字复刻 torch + HPT 的调度行为）。

要回答的问题
------------
HPT 原生调度在联邦场景下会退化成什么？补丁后的全局调度能恢复多少？

HPT 原生（`hpt/run.py` + `hpt/utils/warmup_lr_wrapper.py`）：
    cfg.lr_scheduler.T_max = int(cfg.train.total_iters)   # 联邦下 = 单轮本地步数
    sch = CosineAnnealingLR(...)
    sch = WarmupLR(sch, init_lr=0, num_warmup=cfg.warmup_lr.step, warmup_strategy="linear")
并且**联邦每轮是一个新进程** → scheduler 与 optimizer 每轮重建
→ 50 轮 = 50 次完整的 "warmup → cosine → 0"。

补丁后（`hpt/utils/fedvla_lr.py`）：
    lr = f(全局进度 p)，p 在集中式是 step/total，在联邦是 (r + t/local_steps)/R
→ 联邦 50 轮的 lr 轨迹与同预算集中式**逐点相同**。

本脚本用**真实的 torch 调度器类**（不是手写近似）跑出每一步的 lr，然后对比：

  A. 等预算集中式（原生 HPT 调度，2250 步）
  B. 联邦 IID（原生 HPT 调度，50 轮 × 45 步）
  C. 等预算集中式（全局调度，2250 步）
  D. 联邦 IID（全局调度，50 轮 × 45 步）

并验证 C 与 D 的 lr 序列逐点相同（这是补丁正确性的判据）。

用法：
    python lr_analysis.py                # 打印对比表
    python lr_analysis.py --plot out.png # 画 lr 轨迹对比图
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET_LR = 1e-5
ETA_MIN = 1e-6
WARMUP_FRAC = 0.25


# ── HPT 原生 WarmupLR（逐字复制自 hpt/utils/warmup_lr_wrapper.py）────────
class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, scheduler, init_lr=1e-3, num_warmup=1, warmup_strategy="linear"):
        self._scheduler = scheduler
        self._init_lr = init_lr
        self._num_warmup = num_warmup
        self._step_count = 0
        self._warmup_strategy = warmup_strategy
        self._warmup_func = self._warmup_linear
        self._format_param()

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def _format_param(self):
        for group in self._scheduler.optimizer.param_groups:
            group["warmup_max_lr"] = group["lr"]
            group["warmup_initial_lr"] = min(self._init_lr, group["lr"])

    def _warmup_linear(self, start, end, pct):
        return (end - start) * pct + start

    def get_lr(self):
        lrs = []
        step_num = self._step_count
        if step_num <= self._num_warmup:
            for group in self._scheduler.optimizer.param_groups:
                lrs.append(self._warmup_func(
                    group["warmup_initial_lr"], group["warmup_max_lr"],
                    step_num / self._num_warmup))
        else:
            lrs = self._scheduler.get_lr()
        return lrs

    def step(self, *args):
        if self._step_count <= self._num_warmup:
            values = self.get_lr()
            for param_group, lr in zip(self._scheduler.optimizer.param_groups, values):
                param_group["lr"] = lr
            self._step_count += 1
        else:
            self._scheduler.step(*args)


# ── 补丁后的全局调度（逐字复制自 hpt/utils/fedvla_lr.py）─────────────────
class GlobalStepLRScheduler:
    def __init__(self, optimizer, target_lr, eta_min=0.0, warmup_frac=0.25,
                 p_start=0.0, p_end=1.0, num_steps=1, base_lrs=None):
        self.optimizer = optimizer
        self.target_lr = float(target_lr)
        self.eta_min = float(eta_min)
        self.warmup_frac = float(warmup_frac)
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.num_steps = max(1, int(num_steps))
        self._t = 0
        self._base_lrs = list(base_lrs) if base_lrs is not None else [
            float(g["lr"]) for g in optimizer.param_groups]
        self._apply()

    def _progress(self):
        return self.p_start + (self.p_end - self.p_start) * (self._t / self.num_steps)

    def _lr_at(self, p):
        if self.warmup_frac > 0.0 and p < self.warmup_frac:
            return self.target_lr * (p / self.warmup_frac)
        span = max(1e-12, 1.0 - self.warmup_frac)
        q = min(max((p - self.warmup_frac) / span, 0.0), 1.0)
        return self.eta_min + (self.target_lr - self.eta_min) * (1.0 + math.cos(math.pi * q)) / 2.0

    def _apply(self):
        lr = self._lr_at(self._progress())
        for group, base in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = lr * (base / self.target_lr) if self.target_lr > 0 else lr

    def step(self, *args, **kwargs):
        self._t += 1
        self._apply()

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


def _dummy_opt(lr=TARGET_LR):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=lr)


# ── 原生 HPT 调度跑一轮（一个进程）────────────────────────────────────────
def native_round_lrs(num_steps, warmup, t_max):
    """复现一轮本地训练中，每一步实际使用的 lr。
    HPT 的 train() 里 optimizer.step() 先于 scheduler.step()，
    所以 batch t 用的 lr 是第 t 次 scheduler.step() 之后的值；
    batch 0 用的是优化器的初始 lr。"""
    opt = _dummy_opt()
    inner = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=ETA_MIN)
    sch = WarmupLR(inner, init_lr=0, num_warmup=warmup, warmup_strategy="linear")
    lrs = [opt.param_groups[0]["lr"]]        # batch 0
    for _ in range(num_steps):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    return lrs[:num_steps]


def native_central_lrs(total_steps, warmup):
    """集中式：一个进程跑完，warmup 用配置的 step 数，cosine 的 T_max=total_iters"""
    opt = _dummy_opt()
    inner = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=ETA_MIN)
    sch = WarmupLR(inner, init_lr=0, num_warmup=warmup, warmup_strategy="linear")
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(total_steps):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    return lrs[:total_steps]


def global_central_lrs(total_steps, warmup_frac=WARMUP_FRAC):
    opt = _dummy_opt()
    sch = GlobalStepLRScheduler(opt, TARGET_LR, ETA_MIN, warmup_frac,
                                p_start=0.0, p_end=1.0, num_steps=total_steps)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(total_steps):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    return lrs[:total_steps]


def global_fed_lrs(rounds, local_steps, warmup_frac=WARMUP_FRAC):
    """联邦：第 r 轮是一个新进程，p_start=r/R, p_end=(r+1)/R"""
    out = []
    for r in range(rounds):
        opt = _dummy_opt()
        sch = GlobalStepLRScheduler(opt, TARGET_LR, ETA_MIN, warmup_frac,
                                    p_start=r / rounds, p_end=(r + 1) / rounds,
                                    num_steps=local_steps)
        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(local_steps):
            sch.step()
            lrs.append(opt.param_groups[0]["lr"])
        out.extend(lrs[:local_steps])
    return out


def native_fed_lrs(rounds, local_steps, warmup, t_max):
    """联邦 + 原生调度：每轮一个新进程，warmup 与 cosine 全部重置"""
    out = []
    for _ in range(rounds):
        out.extend(native_round_lrs(local_steps, warmup, t_max=t_max))
    return out


def stats(lrs):
    n = len(lrs)
    s = sum(lrs)
    mean = s / n
    eff = sum(1 for x in lrs if x >= 0.5 * TARGET_LR)
    area = s / TARGET_LR
    return {"n": n, "mean": mean, "eff_steps": eff, "area": area,
            "max": max(lrs), "min": min(lrs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-steps", type=int, default=45, help="task1: 3 epoch × 15")
    ap.add_argument("--total-steps", type=int, default=2250, help="等预算集中式总步数")
    ap.add_argument("--warmup", type=int, default=None, help="集中式原生 warmup 步数（默认 total/4）")
    ap.add_argument("--fed-warmup", type=int, default=None,
                    help="联邦原生 warmup 步数（默认 ceil((local*epochs-1)/4)，task1=11）")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    R, LS, TS = args.rounds, args.local_steps, args.total_steps
    warm = args.warmup if args.warmup is not None else (TS + 3) // 4
    # 联邦本地的 warmup：run_fedavg.sh 里 C_WARMUP = ceil(C_TOTAL_ITERS / 4)，
    # 而 C_TOTAL_ITERS = LOCAL_EPOCHS × epoch_size − 1（task1: 3×15−1 = 44 → 11）
    fed_tmax = LS - 1                       # 原生调度下 cosine 的 T_max = 单轮 total_iters
    fed_warm = args.fed_warmup if args.fed_warmup is not None else (fed_tmax + 3) // 4

    print("=" * 96)
    print("FedVLA 学习率调度诊断（真实 torch 调度器逐字复刻）")
    print("=" * 96)
    print(f"场景：联邦 {R} 轮 × 每轮 {LS} 步 = {R*LS} 总步数；等预算集中式 {TS} 步")
    print(f"集中式原生：warmup={warm}  T_max={TS}")
    print(f"联邦原生  ：warmup={fed_warm}  T_max={fed_tmax}（= 单轮 total_iters），每轮进程重建")
    print(f"目标 lr = {TARGET_LR:g}，eta_min = {ETA_MIN:g}")
    print()

    c_native = native_central_lrs(TS, warm)
    f_native = native_fed_lrs(R, LS, fed_warm, t_max=fed_tmax)
    c_global = global_central_lrs(TS)
    f_global = global_fed_lrs(R, LS)

    rows = [
        ("A 等预算集中式 (原生 HPT 调度)", c_native),
        ("B 联邦 IID (原生 HPT 调度)", f_native),
        ("C 等预算集中式 (全局调度)", c_global),
        ("D 联邦 IID (全局调度)", f_global),
    ]
    print(f"{'配置':<34}{'步数':>7}{'平均lr':>12}{'等效步数':>10}{'学习量∫':>10}{'峰值':>11}")
    print("-" * 96)
    for name, lrs in rows:
        st = stats(lrs)
        print(f"{name:<34}{st['n']:>7}{st['mean']:>12.3e}{st['eff_steps']:>10}"
              f"{st['area']:>10.1f}{st['max']:>11.3e}")
    print()
    print("「学习量∫」= Σ lr / target_lr，即整段训练的有效学习率积分（越大越好）")
    print("「等效步数」= lr ≥ 0.5×target_lr 的步数")
    print()

    # 正确性判据：C 与 D 的 lr 序列应逐点相同
    n = min(len(c_global), len(f_global))
    diff = max(abs(c_global[i] - f_global[i]) for i in range(n))
    print("=" * 96)
    print("补丁正确性判据")
    print("=" * 96)
    print(f"C(集中式全局) vs D(联邦全局) 最大 lr 偏差 = {diff:.3e}")
    print("  -> " + ("[OK] 逐点一致：补丁让联邦的 lr 轨迹与同预算集中式完全相同"
                     if diff < 1e-12 else "[!!] 不一致，补丁有 bug"))

    a = stats(c_native)["area"]
    b = stats(f_native)["area"]
    c = stats(c_global)["area"]
    d = stats(f_global)["area"]
    print()
    print("相对『集中式原生』的学习量比例：")
    print("  A 集中式原生          100.0%")
    print(f"  B 联邦原生            {b/a:>6.1%}   <-- 补丁前")
    print(f"  C 集中式全局          {c/a:>6.1%}")
    print(f"  D 联邦全局            {d/a:>6.1%}   <-- 补丁后")
    print()
    print("注意：原生调度下 B 的『学习量积分』并不比 C 低 —— 真正的问题不是积分总量，")
    print("而是**形状**：锯齿波每轮都从 0 重新 warmup 冲高再衰减，而连续 cosine 把")
    print("最后 1/3 的步数留在低 lr 区做精细收敛；再加上 Adam 动量每轮清零。")
    print("=" * 96)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
        ax = axes[0]
        ax.plot(c_native, lw=1.8, color="#2F6FB5", label="A 集中式原生 (2250步)")
        ax.plot(f_native[:150], lw=1.6, color="#C0504D", label="B 联邦原生 (前150步)")
        ax.plot(c_global, lw=1.8, color="#4E9F50", ls="--", label="C 集中式全局")
        ax.plot(f_global[:150], lw=1.6, color="#E08A1E", ls="--", label="D 联邦全局 (前150步)")
        ax.set_xlabel("优化步数")
        ax.set_ylabel("学习率")
        ax.set_title("学习率轨迹：原生锯齿波 vs 全局连续 cosine")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=8.5, frameon=False)
        ax2 = axes[1]
        ax2.plot(c_native, lw=1.8, color="#2F6FB5", label="集中式原生")
        ax2.plot(c_global, lw=1.8, color="#4E9F50", ls="--", label="集中式全局")
        ax2.set_xlabel("优化步数")
        ax2.set_ylabel("学习率")
        ax2.set_title("集中式：原生 vs 全局（warmup 起点差 1 步）")
        ax2.grid(alpha=0.3, ls="--")
        ax2.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"\n已保存图：{args.plot}")

    print("LR_ANALYSIS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
