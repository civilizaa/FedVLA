#!/usr/bin/env python3
"""FedVLA：学习率曲线分析 —— 诊断"联邦本地 lr 衰减过快"的问题。

背景：
  hpt/run.py 硬编码   cfg.lr_scheduler.T_max = int(cfg.train.total_iters)
  所以 cosine 的周期 = 本地总步数。联邦本地只有几十步 -> 每轮本地训练
  都在做一次完整的 "warmup + cosine 衰减到 0" 循环。

  集中式：6000 步，warmup 1000 -> 大部分时间 lr ≈ 1e-5
  联邦  ：每轮 45 步，warmup 11 -> 每轮 lr 升到 1e-5 后 30 步内衰减到 ~0

本脚本按源码逐字复刻 WarmupLR + CosineAnnealingLR 的 lr 轨迹，
算出「等效满学习率步数」（lr >= 0.5 * target_lr 的步数），用于量化差距。
"""
import math

TARGET_LR = 1e-5
ETA_MIN = 1e-6
WARMUP_INIT_LR = 1e-10   # warmup_lr.lr 默认值


def cosine_lr(t, t_max):
    """CosineAnnealingLR 的 lr(t)（源码行为，闭式）"""
    if t >= t_max:
        return ETA_MIN
    return ETA_MIN + (TARGET_LR - ETA_MIN) * (1 + math.cos(math.pi * t / t_max)) / 2


def warmup_wrapped_lr(t, t_max, warmup_steps):
    """WarmupLR 包裹后的实际 lr
    源码 hpt/utils/warmup_lr_wrapper.py:
      - 前 num_warmup 步：线性从 warmup_initial_lr 升到 warmup_max_lr
      - 之后：直接用被包裹 scheduler 的 lr
    """
    if t < warmup_steps:
        # _warmup_linear: lr = initial + (max - initial) * t / num_warmup
        return WARMUP_INIT_LR + (TARGET_LR - WARMUP_INIT_LR) * t / max(1, warmup_steps)
    return cosine_lr(t, t_max)


def analyze(name, total_iters, warmup_steps, rounds=1):
    """total_iters: 单次本地训练的总步数
    warmup_steps: warmup_lr.step
    rounds: 该配置重复多少轮（联邦场景）
    """
    t_max = total_iters          # run.py 硬编码
    lrs = [warmup_wrapped_lr(t, t_max, warmup_steps) for t in range(total_iters)]
    eff = sum(1 for x in lrs if x >= 0.5 * TARGET_LR)
    total_steps = total_iters * rounds
    return {
        "name": name,
        "t_max": t_max,
        "warmup": warmup_steps,
        "rounds": rounds,
        "total_steps": total_steps,
        "eff_steps": eff * rounds,
        "lr_first": lrs[0],
        "lr_mid": lrs[total_iters // 2],
        "lr_last": lrs[-1],
        "lr_mean": sum(lrs) / len(lrs),
        "lr_sum": sum(lrs) * rounds,
    }


def main():
    print("=" * 92)
    print("学习率曲线诊断：联邦本地训练 vs 集中式")
    print("=" * 92)
    print()
    print(f"{'配置':<34}{'T_max':>8}{'warmup':>9}{'总步数':>9}{'等效步数':>10}{'平均lr':>12}")
    print("-" * 92)

    configs = [
        # (名称, 单次本地 total_iters, warmup, 重复轮数)
        ("集中式 task1 (HPT 官方配方)", 6000, 1000, 1),
        ("集中式 task2", 6000, 1000, 1),
        ("集中式 task3", 6000, 1000, 1),
        ("等预算集中式 task1 (2250步)", 2250, 563, 1),
        ("等预算集中式 task3 (3600步)", 3600, 900, 1),
        ("联邦 task1 本地 (当前配置)", 44, 11, 50),
        ("联邦 task2 本地 (当前配置)", 56, 14, 50),
        ("联邦 task3 本地 (当前配置)", 71, 18, 50),
        # 修复方案：T_max 覆盖整个联邦过程（需要打补丁）
        ("联邦 task1 (修复: T_max=2200)", 44, 11, 50),
    ]

    rows = []
    for name, ti, wu, r in configs:
        if "修复" in name:
            # 手动指定 t_max = 本地步数 × 轮数
            t_max = ti * r
            lrs = [warmup_wrapped_lr(t % ti, t_max, wu) for t in range(ti * r)]
            eff = sum(1 for x in lrs if x >= 0.5 * TARGET_LR)
            row = {
                "name": name, "t_max": t_max, "warmup": wu, "rounds": r,
                "total_steps": ti * r, "eff_steps": eff,
                "lr_mean": sum(lrs) / len(lrs),
                "lr_sum": sum(lrs),
            }
        else:
            row = analyze(name, ti, wu, r)
        rows.append(row)
        print(f"{row['name']:<34}{row['t_max']:>8}{row['warmup']:>9}"
              f"{row['total_steps']:>9}{row['eff_steps']:>10}{row['lr_mean']:>12.3e}")

    print()
    print("「等效步数」= lr >= 0.5×目标lr 的步数（衡量真正在学习的量）")
    print()
    base = rows[0]
    print("相对集中式 task1 的学习量比例：")
    print("-" * 92)
    for row in rows:
        ratio = row["eff_steps"] / base["eff_steps"]
        lr_ratio = row["lr_sum"] / base["lr_sum"]
        print(f"  {row['name']:<34} 等效步数 {ratio:>6.1%}   lr 累积量 {lr_ratio:>7.1%}")
    print()
    print("结论：")
    print("  当前联邦配置的等效学习量只有集中式的 "
          f"{rows[5]['eff_steps']/base['eff_steps']:.0%}，")
    print("  这解释了为什么联邦 loss 高 4 倍、成功率低一个数量级。")
    print("  修复：把 cosine 的 T_max 从「本地总步数」改成「本地步数 × 总轮数」")
    print("        （需要给 hpt/run.py 打补丁，因为 T_max 被硬编码覆盖）。")
    print()
    print("LR_ANALYSIS_DONE")
    return 0


if __name__ == "__main__":
    main()
