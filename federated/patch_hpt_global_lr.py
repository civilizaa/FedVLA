"""给 HPT 打「全局步数感知调度」补丁（幂等）：修掉联邦场景下每轮重启 warmup+cosine 的缺陷。

═══════════════════════════════════════════════════════════════════════════════
缺陷（2026-11-04 门2 实验定位）
═══════════════════════════════════════════════════════════════════════════════
HPT 原生 `hpt/run.py`（逐字）：

    opt = utils.get_optimizer(cfg.optimizer, policy, cfg.optimizer_misc)
    cfg.lr_scheduler.T_max = int(cfg.train.total_iters)
    sch = utils.get_scheduler(cfg.lr_scheduler, optimizer=opt)
    sch = WarmupLR(sch, init_lr=0, num_warmup=cfg.warmup_lr.step, warmup_strategy="linear")

问题有三层：

1) **T_max 被硬编码成"单轮本地步数"**
   `cfg.train.total_iters` 在联邦下 = 本地一轮的 batch step 数（task1=44/45），
   所以 cosine 的周期只有 44 步。而 `hpt/run.py` 的循环条件是
   `train_steps > cfg.train.total_iters` 才 break，于是"跑满 N 个 epoch"
   需要 total_iters = N×epoch_size − 1。

2) **每轮是一个新进程 → 调度器时钟每轮归零**
   联邦第 r 轮由 4 个客户端各起一次 `python -m hpt.run`，每次都重建 optimizer
   和 scheduler。于是 50 轮 = 50 次完整的 "warmup 升到 lr → cosine 衰减到 0"，
   也就是 **50 次 LR 重启**。等价于把一条连续 cosine 切成 50 段锯齿波。
   这条锯齿波的 lr 积分与连续 cosine 接近，但**形状**完全不同：
   连续 cosine 后 1/3 的步数在低 lr 区间做精细收敛，锯齿波则每一轮都把
   模型从"高 lr 冲出去"再"低 lr 拉回来"，并且每次都从零开始 warmup。

3) **Adam 的动量状态每轮被清零**
   `opt = utils.get_optimizer(...)` 在进程内新建 → exp_avg / exp_avg_sq 全部丢失。
   每轮只有 45 步，前 ~10 步都在重新估计二阶矩，等效步数进一步缩水。
   而真实的联邦客户端的本地优化器是**持久的**（FL 框架里 optimizer 不会每轮重建），
   所以这不但伤性能，还让实验设置偏离真实 FL。

═══════════════════════════════════════════════════════════════════════════════
补丁做了什么
═══════════════════════════════════════════════════════════════════════════════
引入一个自实现的调度器 `hpt.utils.fedvla_lr.GlobalStepLRScheduler`：
学习率是**全局进度 p ∈ [0,1]** 的函数，与"进程内步数"解耦：

    p <  warmup_frac :  lr = target_lr * p / warmup_frac                (线性 warmup)
    p >= warmup_frac :  lr = eta_min + (target_lr - eta_min)
                                  * (1 + cos(pi * (p - warmup_frac) / (1 - warmup_frac))) / 2

* 集中式：p = step / total_steps
* 联邦第 r 轮：p = (r + t / local_steps) / NUM_ROUNDS

于是**联邦 50 轮的 lr 轨迹与同预算集中式逐点相同** —— 这是关键：
这样 IID 联邦与集中式的差异就只剩下"聚合"本身，异构实验也就测的是纯异构效应。

同时支持跨轮持久化 Adam 状态（`FEDVLA_OPT_STATE`）。

用法（在 AutoDL 上）：
    python /root/HPT/federated/patch_hpt_global_lr.py             # 打补丁
    python /root/HPT/federated/patch_hpt_global_lr.py --verify    # 只检查
    python /root/HPT/federated/patch_hpt_global_lr.py --revert    # 回滚

打补丁后必须用环境变量开启（默认关闭，不改变 HPT 原生行为）：
    FEDVLA_GLOBAL_LR=1                       # 开启全局调度
    FEDVLA_P_START=0.02  FEDVLA_P_END=0.04  # 本进程对应的全局进度区间
    FEDVLA_NUM_STEPS=45                     # 本进程执行的 step 数
    FEDVLA_WARMUP_FRAC=0.25                 # warmup 占全局进度的比例
    FEDVLA_OPT_STATE=/path/to/opt_c0.pt     # Adam 状态持久化（可选）
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

DEFAULT_HPT_DIR = os.path.join(os.path.expanduser("~"), "HPT")

MARK = "[FedVLA patch]"

# ── 补丁锚点 1：optimizer / scheduler 构造 ────────────────────────────────
ANCHOR1 = (
    "    opt = utils.get_optimizer(cfg.optimizer, policy, cfg.optimizer_misc)\n"
    "    cfg.lr_scheduler.T_max = int(cfg.train.total_iters)\n"
    "    sch = utils.get_scheduler(cfg.lr_scheduler, optimizer=opt)\n"
    "    sch = WarmupLR(sch, init_lr=0, num_warmup=cfg.warmup_lr.step, warmup_strategy=\"linear\")\n"
)

BLOCK1 = '''    opt = utils.get_optimizer(cfg.optimizer, policy, cfg.optimizer_misc)
    cfg.lr_scheduler.T_max = int(cfg.train.total_iters)
    sch = utils.get_scheduler(cfg.lr_scheduler, optimizer=opt)
    sch = WarmupLR(sch, init_lr=0, num_warmup=cfg.warmup_lr.step, warmup_strategy="linear")

    # [FedVLA patch] 全局步数感知的 LR 调度 + Adam 状态跨轮持久化
    # 背景：HPT 把 cosine 的 T_max 硬编码为 train.total_iters，而联邦下
    # total_iters 是【单轮本地步数】；加上每轮一个新进程（scheduler/optimizer
    # 每轮重建），导致 50 轮 = 50 次 "warmup -> cosine -> 0" 的 LR 重启，
    # 且 Adam 动量每轮清零。补丁让 lr 成为【全局进度 p】的函数，
    # 与进程内步数解耦，从而与同预算集中式的 lr 轨迹逐点一致。
    _fedvla_on = os.environ.get("FEDVLA_GLOBAL_LR", "") == "1"
    if _fedvla_on:
        from hpt.utils.fedvla_lr import GlobalStepLRScheduler
        _p_start = float(os.environ.get("FEDVLA_P_START", "0") or 0.0)
        _p_end = float(os.environ.get("FEDVLA_P_END", "1") or 1.0)
        _nsteps = int(os.environ.get("FEDVLA_NUM_STEPS", "0") or 0)
        if _nsteps <= 0:
            _nsteps = int(cfg.train.total_iters) + 1
        _warm = float(os.environ.get("FEDVLA_WARMUP_FRAC", "0.25") or 0.25)
        _eta_min = float(cfg.lr_scheduler.get("eta_min", 0.0) or 0.0)
        sch = GlobalStepLRScheduler(
            optimizer=opt,
            target_lr=float(cfg.optimizer.lr),
            eta_min=_eta_min,
            warmup_frac=_warm,
            p_start=_p_start,
            p_end=_p_end,
            num_steps=_nsteps,
        )
        print(f"[FedVLA patch] GlobalStepLR enabled: p_start={_p_start:.4f} "
              f"p_end={_p_end:.4f} num_steps={_nsteps} warmup_frac={_warm} "
              f"target_lr={cfg.optimizer.lr} eta_min={_eta_min}")
    # [FedVLA patch] 优化器状态跨轮持久化（真实 FL 客户端的本地优化器是持久的）
    _opt_state = os.environ.get("FEDVLA_OPT_STATE", "")
    if _opt_state and os.path.exists(_opt_state):
        try:
            _sd = torch.load(_opt_state, map_location="cpu", weights_only=False)
            opt.load_state_dict(_sd)
            print(f"[FedVLA patch] optimizer state loaded from {_opt_state}")
        except Exception as _e:
            print(f"[FedVLA patch] optimizer state load FAILED ({type(_e).__name__}: {_e}); "
                  "continuing with fresh optimizer state")
'''

# ── 补丁锚点 2：训练结束后保存 optimizer 状态 ─────────────────────────────
ANCHOR2 = '    # log and finish\n    print("model saved to :", policy_path)\n'

BLOCK2 = '''    # [FedVLA patch] 保存 optimizer 状态，供下一轮续训（跨轮保持 Adam 动量）
    if _opt_state:
        try:
            _sd = opt.state_dict()
            _st = _sd.get("state", {}) or {}
            for _k, _v in _st.items():
                if torch.is_tensor(_v):
                    _st[_k] = _v.detach().cpu()
            torch.save(_sd, _opt_state)
            print(f"[FedVLA patch] optimizer state saved to {_opt_state}")
        except Exception as _e:
            print(f"[FedVLA patch] optimizer state save FAILED ({type(_e).__name__}: {_e})")

    # log and finish
    print("model saved to :", policy_path)
'''

# ── 新文件：hpt/utils/fedvla_lr.py ────────────────────────────────────────
SCHEDULER_SRC = '''"""[FedVLA patch] 全局步数感知的 cosine-with-warmup 学习率调度器。

这个文件由 federated/patch_hpt_global_lr.py 自动写入 hpt/utils/，请勿手工编辑。

设计要点
--------
学习率是**全局进度 p ∈ [0,1]** 的函数，而不是"进程内步数"的函数：

    p <  warmup_frac :  lr = target_lr * p / warmup_frac
    p >= warmup_frac :  lr = eta_min + (target_lr - eta_min)
                                  * (1 + cos(pi * (p - warmup_frac) / (1 - warmup_frac))) / 2

* 集中式训练： p = step / total_steps
* 联邦第 r 轮： p = (r + t / local_steps) / NUM_ROUNDS

这样联邦 50 轮的 lr 轨迹与同预算的集中式训练**逐点相同**，
IID 联邦与集中式之间的差异就只剩下"参数平均"本身。
"""
import math


class GlobalStepLRScheduler:
    """按全局进度计算学习率的调度器（跨联邦轮次连续，无重启）。"""

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
            float(g["lr"]) for g in optimizer.param_groups
        ]
        self._apply()

    # ---- 进度与学习率 ----
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
            if self.target_lr > 0:
                group["lr"] = lr * (base / self.target_lr)
            else:
                group["lr"] = lr

    # ---- torch 调度器接口 ----
    def step(self, *args, **kwargs):
        self._t += 1
        self._apply()

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def get_lr(self):
        return self.get_last_lr()

    def state_dict(self):
        return {"_t": self._t, "_base_lrs": self._base_lrs}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
'''


def find_hpt_dir(explicit: str | None) -> str:
    if explicit:
        return explicit
    for cand in (DEFAULT_HPT_DIR, "/root/HPT", os.getcwd()):
        if os.path.isfile(os.path.join(cand, "hpt", "run.py")):
            return cand
    raise FileNotFoundError("找不到 HPT 源码目录（需要含 hpt/run.py），请用 --hpt-dir 指定")


def main(argv=None):
    ap = argparse.ArgumentParser(description="给 HPT 打全局步数感知调度补丁")
    ap.add_argument("--hpt-dir", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args(argv)

    hpt_dir = find_hpt_dir(args.hpt_dir)
    run_py = os.path.join(hpt_dir, "hpt", "run.py")
    sched_py = os.path.join(hpt_dir, "hpt", "utils", "fedvla_lr.py")
    backup = run_py + ".fedvla.bak"

    if args.revert:
        if not os.path.exists(backup):
            print(f"[!!] 无备份，无法回滚：{backup}")
            return 2
        shutil.copy(backup, run_py)
        print(f"[OK] 已回滚：{backup} -> {run_py}")
        print(f"     注意：{sched_py} 保留无害（不被引用）")
        return 0

    src = open(run_py, "r", encoding="utf-8").read()
    patched = MARK in src

    if args.verify:
        ok_sched = os.path.exists(sched_py)
        if patched and ok_sched:
            print("[OK] 补丁已存在：run.py 已注入 + hpt/utils/fedvla_lr.py 已生成")
            return 0
        print(f"[!!] 补丁不完整：run.py patched={patched}  scheduler_exists={ok_sched}")
        return 1

    if patched:
        print("[OK] 补丁已存在，无需重复打（幂等）")
        if not os.path.exists(sched_py):
            os.makedirs(os.path.dirname(sched_py), exist_ok=True)
            with open(sched_py, "w", encoding="utf-8") as f:
                f.write(SCHEDULER_SRC)
            print(f"[OK] 补写调度器文件：{sched_py}")
        return 0

    # 写调度器文件
    os.makedirs(os.path.dirname(sched_py), exist_ok=True)
    with open(sched_py, "w", encoding="utf-8") as f:
        f.write(SCHEDULER_SRC)
    print(f"[OK] 已写入调度器：{sched_py}")

    if ANCHOR1 not in src:
        print("[!!] 找不到锚点 1（optimizer/scheduler 构造段），HPT 版本可能不同，请手工检查：")
        print(f"     {run_py}")
        return 3
    if ANCHOR2 not in src:
        print("[!!] 找不到锚点 2（训练结束保存段），HPT 版本可能不同，请手工检查：")
        print(f"     {run_py}")
        return 3

    if not os.path.exists(backup):
        shutil.copy(run_py, backup)
        print(f"[OK] 已备份 -> {backup}")

    src = src.replace(ANCHOR1, BLOCK1, 1)
    src = src.replace(ANCHOR2, BLOCK2, 1)

    with open(run_py, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[OK] 已打补丁：{run_py}")
    print("     1) 新增 hpt/utils/fedvla_lr.py（GlobalStepLRScheduler）")
    print("     2) run.py：FEDVLA_GLOBAL_LR=1 时用全局进度调度替换原生 WarmupLR+cosine")
    print("     3) run.py：FEDVLA_OPT_STATE 指定的路径用于 Adam 状态跨轮持久化")
    print()
    print("下一步（云端）：")
    print("  bash /root/HPT/federated/run_fedavg.sh   # 驱动端已带 FEDVLA_* 环境变量")
    print("  或手动单跑验证：")
    print("  FEDVLA_GLOBAL_LR=1 FEDVLA_P_START=0 FEDVLA_P_END=0.02 FEDVLA_NUM_STEPS=45 \\")
    print("    python -m hpt.run ... ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
