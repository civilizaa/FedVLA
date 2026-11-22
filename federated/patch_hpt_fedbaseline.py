"""给 HPT 打「联邦基线（FedProx / SCAFFOLD）」补丁（幂等，可验证、可回滚）。

═══════════════════════════════════════════════════════════════════════════════
为什么需要打补丁
═══════════════════════════════════════════════════════════════════════════════
FedProx 与 SCAFFOLD 都作用在**本地优化的梯度**上：

  * FedProx：往梯度里加 μ(w − w_global)
  * SCAFFOLD：把梯度方向改成 g − c_k + c

而 HPT 的训练循环在 `hpt/train_test.py::train()` 里：

    domain_loss = model.compute_loss(batch)
    optimizer.zero_grad()
    domain_loss.backward()          ← 梯度在这里产生
                                     ← 【补丁插入点】
    optimizer.step()                ← 梯度在这里被应用

数学上，在 `backward()` 之后、`optimizer.step()` 之前修改 `.grad`，
与"把修正项并入 loss 再 backward"**完全等价**（因为两者都是加一个常数项到梯度）。

⇒ 因此**不需要改 HPT 的模型定义、不需要改 compute_loss**。
  这点很重要：HPT 的 compute_loss 在 policy.py 里，改它就等于改所有历史实验的
  损失函数 ⇒ 违反"跨批次实验同构"（铁律 D），已跑完的 4 个阶段全部作废。

本补丁只做两件事：
  1. 在 `backward()` 与 `optimizer.step()` 之间插入一个**钩子调用**；
  2. 在 `train()` 结尾把 SCAFFOLD 的本地控制变量更新所需信息落盘。

默认完全关闭 ⇒ 补丁存在但不开功能时，训练行为与不打补丁**逐字节一致**。

═══════════════════════════════════════════════════════════════════════════════
环境变量开关（默认全关 ⇒ 零影响）
═══════════════════════════════════════════════════════════════════════════════
  FEDVLA_BASELINE          = "" | "fedprox" | "scaffold"
  FEDVLA_PROX_MU           = 近端强度 μ（默认 0.01；仅 fedprox 用）
  FEDVLA_PROX_ANCHOR       = w_global 的落盘路径（.pt），由 run_fedavg.sh 写入
  FEDVLA_SCAFFOLD_C        = 全局控制变量 c 的路径（.pt）
  FEDVLA_SCAFFOLD_CV_OUT   = 本地控制变量的输出路径（.pt）
  FEDVLA_SCAFFOLD_WGLOBAL  = 本轮起始全局参数的路径（.pt），用于 c_k 更新
  FEDVLA_SCAFFOLD_STEPS    = 本地实际步数 K
  FEDVLA_SCAFFOLD_LR       = 本地平均学习率 η

═══════════════════════════════════════════════════════════════════════════════
用法（在 AutoDL 上）
═══════════════════════════════════════════════════════════════════════════════
    python /root/HPT/federated/patch_hpt_fedbaseline.py            # 打补丁
    python /root/HPT/federated/patch_hpt_fedbaseline.py --verify    # 只检查
    python /root/HPT/federated/patch_hpt_fedbaseline.py --revert    # 回滚
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

MARK = "[FedVLA-BASE]"

# ── 补丁锚点：train() 里 backward 与 step 之间 ──────────────────────────
ANCHOR = """        domain_loss.backward()

        optimizer.step()
"""

BLOCK = """        domain_loss.backward()

        # [FedVLA-BASE] 联邦基线梯度修正钩子（默认关闭 ⇒ 行为与原版一致）
        try:
            from hpt.utils.fedvla_baseline import apply_baseline_hook
            apply_baseline_hook(model, opt)
        except Exception as _be:
            print("[FedVLA-BASE] hook skipped:", type(_be).__name__, _be)

        optimizer.step()
"""

# ── 补丁锚点 2：train() 结尾（导出 SCAFFOLD 本地控制变量）───────────────
ANCHOR2 = """    return {k: np.mean(v) for k, v in info_log.items() if len(v) > 1}
"""

BLOCK2 = """    # [FedVLA-BASE] SCAFFOLD：本地训练结束后更新本地控制变量 c_k
    try:
        from hpt.utils.fedvla_baseline import finish_baseline_round
        finish_baseline_round(model)
    except Exception as _be:
        print("[FedVLA-BASE] finish skipped:", type(_be).__name__, _be)

    return {k: np.mean(v) for k, v in info_log.items() if len(v) > 1}
"""

# ── 新文件：hpt/utils/fedvla_baseline.py ────────────────────────────────
# 钩子代码放在独立文件 fedvla_baseline_hook.py 里（不内嵌字符串）——
# 内嵌在补丁脚本里会造成三引号嵌套地狱，且无法被语法检查/lint 覆盖。
HOOK_SRC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fedvla_baseline_hook.py")


def _apply(path: str, anchor: str, block: str, tag: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if block in src:
        print(f"  [skip] {tag}：已打过补丁")
        return False
    if anchor not in src:
        raise SystemExit(
            f"[FATAL] {tag}：在 {path} 里找不到锚点。\n"
            f"  HPT 版本可能变了 ⇒ **必须先核对再改**，绝不能盲改（否则打错补丁不报错）。\n"
            f"  锚点原文：\n{anchor}")
    src = src.replace(anchor, block, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  [ok] {tag}：已注入 {path}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="给 HPT 打 FedProx/SCAFFOLD 补丁")
    ap.add_argument("--hpt-dir", default=os.path.join(os.path.expanduser("~"), "HPT"))
    ap.add_argument("--verify", action="store_true", help="只检查，不改文件")
    ap.add_argument("--revert", action="store_true", help="回滚补丁")
    args = ap.parse_args(argv)

    hpt_dir = os.path.abspath(args.hpt_dir)
    train_test = os.path.join(hpt_dir, "hpt", "train_test.py")
    utils_dir = os.path.join(hpt_dir, "hpt", "utils")
    hook_file = os.path.join(utils_dir, "fedvla_baseline.py")

    if not os.path.exists(train_test):
        raise SystemExit(f"[FATAL] 找不到 {train_test}（--hpt-dir 是不是指错了？）")

    if args.revert:
        if not os.path.exists(hook_file + ".bak"):
            print("[revert] 没有备份文件，无需回滚")
            return
        shutil.copy(hook_file + ".bak", hook_file)
        shutil.copy(train_test + ".bak", train_test)
        print("[revert] 已回滚")
        return

    if args.verify:
        with open(train_test, "r", encoding="utf-8") as f:
            src = f.read()
        ok1 = BLOCK in src
        ok2 = BLOCK2 in src
        ok3 = os.path.exists(hook_file)
        print(f"[verify] train_test 钩子1: {'✅' if ok1 else '❌'}")
        print(f"[verify] train_test 钩子2: {'✅' if ok2 else '❌'}")
        print(f"[verify] fedvla_baseline.py: {'✅' if ok3 else '❌'}")
        sys.exit(0 if (ok1 and ok2 and ok3) else 1)

    print(f"[FedVLA-BASE] 打补丁到 {hpt_dir}")
    os.makedirs(utils_dir, exist_ok=True)
    if not os.path.exists(train_test + ".bak"):
        shutil.copy(train_test, train_test + ".bak")
    if not os.path.exists(hook_file + ".bak"):
        shutil.copy(hook_file, hook_file + ".bak") if os.path.exists(hook_file) else None

    # 写钩子模块（从独立文件复制，避免嵌套三引号）
    if os.path.exists(hook_file):
        print("  [skip] fedvla_baseline.py：已存在")
    else:
        if not os.path.exists(HOOK_SRC_FILE):
            raise SystemExit(f"[FATAL] 找不到钩子源文件 {HOOK_SRC_FILE}")
        shutil.copy(HOOK_SRC_FILE, hook_file)
        print(f"  [ok] 写入 {hook_file}（源：{os.path.basename(HOOK_SRC_FILE)}）")

    _apply(train_test, ANCHOR, BLOCK, "梯度修正钩子")
    _apply(train_test, ANCHOR2, BLOCK2, "控制变量导出")

    # 自检：导入不能炸
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("fedvla_baseline", hook_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("  [ok] fedvla_baseline.py 导入自检通过")
    except Exception as e:
        raise SystemExit(f"[FATAL] 钩子模块导入失败：{type(e).__name__}: {e}")

    print("[FedVLA-BASE] 补丁完成 ✅（默认关闭，需 FEDVLA_BASELINE 显式开启）")


if __name__ == "__main__":
    main()
