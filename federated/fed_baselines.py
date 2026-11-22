"""FedVLA 第 5 阶段：联邦基线方法（FedProx / SCAFFOLD）的核心数学。

═══════════════════════════════════════════════════════════════════════════════
为什么需要这个文件
═══════════════════════════════════════════════════════════════════════════════
论文路线图 6.1「主实验矩阵」要求与 6 个方法对比：

    Centralized / FedAvg / **FedProx** / **SCAFFOLD** / EDA / 按簇聚合

第 3、4 阶段已实现并验证了 FedAvg、EDA、簇感知；
**FedProx 与 SCAFFOLD 此前完全缺失**（全仓 grep 零命中）——
而这两个恰恰是通信评委最熟悉的联邦基线，缺了会被直接质疑"你的方法比 FedAvg 好，
但比 FedProx/SCAFFOLD 好吗？"

本文件把两者的数学抽成**纯逻辑函数**（只依赖 torch，不依赖 HPT、不依赖 GPU），
好处（铁律 E：实验代码先离线单测）：
  * 数学正确性可以在**合成数据**上 5 分钟验完，不用烧 6 小时云端算力；
  * HPT 侧只需要一个极薄的胶水层（patch），把这里的函数接到训练循环里；
  * 两个方法共享同一套「共享参数键」定义，不会与 aggregate.py 的口径漂移。

═══════════════════════════════════════════════════════════════════════════════
FedProx（Li et al., MLSys 2020）
═══════════════════════════════════════════════════════════════════════════════
把 FedAvg 的本地目标从 min_w F_k(w) 改成

    min_w  F_k(w) + (μ/2)·‖w − w_global‖²

其中 w_global 是**本轮开始时**服务器下发的全局模型，μ 是近端强度。
梯度：∇ = ∇F_k(w) + μ·(w − w_global)

直觉：限制本地模型偏离全局模型的幅度 ⇒ 异构下更稳，通信轮次间更平滑。

实现要点（本作特有）：
  * FedVLA 的「共享参数」是 trunk.* + normalizer.* + modalities_tokens.*，
    而 stems.*/heads.* 是**个性化**的（各客户端域不同，不可跨客户端平均）。
    ⇒ 近端项**只作用于共享参数**：个性化部分往全局模型拉是错的
    （全局模型里根本没有本客户端的域专属 head）。
  * 这个键集合必须与 aggregate.py 的 `_is_shared_param` **逐字一致**，
    否则会出现"聚合时算共享、近端时算私有"的错配（静默失真）。
    ⇒ 本文件从 aggregate.py **直接 import** 该函数，不复制粘贴。

═══════════════════════════════════════════════════════════════════════════════
SCAFFOLD（Karimireddy et al., ICML 2020）
═══════════════════════════════════════════════════════════════════════════════
用控制变量修正**客户端漂移**：

  * 服务器持有全局控制变量 c；每个客户端 k 持有本地控制变量 c_k。
  * 本地更新用的方向：  g_k ← g_k − c_k + c      （把漂移减掉）
  * 本地训练结束后更新本地控制变量：
        c_k⁺ = c_k − c + (w_global − w_k)/(K·η)
    其中 K = 本地步数，η = 学习率。
  * 服务器聚合：  c⁺ = (1/N)·Σ_k c_k⁺        （等权）
    再加上标准的参数聚合：  w⁺ = (1/N)·Σ_k w_k

实现要点（本作特有）：
  * 控制变量的键集合必须与**参与聚合的参数**完全一致 —— 本作聚合的是
    trunk.*（见 aggregate.py 的 average_trunk），所以控制变量也只覆盖 trunk.*
    （normalizer/modalities_tokens 在 aggregate 里是"同名同形才平均"，
     跨异构客户端形状不一致，套控制变量会引入形状错误 ⇒ 排除。）
  * η（学习率）在 HPT 里是**随全局进度变化的调度器输出**，不是常数。
    论文原文公式假定常数 η。作 faithful 实现：取**本地最后一轮的平均 lr**
    （由训练侧统计并落盘），而不是拍一个常数。
  * 初始 c_k = 0（等价于标准 SCAFFOLD 的零初始化）。

═══════════════════════════════════════════════════════════════════════════════
与既有 aggregate.py 的关系
═══════════════════════════════════════════════════════════════════════════════
  * FedProx / SCAFFOLD **不改变服务器的聚合权重**（仍按现有 FEDAVG_WEIGHTS 口径），
    它们改的是**客户端本地更新**。所以 aggregate.py 不需要新分支，
    只需要在 SCAFFOLD 模式下额外聚合控制变量（见 aggregate_control_variates）。
  * 本文件对 aggregate.py 是**只读依赖**（import `_is_shared_param`），
    不产生循环 import：aggregate.py 不 import 本文件。

作者：FedVLA 实验脚本
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, Sequence

import torch

# 与 aggregate.py 同目录，直接复用它的共享参数判定（避免口径漂移）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import _is_shared_param  # noqa: E402

__all__ = [
    "shared_keys",
    "scaffold_keys",
    "apply_prox_term",
    "prox_penalty",
    "scaffold_direction",
    "scaffold_update_local_cv",
    "scaffold_update_global_cv",
    "aggregate_control_variates",
]


# ═══════════════════════════════════════════════════════════════════════════
# 键集合定义
# ═══════════════════════════════════════════════════════════════════════════
def shared_keys(state: dict) -> list[str]:
    """上行共享参数的键集合（会参与跨客户端平均的那些）。

    直接复用 aggregate._is_shared_param：trunk.* / normalizer.* / modalities_tokens.*
    （排除 stems.* / heads.* —— 那是域专属的个性化参数）。

    ⚠️ 必须与 aggregate.py 用**同一个函数**：若这里自己写一份正则，
    迟早会出现"聚合按 A 口径、近端按 B 口径"的静默失真。
    """
    return [k for k in state.keys() if _is_shared_param(k)]


def scaffold_keys(state: dict) -> list[str]:
    """SCAFFOLD 控制变量覆盖的键集合 = **参与参数聚合的键**。

    aggregate.py 的参数聚合（average_trunk）只平均 `trunk.` 前缀的键；
    normalizer.* / modalities_tokens.* 走的是"同名同形才平均"分支，
    异构客户端之间形状不一致 ⇒ 套控制变量必然形状错配。

    ⇒ 控制变量**只覆盖 trunk.***，与参数聚合口径逐字一致。
    """
    return [k for k in state.keys() if k.startswith("trunk.")]


# ═══════════════════════════════════════════════════════════════════════════
# FedProx
# ═══════════════════════════════════════════════════════════════════════════
def prox_penalty(state: dict, global_state: dict, mu: float,
                 keys: Sequence[str] | None = None) -> torch.Tensor:
    """近端项数值： (μ/2)·Σ‖w − w_global‖²。

    纯诊断用（打印"本地模型偏离全局多远"）；训练时用 apply_prox_term 走梯度。
    """
    if mu <= 0:
        return torch.zeros((), dtype=torch.float64)
    ks = list(keys) if keys is not None else shared_keys(state)
    total = torch.zeros((), dtype=torch.float64)
    for k in ks:
        if k not in global_state:
            continue
        w = state[k].detach().float()
        wg = global_state[k].detach().float()
        total = total + (w - wg).pow(2).sum().double()
    return 0.5 * mu * total


def apply_prox_term(params: Iterable[torch.nn.Parameter],
                    named_params: dict[str, torch.nn.Parameter],
                    global_state: dict,
                    mu: float,
                    keys: Sequence[str] | None = None) -> float:
    """把近端项的梯度 μ·(w − w_global) 加到 .grad 上。

    HPT 的训练循环（hpt/train_test.py::train）里是：

        domain_loss = model.compute_loss(batch)
        optimizer.zero_grad()
        domain_loss.backward()
        optimizer.step()

    而 `model.compute_loss` 在 policy.py 里，我们**不打算改 HPT 的模型定义**
    （改一处 loss 就得重跑全部历史实验，违反"同构"要求）。

    ⇒ 正确做法：在 `backward()` 之后、`optimizer.step()` 之前，
      往 .grad 上加 μ(w − w_global)。这在数学上**完全等价**于
      把近端项并入 loss（因为 ∂/∂w [ (μ/2)‖w−w_g‖² ] = μ(w−w_g)），
      且不需要碰 HPT 的模型/损失代码 ⇒ 对既有实验零影响。

    返回本次加进去的近端梯度范数（用于日志/自检）。
    """
    if mu <= 0:
        return 0.0
    ks = set(keys) if keys is not None else set(shared_keys(global_state))
    total_sq = 0.0
    for name, p in named_params.items():
        if name not in ks:
            continue
        if p.grad is None:
            continue
        if name not in global_state:
            continue
        wg = global_state[name].detach().to(p.device, dtype=p.dtype)
        g = mu * (p.detach() - wg)
        p.grad.add_(g)
        total_sq += float(g.detach().float().pow(2).sum())
    return total_sq ** 0.5


# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD
# ═══════════════════════════════════════════════════════════════════════════
def scaffold_direction(grad: torch.Tensor, c_local: torch.Tensor | None,
                       c_global: torch.Tensor | None) -> torch.Tensor:
    """SCAFFOLD 修正后的更新方向：  g ← g − c_k + c。

    c_k / c 为 None 时按 0 处理（零初始化，等价于退化成 FedAvg 的第一轮）。
    """
    out = grad
    if c_local is not None:
        out = out - c_local.to(out.device, out.dtype)
    if c_global is not None:
        out = out + c_global.to(out.device, out.dtype)
    return out


def scaffold_update_local_cv(c_local: dict[str, torch.Tensor],
                             c_global: dict[str, torch.Tensor],
                             w_global: dict[str, torch.Tensor],
                             w_local: dict[str, torch.Tensor],
                             local_steps: int,
                             lr: float,
                             keys: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
    """客户端侧本地控制变量更新：

        c_k⁺ = c_k − c + (w_global − w_k) / (K·η)

    参数
    ----
    c_local / c_global : 控制变量（形状与对应参数一致）
    w_global           : 本轮开始时服务器下发的全局参数
    w_local            : 本地训练结束后的参数
    local_steps (K)    : 本地实际执行的优化步数
    lr (η)             : 学习率（本作是调度器输出，取本地平均值）

    ⚠️ 若 K·η ≤ 0 或参数缺失 → **跳过该键**（保持原 c_k），
       而不是塞 0 —— 塞 0 会让控制变量突然归零，等效于人为削弱 SCAFFOLD。
    """
    if local_steps <= 0 or lr <= 0:
        return {k: v.clone() for k, v in c_local.items()}
    ks = list(keys) if keys is not None else list(c_local.keys())
    out: dict[str, torch.Tensor] = {}
    denom = float(local_steps) * float(lr)
    for k in ks:
        if k not in c_global or k not in w_global or k not in w_local:
            out[k] = c_local[k].clone()
            continue
        c_k = c_local[k]
        c = c_global[k]
        w_g = w_global[k].detach().float()
        w_l = w_local[k].detach().float()
        out[k] = (c_k.float() - c.float() + (w_g - w_l) / denom).to(c_k.dtype)
    return out


def scaffold_update_global_cv(local_cvs: Sequence[dict[str, torch.Tensor]],
                              weights: Sequence[float] | None = None
                              ) -> dict[str, torch.Tensor]:
    """服务器侧全局控制变量聚合：  c⁺ = Σ_k p_k · c_k⁺  （p_k 归一化）。

    weights 为 None → 等权（标准 SCAFFOLD）。
    """
    if not local_cvs:
        return {}
    if weights is None:
        w = [1.0] * len(local_cvs)
    else:
        if len(weights) != len(local_cvs):
            raise ValueError(f"weights 长度 {len(weights)} != 客户端数 {len(local_cvs)}")
        w = [float(x) for x in weights]
    s = sum(w)
    if s <= 0:
        raise ValueError("控制变量聚合权重之和必须 > 0")
    w = [x / s for x in w]
    out: dict[str, torch.Tensor] = {}
    for k in local_cvs[0].keys():
        acc = None
        for i, cv in enumerate(local_cvs):
            if k not in cv:
                continue
            term = cv[k].detach().float() * w[i]
            acc = term if acc is None else acc + term
        if acc is not None:
            out[k] = acc
    return out


def aggregate_control_variates(client_dirs: Sequence[str],
                               out_path: str,
                               weights: Sequence[float] | None = None,
                               filename: str = "scaffold_cv.pt") -> dict:
    """从各客户端目录读 `scaffold_cv.pt`，聚合成全局控制变量并落盘。

    每个客户端把自己的**本地控制变量**（已按 SCAFFOLD 公式更新过）写在自己的
    输出目录里（`scaffold_cv.pt`），服务器读完聚合成全局的，下一轮下发。

    返回统计信息（键数、范数），用于日志与自检。
    """
    import json

    cvs = []
    for d in client_dirs:
        p = os.path.join(d, filename)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"客户端 {d} 缺 {filename} —— SCAFFOLD 模式要求每轮结束时导出本地控制变量。"
                f"\n  检查：客户端训练后是否执行了 export_control_variates 步骤。")
        blob = torch.load(p, map_location="cpu", weights_only=False)
        cv = blob.get("cv", blob) if isinstance(blob, dict) else blob
        cvs.append(cv)

    g = scaffold_update_global_cv(cvs, weights)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save({"cv": g}, out_path)
    norm = float(sum(float(v.detach().float().pow(2).sum()) for v in g.values()) ** 0.5)
    info = {"n_keys": len(g), "norm": norm, "out": out_path,
            "n_clients": len(cvs)}
    print(f"[scaffold] 控制变量聚合：{len(cvs)} 客户端 × {len(g)} 键，"
          f"‖c‖₂ = {norm:.4e} -> {out_path}")
    return info


def _cli_aggregate_cv(args):
    """服务端：聚合并下发全局控制变量 c。"""
    weights = None
    if args.weights:
        weights = [float(x) for x in args.weights]
        if len(weights) != len(args.clients):
            raise SystemExit(f"[FATAL] weights 数量({len(weights)}) != 客户端数({len(args.clients)})")
    # 客户端的本地控制变量文件由训练侧写到 <client_dir>/scaffold_cv.pt
    cvs = []
    for d in args.clients:
        p = os.path.join(d, "scaffold_cv.pt")
        if not os.path.exists(p):
            raise SystemExit(
                f"[FATAL] 客户端 {d} 缺 scaffold_cv.pt —— SCAFFOLD 模式要求每个客户端在"
                f"本地训练结束后导出控制变量。\n"
                f"  检查：HPT 基线补丁是否已打（patch_hpt_fedbaseline.py）"
                f"、FEDVLA_SCAFFOLD_CV_OUT 是否设置。")
        blob = torch.load(p, map_location="cpu", weights_only=False)
        cvs.append(blob.get("cv", blob) if isinstance(blob, dict) else blob)
    g = scaffold_update_global_cv(cvs, weights)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save({"cv": g}, args.out)
    norm = float(sum(float(v.detach().float().pow(2).sum()) for v in g.values()) ** 0.5)
    print(f"[scaffold] 全局控制变量聚合完成：{len(cvs)} 客户端 × {len(g)} 键，"
          f"‖c‖₂={norm:.4e} -> {args.out}")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="FedVLA 联邦基线工具（FedProx / SCAFFOLD）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("aggregate-cv", help="聚合各客户端的本地控制变量 → 全局 c")
    p1.add_argument("--clients", nargs="+", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--weights", nargs="*", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "aggregate-cv":
        _cli_aggregate_cv(args)


def _selftest() -> None:
    """最小自检（可离线跑，无 GPU、无 HPT）。"""
    torch.manual_seed(0)
    # FedProx：近端梯度应等于 μ(w − w_g)
    w = torch.nn.Parameter(torch.randn(4))
    w.grad = torch.zeros(4)
    g_state = {"trunk.a": torch.randn(4)}
    apply_prox_term([w], {"trunk.a": w}, g_state, mu=0.1, keys=["trunk.a"])
    expect = 0.1 * (w.detach() - g_state["trunk.a"])
    assert torch.allclose(w.grad, expect), (w.grad, expect)

    # SCAFFOLD：方向修正
    g = torch.ones(3)
    c_l = torch.full((3,), 0.5)
    c_g = torch.full((3,), 0.2)
    assert torch.allclose(scaffold_direction(g, c_l, c_g), torch.full((3,), 0.7))

    # SCAFFOLD：本地控制变量更新
    c_local = {"trunk.a": torch.zeros(2)}
    c_global = {"trunk.a": torch.zeros(2)}
    w_global = {"trunk.a": torch.tensor([1.0, 2.0])}
    w_local = {"trunk.a": torch.tensor([0.0, 0.0])}
    new = scaffold_update_local_cv(c_local, c_global, w_global, w_local,
                                   local_steps=10, lr=0.1, keys=["trunk.a"])
    assert torch.allclose(new["trunk.a"], torch.tensor([1.0, 2.0]) / 1.0)

    # SCAFFOLD：全局控制变量聚合（等权）
    a = {"trunk.a": torch.tensor([1.0, 0.0])}
    b = {"trunk.a": torch.tensor([3.0, 0.0])}
    agg = scaffold_update_global_cv([a, b])
    assert torch.allclose(agg["trunk.a"], torch.tensor([2.0, 0.0]))
    print("[selftest] fed_baselines 自检通过 ✅")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate-cv":
        main()
    else:
        _selftest()
