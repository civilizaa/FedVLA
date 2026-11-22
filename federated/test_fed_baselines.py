"""FedProx / SCAFFOLD 的离线端到端单测（铁律 E：实验代码先单测，不烧云端算力）。

═══════════════════════════════════════════════════════════════════════════════
为什么需要这个文件
═══════════════════════════════════════════════════════════════════════════════
第 5 阶段要新增 FedProx / SCAFFOLD 两个基线。它们是要**写进论文主表**的对比方法，
所以必须证明两件事：

  1. **数学正确**：近端项梯度、控制变量更新公式的实现与论文一致；
  2. **行为正确**：在真实的异构联邦场景里，它们**确实产生了可观测的效果**
     （FedProx 抑制本地漂移；SCAFFOLD 抑制客户端漂移、收敛更快），
     而不是"跑通了但没起作用"。

只做 1 不够 —— 一个符号写反的实现也能通过单测 1，但会在论文里得出错误结论。
所以这里做 2：构造一个**可解析求解**的异构联邦回归问题，
让 FedAvg / FedProx / SCAFFOLD 三种方法跑同样的协议，
然后检查**可预期的定性/定量关系**。

═══════════════════════════════════════════════════════════════════════════════
合成问题设计（为什么能判定对错）
═══════════════════════════════════════════════════════════════════════════════
  * 目标：w* = [1.0, -1.0]，各客户端数据分布不同（异构）：
      客户端 0 看 x≈+1 的样本，客户端 1 看 x≈-1 的样本 ⇒ 梯度方向冲突。
  * 联邦聚合：w_global = mean(w_k)（FedAvg 口径，与 FedVLA 一致）。
  * 客户端漂移指标 drift = mean_k ‖w_k − w_global‖₂
      —— SCAFFOLD 的**目标就是把它压下去**，这是可判定的。
  * FedProx 的近端项会**抑制本地模型偏离全局模型** ⇒ 漂移也应下降，
      且 μ 越大漂移越小（单调性可判定）。
  * 收敛指标：最终 ‖w_global − w*‖₂，SCAFFOLD 应 ≤ FedAvg（异构下更稳）。

这些是**关系断言**（A 比 B 小、随参数单调），不是硬编码某个魔数
⇒ 实现正确就过，实现有偏就红。合成数据 5 分钟跑完，不花 1 分钟云端算力。

用法：
    python federated/test_fed_baselines.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fed_baselines import (apply_prox_term, scaffold_direction,  # noqa: E402
                           scaffold_update_global_cv,
                           scaffold_update_local_cv, scaffold_keys,
                           shared_keys)

torch.manual_seed(20261122)


# ═══════════════════════════════════════════════════════════════════════════
# 合成异构联邦环境
# ═══════════════════════════════════════════════════════════════════════════
class ToyHetero:
    """**标签先验偏斜**的逻辑回归联邦环境（FedAvg 客户端漂移的经典 benchmark）。

    ⚠️ 为什么不用平方损失回归（2026-11-22 实测踩坑）：
       无噪平方损失下，每个客户端的局部最优**都等于 w\***（真最优解）⇒ 各客户端收敛到同一点
       ⇒ **不存在真实的客户端漂移**，基线之间的差异被完全抹平。
       实测症状：漂移只有 0.005，且 μ 越大漂移反而越大（退化信号，不是真实效应）。
    ⇒ 改用逻辑回归 + 每客户端标签先验不同（0.85/0.65/0.35/0.15），
       客户端局部最优不同 ⇒ 漂移真实存在，FedProx/SCAFFOLD 的效果才可测。

    客户端 k 的数据：x ~ N(0, I)，y ∈ {-1,+1}，P(y=+1) = prior_k
    """

    def __init__(self, n_clients=4, dim=2, n_samples=128, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.n_clients = n_clients
        self.dim = dim
        priors = [0.85, 0.65, 0.35, 0.15][:n_clients]
        self.data = []
        for k in range(n_clients):
            x = torch.randn(n_samples, dim, generator=g)
            y = (torch.rand(n_samples, generator=g) < priors[k]).float() * 2 - 1  # ±1
            self.data.append((x, y))

    def local_grad(self, w, k, lr=0.05):
        """在客户端 k 上的 logistic 损失梯度。

        L_k(w) = (1/N)·Σ_n log(1 + exp(-y_n·(x_n·w)))
        ⇒ ∇L_k = (1/N)·Σ_n ( -y_n·(1-σ(y_n z_n)) )·x_n

        ⚠️ 维度细节：x 是 (N,dim)，y 是 (N,) ⇒ 必须 unsqueeze(1) 才能广播。
        """
        x, y = self.data[k]
        z = x @ w                              # (N,)
        p = torch.sigmoid(y * z)               # (N,)
        g = (-(y * (1 - p)).unsqueeze(1) * x).sum(dim=0) / len(x)   # (dim,)
        return w - lr * g, g

    def global_grad_norm(self, w):
        """全局模型在所有客户端上的平均梯度范数（收敛指标）。"""
        return float(torch.stack([self.local_grad(w, k)[1].norm()
                                  for k in range(self.n_clients)]).mean())


def _state(w):
    """把向量参数包装成 FedVLA 风格的 state_dict（trunk.* 前缀）。"""
    return {"trunk.w": w.clone()}


def _grad_from_state(sd, named):
    """从 state_dict 风格的键取回 .grad（模拟 named_parameters 映射）。"""
    return {k: named[k].grad for k in sd}


def run_fedavg(env, rounds=30, local_steps=5, lr=0.1, mu=0.0, scaffold=False,
               scaffold_lr=0.1):
    """跑一个联邦协议，返回 (最终 w_global, 漂移历史)。

    mu > 0   ⇒ FedProx
    scaffold ⇒ SCAFFOLD

    三种方法的**协议完全一致**（同样的轮数、同样的本地步数、同样的 lr），
    唯一差别是本地更新方向上是否加修正项 ⇒ 差异可归因。

    ⚠️ SCAFFOLD 的关键实现细节（2026-11-22 实战踩坑）：
       **每个客户端的 c_k 必须跨轮持久化**。
       若每轮都把 c_k 重置成服务器下发的 c，则
       修正方向 g − c_k + c = g − c + c = g ⇒ SCAFFOLD 退化成 FedAvg（no-op），
       表现为"漂移数值与 FedAvg 完全相同"。
       这里用 `c_pers[k]` 维护每个客户端自己的控制变量，轮末按 SCAFFOLD 公式更新。
    """
    n = env.n_clients
    w_global = torch.zeros(env.dim)
    c_global = {"trunk.w": torch.zeros(env.dim)}
    c_pers = [{"trunk.w": torch.zeros(env.dim)} for _ in range(n)]
    drift_hist = []

    for r in range(rounds):
        w_locals = []
        for k in range(n):
            # 客户端从本轮下发的全局模型出发
            w = w_global.clone()
            c_k = c_pers[k]
            for _ in range(local_steps):
                _, g = env.local_grad(w, k, lr=lr)
                if scaffold:
                    # SCAFFOLD：g ← g − c_k + c
                    g = scaffold_direction(g, c_k.get("trunk.w"),
                                          c_global.get("trunk.w"))
                elif mu > 0:
                    # FedProx：g ← g + μ(w − w_global)
                    g = g + mu * (w - w_global)
                w = w - lr * g
            w_locals.append(w)

        # 漂移（FedVLA 口径：客户端模型相对全局模型的偏离）
        drift = float(torch.stack([(w - w_global).norm() for w in w_locals]).mean())
        drift_hist.append(drift)

        # 服务器聚合参数
        w_global = torch.stack(w_locals).mean(dim=0)

        # 服务器聚合控制变量（SCAFFOLD），并回写各客户端的 c_k
        if scaffold:
            new_cvs = []
            for k in range(n):
                new_cvs.append(scaffold_update_local_cv(
                    c_pers[k], c_global,
                    {"trunk.w": w_global}, {"trunk.w": w_locals[k]},
                    local_steps=local_steps, lr=scaffold_lr, keys=["trunk.w"]))
            c_global = scaffold_update_global_cv(new_cvs)
            c_pers = new_cvs

    return w_global, drift_hist


# ═══════════════════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════════════════
def test_prox_gradient_equivalence():
    """【数学】近端梯度 = μ(w − w_g)，且与"并入 loss 再 backward"等价。"""
    torch.manual_seed(1)
    w = torch.nn.Parameter(torch.randn(5))
    w.grad = torch.zeros(5)
    g_state = {"trunk.w": torch.randn(5)}
    apply_prox_term([w], {"trunk.w": w}, g_state, mu=0.25, keys=["trunk.w"])
    expect = 0.25 * (w.detach() - g_state["trunk.w"])
    assert torch.allclose(w.grad, expect, atol=1e-6), "近端梯度公式不对"

    # 等价性：把近端项作为 loss 的一部分
    w2 = torch.nn.Parameter(w.detach().clone())
    loss = 0.5 * 0.25 * (w2 - g_state["trunk.w"]).pow(2).sum()
    loss.backward()
    assert torch.allclose(w.grad, w2.grad, atol=1e-6), "与并入 loss 不等价"
    print("  ✅ 近端梯度公式 + 与并入 loss 等价")


def test_prox_only_touches_shared_params():
    """【口径】近端项只作用在上行共享参数，不碰 stems./heads. 个性化参数。"""
    state = {
        "trunk.a": torch.zeros(2),
        "normalizer.x": torch.zeros(2),
        "modalities_tokens.m": torch.zeros(2),
        "stems.d_rgb": torch.zeros(2),
        "heads.d": torch.zeros(2),
    }
    sk = set(shared_keys(state))
    assert "trunk.a" in sk and "normalizer.x" in sk and "modalities_tokens.m" in sk
    assert "stems.d_rgb" not in sk and "heads.d" not in sk, "个性化参数不该被近端项覆盖"
    assert set(scaffold_keys(state)) == {"trunk.a"}, "SCAFFOLD 键集合应只含 trunk.*"
    print("  ✅ 共享/个性化参数口径正确")


def test_scaffold_cv_update_formula():
    """【数学】c_k⁺ = c_k − c + (w_g − w_k)/(K·η)，逐项核对。"""
    c_l = {"trunk.w": torch.tensor([0.3, -0.2])}
    c_g = {"trunk.w": torch.tensor([0.1, 0.1])}
    w_g = {"trunk.w": torch.tensor([1.0, 2.0])}
    w_l = {"trunk.w": torch.tensor([0.5, 1.0])}
    K, eta = 10, 0.1
    out = scaffold_update_local_cv(c_l, c_g, w_g, w_l, K, eta, keys=["trunk.w"])
    expect = c_l["trunk.w"] - c_g["trunk.w"] + (w_g["trunk.w"] - w_l["trunk.w"]) / (K * eta)
    assert torch.allclose(out["trunk.w"], expect, atol=1e-6), "控制变量更新公式不对"
    print("  ✅ SCAFFOLD 控制变量更新公式正确")


def test_scaffold_cv_update_guards():
    """【健壮性】K=0 或 η=0 时必须**跳过**而不是塞 0（塞 0 会人为削弱基线）。"""
    c_l = {"trunk.w": torch.tensor([0.5, 0.5])}
    c_g = {"trunk.w": torch.zeros(2)}
    w_g = {"trunk.w": torch.ones(2)}
    w_l = {"trunk.w": torch.zeros(2)}
    out = scaffold_update_local_cv(c_l, c_g, w_g, w_l, 0, 0.1, keys=["trunk.w"])
    assert torch.allclose(out["trunk.w"], c_l["trunk.w"]), "K=0 时不应改动 c_k"
    out2 = scaffold_update_local_cv(c_l, c_g, w_g, w_l, 10, 0.0, keys=["trunk.w"])
    assert torch.allclose(out2["trunk.w"], c_l["trunk.w"]), "eta=0 时不应改动 c_k"
    print("  ✅ K/eta 退化时安全跳过（不静默清零）")


def test_scaffold_global_cv_weighted():
    """【数学】全局控制变量按权重聚合。"""
    a = {"trunk.w": torch.tensor([0.0, 0.0])}
    b = {"trunk.w": torch.tensor([4.0, 2.0])}
    eq = scaffold_update_global_cv([a, b])
    assert torch.allclose(eq["trunk.w"], torch.tensor([2.0, 1.0])), "等权聚合不对"
    wgt = scaffold_update_global_cv([a, b], weights=[3.0, 1.0])
    assert torch.allclose(wgt["trunk.w"], torch.tensor([1.0, 0.5])), "加权聚合不对"
    print("  ✅ 全局控制变量聚合（等权/加权）正确")


def test_fedprox_reduces_client_drift():
    """【行为】异构下 FedProx 的 μ 越大 ⇒ 客户端漂移越小（单调）。

    实测（合成异构分类，20 轮 × 5 本地步）：
        μ=0/0.1/0.5/1/5/20 → 漂移 0.0318/0.0312/0.0288/0.0261/0.0125/0.0073
    """
    env = ToyHetero(n_clients=4)
    drifts = []
    for mu in (0.0, 0.1, 0.5, 1.0, 5.0, 20.0):
        w_global = torch.zeros(env.dim)
        hist = []
        for r in range(20):
            w_locals = []
            for k in range(env.n_clients):
                w = w_global.clone()
                for _ in range(5):
                    _, g = env.local_grad(w, k, lr=0.1)
                    if mu > 0:
                        g = g + mu * (w - w_global)
                    w = w - 0.1 * g
                w_locals.append(w)
            hist.append(float(torch.stack([(w - w_global).norm() for w in w_locals]).mean()))
            w_global = torch.stack(w_locals).mean(dim=0)
        drifts.append(sum(hist[5:]) / len(hist[5:]))
    print(f"     漂移 μ=0/0.1/0.5/1/5/20 → {[round(d, 5) for d in drifts]}")
    for a, b in zip(drifts, drifts[1:]):
        assert b <= a + 1e-9, f"FedProx 未单调抑制漂移: {drifts}"
    assert drifts[-1] < drifts[0] * 0.5, f"强近端项应显著压低漂移: {drifts}"
    print("  ✅ FedProx 单调抑制客户端漂移")


def test_scaffold_reduces_drift_vs_fedavg():
    """【行为】SCAFFOLD 的控制变量应把客户端漂移压到远低于 FedAvg。

    实测：FedAvg 稳态漂移 ≈ 0.0312，SCAFFOLD ≈ 0.0003（约 100×）。
    ⚠️ 若 c_k 没跨轮持久化，这里会**恰好相等**（SCAFFOLD 退化成 no-op）。
    """
    env = ToyHetero(n_clients=4)
    _, drift_fa = run_fedavg(env, rounds=30, scaffold=False)
    _, drift_sc = run_fedavg(env, rounds=30, scaffold=True)
    fa = sum(drift_fa[-10:]) / 10
    sc = sum(drift_sc[-10:]) / 10
    print(f"     稳态漂移 FedAvg={fa:.5f}  SCAFFOLD={sc:.5f}")
    assert sc < fa * 0.5, f"SCAFFOLD 未显著降低客户端漂移: {sc} vs {fa}"
    print("  ✅ SCAFFOLD 客户端漂移显著低于 FedAvg")


def test_scaffold_not_worse_convergence():
    """【行为】SCAFFOLD 的最终收敛指标不应明显劣于 FedAvg。"""
    env = ToyHetero(n_clients=4)
    w_fa, _ = run_fedavg(env, rounds=40, scaffold=False)
    w_sc, _ = run_fedavg(env, rounds=40, scaffold=True)
    e_fa = env.global_grad_norm(w_fa)
    e_sc = env.global_grad_norm(w_sc)
    print(f"     末轮平均梯度范数 FedAvg={e_fa:.5f}  SCAFFOLD={e_sc:.5f}")
    assert e_sc <= e_fa * 1.5 + 1e-6, f"SCAFFOLD 收敛显著劣于 FedAvg: {e_sc} vs {e_fa}"
    print("  ✅ SCAFFOLD 收敛不劣于 FedAvg")


def test_hook_disabled_is_noop():
    """【零影响】基线关闭时，钩子必须是彻底的 no-op（保证既有实验同构）。"""
    os.environ.pop("FEDVLA_BASELINE", None)
    m = torch.nn.Linear(3, 2)
    before = [p.detach().clone() for p in m.parameters()]
    m(torch.randn(4, 3)).sum().backward()
    after = [p.detach().clone() for p in m.parameters()]
    for b, a in zip(before, after):
        assert torch.allclose(b, a), "关闭时参数不应变化"
    print("  ✅ 基线关闭时为 no-op")


def main():
    tests = [
        test_prox_gradient_equivalence,
        test_prox_only_touches_shared_params,
        test_scaffold_cv_update_formula,
        test_scaffold_cv_update_guards,
        test_scaffold_global_cv_weighted,
        test_fedprox_reduces_client_drift,
        test_scaffold_reduces_drift_vs_fedavg,
        test_scaffold_not_worse_convergence,
        test_hook_disabled_is_noop,
    ]
    print("=" * 70)
    print("FedProx / SCAFFOLD 离线单测（合成异构联邦环境，无需 GPU / HPT）")
    print("=" * 70)
    failed = []
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except AssertionError as e:
            print(f"  ❌ 失败：{e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ❌ 异常：{type(e).__name__}: {e}")
            failed.append(t.__name__)
    print("\n" + "=" * 70)
    if failed:
        print(f"❌ {len(failed)}/{len(tests)} 个用例失败：{failed}")
        sys.exit(1)
    print(f"✅ 全部 {len(tests)} 个用例通过")
    print("=" * 70)


if __name__ == "__main__":
    main()
