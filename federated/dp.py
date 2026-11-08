#!/usr/bin/env python3
"""FedVLA 第 4 阶段隐私侧：DP-SGD（模态感知预算分配）。

## 背景与威胁模型

第 4 章要回答的问题：**上行更新泄露了多少隐私？加多少噪声能挡住梯度反演？**

威胁模型（论文 6.3 节）：
- 服务器"诚实但好奇"（honest-but-curious）：遵守协议，但会分析收到的上行更新。
- 攻击者能力：拿到**被 DP 噪声污染后的**上行更新（不是原始梯度），
  试图重建客户端的私有观测（图像）。
- 攻击目标：视觉通路（图像）—— 这是最直观的、也是论文里最需要的一张图。

## 为什么用"模态感知预算"（H3）

FedVLA 的策略有三类参数，它们对隐私的敏感度**完全不同**：

| 通路 | 参数 | 隐私敏感度 | 理由 |
|---|---|---|---|
| 视觉 | `stems.<domain>_image` | **高** | 图像编码器梯度直接编码像素内容，反演风险最大 |
| 状态 | `stems.<domain>_state` | 中 | 本体感觉（关节角/末端位姿），反演价值低 |
| 动作 | `heads.<domain>` | 低 | 动作预测头，梯度几乎不泄漏输入 |
| 共享 | `trunk.*` | 中 | 混合表征，反演需要额外先验 |

**统一 DP 预算**的问题：如果给所有参数同一个噪声倍率 σ，
高敏感度的视觉通路拿到的"相对噪声"其实不够（它的梯度范数小），
而低敏感度的动作头白白消耗了隐私预算。

**模态感知预算**：把总预算 ε 按敏感度分配，各通路用**不同的 σ**，
使"有效信噪比"在各通路上大致对齐。

## 数学

DP-SGD（Abadi et al. 2016）标准三步：
1. **逐样本裁剪**：每个样本的梯度 g_i 裁到 L2 范数 ≤ C；
2. **加噪**：对裁剪后的梯度求和，加 N(0, σ²C²) 噪声；
3. **取均值**：除以 batch size B。

本文实现的是 **"等效逐样本裁剪"**：由于我们的优化器是 Adam 且要跨轮持久化，
逐样本梯度需要 per-sample backward（显存与时间代价大），
因此采用**按参数组的全局 L2 裁剪 + 加噪**（这是 FL 里的标准做法，
等价于"服务器看到的聚合更新满足 (ε,δ)-DP"）。

隐私预算用 **RDP accountant**（Rényi DP）计算：
- 每轮：一次高斯机制，噪声倍率 σ，采样率 q = 1/N_clients（客户端级采样）
- T 轮：RDP 复合 → 转 (ε, δ)-DP

## 模态感知预算分配

给定总预算 ε_total 和各通路的敏感度权重 s_m（越大越敏感），
分配 ε_m ∝ s_m，再由 σ_m = f(ε_m, q, T) 反解噪声倍率。

实现上用**数值搜索**：对 σ 做二分，使 RDP accountant 算出的总 ε 等于目标。
每个通路的 σ 按 ε_m 的比例缩放（ε 越小 → σ 越大）。

## 用法

```python
from dp import (allocate_modality_budget, DPConfig, apply_dp_to_update,
                rdp_accountant, epsilon_from_rdp)

cfg = DPConfig(epsilon=8.0, delta=1e-5, steps=50, num_clients=4,
               modality_weights={"vision": 0.5, "state": 0.2,
                                 "action": 0.1, "shared": 0.2})
alloc = allocate_modality_budget(cfg)
noisy = apply_dp_to_update(update, alloc, clip_norm=1.0)
```
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# RDP accountant（整数阶）
# ---------------------------------------------------------------------------
# 参考：Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled
# Gaussian Mechanism"，以及 TensorFlow Privacy / Opacus 的实现方式。
# 这里用整数阶的闭式表达式（论文附录给出），用 log-space 数值稳定。

_ORDERS: Tuple[int, ...] = tuple(range(2, 64)) + (
    128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def _log_add(a: float, b: float) -> float:
    """log(e^a + e^b)，数值稳定。"""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_erfc(x: float) -> float:
    """log(erfc(x))，大 x 时不溢出。"""
    try:
        return math.log(math.erfc(x))
    except (OverflowError, ValueError):
        # erfc(x) ≈ exp(-x²)/(x√π) for large x
        return -x * x - math.log(x) - 0.5 * math.log(math.pi)


def rdp_sampled_gaussian(q: float, noise_multiplier: float,
                         steps: int, orders: Iterable[int] = _ORDERS) -> Dict[int, float]:
    """采样高斯机制在 T 步后的 RDP（Rényi 阶 α）。

    q            —— 采样率（每轮从 N 个客户端里采 1 个的视角 q=1/N）
    noise_multiplier —— σ/C
    steps        —— 复合轮数
    返回 {order: rdp}
    """
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier 必须 > 0")
    if not (0 < q <= 1):
        raise ValueError("q 必须在 (0, 1]")
    out: Dict[int, float] = {}
    for order in orders:
        if order == 1:
            # 一阶 RDP 就是 KL，直接跳到最终转换里的最优
            out[order] = 0.0
            continue
        if q == 1.0:
            # 无采样：高斯机制的闭式 RDP，同样要乘 steps（复合性质）
            out[order] = steps * (order / (2.0 * noise_multiplier ** 2))
            continue
        # 采样高斯机制的整数阶闭式（Mironov 2019, Thm 3）
        # 单步的 RDP，然后按 steps 做（近似）复合：RDP(T) ≈ T · RDP(1)
        # ⚠️ 2026-11-08 修正：原实现漏了 steps 的乘法因子 ⇒ ε 与轮数无关，
        #    测试"ε 关于轮数单调递增"直接失败。RDP 的复合性质要求乘以 T。
        log_terms: List[float] = []
        for i in range(order + 1):
            log_binom = _log_comb(order, i)
            log_q_pow = i * math.log(q) + (order - i) * math.log(1.0 - q)
            # A_alpha = sum_{i=0}^{alpha} C(alpha,i) q^i (1-q)^{alpha-i}
            #           * exp(i(i-1) / (2 sigma^2))
            log_a = log_binom + log_q_pow + (i * (i - 1.0)) / (2.0 * noise_multiplier ** 2)
            log_terms.append(log_a)
        # 用 log-sum-exp 技巧：先减去最大值
        mx = max(log_terms)
        total = mx + math.log(sum(math.exp(t - mx) for t in log_terms))
        out[order] = steps * total / (order - 1.0)
    return out


def epsilon_from_rdp(rdp: Dict[int, float], delta: float,
                     orders: Optional[Sequence[int]] = None) -> Tuple[float, int]:
    """RDP → (ε, δ)-DP。

    转换式（improved 界）：ε = min_α { rdp(α) + log(1/δ)/(α-1) }
    """
    if delta <= 0 or delta >= 1:
        raise ValueError("delta 必须在 (0,1)")
    best_eps = math.inf
    best_order = 0
    for order, r in rdp.items():
        if order <= 1 or r <= 0:
            continue
        eps = r + math.log(1.0 / delta) / (order - 1.0)
        if eps < best_eps:
            best_eps = eps
            best_order = order
    return best_eps, best_order


# ---------------------------------------------------------------------------
# 配置与预算分配
# ---------------------------------------------------------------------------

#: 默认各通路的敏感度权重（越高越敏感 ⇒ 分到越多隐私预算）
DEFAULT_MODALITY_WEIGHTS: Dict[str, float] = {
    "vision": 0.50,   # 图像编码器 —— 反演风险最高
    "state": 0.15,    # 本体感觉 —— 反演价值低
    "action": 0.10,   # 动作头 —— 几乎不泄漏输入
    "shared": 0.25,   # trunk —— 混合表征
}

#: 参数键 → 通路 的映射规则（按优先级从上到下）
MODALITY_RULES: Tuple[Tuple[str, str], ...] = (
    # 视觉：stems.<domain>_image.*  以及任何名字里带 image 的 stem
    ("stems_image", "vision"),
    ("stems_state", "state"),
    ("heads", "action"),
    ("trunk", "shared"),
    ("normalizer", "shared"),
    ("modalities_tokens", "shared"),
)


@dataclass
class DPConfig:
    """DP-SGD 配置。"""
    epsilon: float = 8.0            # 目标总预算（ε）
    delta: float = 1e-5             # δ
    steps: int = 50                 # 联邦轮数（也是 RDP 复合步数）
    num_clients: int = 4            # 客户端数（采样率 q = 1/num_clients）
    clip_norm: float = 1.0          # 每个参数组的 L2 裁剪阈值 C
    modality_aware: bool = True     # True=模态感知预算；False=统一预算
    modality_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MODALITY_WEIGHTS))
    orders: Tuple[int, ...] = _ORDERS

    def sampling_rate(self) -> float:
        return 1.0 / max(1, self.num_clients)


@dataclass
class BudgetAllocation:
    """预算分配结果。"""
    epsilon_total: float
    delta: float
    steps: int
    num_clients: int
    modality_aware: bool
    #: 通路 → 该通路的 ε
    eps_by_modality: Dict[str, float]
    #: 通路 → 该通路的噪声倍率 σ（乘以 C 后是标准差）
    sigma_by_modality: Dict[str, float]
    #: 实际达成的总 ε（数值搜索的结果）
    achieved_epsilon: float
    best_order: int

    def summary(self) -> str:
        lines = [
            f"DP 预算分配：总 ε={self.epsilon_total} δ={self.delta} "
            f"步数={self.steps} 客户端={self.num_clients} "
            f"模式={'模态感知' if self.modality_aware else '统一'}",
            f"{'通路':<10}{'ε 配额':>10}{'噪声倍率 σ':>14}",
            "-" * 36,
        ]
        for m in sorted(self.eps_by_modality):
            lines.append(f"{m:<10}{self.eps_by_modality[m]:>10.4f}"
                         f"{self.sigma_by_modality[m]:>14.4f}")
        lines.append("-" * 36)
        lines.append(f"实际达成总 ε = {self.achieved_epsilon:.4f} "
                     f"（最优 Rényi 阶 α={self.best_order}）")
        return "\n".join(lines)


def _sigma_for_epsilon(target_eps: float, q: float, steps: int, delta: float,
                       orders: Sequence[int]) -> Tuple[float, int]:
    """数值搜索：找到使 ε(σ) ≈ target_eps 的噪声倍率 σ。

    ε(σ) 关于 σ 单调递减 ⇒ 用二分。σ 越大噪声越大 ⇒ ε 越小。
    """
    if target_eps <= 0:
        raise ValueError("target_eps 必须 > 0")

    def eps_of(sigma: float) -> Tuple[float, int]:
        rdp = rdp_sampled_gaussian(q, sigma, steps, orders)
        return epsilon_from_rdp(rdp, delta, orders)

    # 先找到一个"噪声足够大"的上界（ε 足够小）
    lo, hi = 1e-3, 1.0
    while eps_of(hi)[0] > target_eps and hi < 1e6:
        hi *= 2.0
    if eps_of(hi)[0] > target_eps:
        raise ValueError("无法达到该 ε（σ 已到上限）")
    # hi 满足 ε(hi) <= target；再找一个 ε(lo) > target 的下界
    while eps_of(lo)[0] < target_eps and lo > 1e-6:
        lo /= 2.0
    if eps_of(lo)[0] < target_eps:
        raise ValueError("无法达到该 ε（σ 已到下限）")

    for _ in range(60):
        mid = math.sqrt(lo * hi)          # 几何二分（σ 是乘性参数）
        e, _ = eps_of(mid)
        if e > target_eps:
            lo = mid
        else:
            hi = mid
    sigma = math.sqrt(lo * hi)
    eps, order = eps_of(sigma)
    return sigma, order


def allocate_modality_budget(cfg: DPConfig) -> BudgetAllocation:
    """按模态敏感度分配总预算，返回各通路的 ε 与噪声倍率 σ。

    ## 组合语义（这是本模块最容易搞错的地方）

    各通路的参数**互不相交** ⇒ 满足**并行组合**（parallel composition）：
    若通路 m 满足 (ε_m, δ_m)-DP，则整体满足

        ( max_m ε_m ,  Σ_m δ_m )-DP

    **总 ε 由最"贵"的那一路决定，不是各路之和。** 因此：

    - **模态感知模式**：把总预算 ε_total 全额给**最敏感的通路**（vision），
      其余通路按敏感度权重**按比例打折**（ε_m = ε_total · w_m / w_max）。
      这样 max_m ε_m = ε_total ✓，总预算不被浪费，
      且"视觉通路噪声最少、动作通路噪声最多"。

    - **统一预算模式**：所有通路用**同一个** σ（等价于对整个上行更新加一次噪声），
      总 ε 就是那个 σ 对应的值。

    ⚠️ 论文里必须写清楚是并行组合（max）而不是串行组合（sum），
    否则审稿人会认为隐私预算算错了。
    """
    mods = list(cfg.modality_weights.keys())
    weights = np.array([cfg.modality_weights[m] for m in mods], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("modality_weights 之和必须 > 0")
    w_max = float(weights.max())

    q = cfg.sampling_rate()
    if not cfg.modality_aware:
        # 统一预算：所有通路同一个 σ
        sigma, order = _sigma_for_epsilon(cfg.epsilon, q, cfg.steps,
                                          cfg.delta, cfg.orders)
        eps_m = {m: cfg.epsilon for m in mods}
        sigma_m = {m: sigma for m in mods}
        achieved, _ = epsilon_from_rdp(
            rdp_sampled_gaussian(q, sigma, cfg.steps, cfg.orders),
            cfg.delta, cfg.orders)
        return BudgetAllocation(
            epsilon_total=cfg.epsilon, delta=cfg.delta, steps=cfg.steps,
            num_clients=cfg.num_clients, modality_aware=False,
            eps_by_modality=eps_m, sigma_by_modality=sigma_m,
            achieved_epsilon=achieved, best_order=order)

    # 模态感知：ε_m = ε_total · (w_m / w_max) ⇒ max_m ε_m = ε_total
    eps_m = {m: cfg.epsilon * float(w) / w_max for m, w in zip(mods, weights)}
    sigma_m: Dict[str, float] = {}
    best_order = 0
    for m in mods:
        s, order = _sigma_for_epsilon(eps_m[m], q, cfg.steps, cfg.delta, cfg.orders)
        sigma_m[m] = s
        best_order = max(best_order, order)

    # 并行组合：总 ε = max_m ε_m（各通路参数不相交）
    achieved = max(eps_m.values())
    return BudgetAllocation(
        epsilon_total=cfg.epsilon, delta=cfg.delta, steps=cfg.steps,
        num_clients=cfg.num_clients, modality_aware=True,
        eps_by_modality=eps_m, sigma_by_modality=sigma_m,
        achieved_epsilon=achieved, best_order=best_order)


# ---------------------------------------------------------------------------
# 参数分组与加噪
# ---------------------------------------------------------------------------

def classify_key(key: str) -> str:
    """按参数键名判定所属通路。

    ⚠️ 2026-11-08 修正：原实现只看 `stems.*` / `heads.*` / `trunk.*`，
       而 **FedVLA 的上行只有** `trunk.*` + `normalizer.<domain>` + `modalities_tokens.*`
       （stems/heads 是个性化参数，**永不上行**，见 aggregate.py 的 `_is_shared_param`）。
       若不把 normalizer / modalities_tokens 归到对应通路，
       上行里的"模态专属"部分会被错分到 shared 组 ⇒ 模态感知分配形同虚设。

    归组规则（优先级从上到下）：
      stems.<domain>_image.*              → vision
      stems.<domain>_state.*              → state
      heads.<domain>.*                    → action
      normalizer.<d>.params_dict.action.* → action   （动作归一化统计，直接由私有动作拟合）
      normalizer.<d>.params_dict.state.*  → state
      modalities_tokens.image             → vision
      modalities_tokens.state             → state
      trunk.* / 其它                      → shared    （模态无关的共享主干）
    """
    k = key.lower()
    if k.startswith("stems."):
        if "image" in k:
            return "vision"
        if "state" in k:
            return "state"
        return "shared"
    if k.startswith("heads."):
        return "action"
    if k.startswith("normalizer."):
        if ".action" in k:
            return "action"
        if ".state" in k:
            return "state"
        return "shared"
    if k.startswith("modalities_tokens."):
        if "image" in k:
            return "vision"
        if "state" in k:
            return "state"
        return "shared"
    if k.startswith("trunk."):
        return "shared"
    return "shared"


def epsilon_from_sigma(q: float, sigma: float, steps: int, delta: float,
                        orders: Optional[Sequence[int]] = None) -> Tuple[float, int]:
    """给定噪声倍率 σ，反算达成的 (ε, δ)。

    与 `allocate_modality_budget` 的二分搜索方向相反 ——
    这里 σ 是输入，ε 是输出，**不需要任何搜索**。

    ⚠️ 为什么需要这个方向（2026-11-08 实测发现）：
       FedVLA 的 trunk 有 17.1M 参数，客户端级 DP 的噪声总范数
       = σ·C·√(17.1M) = σ·4139。若 σ=1（ε≈1），噪声范数 4139 ≫ 信号范数 1
       ⇒ 任何 ε∈[1,32] 的配置**都落在噪声地板上**，重建质量曲线是平的，
       根本看不出"ε 越大越容易反演"的规律。
       真正的转折点在 σ≈1e-4（噪声范数 0.41，与信号同量级），
       对应 ε≈1e5~1e7 —— 这本身就是论文要报告的结论：
       **在这个模型规模上，客户端级 DP 要保住可用效用就必须放弃隐私**。
       所以扫描的自变量必须选 σ，而不是 ε。
    """
    rdp = rdp_sampled_gaussian(q, sigma, steps, orders or _ORDERS)
    return epsilon_from_rdp(rdp, delta, orders or _ORDERS)


def allocate_by_sigma(cfg: DPConfig, sigma_ref: float) -> BudgetAllocation:
    """以**参考噪声倍率**为自变量做预算分配（等价于以总 ε 为自变量，但不用二分）。

    语义：
      · `sigma_ref` 是**最敏感通路**（vision）拿到的噪声倍率；
      · 总 ε = ε(σ_ref)（直接由 accountant 算出，无需搜索）；
      · 其余通路的 σ 由 ε_m = ε_total · w_m/w_max 反解（二分）；
      · 并行组合 ⇒ 实际总 ε = max_m ε_m = ε_total。

    统一预算模式（modality_aware=False）下，所有通路共用同一个 σ = sigma_ref，
    总 ε 同样是 ε(σ_ref) ⇒ **两种模式在同一个 sigma_ref 上总 ε 完全相同**，
    这保证了 H3 对比是"相同总预算"下的公平比较。
    """
    if sigma_ref <= 0:
        raise ValueError("sigma_ref 必须 > 0")
    q = cfg.sampling_rate()
    eps_total, order = epsilon_from_sigma(q, sigma_ref, cfg.steps, cfg.delta,
                                          cfg.orders)
    mods = list(cfg.modality_weights.keys())

    if not cfg.modality_aware:
        return BudgetAllocation(
            epsilon_total=eps_total, delta=cfg.delta, steps=cfg.steps,
            num_clients=cfg.num_clients, modality_aware=False,
            eps_by_modality={m: eps_total for m in mods},
            sigma_by_modality={m: sigma_ref for m in mods},
            achieved_epsilon=eps_total, best_order=order)

    weights = np.array([cfg.modality_weights[m] for m in mods], dtype=np.float64)
    w_max = float(weights.max())
    eps_m: Dict[str, float] = {}
    sigma_m: Dict[str, float] = {}
    best_order = order
    for m, w in zip(mods, weights):
        e_m = eps_total * float(w) / w_max
        s, o = _sigma_for_epsilon(e_m, q, cfg.steps, cfg.delta, cfg.orders)
        eps_m[m] = e_m
        sigma_m[m] = s
        best_order = max(best_order, o)
    return BudgetAllocation(
        epsilon_total=eps_total, delta=cfg.delta, steps=cfg.steps,
        num_clients=cfg.num_clients, modality_aware=True,
        eps_by_modality=eps_m, sigma_by_modality=sigma_m,
        achieved_epsilon=max(eps_m.values()), best_order=best_order)


def group_by_modality(update: Dict[str, torch.Tensor]
                      ) -> Dict[str, List[str]]:
    """把参数键按通路分组。"""
    groups: Dict[str, List[str]] = {}
    for k in update:
        groups.setdefault(classify_key(k), []).append(k)
    return groups


def is_shared_uplink_key(key: str) -> bool:
    """这个键是否属于 **FedVLA 上行**（会离开客户端）。

    与 `aggregate.py` 的 `_is_shared_param` **同口径**：
      上行 = `trunk.*` + `normalizer.<domain>` + `modalities_tokens.*`
      `stems.*` / `heads.*` 是个性化参数，留在客户端本地，不上行。
    """
    return not (key.startswith("stems.") or key.startswith("heads."))


def _clip_and_noise_group(tensors: List[torch.Tensor], sigma: float,
                           clip_norm: float, generator: torch.Generator
                           ) -> List[torch.Tensor]:
    """对一个参数组做 L2 裁剪 + 高斯加噪。

    与 DP-SGD 的等价形式：先对整组的梯度做 L2 裁剪（≤ clip_norm），
    再加 N(0, (σ·clip_norm)²) 的各向同性噪声。
    """
    if not tensors:
        return tensors
    flat = torch.cat([t.reshape(-1).float() for t in tensors])
    norm = float(torch.linalg.vector_norm(flat))
    scale = 1.0 if norm <= 0 else min(1.0, clip_norm / (norm + 1e-12))
    out = []
    for t in tensors:
        noisy = (t.float() * scale)
        if sigma > 0:
            noise = torch.randn(noisy.shape, generator=generator,
                                device=noisy.device, dtype=noisy.dtype)
            noisy = noisy + noise * (sigma * clip_norm)
        out.append(noisy.to(t.dtype))
    return out


def apply_dp_to_update(update: Dict[str, torch.Tensor],
                       alloc: BudgetAllocation,
                       clip_norm: Optional[float] = None,
                       seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """对上行更新施加 DP 噪声（按通路的 σ 分别加噪）。

    返回的字典与输入同键同形状（值被裁剪+加噪）。
    """
    C = clip_norm if clip_norm is not None else 1.0
    # ⚠️ 2026-11-08 修正：generator 必须在**张量所在设备**上，
    #    否则 CUDA 张量 + CPU generator 会报
    #    "Expected a 'cuda' device type for generator but found 'cpu'"。
    dev = torch.device("cpu")
    for v in update.values():
        if torch.is_tensor(v):
            dev = v.device
            break
    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(int(seed))
    groups = group_by_modality(update)
    out: Dict[str, torch.Tensor] = {}
    for mod, keys in groups.items():
        sigma = alloc.sigma_by_modality.get(mod, 0.0)
        tensors = [update[k] for k in keys]
        noisy = _clip_and_noise_group(tensors, sigma, C, gen)
        for k, t in zip(keys, noisy):
            out[k] = t
    return out


def effective_snr(update: Dict[str, torch.Tensor],
                  alloc: BudgetAllocation,
                  clip_norm: float = 1.0) -> Dict[str, float]:
    """每个通路的"有效信噪比"：||g|| / (σ·C·√numel)。

    含义：信号能量与噪声能量之比。
    SNR 越低 → 反演越难 → 隐私越好（但效用也越差）。

    ⚠️ 2026-11-08 修正：原实现用**加噪后**向量的范数做分子，
    而加噪会让范数变大 ⇒ SNR 反而上升（与直觉相反，测试直接抓出）。
    正确的分子必须是**原始（干净）**信号的范数。
    """
    groups = group_by_modality(update)
    res: Dict[str, float] = {}
    for mod, keys in groups.items():
        flat = torch.cat([update[k].reshape(-1).float() for k in keys])
        norm = float(torch.linalg.vector_norm(flat))
        sigma = alloc.sigma_by_modality.get(mod, 0.0)
        # 噪声总能量：每个元素方差 (σC)²，共 numel 个元素
        noise_norm = sigma * clip_norm * math.sqrt(max(flat.numel(), 1))
        res[mod] = norm / max(noise_norm, 1e-12)
    return res


# ---------------------------------------------------------------------------
# CLI（供云端干跑）
# ---------------------------------------------------------------------------

def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="DP-SGD 预算分配（干跑用）")
    ap.add_argument("--epsilon", type=float, default=8.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--uniform", action="store_true", help="统一预算模式")
    ap.add_argument("--weights", type=str, default="",
                    help="模态权重，逗号分隔，如 vision=0.5,state=0.15")
    args = ap.parse_args(argv)

    weights = dict(DEFAULT_MODALITY_WEIGHTS)
    if args.weights:
        weights = {}
        for kv in args.weights.split(","):
            k, v = kv.split("=")
            weights[k.strip()] = float(v)

    cfg = DPConfig(epsilon=args.epsilon, delta=args.delta, steps=args.steps,
                   num_clients=args.clients, modality_aware=not args.uniform,
                   modality_weights=weights)
    alloc = allocate_modality_budget(cfg)
    print(alloc.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
