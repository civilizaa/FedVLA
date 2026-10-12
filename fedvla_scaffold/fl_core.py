"""FedVLA 核心算法的可运行骨架（NumPy 版）

用途：在装 MuJoCo / HPT 之前，先把两个核心算法验证清楚
  1. DGMoE 双门控路由（FedVLA Sec. 3.2，公式 3-6）
  2. Expert-Driven Aggregation 专家驱动聚合（FedVLA Sec. 3.3，公式 7-9）

这里不依赖 torch，用随机张量跑通数学。接真实模型时，
把 dual_gate / eda_weights / aggregate_trunk 三个函数换成 torch 版本即可，
接口与调用顺序保持一致。

对应原文记号：
  x    : token 表示            (D,)
  W_t  : token 侧门控权重      (K, D)
  W_g  : 跨层门控残差变换      (K, K)
  W_e  : 专家自感知参数        (K,)
  lam  : 缩放因子，原文取 0.5
  V    : 专家选择计数矩阵      (L, K)
  w    : 聚合权重              (L, N)
"""

from __future__ import annotations
import numpy as np

Array = np.ndarray


def softmax(v: Array) -> Array:
    v = np.asarray(v, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


def token_gate(x: Array, W_t: Array, prev_g: Array | None = None,
               W_g: Array | None = None) -> Array:
    """公式 3：token 侧门控。第一层 Gt = Wt x，之后 Gt = Wt x + Wg G_{t-1}。"""
    scores = W_t @ x
    if prev_g is not None and W_g is not None:
        scores = scores + W_g @ prev_g
    return scores


def expert_gate(st: Array, W_e: Array, lam: float = 0.5) -> Array:
    """公式 4：专家侧门控 se = sign(st - lam * We)。返回布尔激活掩码。"""
    return (st - lam * W_e) > 0


def dual_gate(x: Array, W_t: Array, W_e: Array, lam: float = 0.5,
              prev_g: Array | None = None, W_g: Array | None = None):
    """公式 3-6 合起来：一次前向得到 st / 激活掩码 / 门控值 g。

    返回 (st, mask, g)：
      st   所有专家上的 softmax 分数
      mask 布尔数组，哪些专家被激活
      g    门控值，未激活处置 0
    """
    raw = token_gate(x, W_t, prev_g, W_g)
    st = softmax(raw)
    mask = expert_gate(st, W_e, lam)
    g = np.where(mask, st, 0.0)
    return st, mask, g


def expert_density(masks: Array) -> float:
    """平均每个 token 激活的专家数。原文报告的数值在 1.22 附近。"""
    masks = np.asarray(masks, dtype=bool)
    return float(masks.sum(axis=-1).mean())


def accumulate_counts(counts: Array, mask: Array) -> Array:
    """把一个 token 的激活掩码累加进专家选择计数 V。"""
    return counts + np.asarray(mask, dtype=np.float64)


def eda_weights(Vs: Array, eps: float = 1e-12) -> Array:
    """公式 7-9：专家驱动聚合权重。

    参数
      Vs : (N, L, K) 每个客户端每层的专家选择计数
    返回
      w  : (L, N) 每一层每个客户端的聚合权重，沿 N 维求和为 1
    """
    Vs = np.asarray(Vs, dtype=np.float64)
    n_clients, n_layers, _ = Vs.shape

    norms = np.linalg.norm(Vs, axis=2, keepdims=True)          # (N, L, 1)
    Vn = Vs / np.maximum(norms, eps)                            # 单位化
    # 相似度矩阵 s[l, i, j]
    sim = np.einsum("nlk,mlk->lnm", Vn, Vn)

    totals = sim.sum(axis=2)                                    # (L, N) 每个 i 与所有 j 的相似度之和
    w = totals / np.maximum(totals.sum(axis=1, keepdims=True), eps)
    return w


def aggregate_trunk(client_trunks: Array, w_layer: Array) -> Array:
    """按某一层的 EDA 权重聚合 trunk 参数。

      client_trunks : (N, ...) 第 l 层的参数，客户端维在第 0 维
      w_layer       : (N,)     该层的聚合权重
    """
    client_trunks = np.asarray(client_trunks, dtype=np.float64)
    w_layer = np.asarray(w_layer, dtype=np.float64)
    return np.tensordot(w_layer, client_trunks, axes=(0, 0))


def fedavg(client_trunks: Array, sizes: Array | None = None) -> Array:
    """基线：普通 FedAvg。sizes 为各客户端样本量，None 则等权。"""
    client_trunks = np.asarray(client_trunks, dtype=np.float64)
    if sizes is None:
        return client_trunks.mean(axis=0)
    sizes = np.asarray(sizes, dtype=np.float64)
    return np.tensordot(sizes / sizes.sum(), client_trunks, axes=(0, 0))


def run_round(client_trunks: list[Array], Vs: Array, sizes: Array | None = None):
    """跑一轮服务器端聚合，返回每层的聚合结果。

    client_trunks[i] 是客户端 i 的 trunk 参数列表（按层），
    这里要求各客户端层数一致；stem 与 head 按原文不参与聚合。
    """
    n_layers = len(client_trunks[0])
    w = eda_weights(Vs)
    out = []
    for l in range(n_layers):
        stacked = np.stack([t[l] for t in client_trunks], axis=0)
        out.append(aggregate_trunk(stacked, w[l]))
    return out, w
