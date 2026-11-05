"""FedVLA：EDA（专家驱动聚合，Expert-Driven Aggregation）的权重计算与加权聚合。

公式（与 fedvla_scaffold/fl_core.py 的 toy 实现严格一致）
----------------------------------------------------------
输入 V ∈ R^{N×L×K}：N 个客户端、L 个 MoE 层、K 个专家的**选择计数**

    Vn[l,i,:] = V[l,i,:] / ||V[l,i,:]||                 (按层按客户端单位化)
    sim[l,i,j] = Vn[l,i,:] · Vn[l,j,:]                 (层 l 内客户端 i,j 的余弦相似度)
    totals[l,i] = Σ_j sim[l,i,j]                       (客户端 i 与所有人的相似度之和)
    w[l,i]     = totals[l,i] / Σ_i totals[l,i]         (沿客户端维归一化)

⚠️ **重要提醒（论文里必须照实写）**
   `w_i ∝ Σ_j s_ij` 度量的是**中心性**（client i 与大家的平均相似程度），
   **不是簇归属**。横跨两个簇的"中间人"客户端反而会拿到最高权重。
   所以不要把 EDA 描述成"簇内加权"——它的实际行为是"越像大家，权重越大"。
   这本身是一条可写的发现：EDA 是一种**软化的中心性加权**，
   它抑制孤立客户端（与谁都不像 → 权重低），但不区分不同的簇。

聚合时如何用这组权重
--------------------
MoE 层的权重是**逐层**的（L 个不同的 N 维权重向量），但参数张量可能横跨多层：
  - `blocks.{i}.*` 的参数只影响第 i 层的 token → 用第 i 层的权重
  - 其它（embedding / cross-attn / final norm 等共享参数）→ 用均匀权重
键名到层号的映射由 `layer_of_key()` 提供。
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence

import torch

# transformer block 的键名模式：blocks.<层号>.
_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


def layer_of_key(key: str) -> Optional[int]:
    """从参数键名里解析它属于哪个 transformer block；不属于任何 block 则返回 None。"""
    m = _BLOCK_RE.search(key)
    return int(m.group(1)) if m else None


# ── 权重计算 ──────────────────────────────────────────────────────────────
def similarity_matrix(V: torch.Tensor) -> torch.Tensor:
    """V: (N, L, K) → sim: (L, N, N)"""
    if V.dim() != 3:
        raise ValueError(f"V 必须是 (N, L, K)，实际 {tuple(V.shape)}")
    V = V.double()
    norms = V.norm(dim=2, keepdim=True)                 # (N, L, 1)
    Vn = V / norms.clamp_min(1e-12)
    # sim[l,i,j] = Σ_k Vn[i,l,k]·Vn[j,l,k]
    sim = torch.einsum("nlk,mlk->lnm", Vn, Vn)
    return sim


def eda_weights(V: torch.Tensor, eps: float = 1e-12,
                clamp_nonneg: bool = True) -> torch.Tensor:
    """EDA 权重。V: (N, L, K) → w: (L, N)，每行沿 N 维和为 1。

    ⚠️ **非负性不变量**（2026-11-05 逐字推导 + 实测）：
    真实场景下 V 是**专家计数**，元素非负 → 任意两个客户端的余弦相似度
    `⟨V_i, V_j⟩ / (|V_i||V_j|) ≥ 0`（非负向量内积非负）→ `totals ≥ 0`
    → 权重必然非负。**因此负权重（权重外推）在本管线里不会出现。**

    但 `clamp_nonneg=True` 保留为防御：万一将来 V 变成有符号
    （如改成梯度表示），负权重会让 `aggregate_state_dicts` 做**权重外推**，
    是"训练不收敛"的经典诱因。clamp 对合法的非负输入是无操作，
    只在异常输入时才生效，并顺带重新归一化保证行和为 1。
    """
    sim = similarity_matrix(V)                          # (L, N, N)
    totals = sim.sum(dim=2)                             # (L, N)
    if clamp_nonneg:
        totals = totals.clamp_min(0.0)
    denom = totals.sum(dim=1, keepdim=True).clamp_min(eps)
    return totals / denom


def eda_diagnostics(V: torch.Tensor, client_tasks: Optional[Sequence[str]] = None) -> dict:
    """给出 EDA 权重的可解释诊断（论文里用来说明它的真实行为）。

    返回：
      weights        : (L, N)
      mean_weights   : (N,) 各层平均权重
      within_task    : 若给了 client_tasks，给出每个客户端与"同任务客户端"的平均相似度
                       与"异任务客户端"的平均相似度的对比
    """
    w = eda_weights(V)
    sim = similarity_matrix(V)                          # (L, N, N)
    out = {
        "weights": w,
        "mean_weights": w.mean(dim=0),
        "sim": sim,
    }
    if client_tasks is not None:
        tasks = list(client_tasks)
        n = len(tasks)
        within, across = [], []
        for i in range(n):
            same = [j for j in range(n) if j != i and tasks[j] == tasks[i]]
            diff = [j for j in range(n) if j != i and tasks[j] != tasks[i]]
            within.append(sim[:, i, same].mean().item() if same else float("nan"))
            across.append(sim[:, i, diff].mean().item() if diff else float("nan"))
        out["within_task_sim"] = within
        out["across_task_sim"] = across
    return out


# ── 加权聚合 ──────────────────────────────────────────────────────────────
def aggregate_state_dicts(
    sds: Sequence[Dict[str, torch.Tensor]],
    weights: Optional[torch.Tensor] = None,
    num_layers: Optional[int] = None,
    block_map: Optional[Sequence[int]] = None,
    shared_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """按 EDA 权重聚合多个客户端的 state_dict。

    参数
      sds            : N 个客户端的 state_dict（键集合一致）
      weights        : (L, N) 的 EDA 权重；None 表示均匀权重
      num_layers     : MoE 层数（L）。None 时从 weights 推断。
      block_map      : 行号 -> transformer block 号的映射（长度 L）。
                       **必填于真实场景**：EDA 权重只覆盖"带 MoE 的那些 block"，
                       而参数键里的 `blocks.N.` 是全部 transformer block 的编号。
                       MoE 只加在后一半时，EDA 的第 0 行对应 block 8 而不是 block 0，
                       少了这张映射就会把权重用到错误的层上。
                       None 时退化为"行号 == block 号"（仅适用于所有 block 都带 MoE）。
      shared_weights : 非 block 参数（embedding / final norm 等）使用的权重 (N,)。
                       None 表示沿各行取平均（等价于"整体中心性权重"）。
    返回
      聚合后的 state_dict（浮点张量做加权平均，整数/布尔张量取第一个客户端的）
    """
    n = len(sds)
    if n == 0:
        raise ValueError("没有可聚合的 state_dict")
    if weights is None:
        weights = torch.full((1, n), 1.0 / n, dtype=torch.float64)
    w = weights.double()
    if w.dim() != 2 or w.shape[1] != n:
        raise ValueError(f"weights 形状应为 (L, {n})，实际 {tuple(w.shape)}")
    L = num_layers if num_layers is not None else w.shape[0]
    if w.shape[0] != L:
        raise ValueError(f"weights 行数 {w.shape[0]} 与 num_layers={L} 不一致")
    if block_map is None:
        block_map = list(range(L))
    if len(block_map) != L:
        raise ValueError(f"block_map 长度 {len(block_map)} 与层数 {L} 不一致")
    row_of_block = {int(b): r for r, b in enumerate(block_map)}

    if shared_weights is None:
        shared = w.mean(dim=0)
    else:
        shared = shared_weights.double()

    out: Dict[str, torch.Tensor] = {}
    for key in sds[0].keys():
        ref = sds[0][key]
        if not torch.is_floating_point(ref):
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)
        li = layer_of_key(key)
        if li is not None and li in row_of_block:
            ww = w[row_of_block[li]]
        else:
            # 非 MoE block 的参数 / 共享参数：用整体权重
            ww = shared
        ww = ww / ww.sum().clamp_min(eps)
        for i, sd in enumerate(sds):
            if key not in sd:
                raise KeyError(f"客户端 {i} 缺少键 {key}")
            acc += ww[i].to(ref.device) * sd[key].double()
        out[key] = acc.to(ref.dtype)
    return out


def block_indices_from_names(names: Sequence[str]) -> List[int]:
    """从专家计数的模块路径里解析出 transformer block 号（按升序、去重）。

    例：`trunk.trunk.blocks.8.mlp.moe` -> 8
    """
    out = set()
    for nm in names:
        li = layer_of_key(nm)
        if li is not None:
            out.add(li)
    return sorted(out)


# ── 簇感知聚合（H2 的执行路径）────────────────────────────────────────────
def cluster_clients(V: torch.Tensor, num_clusters: Optional[int] = None,
                    threshold: Optional[float] = None) -> List[int]:
    """按「专家计数的跨层平均相似度」对客户端做层次聚类（平均连接）。

    与 EDA 中心性的区别（论文里必须写清楚）：
      · EDA（中心性）：一个全局的相似度加权，不区分簇 —— 横跨两个簇的"中间人"
        反而拿到最高权重。
      · 簇感知聚合：**先分簇，再在簇内加权，簇间等权** —— 于是"大簇"（多数客户端
        共享的任务）不会因为人数多而在全局模型里占据更大比重。

    簇数的确定（默认：合并距离的**最大跳变**，即 elbow 启发式）：
      层次聚类把 N 个客户端合并到 1 个，记录每一步的平均连接距离；
      这些距离是单调不增的，取相邻两步距离差最大的一处切开 → 那里就是
      "同任务组"与"异任务组"的分界。比拍脑袋指定簇数更站得住脚。

    参数
      V            : (N, L, K)
      num_clusters : 目标簇数；None 时自动（最大跳变）或按 threshold
      threshold    : 平均连接距离小于该值就继续合并；None 时用最大跳变
    返回
      每个客户端的簇号（0..K-1）的列表，长度 N
    """
    N, L, K = V.shape
    if N <= 1:
        return [0] * N
    sim = similarity_matrix(V).mean(dim=0)        # (N, N) 跨层平均相似度
    dist = 1.0 - sim
    dist = ((dist + dist.t()) / 2.0).clamp_min(0.0)

    # ---- 平均连接层次聚类（自己实现，避免依赖 sklearn）----
    clusters = [[i] for i in range(N)]

    def linkage(a, b):
        """平均连接：两簇之间所有跨簇对距离的均值"""
        if not a or not b:
            return 0.0
        vals = [float(dist[x, y]) for x in a for y in b]
        return sum(vals) / len(vals)

    merge_dists: List[float] = []
    merge_hist: List[List[List[int]]] = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = linkage(clusters[i], clusters[j])
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
        merge_dists.append(d)
        merge_hist.append([list(c) for c in clusters])

    # ---- 决定切到几个簇 ----
    if num_clusters is not None and num_clusters >= 1:
        target = min(int(num_clusters), N)
    elif threshold is not None:
        # 合并距离首次 <= threshold 时对应的簇数
        target = N
        for k, d in enumerate(merge_dists):
            if d <= threshold:
                target = N - (k + 1)
                break
    else:
        # elbow：相邻合并距离的最大跳变
        if len(merge_dists) <= 1:
            target = 1
        else:
            gaps = [merge_dists[i] - merge_dists[i + 1]
                    for i in range(len(merge_dists) - 1)]
            # 取**第一个**最大跳变（跳变越大分界越清晰；取第一个避免把同簇拆太碎）
            best_i = 0
            for i in range(1, len(gaps)):
                if gaps[i] > gaps[best_i]:
                    best_i = i
            target = N - (best_i + 1)
        target = max(1, min(N, target))

    if target >= N:
        final = [[i] for i in range(N)]        # 每个客户端自成一簇
    else:
        # merge_hist[k] 是"合并了 k+1 次之后"的簇集合，共 N-(k+1) 个簇
        final = merge_hist[N - target - 1]
    out = [0] * N
    for cid, cl in enumerate(final):
        for i in cl:
            out[i] = cid
    return out


def eda_cluster_weights(V: torch.Tensor, clusters: Sequence[int],
                        eps: float = 1e-12) -> torch.Tensor:
    """簇感知权重：簇内按中心性加权，**簇间等权**。

    w[l,i] = (1/C) · ( Σ_{j∈簇(i)} sim[l,i,j] / Σ_{i'∈簇(i)} Σ_{j∈簇(i')} sim[l,i',j] )

    含义：先在每个簇内部决定"谁更有代表性"，再让每个簇对全局模型贡献同样的比重。
    这与中心性 EDA 的关键差别：大簇（人数多的任务）不会因为人数而主导全局模型。

    返回 (L, N)，每行沿 N 维和为 1。
    """
    N, L, K = V.shape
    sim = similarity_matrix(V)                      # (L, N, N)
    clusters = list(clusters)
    C = max(clusters) + 1 if clusters else 1
    w = torch.zeros(L, N, dtype=torch.float64)
    for cid in range(C):
        members = [i for i in range(N) if clusters[i] == cid]
        if not members:
            continue
        idx = torch.tensor(members, dtype=torch.long)
        sub = sim[:, idx][:, :, idx]                 # (L, m, m)
        totals = sub.sum(dim=2).clamp_min(0.0)       # (L, m) —— 同 eda_weights 的非负不变量
        denom = totals.sum(dim=1, keepdim=True).clamp_min(eps)   # (L, 1)
        w[:, idx] = (totals / denom) / C
    # 归一化（浮点误差修正）
    w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
    return w


def cluster_report(V: torch.Tensor, clusters: Sequence[int],
                   client_tasks: Optional[Sequence[str]] = None) -> str:
    """簇感知聚合的可读诊断（写进论文附录 / 日志）。"""
    clusters = list(clusters)
    C = max(clusters) + 1 if clusters else 1
    sim = similarity_matrix(V).mean(dim=0)
    lines = [f"簇数 = {C}"]
    for cid in range(C):
        members = [i for i in range(len(clusters)) if clusters[i] == cid]
        labels = ([client_tasks[i] for i in members]
                  if client_tasks else [f"c{i}" for i in members])
        # 簇内平均相似度 vs 全局平均相似度
        within = []
        for i in members:
            same = [j for j in members if j != i]
            if same:
                within.append(float(sim[i, same].mean()))
        across = [float(sim[i, j].mean())
                  for i in members for j in range(len(clusters)) if j not in members]
        lines.append(f"  簇{cid}: 成员={members} 标签={labels}"
                     f" 簇内相似度={sum(within)/len(within):.4f}" if within else
                     f"  簇{cid}: 成员={members} 标签={labels}")
        if across:
            lines[-1] += f" 簇外相似度={sum(across)/len(across):.4f}"
    return "\n".join(lines)


def format_weight_report(weights: torch.Tensor, client_tasks: Sequence[str],
                          layer_names: Optional[Sequence[str]] = None) -> str:
    """把 EDA 权重打成可读的文本表（写进论文附录 / 日志）。"""
    w = weights.double()
    L, N = w.shape
    names = list(client_tasks)
    lines = []
    lines.append("层".ljust(10) + "".join(f"{n:>14}" for n in names))
    lines.append("-" * (10 + 14 * N))
    for l in range(L):
        lname = layer_names[l] if layer_names else f"layer{l}"
        lines.append(f"{lname:<10}" + "".join(f"{w[l, i]:>14.4f}" for i in range(N)))
    lines.append("-" * (10 + 14 * N))
    lines.append(f"{'平均':<10}" + "".join(f"{w[:, i].mean():>14.4f}" for i in range(N)))
    return "\n".join(lines)
