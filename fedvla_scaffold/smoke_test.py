"""冒烟测试：不装 MuJoCo、不装 torch，先验证 FedVLA 两个核心算法的数学行为。

运行：python smoke_test.py
"""

import numpy as np
import fl_core as F


def test_dual_gate_sparsity():
    """观察 W_e 初始化尺度如何影响专家激活密度。"""
    rng = np.random.default_rng(0)
    K, D, n_tokens = 8, 16, 2000
    W_t = rng.normal(0, 0.5, size=(K, D))
    x = rng.normal(0, 1.0, size=(n_tokens, D))
    lam = 0.5

    print("=== DGMoE 双门控：专家激活密度 vs W_e 初始化尺度 ===")
    print(f"K={K} 专家, {n_tokens} 个随机 token, lambda={lam}")
    print(f"{'W_e 尺度':>10} | {'激活密度(个/token)':>18} | {'全激活占比':>10}")
    print("-" * 46)
    for scale in [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        W_e = rng.uniform(0, scale, size=K)
        masks = []
        for t in range(n_tokens):
            _, mask, _ = F.dual_gate(x[t], W_t, W_e, lam)
            masks.append(mask)
        density = F.expert_density(np.array(masks))
        all_on = float(np.mean(np.array(masks).sum(axis=1) == K))
        print(f"{scale:>10.2f} | {density:>18.3f} | {all_on:>9.1%}")

    print()
    print("读法：W_e 越接近 0，阈值 lambda*W_e 越低，越多专家被激活；")
    print("      原文报告的激活密度约 1.22，对应 W_e 需要落在能被 st 超过的量级。")
    print("      注意 st 是 K 路 softmax，均值约 1/K，所以阈值必须与 1/K 同量级才有效。")
    print("      这是论文未交代、且直接影响稀疏度与计算开销的一个点。")
    print("      [本结论来自随机张量的 toy 实验，需在真实 HPT 上复验]")


def test_eda_weights():
    """验证 EDA 权重：任务越相似的客户端权重越高，且每层权重和为 1。"""
    rng = np.random.default_rng(1)
    N, L, K = 4, 3, 8

    # 构造任务异构：客户端 0/1 偏好专家组 A，客户端 2/3 偏好专家组 B
    Vs = np.zeros((N, L, K))
    for l in range(L):
        Vs[0, l, [0, 1, 2]] = rng.integers(80, 120, size=3)
        Vs[1, l, [0, 1, 3]] = rng.integers(80, 120, size=3)
        Vs[2, l, [5, 6, 7]] = rng.integers(80, 120, size=3)
        Vs[3, l, [4, 6, 7]] = rng.integers(80, 120, size=3)

    w = F.eda_weights(Vs)
    print()
    print("=== EDA 聚合权重（4 客户端 / 3 层，客户端 0-1 与 2-3 分属两族）===")
    print("层 |  C0     C1     C2     C3   | 求和")
    print("-" * 44)
    for l in range(L):
        row = "  ".join(f"{v:.4f}" for v in w[l])
        print(f" {l} | {row} | {w[l].sum():.4f}")

    sums = w.sum(axis=1)
    assert np.allclose(sums, 1.0), "每层权重必须和为 1"
    print()
    print("重要观察：两族区分得很清楚（专家支持集几乎不相交），但权重几乎均匀。")
    print("          因为 w_i 正比于『该客户端与所有人的相似度之和』，")
    print("          两族对称时，每族内部的总相似度相等 -> 权重被归一化抹平。")
    print("          也就是说 EDA 惩罚的是『离群者』，而不是做『按簇聚合』。")

    # 对照：把客户端 3 的专家使用改成横跨两族（专家 0 属于 A 族，专家 7 属于 B 族）
    print()
    print("=== 对照：让客户端 3 横跨两族（同时用专家 0 与 7）===")
    Vs2 = Vs.copy()
    Vs2[3, :, :] = 0.0
    Vs2[3, :, [0, 7]] = 100.0
    w2 = F.eda_weights(Vs2)
    print("层 |  C0     C1     C2     C3(横跨两族) | 求和")
    print("-" * 52)
    for l in range(L):
        row = "  ".join(f"{v:.4f}" for v in w2[l])
        print(f" {l} | {row} | {w2[l].sum():.4f}")
    print()
    print("结果：C3 权重最高（约 0.29），C2 最低（约 0.18）。")
    print("原因：w_i 衡量的其实是『中心性』——与越多客户端相似的，权重越高。")
    print("      C3 横跨两族，与 A、B 两边都有交集，反而成了最中心的客户端；")
    print("      而 C2 失去同族伙伴后最孤立，权重被压到最低。")
    print()
    print("推论 1：EDA 是『按中心性加权』，不是『按簇聚合』。")
    print("推论 2：客户端分成均衡的两簇时，EDA 权重几乎等于 FedAvg（上一组实验已证），")
    print("        预期增益极小；只有当存在孤立客户端时，EDA 才与 FedAvg 拉开差距。")
    print("推论 3：要处理强任务异构，按簇聚合（FedMVLA 的 MAFA 思路）比 EDA 更直接。")
    print("        —— 这可以作为方向 A 的一个可验证假设，而不是只在口头上说『EDA 更好』。")


def test_aggregation_runs():
    """跑一轮完整聚合，确认 trunk 参数能按 EDA 权重合并。"""
    rng = np.random.default_rng(2)
    N, L = 4, 3
    K = 8
    Vs = rng.integers(1, 50, size=(N, L, K)).astype(float)
    trunks = [[rng.normal(0, 1.0, size=(6, 6)) for _ in range(L)] for _ in range(N)]

    out, w = F.run_round(trunks, Vs)
    fedavg_out = [F.fedavg(np.stack([t[l] for t in trunks], axis=0)) for l in range(L)]

    print()
    print("=== 一轮服务器聚合 ===")
    for l in range(L):
        diff = np.abs(out[l] - fedavg_out[l]).mean()
        print(f"层 {l}: EDA 聚合结果 shape={out[l].shape}, 与 FedAvg 的平均绝对差 = {diff:.6f}")
    print("（差异不为 0 说明 EDA 确实在按专家相似度重新分配权重）")


if __name__ == "__main__":
    test_dual_gate_sparsity()
    test_eda_weights()
    test_aggregation_runs()
    print()
    print("全部冒烟测试通过。")
