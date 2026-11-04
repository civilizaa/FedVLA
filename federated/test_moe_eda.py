#!/usr/bin/env python3
"""FedVLA：MoE 模块 + EDA 聚合的本地单元测试（不需要 GPU / 不需要 HPT）。

跑法：
    python federated/test_moe_eda.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fedvla_moe import (TopKMoE, MoEBlock, moe_param_keys,
                        zero_moe_output_projections, has_moe,
                        collect_expert_counts, stack_expert_counts)
from eda import eda_weights, layer_of_key, aggregate_state_dicts, similarity_matrix

FAILED = []


def check(name, cond, extra=""):
    if cond:
        print(f"  [OK] {name}")
    else:
        print(f"  [FAIL] {name} {extra}")
        FAILED.append(name)


class DenseMlp(nn.Module):
    """HPT 的 Mlp 同构（用于对照）"""
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_moe_basic():
    print("\n[1] TopKMoE 基本性质")
    torch.manual_seed(0)
    dim, hidden, E, k = 32, 64, 8, 2
    moe = TopKMoE(dim, hidden, dim, num_experts=E, top_k=k)
    x = torch.randn(4, 7, dim)          # (B, T, D)
    y = moe(x)
    check("输出形状与输入一致", y.shape == x.shape, f"{tuple(y.shape)} vs {tuple(x.shape)}")

    y.sum().backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all() for p in moe.parameters()]
    check("所有参数都有有限梯度", all(grads))

    moe.reset_counts()
    moe.train()
    _ = moe(x)
    counts = moe.get_counts()
    check("计数总和 = token 数 × top_k",
          abs(counts.sum().item() - (4 * 7 * k)) < 1e-4,
          f"{counts.sum().item()} vs {4*7*k}")
    check("计数非负", bool((counts >= 0).all()))


def test_moe_block_dense_equivalence():
    print("\n[2] MoEBlock 训练起点严格等于稠密 FFN（公平性保证）")
    torch.manual_seed(1)
    dim, hidden, E, k = 24, 48, 8, 2
    dense = DenseMlp(dim, hidden)
    block = MoEBlock(dim, hidden, dim, num_experts=E, top_k=k, expert_ratio=0.25)

    # 稠密权重原样载入（键名一致，不需要任何形状变换）
    dense_sd = {f"fc1.weight": dense.fc1.weight.detach().clone(),
                "fc1.bias": dense.fc1.bias.detach().clone(),
                "fc2.weight": dense.fc2.weight.detach().clone(),
                "fc2.bias": dense.fc2.bias.detach().clone()}
    missing, unexpected = block.load_state_dict(dense_sd, strict=False)
    check("没有多余键", len(unexpected) == 0, f"{list(unexpected)[:4]}")
    check("缺失键全部属于 MoE 分支",
          all(k.startswith("moe.") for k in missing), f"{sorted(missing)[:4]}")
    check("moe_param_keys 与缺失键一致",
          set(missing) == moe_param_keys(block),
          f"{sorted(set(missing) ^ moe_param_keys(block))[:4]}")

    x = torch.randn(3, 5, dim)
    block.eval()
    dense.eval()
    with torch.no_grad():
        y_block = block(x)
        y_dense = dense(x)
    err = (y_block - y_dense).abs().max().item()
    check("输出与稠密 FFN 数值一致", err < 1e-6, f"max err={err:.3e}")


def test_zero_init_helper():
    print("\n[3] 零初始化助手（_init_weights 之后要补一次）")
    torch.manual_seed(3)
    block = MoEBlock(16, 32, 16, num_experts=4, top_k=2, expert_ratio=0.25)
    # 模拟 HPT 的 apply(_init_weights)：把 fc2 重新随机初始化
    for e in block.moe.experts:
        nn.init.normal_(e.fc2.weight, std=0.02)
        nn.init.normal_(e.fc2.bias, std=0.02)
    x = torch.randn(2, 3, 16)
    block.eval()
    with torch.no_grad():
        # 注意：要测的是 **MoE 分支本身**的输出，
        # 而不是 block 的整体输出（整体 = 稠密 FFN + MoE，稠密部分本来就不为 0）
        m1 = block.moe(x)
    check("未补零时 MoE 分支有输出", float(m1.abs().sum()) > 0,
          f"{float(m1.abs().sum()):.3e}")
    n = zero_moe_output_projections(block)
    with torch.no_grad():
        m2 = block.moe(x)
        y2 = block(x)
    check("补零后 MoE 分支输出为 0", float(m2.abs().sum()) < 1e-6,
          f"{float(m2.abs().sum()):.3e}")
    check("统计到 1 个 MoE 分支", n == 1, f"{n}")

    # 补零后整个 block 应当退化为"只有稠密 FFN"的行为：
    # 与单独构造的稠密等价块逐位一致
    dense_only = DenseMlp(16, 32)
    dense_only.load_state_dict({
        "fc1.weight": block.fc1.weight.detach().clone(),
        "fc1.bias": block.fc1.bias.detach().clone(),
        "fc2.weight": block.fc2.weight.detach().clone(),
        "fc2.bias": block.fc2.bias.detach().clone(),
    })
    dense_only.eval()
    with torch.no_grad():
        y_dense = dense_only(x)
    err = (y2 - y_dense).abs().max().item()
    check("补零后 block 输出 == 纯稠密 FFN 输出", err < 1e-6, f"max err={err:.3e}")


def test_has_moe():
    print("\n[4] has_moe 检测")
    block = MoEBlock(8, 16, 8, num_experts=4, top_k=2)
    check("MoEBlock 被检出", has_moe(block))
    check("普通模块不被检出", not has_moe(nn.Linear(3, 3)))


def test_eda_weights():
    print("\n[5] EDA 权重")
    # 3 个客户端，2 层，4 专家：c0/c1 同簇，c2 孤立
    V = torch.tensor([
        [[10.0, 10.0, 0.0, 0.0]], [[10.0, 9.0, 0.0, 0.0]],
        [[10.0, 10.0, 0.0, 0.0]], [[10.0, 9.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0, 1.0]],
    ]).reshape(3, 2, 4)
    w = eda_weights(V)
    check("权重形状 = (L, N)", tuple(w.shape) == (2, 3), f"{tuple(w.shape)}")
    check("每层权重和为 1",
          bool(torch.allclose(w.sum(dim=1), torch.ones(2, dtype=torch.float64), atol=1e-9)),
          f"{w.sum(dim=1)}")
    check("c0/c1 权重 > c2（中心性更高）",
          bool(w[:, 0].mean() > w[:, 2].mean() and w[:, 1].mean() > w[:, 2].mean()),
          f"{w.mean(dim=0)}")
    sim = similarity_matrix(V)
    check("相似度矩阵对称", bool(torch.allclose(sim[0], sim[0].T, atol=1e-9)))
    check("相似度对角线为 1",
          bool(torch.allclose(torch.diagonal(sim[0]),
                              torch.ones(3, dtype=torch.float64), atol=1e-6)))


def test_layer_of_key():
    print("\n[6] 键名 -> 层号")
    check("block 键", layer_of_key("trunk.trunk.blocks.7.mlp.moe.experts.0.fc1.weight") == 7)
    check("嵌套前缀也能识别", layer_of_key("blocks.0.attn.qkv.weight") == 0)
    check("非 block 键返回 None", layer_of_key("trunk.trunk.embeddings.state") is None)


def test_aggregate():
    print("\n[7] 加权聚合")
    n, dim = 3, 4
    sds = []
    for i in range(n):
        sds.append({
            "blocks.0.mlp.fc1.weight": torch.full((dim, dim), float(i)),
            "blocks.1.mlp.fc1.weight": torch.full((dim, dim), float(i)),
            "shared.embed.weight": torch.full((dim,), float(i)),
        })
    w = torch.tensor([[0.8, 0.1, 0.1],
                      [0.1, 0.1, 0.8]])
    out = aggregate_state_dicts(sds, w, num_layers=2)
    check("第 0 层按第 0 层权重聚合",
          abs(out["blocks.0.mlp.fc1.weight"][0, 0].item() - (0.1 * 1 + 0.1 * 2)) < 1e-6,
          f"{out['blocks.0.mlp.fc1.weight'][0,0].item()}")
    check("第 1 层按第 1 层权重聚合",
          abs(out["blocks.1.mlp.fc1.weight"][0, 0].item() - (0.1 * 1 + 0.8 * 2)) < 1e-6,
          f"{out['blocks.1.mlp.fc1.weight'][0,0].item()}")
    shared_w = w.mean(dim=0)
    check("共享参数用各层平均权重",
          abs(out["shared.embed.weight"][0].item() -
              float((shared_w * torch.tensor([0., 1., 2.])).sum())) < 1e-6,
          f"{out['shared.embed.weight'][0].item()}")


def test_counts_collect():
    print("\n[8] 专家计数收集")
    block = MoEBlock(16, 32, 16, num_experts=4, top_k=2)
    x = torch.randn(2, 6, 16)
    block.train()
    block(x)
    counts = collect_expert_counts(block)
    check("收集到 1 个 MoE 层", len(counts) == 1, f"{list(counts)}")
    M = stack_expert_counts(counts)
    check("堆成 (L, K)", tuple(M.shape) == (1, 4), f"{tuple(M.shape)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_moe_basic()
    test_moe_block_dense_equivalence()
    test_zero_init_helper()
    test_has_moe()
    test_eda_weights()
    test_layer_of_key()
    test_aggregate()
    test_counts_collect()
    print()
    if FAILED:
        print(f"!! {len(FAILED)} 项失败: {FAILED}")
        sys.exit(1)
    print("全部通过  ALL_TESTS_PASSED")
