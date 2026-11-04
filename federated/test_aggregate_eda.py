#!/usr/bin/env python3
"""FedVLA：服务器端聚合（FedAvg / EDA）的端到端测试。

用合成的 4 客户端数据走完整的 aggregate.py 流程，验证：

  [1] FedAvg 等权聚合的数值正确性
  [2] EDA 模式：逐层权重真的被用到了对应的 block 上（**层号映射是这里最容易错的地方**）
  [3] EDA 的非 trunk 参数用平均权重
  [4] 产物齐全：model.pth / trunk.pth / eda_weights.pt / eda_report.txt
  [5] 缺失 expert_counts.pt 时给出可读报错（而不是静默回退）

不需要 GPU、不需要 HPT。

跑法：
    python federated/test_aggregate_eda.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import aggregate as agg  # noqa: E402
from eda import block_indices_from_names, eda_weights  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    if cond:
        print(f"  [OK] {name}")
    else:
        print(f"  [FAIL] {name} {extra}")
        FAILED.append(name)


# ── 合成数据 ──────────────────────────────────────────────────────────────
N_CLIENTS = 4
N_MOE_LAYERS = 2            # MoE 层数（L）
N_EXPERTS = 8               # 专家数（K）
MOE_BLOCKS = [8, 9]         # 带 MoE 的 transformer block 号（HPT-Base 有 16 层，MoE 加在后一半）
DIM = 4


def make_client(dirpath: str, idx: int):
    """造一个客户端目录：model.pth + expert_counts.pt

    专家计数的设计要点（让 EDA 权重既**非均匀**、又**逐层不同**）：
      · 第 0 层：客户端 0/1/2 都偏好专家 0，客户端 3 偏好专家 5 → 客户端 3 孤立
      · 第 1 层：客户端 0/1/3 都偏好专家 1，客户端 2 偏好专家 6 → 客户端 2 孤立
    这样：
      - 行 0 与行 1 的权重向量明显不同 → 若层号映射搞错（block8 用了行 1），
        数值对不上，测试能抓到；
      - 存在明显非均匀 → 能验证"聚合结果 != 均匀平均"。
    """
    os.makedirs(dirpath, exist_ok=True)

    # 客户端 i 的模型：blocks.8 / blocks.9 / 一个共享参数 / 一个 stem
    sd = {
        # block 8：值全为 i（便于验算）
        "trunk.trunk.blocks.8.mlp.fc1.weight": torch.full((DIM, DIM), float(idx)),
        "trunk.trunk.blocks.8.mlp.moe.gate.weight": torch.full((N_EXPERTS, DIM), float(idx)),
        # block 9：值全为 10+i
        "trunk.trunk.blocks.9.mlp.fc1.weight": torch.full((DIM, DIM), float(10 + idx)),
        "trunk.trunk.blocks.9.mlp.moe.gate.weight": torch.full((N_EXPERTS, DIM), float(10 + idx)),
        # 非 MoE 的 block 0：值全为 100+i（应该拿"整体平均权重"）
        "trunk.trunk.blocks.0.mlp.fc1.weight": torch.full((DIM, DIM), float(100 + idx)),
        # 共享参数
        "trunk.trunk.embeddings.state": torch.full((DIM,), float(200 + idx)),
        # 非 trunk（按域个性化）
        "stems.task1_state": torch.full((DIM,), float(300 + idx)),
        "heads.task1_w": torch.full((DIM,), float(400 + idx)),
    }
    torch.save(sd, os.path.join(dirpath, "model.pth"))

    counts = torch.zeros(N_MOE_LAYERS, N_EXPERTS)
    # 第 0 层：0/1/2 -> 专家 0，3 -> 专家 5
    pref0 = 5 if idx == 3 else 0
    counts[0, pref0] = 100.0
    counts[0, (pref0 + 1) % N_EXPERTS] = 10.0
    # 第 1 层：0/1/3 -> 专家 1，2 -> 专家 6
    pref1 = 6 if idx == 2 else 1
    counts[1, pref1] = 100.0
    counts[1, (pref1 + 1) % N_EXPERTS] = 10.0
    names = [f"trunk.trunk.blocks.{b}.mlp.moe" for b in MOE_BLOCKS]
    torch.save({"counts": counts, "names": names},
               os.path.join(dirpath, "expert_counts.pt"))

    with open(os.path.join(dirpath, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("# fake config\n")


def build_dirs(tmp: str):
    dirs = []
    for i in range(N_CLIENTS):
        d = os.path.join(tmp, f"client_{i}")
        make_client(d, i)
        dirs.append(d)
    return dirs


# ── 测试 ──────────────────────────────────────────────────────────────────
def test_fedavg(dirs, tmp):
    print("\n[1] FedAvg 等权聚合")
    out = os.path.join(tmp, "out_fedavg")
    agg.aggregate(dirs, out)
    full = torch.load(os.path.join(out, "model.pth"), map_location="cpu", weights_only=False)
    v = full["trunk.trunk.blocks.8.mlp.fc1.weight"][0, 0].item()
    check("blocks.8 = 等权平均 (0+1+2+3)/4 = 1.5", abs(v - 1.5) < 1e-6, f"{v}")
    v9 = full["trunk.trunk.blocks.9.mlp.fc1.weight"][0, 0].item()
    check("blocks.9 = 等权平均 (10+11+12+13)/4 = 11.5", abs(v9 - 11.5) < 1e-6, f"{v9}")
    v0 = full["trunk.trunk.blocks.0.mlp.fc1.weight"][0, 0].item()
    check("blocks.0（非 MoE block）也正确", abs(v0 - 101.5) < 1e-6, f"{v0}")
    # 非 trunk 的同名同形键是"按权重平均"（见 build_global_model 文档）
    check("stem 同名同形 -> 按权重平均 301.5",
          abs(full["stems.task1_state"][0].item() - 301.5) < 1e-6,
          f"{full['stems.task1_state'][0].item()}")


def test_eda(dirs, tmp):
    print("\n[2] EDA 逐层聚合（层号映射是重点）")
    out = os.path.join(tmp, "out_eda")
    tasks = ["task1", "task1", "task2", "task2"]
    agg.aggregate(dirs, out, eda_mode=True, client_tasks=tasks)

    blob = torch.load(os.path.join(out, "eda_weights.pt"), map_location="cpu", weights_only=False)
    w = blob["weights"]
    block_map = blob["block_map"]
    check("权重形状 = (L, N) = (2, 4)", tuple(w.shape) == (N_MOE_LAYERS, N_CLIENTS),
          f"{tuple(w.shape)}")
    check("block_map 正确解析为 [8, 9]", block_map == MOE_BLOCKS, f"{block_map}")
    check("每层权重和为 1",
          bool(torch.allclose(w.sum(dim=1), torch.ones(N_MOE_LAYERS, dtype=torch.float64),
                              atol=1e-9)), f"{w.sum(dim=1)}")
    check("权重非均匀（存在差异）", float(w.std()) > 1e-6, f"std={float(w.std()):.3e}")
    check("第 0 层：孤立的客户端 3 权重低于客户端 0",
          float(w[0, 3]) < float(w[0, 0]), f"{w[0].tolist()}")
    check("第 1 层：孤立的客户端 2 权重低于客户端 0",
          float(w[1, 2]) < float(w[1, 0]), f"{w[1].tolist()}")
    check("第 0 行与第 1 行权重不同（层号映射错了就能抓到）",
          float((w[0] - w[1]).abs().max()) > 1e-6, f"{w.tolist()}")

    full = torch.load(os.path.join(out, "model.pth"), map_location="cpu", weights_only=False)

    # 关键：blocks.8 必须用 **第 0 行** EDA 权重（对应 block 8），
    # 而不是"第 8 行"（不存在）或均匀权重。
    w8 = w[0].double()
    w8 = w8 / w8.sum()
    expect8 = float(sum(w8[i].item() * float(i) for i in range(N_CLIENTS)))
    v8 = full["trunk.trunk.blocks.8.mlp.fc1.weight"][0, 0].item()
    check("blocks.8 用的是 EDA 第 0 行权重（而非均匀）", abs(v8 - expect8) < 1e-6,
          f"{v8} vs {expect8}")
    check("blocks.8 的聚合结果确实不等于均匀平均 1.5",
          abs(v8 - 1.5) > 1e-6, f"{v8}")

    w9 = w[1].double()
    w9 = w9 / w9.sum()
    expect9 = float(sum(w9[i].item() * float(10 + i) for i in range(N_CLIENTS)))
    v9 = full["trunk.trunk.blocks.9.mlp.fc1.weight"][0, 0].item()
    check("blocks.9 用的是 EDA 第 1 行权重", abs(v9 - expect9) < 1e-6, f"{v9} vs {expect9}")

    print("\n[3] 非 MoE block / 共享参数")
    wm = w.double().mean(dim=0)
    wm = wm / wm.sum()
    # 注意：聚合结果会 cast 回 float32（与原权重 dtype 一致），
    # 所以用相对容差比较（绝对容差 1e-6 对 ~100 量级的 float32 太紧）。
    def close(a, b):
        return abs(a - b) <= 1e-5 * max(1.0, abs(b))
    expect0 = float(sum(wm[i].item() * float(100 + i) for i in range(N_CLIENTS)))
    v0 = full["trunk.trunk.blocks.0.mlp.fc1.weight"][0, 0].item()
    check("非 MoE 的 blocks.0 用整体平均权重", close(v0, expect0), f"{v0} vs {expect0}")
    expect_sh = float(sum(wm[i].item() * float(200 + i) for i in range(N_CLIENTS)))
    vsh = full["trunk.trunk.embeddings.state"][0].item()
    check("共享参数用整体平均权重", close(vsh, expect_sh), f"{vsh} vs {expect_sh}")

    print("\n[4] 产物齐全")
    for f in ("model.pth", "trunk.pth", "eda_weights.pt", "eda_report.txt", "config.yaml"):
        check(f"存在 {f}", os.path.exists(os.path.join(out, f)))
    trunk_only = torch.load(os.path.join(out, "trunk.pth"), map_location="cpu",
                             weights_only=False)
    check("trunk.pth 的键去掉了最外层 trunk. 前缀",
          all(k.startswith("trunk.") for k in trunk_only),
          f"{list(trunk_only)[:2]}")
    check("trunk.pth 保留了 MoE 参数",
          any(".moe." in k for k in trunk_only), f"{list(trunk_only)[:2]}")
    with open(os.path.join(out, "eda_report.txt"), "r", encoding="utf-8") as f:
        rep = f.read()
    check("报告里有各层权重表", "block8" in rep and "block9" in rep, rep[:120])


def test_missing_counts(dirs, tmp):
    print("\n[5] 缺 expert_counts.pt 时报错可读")
    bad = os.path.join(tmp, "bad_client")
    make_client(bad, 0)
    os.remove(os.path.join(bad, "expert_counts.pt"))
    try:
        agg.aggregate([bad, bad, bad, bad], os.path.join(tmp, "out_bad"),
                      eda_mode=True)
        check("应当抛出 FileNotFoundError", False, "居然没报错")
    except FileNotFoundError as e:
        check("抛出 FileNotFoundError 且提示清楚",
              "expert_counts.pt" in str(e) and "FEDVLA_MOE=1" in str(e),
              str(e)[:120])


def test_block_map_helper():
    print("\n[6] block 号解析")
    names = ["trunk.trunk.blocks.8.mlp.moe", "trunk.trunk.blocks.9.mlp.moe"]
    check("从模块路径解析出 block 号", block_indices_from_names(names) == [8, 9],
          f"{block_indices_from_names(names)}")


def test_cluster_variant():
    print("\n[7] 簇感知聚合（簇内加权 + 簇间等权）")
    from eda import cluster_clients, eda_cluster_weights

    # 3 个客户端：c0/c1 同任务，c2 另一任务
    V = torch.zeros(3, 2, 8, dtype=torch.float64)
    V[0, :, 0] = 100.0
    V[0, :, 1] = 10.0
    V[1, :, 0] = 100.0
    V[1, :, 1] = 10.0
    V[2, :, 5] = 100.0
    V[2, :, 6] = 10.0

    cl = cluster_clients(V)
    check("自动分簇：c0/c1 同簇，c2 单独", cl[0] == cl[1] and cl[2] != cl[0],
          f"{cl}")
    check("簇数为 2", len(set(cl)) == 2, f"{cl}")

    cl3 = cluster_clients(V, num_clusters=3)
    check("指定 3 簇时每个客户端自成一簇",
          len(set(cl3)) == 3, f"{cl3}")

    wc = eda_cluster_weights(V, cl)
    check("簇感知权重形状 = (L, N)", tuple(wc.shape) == (2, 3), f"{tuple(wc.shape)}")
    check("每层权重和为 1",
          bool(torch.allclose(wc.sum(dim=1), torch.ones(2, dtype=torch.float64),
                              atol=1e-9)), f"{wc.sum(dim=1)}")
    # 簇 {c0,c1} 与簇 {c2} 各占一半总量
    check("同簇两成员权重之和 = 另一簇的权重（簇间等权）",
          abs(float(wc[0, 0] + wc[0, 1]) - float(wc[0, 2])) < 1e-9,
          f"{wc[0].tolist()}")

    # 与中心性 EDA 的关键差异：
    # 中心性 —— 孤立的 c2 谁都不像 → 权重最低
    # 簇感知 —— 每个簇等权 → c2（单成员簇）拿到的总权重份额更大
    from eda import eda_weights
    we = eda_weights(V)
    check("中心性 EDA：孤立客户端 c2 权重最低",
          float(we[0, 2]) < float(we[0, 0]), f"{we[0].tolist()}")
    check("簇感知：c2 不再被压到最低（与中心性行为不同）",
          float(wc[0, 2]) > float(we[0, 2]),
          f"cluster={float(wc[0,2]):.4f} vs centrality={float(we[0,2]):.4f}")


def main():
    tmp = tempfile.mkdtemp(prefix="fedvla_agg_")
    try:
        dirs = build_dirs(tmp)
        test_fedavg(dirs, tmp)
        test_eda(dirs, tmp)
        test_missing_counts(dirs, tmp)
        test_block_map_helper()
        test_cluster_variant()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"!! {len(FAILED)} 项失败: {FAILED}")
        return 1
    print("全部通过  ALL_TESTS_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
