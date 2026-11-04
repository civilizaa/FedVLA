"""FedVLA 服务器端聚合（FedAvg / EDA）。

每个联邦轮结束后调用一次：把 4 个客户端的 `model.pth`（完整 state_dict）读出，
只平均 **trunk** 参数（键前缀 `trunk.`），然后写回全局目录：

  <out>/trunk.pth    —— trunk-only state_dict（键前缀 `trunk.`，对应 Policy.trunk）
  <out>/model.pth    —— 完整 state_dict（平均后的 trunk + 各客户端 stem/head 并集），用于最终 eval
  <out>/config.yaml  —— 从 client0 复制（HPT init_policy 用它实例化同架构）
  <out>/config.json  —— 若存在也复制

为什么只聚合 trunk：HPT 的 stem/head 是 **按域（domain）的 ModuleDict**（stems.<domain>_*、
heads.<domain>），不同客户端的域不同，无法跨客户端平均，且 FedVLA 本就是"trunk 共享聚合、
stem/head 个性化"。所以全局模型 = 平均 trunk + 各客户端 stem/head 并集。

两种聚合模式
------------
  * FedAvg（默认）：所有层、所有参数用同一组（可样本加权的）权重。
  * EDA（`--eda`）：**逐层**加权 —— 每个 MoE 层用自己那一行 EDA 权重，
    权重来自各客户端的专家选择计数（`expert_counts.pt`）。
    这就是 H2「按簇/中心性聚合更优」的实验路径。

依赖：torch（在 hpt 环境里跑）。用法见 federated/README.md。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import (aggregate_state_dicts, block_indices_from_names,  # noqa: E402
                 cluster_clients, cluster_report, eda_cluster_weights,
                 eda_diagnostics, eda_weights, format_weight_report)


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到客户端权重：{path}")
    # weights_only=False：HPT 的 model.pth 是普通 state_dict，torch>=2.1 默认即可载入
    return torch.load(path, map_location="cpu", weights_only=False)


def _is_trunk_key(k: str) -> bool:
    return k.startswith("trunk.")


def average_trunk(client_states: list[dict], weights: list[float] | None = None) -> dict:
    """对 trunk 参数做（可选样本加权的）均值。

    client_states[i] 是完整 state_dict；trunk 键形如 `trunk.trunk.xxx`。
    返回的字典键同 client_states 里的 trunk 键（带 `trunk.` 前缀），供直接写回完整 model.pth。
    """
    trunk_keys = [k for k in client_states[0] if _is_trunk_key(k)]
    if not trunk_keys:
        raise KeyError("未在客户端 state_dict 中找到任何 `trunk.` 前缀参数；请检查 HPT 版本。")

    if weights is None:
        weights = [1.0] * len(client_states)
    assert len(weights) == len(client_states), "weights 与客户端数不一致"
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("weights 之和必须 > 0")
    wnorm = [w / wsum for w in weights]

    avg: dict = {}
    for k in trunk_keys:
        # 形状必须一致才能平均
        try:
            stacked = torch.stack([s[k] for s in client_states], dim=0)
        except RuntimeError as e:
            raise RuntimeError(f"客户端 trunk 参数形状不一致，键={k}: {e}")
        avg[k] = sum(wnorm[i] * client_states[i][k] for i in range(len(client_states)))
    return avg


def average_trunk_eda(client_states: list[dict], weights: torch.Tensor,
                      block_map: list[int]) -> dict:
    """EDA 逐层加权聚合 trunk。

    weights  : (L, N)，第 r 行对应 block_map[r] 号 transformer block
    block_map: 行号 -> transformer block 号

    非 MoE block 的参数与共享参数用"沿层平均"的权重
    （等价于整体中心性权重；见 eda.py 的说明）。
    """
    trunk_keys = [k for k in client_states[0] if _is_trunk_key(k)]
    if not trunk_keys:
        raise KeyError("未在客户端 state_dict 中找到任何 `trunk.` 前缀参数；请检查 HPT 版本。")

    sub = [{k: v for k, v in sd.items() if _is_trunk_key(k)} for sd in client_states]
    avg = aggregate_state_dicts(sub, weights=weights, num_layers=weights.shape[0],
                                 block_map=block_map)
    # 键前缀在 aggregate_state_dicts 里被保留（我们只截取了 trunk.* 的子集）
    return avg


def load_expert_counts(client_dirs: list[str]):
    """读各客户端的 expert_counts.pt，返回 (V, names)。

    V: (N, L, K) 的选择计数张量；names: 对应的模块路径列表。
    """
    mats, names = [], None
    for d in client_dirs:
        p = os.path.join(d, "expert_counts.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"找不到专家计数文件：{p}\n"
                f"  EDA 模式需要客户端开启 FEDVLA_MOE=1 并导出 expert_counts.pt。")
        blob = torch.load(p, map_location="cpu", weights_only=False)
        M = blob["counts"] if isinstance(blob, dict) else blob
        if M.dim() != 2:
            raise ValueError(f"{p}: counts 应为 (L, K)，实际 {tuple(M.shape)}")
        mats.append(M.double())
        if names is None:
            names = list(blob["names"]) if isinstance(blob, dict) else []
    L = mats[0].shape[0]
    for i, m in enumerate(mats):
        if m.shape != mats[0].shape:
            raise ValueError(f"客户端 {i} 的专家计数形状 {tuple(m.shape)} "
                             f"与客户端 0 的 {tuple(mats[0].shape)} 不一致")
    V = torch.stack(mats, dim=0)          # (N, L, K)
    return V, (names or [])



def _tensor_equal_shape(vals: list) -> bool:
    shp = None
    for v in vals:
        if not torch.is_tensor(v):
            return False
        if shp is None:
            shp = tuple(v.shape)
        elif tuple(v.shape) != shp:
            return False
    return True


def build_global_model(client_states: list[dict], trunk_avg: dict,
                       weights: list[float] | None = None) -> dict:
    """用 client0 的完整 state_dict 打底，写入平均 trunk，并并集其余客户端的非 trunk 键。

    为什么不能只并集 `stems.` / `heads.`：
        门0 实测 HPT 完整 state_dict 的顶层前缀分布为
        `trunk(224) / stems(18) / normalizer(12) / heads(8) / modalities_tokens(2)`。
        其中 `normalizer.<domain>`、`modalities_tokens.<modality>` 同样是**按域/按模态分开存**的，
        异构客户端（task1/task2/task3 各一个）只并集 stems/heads 会导致：
          - 全局 model.pth 缺少其它域的 normalizer → 最终 eval 在该域上直接崩；
          - modalities_tokens 缺失 → trunk 里依赖 modality token 的分支失效。
        所以这里对**所有非 trunk 键**做并集。

    同名键的处理：
        形状一致 → 按权重平均（normalizer 统计量、共享 modality token 这类同形参数平均是合理的）；
        形状不一致 → 取第一个存在的（不平均，避免破坏结构），并打印警告。
    """
    full = dict(client_states[0])  # 浅拷贝，避免改到原 dict
    for k, v in trunk_avg.items():
        full[k] = v

    if weights is None:
        weights = [1.0] * len(client_states)
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("weights 之和必须 > 0")
    wnorm = [w / wsum for w in weights]

    # 收集所有非 trunk 键
    groups: dict = {}
    for i, s in enumerate(client_states):
        for k, v in s.items():
            if _is_trunk_key(k):
                continue
            groups.setdefault(k, []).append((i, v))

    conflicts = []
    for k, items in groups.items():
        vals = [v for _, v in items]
        if len(items) == 1:
            full[k] = vals[0]
            continue
        if _tensor_equal_shape(vals):
            full[k] = sum(wnorm[i] * v for i, v in items)
        else:
            full[k] = vals[0]
            conflicts.append(k)

    if conflicts:
        print(f"[aggregate][WARN] {len(conflicts)} 个同名非 trunk 键形状不一致，已取首个客户端的值：")
        for k in conflicts[:10]:
            print(f"           - {k}")
    print(f"[aggregate] 非 trunk 键并集完成，共 {len(groups)} 个键"
          f"（前缀分布：{_prefix_hist(groups)}）")
    return full


def _prefix_hist(groups: dict) -> list:
    from collections import Counter
    return Counter(k.split(".")[0] for k in groups).most_common(10)


def extract_trunk_only(trunk_avg: dict) -> dict:
    """把完整 state_dict 里的 `trunk.trunk.xxx` 变成 trunk.pth 需要的 `trunk.xxx`。

    Policy.trunk 是 ModuleDict（内部键名 `trunk`），其 state_dict 键为 `trunk.xxx`；
    load_trunk 用 `self.trunk.load_state_dict(torch.load(path))` 载入。
    """
    out = {}
    for k, v in trunk_avg.items():
        assert k.startswith("trunk."), f"意外的 trunk 键：{k}"
        out[k[len("trunk."):]] = v  # 'trunk.trunk.xxx' -> 'trunk.xxx'
    return out


def extract_extras(full: dict) -> dict:
    """挑出**不属于 trunk、但属于策略顶层且跨客户端同形**的参数。

    为什么要单独导出：
        源码确认 `self.modalities_tokens` 是 `nn.ParameterDict`（按 modality 命名），
        `self.action_tokens` 条件式为 ParameterDict（按 domain 命名）——
        它们**不在 `self.trunk` 内部**，所以 trunk.pth 里没有它们。
        而 `load_trunk` 只载入 trunk → 客户端续训时模态 token 会保持随机初始化，
        导致同一个共享 trunk 在各客户端拿到不一致的输入。

        本函数导出这些键到 `extras.pth`，补丁后的 `load_trunk` 会顺带载入
        （用 policy.load_state_dict(strict=False)，键路径 `modalities_tokens.<modality>` 天然匹配）。
    """
    out = {}
    for k, v in full.items():
        if k.startswith("modalities_tokens.") or k.startswith("action_tokens."):
            out[k] = v
    return out



def aggregate(client_dirs: list[str], out_dir: str, weights: list[float] | None = None,
              eda_mode: bool = False, client_tasks: list[str] | None = None,
              eda_cluster: bool = False, num_clusters: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    client_models = [os.path.join(d, "model.pth") for d in client_dirs]
    states = [_load_state(m) for m in client_models]

    if eda_mode:
        print(f"[aggregate] EDA 模式：客户端数={len(states)}"
              f"{'（簇感知：簇内加权 + 簇间等权）' if eda_cluster else '（中心性加权）'}")
        V, names = load_expert_counts(client_dirs)
        block_map = block_indices_from_names(names) if names else list(range(V.shape[1]))
        clusters = None
        if eda_cluster:
            clusters = cluster_clients(V, num_clusters=num_clusters)
            print(f"[aggregate] 客户端分簇结果: "
                  f"{dict(zip([f'c{i}' for i in range(len(states))], clusters))}")
            print(cluster_report(V, clusters, client_tasks))
            w = eda_cluster_weights(V, clusters)
        else:
            w = eda_weights(V)                    # (L, N)
        if len(block_map) != w.shape[0]:
            raise ValueError(f"从专家计数名解析出 {len(block_map)} 个 block，"
                             f"但权重的层数是 {w.shape[0]}；两者必须一致")
        print(f"[aggregate] EDA 权重形状={tuple(w.shape)}  对应 block={block_map}")
        print(format_weight_report(w, client_tasks or [f"c{i}" for i in range(len(states))],
                                   layer_names=[f"block{b}" for b in block_map]))
        diag = eda_diagnostics(V, client_tasks)
        # 非 trunk 参数（normalizer 统计量 / modalities_tokens）也用 EDA 的平均权重，
        # 保证"整份模型"用的是同一套客户端重要性判断。
        weights = diag["mean_weights"].tolist()
        trunk_avg = average_trunk_eda(states, w, block_map)
        # 把权重与诊断落盘，论文表格/图直接引用
        torch.save({"weights": w, "block_map": block_map,
                    "counts": V, "names": names, "clusters": clusters},
                   os.path.join(out_dir, "eda_weights.pt"))
        with open(os.path.join(out_dir, "eda_report.txt"), "w", encoding="utf-8") as f:
            f.write(format_weight_report(
                w, client_tasks or [f"c{i}" for i in range(len(states))],
                layer_names=[f"block{b}" for b in block_map]))
            f.write("\n\n平均权重: ")
            f.write(", ".join(f"c{i}={v:.4f}" for i, v in
                              enumerate(diag["mean_weights"].tolist())))
            f.write("\n")
            if clusters is not None:
                f.write("\n簇划分: " + str(clusters) + "\n")
                f.write(cluster_report(V, clusters, client_tasks) + "\n")
            if client_tasks:
                f.write("同任务相似度: " + ", ".join(f"{v:.4f}" for v in diag["within_task_sim"]))
                f.write("\n异任务相似度: " + ", ".join(f"{v:.4f}" for v in diag["across_task_sim"]))
                f.write("\n")
    else:
        print(f"[aggregate] FedAvg 模式：客户端数={len(states)}  权重={weights}")
        trunk_avg = average_trunk(states, weights)

    full = build_global_model(states, trunk_avg, weights)
    trunk_only = extract_trunk_only(trunk_avg)

    torch.save(full, os.path.join(out_dir, "model.pth"))
    torch.save(trunk_only, os.path.join(out_dir, "trunk.pth"))
    extras = extract_extras(full)
    if extras:
        torch.save(extras, os.path.join(out_dir, "extras.pth"))
        print(f"[aggregate] 已写出 extras.pth（{len(extras)} 个顶层共享参数）："
              f"{sorted(extras.keys())}")
    # 复制 config.yaml / config.json（HPT init_policy 与 run_eval 都需要）
    for fname in ("config.yaml", "config.json"):
        src = os.path.join(client_dirs[0], fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))
            print(f"[aggregate] 复制 {fname}")
    print(f"[aggregate] 已写出全局模型 -> {out_dir}  (trunk.pth / model.pth / config*)")


def main(argv=None):
    p = argparse.ArgumentParser(description="FedVLA 聚合（FedAvg / EDA）")
    p.add_argument("--clients", nargs="+", required=True,
                   help="各客户端输出目录（含 model.pth），空格分隔")
    p.add_argument("--out", required=True, help="全局模型输出目录")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="可选：各客户端样本量权重（FedAvg 样本加权）；省略则等权")
    p.add_argument("--eda", action="store_true",
                   help="用 EDA（专家驱动聚合）替代等权 FedAvg：逐层按专家计数相似度加权")
    p.add_argument("--eda-cluster", action="store_true",
                   help="簇感知聚合：先分簇，簇内中心性加权 + 簇间等权（需配合 --eda）")
    p.add_argument("--num-clusters", type=int, default=None,
                   help="簇感知聚合的目标簇数；省略则自动")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="可选：各客户端对应的任务名，用于 EDA 诊断报告")
    args = p.parse_args(argv)

    if args.weights is not None and len(args.weights) != len(args.clients):
        p.error("--weights 数量必须与 --clients 一致")
    if args.tasks is not None and len(args.tasks) != len(args.clients):
        p.error("--tasks 数量必须与 --clients 一致")
    if args.eda_cluster and not args.eda:
        p.error("--eda-cluster 需要配合 --eda 一起使用")

    aggregate(args.clients, args.out, args.weights, eda_mode=args.eda,
              client_tasks=args.tasks, eda_cluster=args.eda_cluster,
              num_clusters=args.num_clusters)



if __name__ == "__main__":
    main()
