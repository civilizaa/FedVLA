"""FedVLA 服务器端聚合（FedAvg）。

每个联邦轮结束后调用一次：把 4 个客户端的 `model.pth`（完整 state_dict）读出，
只平均 **trunk** 参数（键前缀 `trunk.`），然后写回全局目录：

  <out>/trunk.pth    —— trunk-only state_dict（键前缀 `trunk.`，对应 Policy.trunk）
  <out>/model.pth    —— 完整 state_dict（平均后的 trunk + 各客户端 stem/head 并集），用于最终 eval
  <out>/config.yaml  —— 从 client0 复制（HPT init_policy 用它实例化同架构）
  <out>/config.json  —— 若存在也复制

为什么只聚合 trunk：HPT 的 stem/head 是 **按域（domain）的 ModuleDict**（stems.<domain>_*、
heads.<domain>），不同客户端的域不同，无法跨客户端平均，且 FedVLA 本就是"trunk 共享聚合、
stem/head 个性化"。所以全局模型 = 平均 trunk + 各客户端 stem/head 并集。

依赖：torch（在 hpt 环境里跑）。用法见 federated/README.md。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch


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



def aggregate(client_dirs: list[str], out_dir: str, weights: list[float] | None = None):
    os.makedirs(out_dir, exist_ok=True)
    client_models = [os.path.join(d, "model.pth") for d in client_dirs]
    states = [_load_state(m) for m in client_models]

    print(f"[aggregate] 客户端数={len(states)}  权重={weights}")
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
    p = argparse.ArgumentParser(description="FedVLA FedAvg 聚合")
    p.add_argument("--clients", nargs="+", required=True,
                   help="各客户端输出目录（含 model.pth），空格分隔")
    p.add_argument("--out", required=True, help="全局模型输出目录")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="可选：各客户端样本量权重（FedAvg 样本加权）；省略则等权")
    args = p.parse_args(argv)

    if args.weights is not None and len(args.weights) != len(args.clients):
        p.error("--weights 数量必须与 --clients 一致")

    aggregate(args.clients, args.out, args.weights)


if __name__ == "__main__":
    main()
