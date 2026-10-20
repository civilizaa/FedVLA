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


def build_global_model(client_states: list[dict], trunk_avg: dict) -> dict:
    """用 client0 的完整 state_dict 打底，写入平均 trunk，并并集其余客户端的 stem/head。"""
    full = dict(client_states[0])  # 浅拷贝，避免改到原 dict
    for k, v in trunk_avg.items():
        full[k] = v
    # 并集 stem/head：其余客户端里、full 还没有的域参数补进来
    for s in client_states[1:]:
        for k, v in s.items():
            if k.startswith("stems.") or k.startswith("heads."):
                if k not in full:
                    full[k] = v
    return full


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


def aggregate(client_dirs: list[str], out_dir: str, weights: list[float] | None = None):
    os.makedirs(out_dir, exist_ok=True)
    client_models = [os.path.join(d, "model.pth") for d in client_dirs]
    states = [_load_state(m) for m in client_models]

    print(f"[aggregate] 客户端数={len(states)}  权重={weights}")
    trunk_avg = average_trunk(states, weights)
    full = build_global_model(states, trunk_avg)
    trunk_only = extract_trunk_only(trunk_avg)

    torch.save(full, os.path.join(out_dir, "model.pth"))
    torch.save(trunk_only, os.path.join(out_dir, "trunk.pth"))
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
