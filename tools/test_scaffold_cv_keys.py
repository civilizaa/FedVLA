#!/usr/bin/env python3
"""SCAFFOLD c_k 键空间错配的离线单测（铁律 E + 铁律 R）。

复现 2026-11-23 云端真实崩溃：
    File "/root/fedcore/fed_baselines.py", line 237, in scaffold_update_local_cv
        out[k] = c_local[k].clone()
    KeyError: 'trunk.trunk.blocks.8.mlp.moe.gate.weight'

场景：round 0 的初始模型**不含 MoE** ⇒ c_k 只有 224 键；
      round 1 的模型含 MoE ⇒ w_g 有 488 键 ⇒ c_k 缺 264 个 expert 键。

正确语义：此前未出现的参数，其控制变量按 **0** 起算（标准 SCAFFOLD）。
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "federated"))

from fed_baselines import scaffold_update_local_cv, scaffold_keys  # noqa: E402


def build():
    """w_g/w_l 含 MoE（488 键量级）；c_l/c_g 只含非 MoE 部分（224 键量级）。"""
    w_g, w_l = {}, {}
    for i in range(2):
        for j in range(3):
            k = f"trunk.trunk.blocks.{i}.mlp.fc{j}.weight"
            w_g[k] = torch.randn(4, 4)
            w_l[k] = torch.randn(4, 4)
    # MoE 专属键：c_l / c_g 里没有（模拟 round0 无 MoE → round1 才有）
    for i in range(2):
        k = f"trunk.trunk.blocks.{i}.mlp.moe.gate.weight"
        w_g[k] = torch.randn(8, 8)
        w_l[k] = torch.randn(8, 8)
        k2 = f"trunk.trunk.blocks.{i}.mlp.moe.experts.0.fc1.weight"
        w_g[k2] = torch.randn(8, 4)
        w_l[k2] = torch.randn(8, 4)
    c_l = {k: torch.randn_like(v) for k, v in w_g.items()
           if "moe" not in k}
    c_g = {k: torch.randn_like(v) for k, v in w_g.items()
           if "moe" not in k}
    return c_l, c_g, w_g, w_l


def main() -> int:
    c_l, c_g, w_g, w_l = build()
    keys = scaffold_keys(w_g)
    print(f"w_g 键数={len(w_g)}  c_l 键数={len(c_l)}  参与键数={len(keys)}")

    # 1) 不得抛 KeyError
    try:
        out = scaffold_update_local_cv(c_l, c_g, w_g, w_l, 44, 1e-4, keys)
    except KeyError as e:
        print(f"[FAIL] 仍然 KeyError: {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 抛异常 {type(e).__name__}: {e}")
        return 1

    # 2) 输出必须覆盖 w_g 的全部参与键（MoE 键也必须有控制变量）
    missing = [k for k in keys if k not in out]
    if missing:
        print(f"[FAIL] 输出缺 {len(missing)} 个键，例：{missing[:3]}")
        return 1
    print(f"[OK] 无 KeyError，输出覆盖全部 {len(out)} 个参与键")

    # 3) 缺失 c_k 的键必须**按 0 起算**：
    #    c_k = 0 - c(=0) + (w_g - w_l)/(K*eta)
    moe_keys = [k for k in keys if "moe" in k]
    for k in moe_keys[:3]:
        expect = (w_g[k].float() - w_l[k].float()) / (44 * 1e-4)
        if not torch.allclose(out[k].float(), expect, atol=1e-3):
            print(f"[FAIL] {k} 的 c_k 起点不是 0 起算")
            return 1
    print(f"[OK] {len(moe_keys)} 个 MoE 键的 c_k 均按 0 起算（标准 SCAFFOLD）")

    # 4) 反向：c_k 存在时必须保留原值（不能被静默清零）
    out2 = scaffold_update_local_cv(c_l, c_g, w_g, w_l, 44, 1e-4, keys)
    for k in [k for k in keys if "moe" not in k][:3]:
        expect = (c_l[k].float() - c_g[k].float()
                  + (w_g[k].float() - w_l[k].float()) / (44 * 1e-4))
        if not torch.allclose(out2[k].float(), expect, atol=1e-3):
            print(f"[FAIL] {k} 的 c_k 更新公式不对")
            return 1
    print("[OK] 已存在 c_k 的键更新公式正确（c_k⁺ = c_k − c + (w_g−w_l)/(K·η)）")

    print("\n===== 结论：键空间错配场景已修复 ✅ =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
