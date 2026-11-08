#!/usr/bin/env python
"""MoE 评估链路体检：确认 policy 构造路径与 MoE 键的匹配情况。

背景（2026-11-06 核查）：
  - `hpt/run_eval.py` 用 `hydra.utils.instantiate(cfg.network)` 构造 policy
    → Policy.__init__ → `_create_policy_trunk` → 只有 `FEDVLA_MOE == "1"`
      时才创建 MoEBlock
  - `Policy.load_model` 用 `self.load_state_dict(sd, strict=False)`
    → **缺失的 MoE 键会被静默吞掉**，评估不报错，但评的是残缺模型

本脚本做三件事：
  1) 扫描已产出的 MoE 模型权重，统计 MoE 键的数量与所在 block
  2) 分别在 `FEDVLA_MOE=0` / `=1` 两种环境下构造 policy，
     对比 state_dict 键集合与实际加载的 missing/unexpected keys
  3) 打印结论：评估是否必须显式注入 MoE 环境变量
"""
import os
import sys
import json

import torch

# hpt 是 /root/HPT 下的源码包，脚本从 /root/HPT 运行时要能 import 到
sys.path.insert(0, "/root/HPT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")


def scan_ckpt(path):
    sd = torch.load(path, map_location="cpu")
    keys = list(sd.keys())
    moe = [k for k in keys if "moe" in k.lower()]
    blocks = sorted({k.split(".")[3] for k in moe if len(k.split(".")) > 4})
    return {
        "path": path,
        "total_keys": len(keys),
        "moe_keys": len(moe),
        "blocks_with_moe": blocks,
        "moe_sample": moe[:4],
    }


def build_and_load(cfg_path, model_path, moe_on):
    """在给定 MoE 开关下构造 policy 并加载模型，报告键匹配情况。"""
    os.environ["FEDVLA_MOE"] = "1" if moe_on else "0"
    os.environ.setdefault("FEDVLA_MOE_EXPERTS", "8")
    os.environ.setdefault("FEDVLA_MOE_TOPK", "2")
    os.environ.setdefault("FEDVLA_MOE_EXPERT_RATIO", "0.25")
    os.environ.setdefault("FEDVLA_MOE_LAYERS", "half")

    from omegaconf import OmegaConf
    from hpt.models.policy import Policy
    import hydra

    cfg = OmegaConf.load(cfg_path)
    net_cfg = cfg.network
    # run_eval.py 用的是 hydra.utils.instantiate(cfg.network)
    policy = hydra.utils.instantiate(net_cfg)
    if isinstance(policy, dict):
        policy = hydra.utils.instantiate(net_cfg["_target_"], **{
            k: v for k, v in net_cfg.items() if k != "_target_"})

    psd = policy.state_dict()
    p_moe = [k for k in psd if "moe" in k.lower()]
    sd = torch.load(model_path, map_location="cpu")
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    return {
        "moe_on": moe_on,
        "policy_total_keys": len(psd),
        "policy_moe_keys": len(p_moe),
        "loaded_missing": len(missing),
        "loaded_unexpected": len(unexpected),
        "missing_moe": len([k for k in missing if "moe" in k.lower()]),
        "unexpected_moe": len([k for k in unexpected if "moe" in k.lower()]),
        "missing_sample": missing[:5],
        "unexpected_sample": unexpected[:5],
        "n_params_M": round(sum(p.numel() for p in policy.parameters()) / 1e6, 3),
    }


def main():
    out = {}
    cands = sys.argv[1:] or ["/root/HPT/output/p3_moe_fedavg/global_r49"]
    scans = []
    for d in cands:
        for fn in ("model.pth", "trunk.pth"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                scans.append(scan_ckpt(p))
    out["ckpt_scan"] = scans
    print("=== 1) 权重扫描 ===")
    print(json.dumps(scans, indent=2, ensure_ascii=False))

    # 找一份 MoE 模型的 config.yaml + model.pth 做构造对比
    probe = None
    for d in cands:
        cfg_p = os.path.join(d, "config.yaml")
        m_p = os.path.join(d, "model.pth")
        if os.path.exists(cfg_p) and os.path.exists(m_p):
            probe = (cfg_p, m_p)
            break
    if probe is None:
        print("!! 找不到可用于构造对比的 config.yaml + model.pth")
        return 0

    print("=== 2) policy 构造对比 ===")
    cmp_res = []
    for moe_on in (False, True):
        try:
            r = build_and_load(probe[0], probe[1], moe_on)
            cmp_res.append(r)
            print(json.dumps(r, indent=2, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            cmp_res.append({"moe_on": moe_on, "error": f"{type(e).__name__}: {e}"})
            print(f"[moe_on={moe_on}] ERROR {type(e).__name__}: {e}")
    out["construct_cmp"] = cmp_res

    print("=== 3) 结论 ===")
    ok = [r for r in cmp_res if "error" not in r]
    if len(ok) == 2:
        off, on = ok[0], ok[1]
        if off["policy_moe_keys"] == 0 and on["policy_moe_keys"] > 0:
            print("!! 确认：FEDVLA_MOE 未设时 policy 不含任何 MoE 模块；"
                  "而权重含 MoE 键 → strict=False 会静默丢弃 → "
                  "评估的是残缺模型。评估必须显式设 FEDVLA_MOE=1。")
            print(f"   moe_off 参数 {off['n_params_M']}M / unexpected_moe="
                  f"{off['unexpected_moe']}")
            print(f"   moe_on  参数 {on['n_params_M']}M / missing_moe="
                  f"{on['missing_moe']}")
        elif off["policy_moe_keys"] > 0:
            print("?? 两个环境都含 MoE 模块，请复核构造路径")
        else:
            print("?? 未观察到 MoE 键差异，请复核")
    return 0


if __name__ == "__main__":
    sys.exit(main())
