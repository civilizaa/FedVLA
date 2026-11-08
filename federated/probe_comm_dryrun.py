#!/usr/bin/env python
"""第4阶段通信损伤干跑的体检段（独立脚本，避免 relay 传 heredoc）。

检查 /tmp/comm_dryrun/ 下两次聚合的产物：
  1. MoE 专家键一个不少（264 个，blocks 8~15）；
  2. 被丢弃客户端独占的域（task3）被上一轮全局模型正确回退补齐；
  3. 通信量统计写出来了；
  4. 各量化方案在真实权重上的误差与通信量（与门2 同源）。
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/HPT/federated")

TMP = "/tmp/comm_dryrun"
SRC = "/root/HPT/output/p3_moe_eda_cluster"


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def report(tag, path):
    print(f"--- {tag} ---")
    if not os.path.exists(path):
        print("   [FAIL] 输出不存在:", path)
        return False
    sd = load(path)
    mk = [k for k in sd if "moe" in k.lower()]
    blocks = sorted({int(k.split(".")[3]) for k in mk
                     if k.startswith("trunk.trunk.blocks.") and len(k.split(".")) > 4})
    doms = sorted({k.split(".")[1] for k in sd if k.startswith("normalizer.")})
    print(f"   总键数 = {len(sd)}   MoE 键数 = {len(mk)}   MoE blocks = {blocks}")
    print(f"   normalizer 覆盖的域 = {doms}")
    ok = True
    if len(mk) != 264:
        print("   [FAIL] MoE 键数不是 264")
        ok = False
    else:
        print("   [OK] MoE 键数 264")
    if blocks != list(range(8, 16)):
        print(f"   [FAIL] MoE 层不对：{blocks}")
        ok = False
    else:
        print("   [OK] MoE 层 = blocks 8~15")
    # normalizer 实际键名：normalizer.<domain>.params_dict.action.input_stats.mean
    if not any(k.startswith("normalizer.mujoco_metaworld_task3.") for k in sd):
        print("   [FAIL] 缺 task3 的 normalizer（回退补齐失败）")
        ok = False
    else:
        print("   [OK] task3 的 normalizer 在（回退补齐成功）")
    if not any(k.startswith("stems.mujoco_metaworld_task3") for k in sd):
        print("   [FAIL] 缺 task3 的 stems（回退补齐失败）")
        ok = False
    else:
        print("   [OK] task3 的 stems 在（回退补齐成功）")
    return ok


ok = report("丢包聚合", f"{TMP}/out_drop/model.pth")
ok = report("int8 量化聚合", f"{TMP}/out_int8/model.pth") and ok

print()
print("--- 通信量统计 ---")
for f in ("comm_drop.json", "comm_int8.json"):
    p = os.path.join(TMP, f)
    if os.path.exists(p):
        print(f"   {f}: {json.dumps(json.load(open(p)), ensure_ascii=False)[:240]}")
    else:
        print(f"   [FAIL] {f} 不存在")
        ok = False

print()
print("--- 量化误差与通信量（真实权重，门2 参照：int8=0.0050 int4=0.0895 bit1=0.5571）---")
try:
    from comm_channel import Quantizer
    sd = load(f"{SRC}/global_r49/model.pth")
    shared = {k: v for k, v in sd.items()
              if not k.startswith("stems.") and not k.startswith("heads.")}
    for s in ("fp32", "fp16", "int8_g", "int4_g", "bit1"):
        q = Quantizer(s)
        b = q.theoretical_bytes(shared)
        print(f"   {s:<8} rel_err = {q.mean_tensor_rel_err(sd):.4f}   "
              f"单客户端上行 = {b/1048576:>7.2f} MiB   压缩比 = {q.compression_ratio(shared):.4f}")
except Exception as e:
    print(f"   [WARN] 无法计算量化误差: {type(e).__name__}: {e}")

print()
print("DRYRUN_ALL_PASS" if ok else "DRYRUN_SOME_FAILED")
