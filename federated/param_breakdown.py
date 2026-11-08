#!/usr/bin/env python
"""第4阶段：实测模型的参数量分解，核对"上行通信量"的口径。

背景：门2 的 comm_cost.py 把上行量按**纯 trunk** 算（12,644,352 参数 × 4B）。
但 aggregate.py 实际还平均了 normalizer.<domain> 与 modalities_tokens ——
这些是各客户端**不同**的（所以要平均），意味着它们**也要上行**。
本脚本给出实测分解，判断门2 的口径是否低估。
"""
import os
import sys

import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/HPT/output/p3_moe_eda_cluster/global_r49/model.pth"

sd = torch.load(MODEL, map_location="cpu", weights_only=False)
groups = {}
for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    p = k.split(".")[0]
    groups[p] = groups.get(p, 0) + v.numel()

tot = sum(groups.values())
print(f"模型：{MODEL}")
print(f"{'前缀':<22}{'参数量':>14}{'fp32 大小':>12}{'占比':>8}")
print("-" * 60)
for p, n in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"{p:<22}{n:>14,}{n*4/1048576:>10.2f} MiB{n/tot:>8.1%}")
print("-" * 60)
print(f"{'TOTAL':<22}{tot:>14,}{tot*4/1048576:>10.2f} MiB")

trunk = groups.get("trunk", 0)
personal = groups.get("stems", 0) + groups.get("heads", 0)
shared = tot - personal
print()
print(f"纯 trunk（门2 的口径）      = {trunk:>14,}  {trunk*4/1048576:>7.2f} MiB")
print(f"共享上行（真实，trunk+其他） = {shared:>14,}  {shared*4/1048576:>7.2f} MiB")
print(f"个性化 stems+heads（不上行）= {personal:>14,}  {personal*4/1048576:>7.2f} MiB")
print()
print(f"→ 门2 的 comm_cost.py 口径低估倍数 = {shared/trunk:.3f}x")
print()
print("单轮通信量（4 客户端，fp32）：")
for label, up_per in (("门2 口径(纯trunk)", trunk), ("真实口径(共享参数)", shared)):
    up = 4 * up_per * 4
    down = up_per * 4
    print(f"  {label:<22} 上行 {up/1048576:>7.2f} MiB  下行 {down/1048576:>7.2f} MiB"
          f"  单轮合计 {(up+down)/1048576:>7.2f} MiB")
print()
print("normalizer 的实际键名样例：")
nk = [k for k in sd if k.startswith("normalizer.")]
for k in nk[:6]:
    print(f"  {k}")
print(f"  共 {len(nk)} 个 normalizer 键")
