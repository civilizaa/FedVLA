"""校验联邦全局模型的异构完整性。

检查 global_rN/model.pth 是否覆盖所有客户端的域的 stems/heads/normalizer，
以及 trunk 是否真的是 4 个客户端的平均（而非某一个客户端的拷贝）。
"""
import sys
from collections import Counter

import torch

GLOBAL = sys.argv[1] if len(sys.argv) > 1 else "/root/HPT/output/smoke_hetero/global_r1"
CLIENTS = [f"/root/HPT/output/smoke_hetero/client_{c}_r1" for c in range(4)]

g = torch.load(f"{GLOBAL}/model.pth", map_location="cpu", weights_only=False)
print(f"=== 全局模型 {GLOBAL} ===")
print("顶层键前缀:", Counter(k.split(".")[0] for k in g).most_common())

def domains_of(st, prefix):
    """从形如 {prefix}.{domain}.xxx 的键里提取 domain 集合"""
    out = set()
    for k in st:
        parts = k.split(".")
        if parts[0] == prefix and len(parts) > 1:
            out.add(parts[1])
    return sorted(out)

for prefix in ("stems", "heads", "normalizer"):
    gd = domains_of(g, prefix)
    print(f"  {prefix}: 全局覆盖域 = {gd}")

print()
print("=== 各客户端的域 ===")
client_st = []
for i, d in enumerate(CLIENTS):
    st = torch.load(f"{d}/model.pth", map_location="cpu", weights_only=False)
    client_st.append(st)
    print(f"  client{i}: stems={domains_of(st,'stems')} heads={domains_of(st,'heads')} "
          f"normalizer={domains_of(st,'normalizer')}")

print()
print("=== 全局是否覆盖所有客户端的域 ===")
all_domains = set()
for st in client_st:
    all_domains |= set(domains_of(st, "stems"))
g_stems = set(domains_of(g, "stems"))
g_heads = set(domains_of(g, "heads"))
g_norm = set(domains_of(g, "normalizer"))
missing = all_domains - g_stems
print("  客户端域并集:", sorted(all_domains))
print("  全局 stems 缺失:", sorted(missing) if missing else "无 ✓")
if all_domains - g_heads:
    print("  [!] 全局 heads 缺失:", sorted(all_domains - g_heads))
if all_domains - g_norm:
    print("  [!] 全局 normalizer 缺失:", sorted(all_domains - g_norm))

print()
print("=== trunk 是否为 4 客户端的平均（不是拷贝）===")
trunk_keys = [k for k in g if k.startswith("trunk.")]
n = len(client_st)
max_dev = 0.0
worst = None
for k in trunk_keys[:40]:
    vals = [st[k].float() for st in client_st if k in st]
    if len(vals) != n:
        continue
    avg = sum(vals) / n
    dev = float((avg - g[k].float()).abs().max())
    if dev > max_dev:
        max_dev = dev
        worst = k
print(f"  全局 trunk 与客户端均值的最大偏差 = {max_dev:.3e} @ {worst}")
if max_dev < 1e-6:
    print("  [OK] 全局 trunk 确实是 4 客户端的等权平均")
else:
    print("  [!!] 全局 trunk 不等于客户端均值 —— 聚合逻辑有问题")

# 额外验证：全局 trunk 与单个客户端的差异应明显大于与均值的差异
k0 = trunk_keys[0]
v0 = client_st[0][k0].float()
avg0 = sum(st[k0].float() for st in client_st) / n
print(f"  样例键 {k0}:")
print(f"    |global - client0| = {float((g[k0].float()-v0).abs().max()):.3e}")
print(f"    |global - mean4|   = {float((g[k0].float()-avg0).abs().max()):.3e}")

print()
print("=== extras.pth 存在性 ===")
import os
print("  extras.pth:", os.path.exists(f"{GLOBAL}/extras.pth"))
print("  trunk.pth :", os.path.exists(f"{GLOBAL}/trunk.pth"))
print("  config.yaml:", os.path.exists(f"{GLOBAL}/config.yaml"))
print("VERIFY_DONE")
