import torch
import os

p = "/root/HPT/output/smoke_hetero/global_r0/extras.pth"
e = torch.load(p, map_location="cpu", weights_only=False)
print("extras keys:", len(e))
for k, v in e.items():
    print("  ", k, tuple(v.shape))

print()
print("=== client_0_r0/model.pth 顶层键前缀分布 ===")
m = torch.load("/root/HPT/output/smoke_hetero/client_0_r0/model.pth", map_location="cpu", weights_only=False)
from collections import Counter
c = Counter(k.split(".")[0] for k in m)
print(c.most_common())

print()
print("=== normalizer 相关键样例 ===")
nk = [k for k in m if k.startswith("normalizer")]
print("normalizer 键数:", len(nk))
for k in nk[:6]:
    print("  ", k, tuple(m[k].shape))

print()
print("=== modalities_tokens / action_tokens 键 ===")
for k in m:
    if k.startswith("modalities_tokens") or k.startswith("action_tokens"):
        print("  ", k, tuple(m[k].shape))

print()
print("=== config.yaml token_postprocessing ===")
import re
txt = open("/root/HPT/output/smoke_hetero/global_r0/config.yaml").read()
for line in txt.splitlines():
    if "token_postprocessing" in line or "modality" in line.lower():
        print("  ", line)
