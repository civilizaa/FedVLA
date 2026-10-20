"""【运行前必做】验证 HPT 能否从【本地 trunk.pth】续训。

为什么需要：federated/run_fedavg.sh 每轮聚合后，把全局 trunk 写成
`<global>/trunk.pth`，下一轮客户端靠 HPT 的 `init_policy` 走本地续训分支加载它。
但我从 HPT 源码扒到的 `Policy.load_trunk` 只显示了 `hf://` 分支，本地路径是否真能
加载不确定。这一步用最小、最可靠的路径决断：

  1) 用已知可用的 Policy.from_pretrained("hf://...") 拿到已加载的 trunk；
  2) 把 trunk 参数清零；
  3) 把原始 trunk 写成本地 /tmp/trunk_test.pth，调用 load_trunk(本地路径)；
  4) 对比是否装回原值。

输出 LOADED_TRUNK_MATCHES_SOURCE = True  -> 联邦管线可直接用（run_fedavg.sh 成立）。
输出 False                                  -> load_trunk 本地是空操作，需改用兜底自定义循环（见 README）。

用法（在 hpt 环境、有 HPT 源码的目录）：
    python federated/verify_continuation.py
"""
from __future__ import annotations

import os
import tempfile

import torch

from hpt.models.policy import Policy

SRC_HF = "hf://liruiw/hpt-base"


def main():
    print("[verify] 用 from_pretrained(hf) 初始化（已知可用）...")
    src = Policy.from_pretrained(SRC_HF)
    src.trunk.to("cpu")
    trunk_ref = {k: v.clone() for k, v in src.trunk.state_dict().items()}
    print(f"[verify] 参考 trunk 参数量: {sum(v.numel() for v in trunk_ref.values())}")

    # 清零 trunk，模拟"未加载"状态
    for p in src.trunk.parameters():
        p.data.zero_()
    zero_sum = sum(p.data.abs().sum().item() for p in src.trunk.parameters())
    print(f"[verify] 清零后 trunk 绝对值之和 = {zero_sum:.4f} (应≈0)")

    # 写本地 trunk.pth 并 load_trunk(本地路径)
    tmp = tempfile.mkdtemp()
    local_trunk = os.path.join(tmp, "trunk_test.pth")
    torch.save(trunk_ref, local_trunk)
    print(f"[verify] 写本地 trunk.pth -> {local_trunk}")
    print("[verify] 调用 load_trunk(本地路径) ...")
    src.load_trunk(local_trunk)

    loaded = src.trunk.state_dict()
    ok = all(torch.equal(trunk_ref[k], loaded[k]) for k in trunk_ref)
    print(f"LOADED_TRUNK_MATCHES_SOURCE = {ok}")
    if ok:
        print("[verify] ✅ 本地续训可用：联邦管线可直接运行 run_fedavg.sh")
    else:
        print("[verify] ❌ 本地 trunk.pth 未被加载（load_trunk 仅支持 hf://）。")
        print("         需改用 README 中的兜底方案：自定义训练循环直接 load_state_dict 全局 model.pth。")
    return ok


if __name__ == "__main__":
    main()
