"""给 HPT 打「联邦续训」补丁（幂等）：让 `Policy.load_trunk` 支持本地路径 + 载入模态 token。

为什么需要（两个独立缺陷）：

  缺陷 1 —— load_trunk 对本地路径是空操作
      HPT 源码：
          def load_trunk(self, path: str):
              if "hf://" in path:
                  ...
                  self.trunk.load_state_dict(torch.load(path), strict=True)
              # ← 没有 else 分支，本地路径 = 静默什么都不做

      FedVLA 的联邦续训依赖 `train.pretrained_dir=<全局目录>`，
      全局目录里的 trunk.pth 是本地路径 → load_trunk 空操作 →
      每轮客户端都从 hf 原始权重重新开始 → **联邦静默退化成"每轮独立训练"**，
      聚合的权重根本没被用上。这个错不报错，但结果全废。

  缺陷 2 —— 模态 token 不在 trunk 里，续训时保持随机初始化
      源码确认 `self.modalities_tokens` 是 `nn.ParameterDict`（按 modality 命名），
      位于 policy 顶层而非 `self.trunk` 内部 → trunk.pth 里没有它们。
      `load_trunk` 只载入 trunk → 客户端续训时模态 token 仍是随机值，
      同一个共享 trunk 在各客户端拿到不一致的输入。

打补丁后的行为：
      hf://                    → 原逻辑不变
      本地 trunk.pth 存在      → self.trunk.load_state_dict(..., strict=True)
      同目录 extras.pth 存在   → policy.load_state_dict(extras, strict=False)

用法（在 AutoDL 上）：
      python /root/HPT/federated/patch_hpt_load_trunk.py            # 打补丁
      python /root/HPT/federated/patch_hpt_load_trunk.py --verify   # 只检查
      python /root/HPT/federated/patch_hpt_load_trunk.py --revert   # 回滚
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

DEFAULT_TARGET = os.path.join(os.path.expanduser("~"), "HPT", "hpt", "models", "policy.py")

# ── 补丁锚点 ───────────────────────────────────────────────────────────
# 原始 load_trunk 末尾那一行
ANCHOR = "            self.trunk.load_state_dict(torch.load(path), strict=True)\n"

PATCHED_BLOCK = '''            self.trunk.load_state_dict(torch.load(path), strict=True)
        elif os.path.exists(path):
            # [FedVLA patch] 本地 trunk.pth 续训（原版无此分支，会静默空操作）
            try:
                sd = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:  # 老版本 torch 没有 weights_only 参数
                sd = torch.load(path, map_location="cpu")
            self.trunk.load_state_dict(sd, strict=True)
            # [FedVLA patch] 顺带载入顶层共享参数（modalities_tokens / action_tokens）
            extras_path = os.path.join(os.path.dirname(os.path.abspath(path)), "extras.pth")
            if os.path.exists(extras_path):
                try:
                    extras = torch.load(extras_path, map_location="cpu", weights_only=False)
                except TypeError:
                    extras = torch.load(extras_path, map_location="cpu")
                self.load_state_dict(extras, strict=False)
                logging.info(f"[FedVLA patch] loaded extras from {extras_path} "
                             f"({len(extras)} tensors)")
'''

MARK = "[FedVLA patch]"


def find_target(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.exists(DEFAULT_TARGET):
        return DEFAULT_TARGET
    for root in (os.path.expanduser("~/HPT"), "/root/HPT", os.getcwd()):
        p = os.path.join(root, "hpt", "models", "policy.py")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("找不到 hpt/models/policy.py，请用 --target 指定")


def main(argv=None):
    ap = argparse.ArgumentParser(description="给 HPT Policy.load_trunk 打联邦续训补丁")
    ap.add_argument("--target", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args(argv)

    path = find_target(args.target)
    src = open(path, "r", encoding="utf-8").read()
    backup = path + ".fedvla.bak"

    if args.revert:
        if not os.path.exists(backup):
            print(f"[!!] 无备份，无法回滚：{backup}")
            return 2
        shutil.copy(backup, path)
        print(f"[OK] 已回滚：{backup} -> {path}")
        return 0

    patched = MARK in src

    if args.verify:
        if patched:
            print("[OK] 补丁已存在（load_trunk 支持本地路径 + extras）")
            return 0
        print("[!!] 补丁不存在 —— 联邦续训会静默失效，请先打补丁")
        return 1

    if patched:
        print("[OK] 补丁已存在，无需重复打（幂等）")
        return 0

    if ANCHOR not in src:
        print("[!!] 找不到锚点行，HPT 版本可能不同，请手工检查：")
        print(f"     {path}")
        return 3

    # policy.py 顶部是否已 import os / logging
    need_os = "\nimport os" not in src and "import os\n" not in src
    need_logging = "import logging" not in src

    if not os.path.exists(backup):
        shutil.copy(path, backup)
        print(f"[OK] 已备份 -> {backup}")

    src = src.replace(ANCHOR, PATCHED_BLOCK, 1)

    if need_os:
        src = "import os\n" + src
    if need_logging:
        src = "import logging\n" + src

    open(path, "w", encoding="utf-8").write(src)
    print(f"[OK] 已打补丁：{path}")
    print("     1) load_trunk 新增：elif os.path.exists(path): -> trunk.load_state_dict(strict=True)")
    print("     2) 同目录存在 extras.pth 时，额外载入 modalities_tokens / action_tokens")
    print("     下一步：重跑门1，应看到 LOADED_TRUNK_MATCHES_SOURCE = True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
