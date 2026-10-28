#!/usr/bin/env bash
# FedVLA 联邦续训补丁 —— 零传输内联版（把下面整个 <<'EOF' ... EOF 整块粘贴到 AutoDL 终端）
#
# 为什么需要：
#   HPT 源码的 Policy.load_trunk 只处理 "hf://"：
#       if "hf://" in path: ... self.trunk.load_state_dict(torch.load(path), strict=True)
#       # ← 没有 else，本地路径 = 静默空操作
#   而 run_fedavg.sh 靠 train.pretrained_dir=<全局目录> 续训，全局目录的 trunk.pth
#   是本地路径 → load_trunk 什么都不做 → 每轮客户端都从 hf 原始权重重新开始 →
#   **联邦静默退化成"每轮独立训练"，聚合的权重根本没被用上**。不报错，但结果全废。
#
#   同时 modalities_tokens 是 policy 顶层的 ParameterDict（不在 trunk 内），
#   trunk.pth 里没有它 → 续训时模态 token 保持随机初始化。
#
# 本脚本做两件事（幂等，可重复执行）：
#   1) 给 load_trunk 加本地分支 + 顺带载入同目录 extras.pth
#   2) --verify 模式只检查不修改
set -u
HPT_DIR=${HPT_DIR:-/root/HPT}
POLICY="$HPT_DIR/hpt/models/policy.py"
HPT_PY=${HPT_PY:-/root/miniconda3/envs/hpt/bin/python}

if [ ! -f "$POLICY" ]; then
  echo "[!!] 找不到 $POLICY —— 请确认 HPT 仓库位置（可用 HPT_DIR=xxx 指定）"
  exit 1
fi

MODE=${1:-patch}
"$HPT_PY" - "$POLICY" "$MODE" <<'PY'
import os, shutil, sys

path, mode = sys.argv[1], sys.argv[2]
src = open(path, "r", encoding="utf-8").read()
backup = path + ".fedvla.bak"

ANCHOR = "            self.trunk.load_state_dict(torch.load(path), strict=True)\n"
PATCHED = '''            self.trunk.load_state_dict(torch.load(path), strict=True)
        elif os.path.exists(path):
            # [FedVLA patch] 本地 trunk.pth 续训（原版无此分支，会静默空操作）
            try:
                sd = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
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
patched = MARK in src

if mode == "verify":
    print("[OK] 补丁已存在（load_trunk 支持本地路径 + extras）" if patched
          else "[!!] 补丁不存在 —— 联邦续训会静默失效，请先执行本脚本（不带参数）")
    sys.exit(0 if patched else 1)

if mode == "revert":
    if not os.path.exists(backup):
        print(f"[!!] 无备份，无法回滚：{backup}")
        sys.exit(2)
    shutil.copy(backup, path)
    print(f"[OK] 已回滚：{backup} -> {path}")
    sys.exit(0)

if patched:
    print("[OK] 补丁已存在，无需重复打（幂等）")
    sys.exit(0)

if ANCHOR not in src:
    print("[!!] 找不到锚点行，HPT 版本可能不同，请手工检查：")
    print(f"     {path}")
    sys.exit(3)

if "import os" not in src:
    src = "import os\n" + src
if "import logging" not in src:
    src = "import logging\n" + src

if not os.path.exists(backup):
    shutil.copy(path, backup)
    print(f"[OK] 已备份 -> {backup}")

src = src.replace(ANCHOR, PATCHED, 1)
open(path, "w", encoding="utf-8").write(src)
print(f"[OK] 已打补丁：{path}")
print("     1) load_trunk 新增：elif os.path.exists(path): -> trunk.load_state_dict(strict=True)")
print("     2) 同目录存在 extras.pth 时额外载入 modalities_tokens / action_tokens")
print("     下一步：重跑门1(gate_check_inline.sh)，应看到 LOADED_TRUNK_MATCHES_SOURCE = True")
PY
