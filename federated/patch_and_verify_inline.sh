#!/usr/bin/env bash
# =============================================================================
# FedVLA 补丁 + 门1 验证 —— 一键内联版（整块粘贴到 AutoDL 终端即可执行）
# =============================================================================
# 背景（2026-10-27 云端门1 实测 + HPT 源码逐字核实）：
#   HPT 的 Policy.load_trunk 只有 "hf://" 分支，没有 else：
#       def load_trunk(self, path):
#           if "hf://" in path:
#               ...
#               self.trunk.load_state_dict(torch.load(path), strict=True)
#           # ← 本地路径 = 静默空操作
#   而 run_fedavg.sh 靠 train.pretrained_dir=<全局目录> 续训，全局目录的 trunk.pth
#   是本地路径 → load_trunk 什么都不做 → 每轮客户端都从 hf 原始权重重开 →
#   **联邦静默退化成"每轮独立训练"，聚合权重从未被使用**。不报错，但 50 轮全废。
#
# 本脚本：打补丁（幂等/可回滚）→ 立刻跑门1b 验证 → 打印结论
# =============================================================================
set -u
HPT_DIR=${HPT_DIR:-/root/HPT}
HPT_PY=${HPT_PY:-/root/miniconda3/envs/hpt/bin/python}
POLICY="$HPT_DIR/hpt/models/policy.py"

export MUJOCO_GL=egl WANDB_MODE=offline HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8
export PYTHONPATH="$HPT_DIR:${PYTHONPATH:-}"

echo "===== FedVLA 补丁 + 门1 ====="
echo "HPT_DIR=$HPT_DIR"
echo "python  =$HPT_PY"

if [ ! -f "$POLICY" ]; then
  echo "[!!] 找不到 $POLICY —— 请确认 HPT 位置（可用 HPT_DIR=xxx 指定）"
  exit 1
fi
if [ ! -x "$HPT_PY" ]; then
  echo "[!!] 找不到 hpt 环境 python：$HPT_PY"
  exit 1
fi
cd "$HPT_DIR" || exit 1

# ---------------------------------------------------------------- 打补丁 ---
"$HPT_PY" - "$POLICY" <<'PY'
import os, shutil, sys

path = sys.argv[1]
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

if MARK in src:
    print("[OK] 补丁已存在，无需重复打（幂等）")
    sys.exit(0)

if ANCHOR not in src:
    print("[!!] 找不到锚点行 —— HPT 版本可能不同，请手工检查：")
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
print("     1) load_trunk 新增 elif os.path.exists(path) 分支")
print("     2) 同目录存在 extras.pth 时额外载入 modalities_tokens / action_tokens")
PY
PATCH_RC=$?
if [ "$PATCH_RC" -ne 0 ]; then
  echo "[!!] 打补丁失败（退出码 $PATCH_RC），中止。"
  exit "$PATCH_RC"
fi

# ------------------------------------------------------------ 门1b 验证 ---
echo
echo "########## 门1b：本地 trunk.pth 续训等价性 ##########"
"$HPT_PY" - <<'PY'
import os, tempfile, torch
from hpt.models.policy import Policy

try:
    src = Policy.from_pretrained("hf://liruiw/hpt-base")
except Exception as e:
    print("[!!] from_pretrained 失败:", repr(e))
    raise SystemExit(1)

src.trunk.to("cpu")
ref = {k: v.clone() for k, v in src.trunk.state_dict().items()}
nparam = sum(v.numel() for v in ref.values())
print("参考 trunk 参数量:", nparam)
print("trunk 键示例:", list(ref.keys())[:3])

for p in src.trunk.parameters():
    p.data.zero_()
z = sum(p.data.abs().sum().item() for p in src.trunk.parameters())
print("清零后绝对值之和 =", round(z, 6), "(应≈0)")

tmp = tempfile.mkdtemp()
lp = os.path.join(tmp, "trunk_test.pth")
torch.save(ref, lp)

src.load_trunk(lp)
loaded = src.trunk.state_dict()
ok = all(torch.equal(ref[k], loaded[k]) for k in ref)
print(f"LOADED_TRUNK_MATCHES_SOURCE = {ok}")

if ok:
    print("[OK] 本地续训可用 -> 联邦管线(run_fedavg.sh)成立")
    print("     下一步：跑门2 异构烟雾（NUM_ROUNDS=1, CLIENT_DOMAINS=task1 task2 task3 task1）")
    raise SystemExit(0)
else:
    print("[!!] load_trunk 本地仍是空操作 —— 补丁没生效？检查上面是否有报错")
    raise SystemExit(1)
PY
GATE_RC=$?

echo
if [ "$GATE_RC" -eq 0 ]; then
  echo "===== 门1 通过：联邦续训链路可用 ====="
else
  echo "===== 门1 失败（退出码 $GATE_RC）====="
fi
exit "$GATE_RC"
