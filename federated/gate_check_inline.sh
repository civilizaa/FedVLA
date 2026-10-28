#!/usr/bin/env bash
# FedVLA 门0 + 门1 单命令自包含版（无需任何文件传输！）
#
# 用法：把下面整个 <<'EOF' ... EOF 整块粘贴到 AutoDL 终端即可。
# 原理：门0/门1 的检查逻辑直接内联在本脚本里，不依赖 federated/ 任何文件。
#
# 门0：打印 model.pth 的 trunk 键名 + 顶层前缀分布（验证 aggregate.py 假设）
# 门1：验证 HPT 能否从本地 trunk.pth 续训
set -u
HPT_PY=/root/miniconda3/envs/hpt/bin/python
HPT_DIR=/root/HPT
export MUJOCO_GL=egl WANDB_MODE=offline HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8
export PYTHONPATH="$HPT_DIR:${PYTHONPATH:-}"
cd "$HPT_DIR" || exit 1

echo "########## 门0：trunk 键名 ##########"
"$HPT_PY" - <<'PY'
import os, torch
from collections import Counter
p = "output/reach_baseline/model.pth"
if not os.path.exists(p):
    print(f"[!!] 找不到 {p}（需先完成集中式训练）")
    raise SystemExit(0)
sd = torch.load(p, map_location="cpu", weights_only=False)
print("state_dict 总键数:", len(sd))
print("顶层前缀分布:", Counter(k.split(".")[0] for k in sd).most_common(12))
tk = [k for k in sd if k.startswith("trunk")]
print("trunk 键数量:", len(tk))
print("trunk 键示例:", tk[:6])
if not tk:
    print("[!!] 无 'trunk.' 前缀键 => aggregate.py 的 _is_trunk_key 会漏选，需改判定")
else:
    print("[OK] aggregate.py 假设成立；写进 trunk.pth 的键名示例:", tk[0][len("trunk."):])
PY

echo
echo "########## 门1：本地 trunk.pth 续训 ##########"
echo "--- 门1a：补丁存在性 ---"
if grep -q "FedVLA patch" "$HPT_DIR/hpt/models/policy.py" 2>/dev/null; then
  echo "[OK] policy.py 已含 FedVLA 补丁"
else
  echo "[!!] policy.py 未打补丁 —— load_trunk 对本地路径是空操作，联邦会静默退化为每轮独立训练"
  echo "     请先执行：bash federated/patch_inline.sh"
fi
echo "--- 门1b：续训等价性 ---"
"$HPT_PY" - <<'PY'
import os, tempfile, torch
from hpt.models.policy import Policy
try:
    src = Policy.from_pretrained("hf://liruiw/hpt-base")
except Exception as e:
    print("[!!] from_pretrained 失败:", e)
    raise SystemExit(1)
src.trunk.to("cpu")
ref = {k: v.clone() for k, v in src.trunk.state_dict().items()}
print("参考 trunk 参数量:", sum(v.numel() for v in ref.values()))
for p in src.trunk.parameters():
    p.data.zero_()
z = sum(p.data.abs().sum().item() for p in src.trunk.parameters())
print("清零后绝对值之和 =", round(z, 4), "(应≈0)")
tmp = tempfile.mkdtemp()
lp = os.path.join(tmp, "trunk_test.pth")
torch.save(ref, lp)
src.load_trunk(lp)
loaded = src.trunk.state_dict()
ok = all(torch.equal(ref[k], loaded[k]) for k in ref)
print(f"LOADED_TRUNK_MATCHES_SOURCE = {ok}")
if ok:
    print("[OK] 本地续训可用 -> 联邦管线(run_fedavg.sh)成立")
else:
    print("[!!] load_trunk 本地仍是空操作 -> 先执行 bash federated/patch_inline.sh 再重跑门1")
PY
echo
echo "门1 退出码: $?"
