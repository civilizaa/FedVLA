#!/usr/bin/env bash
# 重打 FedVLA 补丁：先回滚到原始 policy.py，再打新补丁。
# 原因：旧补丁用 self.load_state_dict(extras) 会摧毁 normalizer.params_dict
#       （DictOfTensorMixin._load_from_state_dict 无条件重建 params_dict）
#       -> 联邦第 1 轮续训第一步就崩。
set -u
PY=/root/miniconda3/envs/hpt/bin/python
TARGET=/root/HPT/hpt/models/policy.py
PATCH=/root/HPT/federated/patch_hpt_load_trunk.py

echo "===== 1. 回滚 ====="
$PY "$PATCH" --revert --target "$TARGET" || exit 1

echo "===== 2. 打新补丁 ====="
$PY "$PATCH" --target "$TARGET" || exit 1

echo "===== 3. 校验 ====="
$PY "$PATCH" --verify --target "$TARGET" || exit 1
grep -n "loaded extras\|container\[_key\]\|load_state_dict(extras" "$TARGET" | head -10

echo "===== 4. 语法检查 ====="
$PY -c "import ast;ast.parse(open('$TARGET').read());print('policy.py 语法 OK')" || exit 1

echo "===== 5. 确认旧写法已消失 ====="
# 注意：要排除注释行 —— 补丁块里为了警示，注释中出现了 "self.load_state_dict(extras" 字样。
# 只看真正的代码行（去掉行首空白后不以 # 开头）。
if grep -vE '^[[:space:]]*#' "$TARGET" | grep -q "self\.load_state_dict(extras"; then
  echo "[!!] 旧写法仍在 —— 补丁未生效！"
  exit 1
fi
echo "[OK] 旧的 self.load_state_dict(extras) 已移除"
echo "REPATCH_DONE"
