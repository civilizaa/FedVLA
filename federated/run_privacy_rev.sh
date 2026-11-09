#!/usr/bin/env bash
# FedVLA 第4阶段隐私侧：补跑 modality_rev（按实测可攻击性重分配预算）。
#
# ⚠️ 为什么要补跑（2026-11-08）：
#   上一轮手动补跑 modality_rev 时**漏了 --dp-steps 50**（privacy_experiment.py 的
#   argparse 默认是 25）⇒ RDP 复合轮数减半 ⇒ 总 ε 从 5.0e9 变成 2.5e9
#   ⇒ 与 uniform / modality 的 5.0e9 **不可比**（不是等预算对比）。
#   ⇒ 已用 purge_privacy_mode.py 把那 19 条错误记录从 privacy_results.json 剔除，
#      本脚本用完整参数重跑。
#
# 参数必须与 run_privacy_scan.sh（主扫描）**逐字一致**：
#   σ 序列 / 模式 / 目标 / 样本 / δ / dp-steps / clients / clip-norm /
#   inv-steps / inv-lr / match-mode / seed
#
# 幂等：privacy_experiment.py 按 (sample, mode, σ_ref, target) 增量续跑，
#       已存在的组合自动跳过 ⇒ 本脚本可重复执行。
set -u

OUT=${OUT:-/root/HPT/output}
PRIV_DIR=${PRIV_DIR:-$OUT/privacy}
HPT_PY=/root/miniconda3/envs/hpt/bin/python

# ===== MoE 环境变量（与训练侧逐字一致，否则是残缺模型 13.94M）=====
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half
export MUJOCO_GL=egl

mkdir -p "$PRIV_DIR"

echo "[$(date '+%F %T')] ===== modality_rev 补跑（--dp-steps 50）====="
echo "[$(date '+%F %T')] 模型=$OUT/p3_moe_eda_cluster/global_r49"
echo "[$(date '+%F %T')] 结果=$PRIV_DIR/privacy_results.json"

"$HPT_PY" -u /root/fedcore/privacy_experiment.py \
  --model-dir "$OUT/p3_moe_eda_cluster/global_r49" \
  --zarr /root/HPT/data/zarr_mujoco_metaworld_task3_resnet_traj100 \
  --domain mujoco_metaworld_task3 \
  --out "$PRIV_DIR" \
  --sigmas 1e-4,1e-3,1e-2,1e-1,1 \
  --modes modality_rev \
  --targets image,action,state \
  --samples 0,1,2 \
  --delta 1e-5 \
  --dp-steps 50 \
  --clients 4 \
  --clip-norm 1.0 \
  --inv-steps 200 \
  --inv-lr 0.05 \
  --match-mode dense \
  --seed 1234

rc=$?
echo "[$(date '+%F %T')] modality_rev 补跑结束 rc=$rc"
touch "$PRIV_DIR/PRIVACY_REV_DONE"
exit $rc
