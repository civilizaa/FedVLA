#!/usr/bin/env bash
# 快速诊断：P3-D（IID + MoE + FedAvg）的 task1 成功率是否正常？
#
# 背景：P3-A/B/C（异构 + MoE）三个组的 task1/task2 全是 0.000。
# 两种可能：
#   (a) 异构划分下 reach/push 本来就学不会（MoE 链路正常，问题在异构）
#   (b) MoE 推理链路有 bug（模型被破坏，任何任务都学不会）
#
# 判别实验：P3-D 是 **IID 划分 + MoE**（等价于 IID 联邦 + MoE），
# 其无 MoE 对照（g2b_iid_e4_r49）的 task1 成功率是 0.300。
#   · P3-D ≈ 0.300 或更高  → 结论 (a)：MoE 链路正常，是异构的问题
#   · P3-D ≈ 0             → 结论 (b)：MoE 推理链路有 bug，P3-A/B/C 全部作废
#
# 两个 case 用**同一个 seed、同样本数**，直接可比。
set -u

OUT=/root/HPT/output
HPT_PY=/root/miniconda3/envs/hpt/bin/python
DOMAIN=mujoco_metaworld_task1
ENVNAME=reach-v2
SEEDS=${SEEDS:-0}
EPISODES=${EPISODES:-10}

export WANDB_MODE=offline
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# MoE 开关必须与训练侧完全一致（否则 strict=False 会静默丢弃 264 个专家权重）
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half

run_case () {
  local model_dir="$1" tag="$2"
  local logf="$OUT/quickdiag_${tag}.log"
  echo "=== case $tag : $model_dir ==="
  (
    cd /root/HPT || exit 1
    "$HPT_PY" -m hpt.run_eval --multirun \
      --config-name=config \
      --config-path="$model_dir" \
      train.pretrained_dir="$model_dir" \
      domains="$DOMAIN" \
      "rollout_runner.env_names=[\"$ENVNAME\"]" \
      "dataset_generator_func.env_names=[\"$ENVNAME\"]" \
      seed="$SEEDS" \
      rollout_runner.episode_num="$EPISODES" \
      hydra/launcher=joblib \
      hydra.launcher.n_jobs=1 \
      hydra.sweep.dir="$OUT/quickdiag_${tag}_dir" \
      > "$logf" 2>&1
  )
  echo "rc=$?"
  grep -oE "number of total params \(M\): [0-9.]+" "$logf" | head -1
  grep -E "^\| *avg" "$logf" | tail -2
}

echo "########## P3-D：IID + MoE + FedAvg ##########"
run_case "$OUT/p3_moe_iid/global_r49" "p3_moe_iid"

echo
echo "########## 对照：g2b_iid_e4_r49（IID，无 MoE）##########"
unset FEDVLA_MOE
run_case "$OUT/g2b_iid_e4/global_r49" "g2b_iid_e4"

echo
echo "########## 汇总 ##########"
for t in p3_moe_iid g2b_iid_e4; do
  f="$OUT/quickdiag_${t}.log"
  echo "--- $t ---"
  grep -oE "number of total params \(M\): [0-9.]+" "$f" | head -1
  grep -E "^\| *(avg|reach-v2)" "$f" | tail -4
done
echo "QUICKDIAG_DONE"
