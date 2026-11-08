#!/usr/bin/env bash
# 端到端验证：评估链路在【设 / 不设】FEDVLA_MOE 时，是否真的加载了 MoE 专家权重。
#
# 走完整 hydra 路径（hpt.run_eval），与正式评估完全一致，只是把规模缩到
# 1 seed × 2 rollouts，几分钟就能跑完。
#
# 判据：run_eval.py 会打印 "number of params (M): X"
#   不设 FEDVLA_MOE → 12.644M（残缺模型，264 个专家权重被 strict=False 静默丢弃）
#   设   FEDVLA_MOE → 21.082M（完整模型）
#
# 用法：
#   bash verify_moe_eval.sh /root/HPT/output/p3_moe_fedavg/global_r49
set -u

MODEL_DIR="${1:?请传入模型目录}"
OUT=/root/HPT/output
HPT_PY=/root/miniconda3/envs/hpt/bin/python
DOMAIN=mujoco_metaworld_task1
ENVNAME=reach-v2

export WANDB_MODE=offline
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

run_case () {
  local on="$1" tag="moe_${2}"
  local logf="$OUT/verify_${tag}.log"
  if [ "$on" = "1" ]; then
    export FEDVLA_MOE=1
    export FEDVLA_MOE_EXPERTS=8
    export FEDVLA_MOE_TOPK=2
    export FEDVLA_MOE_EXPERT_RATIO=0.25
    export FEDVLA_MOE_LAYERS=half
  else
    unset FEDVLA_MOE FEDVLA_MOE_EXPERTS FEDVLA_MOE_TOPK \
          FEDVLA_MOE_EXPERT_RATIO FEDVLA_MOE_LAYERS
  fi
  echo "=== case: FEDVLA_MOE=$on ==="
  (
    cd /root/HPT || exit 1
    "$HPT_PY" -m hpt.run_eval --multirun \
      --config-name=config \
      --config-path="$MODEL_DIR" \
      train.pretrained_dir="$MODEL_DIR" \
      domains="$DOMAIN" \
      "rollout_runner.env_names=[\"$ENVNAME\"]" \
      "dataset_generator_func.env_names=[\"$ENVNAME\"]" \
      seed="range(1)" \
      rollout_runner.episode_num=2 \
      hydra/launcher=joblib \
      hydra.launcher.n_jobs=1 \
      hydra.sweep.dir="$OUT/verify_${tag}_dir" \
      > "$logf" 2>&1
  )
  echo "rc=$?"
  grep -oE "number of params \(M\): [0-9.]+" "$logf" | head -1
}

P_OFF=$(run_case 0 off)
P_ON=$(run_case 1 on)

echo
echo "==================== 结论 ===================="
echo "不设 FEDVLA_MOE : $P_OFF"
echo "设   FEDVLA_MOE : $P_ON"

off_n=$(echo "$P_OFF" | grep -oE "[0-9.]+$" || echo "")
on_n=$(echo "$P_ON" | grep -oE "[0-9.]+$" || echo "")
echo "off=${off_n:-?}M  on=${on_n:-?}M"

if [ -n "$on_n" ] && [ -n "$off_n" ]; then
  awk -v a="$on_n" -v b="$off_n" 'BEGIN{
    if (a > 15.0 && b < 15.0) { print "VERDICT=PASS  MoE 环境变量确实是评估生效的必要条件"; exit 0 }
    else { print "VERDICT=FAIL  参数差异不符合预期"; exit 1 }
  }'
  exit $?
fi
echo "VERDICT=INCONCLUSIVE  没能读到参数量"
exit 2
