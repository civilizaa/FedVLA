#!/usr/bin/env bash
# FedVLA：统一评估入口（集中式基线与联邦全局模型共用同一口径）
#
# 用法：
#   MODEL_DIR=/root/HPT/output/centralized/task1 DOMAIN=mujoco_metaworld_task1 ENV=reach-v2 \
#     bash eval_model.sh
#
# 口径：3 seeds × 20 rollouts（论文统计底线），报均值 ± 95% CI
#   - seeds   : range(3)（hydra multirun + joblib launcher）
#   - 每 seed : rollout_runner.episode_num=20 次 rollout
#
# 关键点：
#   1) --config-path 必须是【绝对路径】且指向含 config.yaml 的模型目录
#      （HPT 的 --config-path 相对包目录解析，不是 cwd；踩过坑）。
#   2) eval 用 policy.load_model(model.pth) -> strict=False，
#      所以联邦全局模型（trunk 已聚合 + 各域 stem/head 并集）可直接按域评估。
#   3) 必须覆盖 domains 与 rollout_runner.env_names，
#      否则会沿用 config.yaml 里的 task1/reach-v2，评错任务。
set -u

MODEL_DIR="${MODEL_DIR:?请设置 MODEL_DIR 为模型目录（内含 model.pth 与 config.yaml）}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAIN="${DOMAIN:?请设置 DOMAIN，例如 mujoco_metaworld_task1}"
ENVNAME="${ENVNAME:?请设置 ENVNAME，例如 reach-v2}"
N_SEEDS="${N_SEEDS:-3}"
EPISODE_NUM="${EPISODE_NUM:-20}"
N_JOBS="${N_JOBS:-1}"
HPT_PY=/root/miniconda3/envs/hpt/bin/python

export WANDB_MODE=offline
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

TAG=$(basename "$MODEL_DIR")
# ⚠️ 日志名必须带 domain —— 否则同一模型评估多个任务时，后者会覆盖前者，
#    最后只剩最后一个任务的结果（踩过：三个任务评估完只剩 task3 的表格）。
# ⚠️ TAG 相同但模型来源不同（集中式 / 等预算 / 联邦）也会互相覆盖，
#    所以额外允许用 LOG_TAG 覆盖 TAG。
LOG_TAG=${LOG_TAG:-$TAG}
OUT_DIR="/root/HPT/output/eval_${LOG_TAG}_${DOMAIN}"
LOG="/root/HPT/output/eval_${LOG_TAG}_${DOMAIN}.log"

mkdir -p "$OUT_DIR"
echo "===== 评估 $MODEL_DIR ====="
echo "domain=$DOMAIN env=$ENVNAME seeds=$N_SEEDS episodes=$EPISODE_NUM n_jobs=$N_JOBS"
echo "日志: $LOG"

(
  cd /root/HPT || exit 1
  "$HPT_PY" -m hpt.run_eval --multirun \
    --config-name=config \
    --config-path="$MODEL_DIR" \
    train.pretrained_dir="$MODEL_DIR" \
    domains="$DOMAIN" \
    "rollout_runner.env_names=[\"$ENVNAME\"]" \
    "dataset_generator_func.env_names=[\"$ENVNAME\"]" \
    seed="range($N_SEEDS)" \
    rollout_runner.episode_num="$EPISODE_NUM" \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs="$N_JOBS" \
    hydra.sweep.dir="$OUT_DIR" \
    > "$LOG" 2>&1
)
rc=$?
echo "[---] rc=$rc"
echo "===== 成功率汇总 ====="
grep -iE "success|success_rate|reward" "$LOG" | tail -n 40
# 从表格里精确提取每个 seed 的成功率并算均值 ± 95% CI
/root/miniconda3/envs/hpt/bin/python "$SCRIPT_DIR/summarize_eval.py" "$LOG" 2>/dev/null || true
echo "完整日志: $LOG"
