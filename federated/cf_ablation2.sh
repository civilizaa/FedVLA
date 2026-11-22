#!/usr/bin/env bash
# 后验量化消融（并行版）：与 cf_ablation.sh 同一套逻辑，但可指定 scope 列表与 tag 前缀，
# 便于"抢在自动关机前"把关键变体先跑完（不同 tag ⇒ 不同 DST ⇒ 不与正在跑的顺序版冲突）。
#
#   SCOPES="norm trunk" TAG_PREFIX=cf_p3c_int8b bash /root/fedcore/cf_ablation2.sh
set -u

BASE=/root/fedcore/keep/p3c_global_r49
K=/root/fedcore
OUT=/root/HPT/output
PY=/root/miniconda3/envs/hpt/bin/python
LOG=$OUT/cf_ablation2.log
SCOPES=${SCOPES:-"norm trunk"}
TAG_PREFIX=${TAG_PREFIX:-cf_p3c_int8b}
BITS=${BITS:-8}

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half
export OMP_NUM_THREADS=2
export PYTHONPATH="/root/HPT:${PYTHONPATH:-}"

log "===== 并行消融开始 scopes=[$SCOPES] prefix=$TAG_PREFIX bits=$BITS ====="

for scope in $SCOPES; do
    TAG=${TAG_PREFIX}_${scope}
    DST=$K/cf/$TAG
    log "--- 构建 $TAG（scope=$scope bits=$BITS）---"
    "$PY" "$K/cf_quant_variant.py" --src "$BASE" --dst "$DST" --bits "$BITS" --scope "$scope" \
        --out "$OUT/cf_${TAG}_variant.json" >> "$LOG" 2>&1
    log "--- 评估 $TAG ---"
    MODEL_DIR="$DST" DOMAIN=mujoco_metaworld_task3 ENVNAME=button-press-topdown-v2 \
        N_SEEDS=3 EPISODE_NUM=20 N_JOBS=2 LOG_TAG="$TAG" \
        bash "$K/eval_model.sh" > "$OUT/cf_${TAG}_eval_stdout.log" 2>&1
    rc=$?
    n=$(grep -c '| avg' "$OUT/eval_${TAG}_mujoco_metaworld_task3.log" 2>/dev/null || echo 0)
    log "$TAG 评估结束 rc=$rc 表数=$n"
    grep '| avg' "$OUT/eval_${TAG}_mujoco_metaworld_task3.log" 2>/dev/null | tr '\n' ' ' >> "$LOG"
    echo "" >> "$LOG"
    rm -rf "$DST"
done

log "===== 并行消融完成 ====="
