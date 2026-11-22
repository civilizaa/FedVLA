#!/usr/bin/env bash
# ==============================================================================
# 后验量化消融：把"量化把策略打死了"这件事拆成"哪一部分打死的"
#
# 基线：第3阶段的 P3-C 全局模型（异构 + MoE + 簇感知，50 轮，**完全干净**：fp32、无量化、无 DP 噪声）
#       它的 task3 成功率**已有**（0.90 / 0.60 / 0.50 ⇒ 0.667），不必重跑。
#
# 三个变体（同一基线，只改一处）：
#   scope=all   → 所有上行共享参数都做 int8_g（与链路一致）
#   scope=norm  → **只**做 normalizer（trunk 保持干净）
#   scope=trunk → **只**做 trunk + modalities_tokens（normalizer 保持干净）
#
# 判读
# ----
#   · all   ≈ 0.000  ⇒ 后验量化同样能打死策略（与逐轮量化结论一致）
#   · norm  ≈ 0.000  且 trunk ≈ 0.667 ⇒ **normalizer 是致命环节**（可操作的修复：排除归一化统计量）
#   · trunk ≈ 0.000  ⇒ 主干清零同样致命（⇒ 该量化方案整体不可用）
#
# ⚠️ 口径声明：本实验是**后验量化**（对已训练好的模型一次性量化），
#    而通信扫描是**训练中逐轮量化**（损伤会逐轮复合）。
#    后验是"更弱"的损伤设定 ⇒ 若后验都能打死，训练中更会被打死；
#    若后验打不死、训练中却打死了 ⇒ 说明"逐轮复合"是必要条件之一。两者的结论都要写清楚。
# ==============================================================================
set -u

BASE_SRC=/root/HPT/output/p3_moe_eda_cluster/global_r49
BASE=/root/fedcore/keep/p3c_global_r49
K=/root/fedcore
OUT=/root/HPT/output
PY=/root/miniconda3/envs/hpt/bin/python
LOG=$OUT/cf_ablation.log

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

mkdir -p "$OUT" "$K/cf"

# 备份干净基线（防止被别的清理逻辑删掉）
if [ ! -d "$BASE" ]; then
    cp -r "$BASE_SRC" "$BASE" && log "已备份干净基线 P3-C → $BASE"
fi

# MoE 环境变量必须与训练侧逐字一致（否则评估会静默丢掉 264 个专家权重）
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half
export OMP_NUM_THREADS=2
export PYTHONPATH="/root/HPT:${PYTHONPATH:-}"

log "===== 后验量化消融开始（基线 P3-C，task3 基线成功率 0.667）====="

for scope in all norm trunk; do
    TAG=cf_p3c_int8_${scope}
    DST=$K/cf/$TAG
    log "--- 构建变体 $TAG（scope=$scope）---"
    "$PY" "$K/cf_quant_variant.py" --src "$BASE" --dst "$DST" --bits 8 --scope "$scope" \
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
    # 评估完立刻删变体权重（磁盘）
    rm -rf "$DST"
done

log "===== 消融全部完成 ====="
df -h /root | tail -1 >> "$LOG"
