#!/usr/bin/env bash
# ==============================================================================
# 反事实评估：把**未量化**的 normalizer 塞回量化后的全局模型，再跑 task3 评估
#
# 为什么需要
# ----------
# 第4阶段量化三档（int8/int4/bit1）在 task3 上成功率全为 0.000，
# 而 int8 的**权重级**相对误差只有 0.44% ⇒ 二者矛盾。
# 取证指向：量化把 normalizer.state.scale 的多数维度**清零**
#          ⇒ 归一化输出恒为 offset ⇒ 输入信息被抹掉。
# 本脚本做因果验证：唯一变量 = normalizer。
#
# 用法
# ----
#   TAG=cf_int4_restore \
#   SRC=/root/fedcore/keep/c_int4_global_r49 \
#   CLEAN=/root/fedcore/keep/c_int4_client0_r49 \
#   bash /root/fedcore/cf_eval.sh
#
# ⚠️ 铁律 A：MoE 环境变量必须与训练侧**逐字一致**，
#    否则评估会静默丢掉 264 个专家权重（strict=False，零报错），
#    评出来的是 13.94M 的残缺模型 ⇒ 结果作废。
# ⚠️ 铁律 G：必须在 /root/HPT 下调用（python -m hpt.run_eval 依赖 CWD）。
# ==============================================================================
set -u

TAG=${TAG:?设置 TAG（决定日志名）}
SRC=${SRC:?设置 SRC（量化后的全局模型目录）}
CLEAN=${CLEAN:?设置 CLEAN（客户端本地权重目录，干净 normalizer 来源）}
DST=${DST:-/root/fedcore/cf/$TAG}
DOMAIN=${DOMAIN:-mujoco_metaworld_task3}
ENVNAME=${ENVNAME:-button-press-topdown-v2}
N_SEEDS=${N_SEEDS:-3}
EPISODE_NUM=${EPISODE_NUM:-20}
N_JOBS=${N_JOBS:-2}
OUT=${OUT:-/root/HPT/output}
CORE=${CORE:-/root/fedcore}
PY=${PY:-/root/miniconda3/envs/hpt/bin/python}

mkdir -p "$OUT" "$DST"

# ===== MoE 环境变量（必须逐字复刻训练侧）=====
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half
export OMP_NUM_THREADS=2
export PYTHONPATH="/root/HPT:${PYTHONPATH:-}"

echo "[cf] 打补丁：干净 normalizer ← $CLEAN，注入 $DST（源 $SRC）"
"$PY" "$CORE/cf_restore.py" --src "$SRC" --clean "$CLEAN" --dst "$DST" \
      --out "$OUT/cf_${TAG}_restore_report.json"

echo "[cf] 开始评估：domain=$DOMAIN env=$ENVNAME seeds=$N_SEEDS episodes=$EPISODE_NUM"
MODEL_DIR="$DST" DOMAIN="$DOMAIN" ENVNAME="$ENVNAME" \
  N_SEEDS="$N_SEEDS" EPISODE_NUM="$EPISODE_NUM" N_JOBS="$N_JOBS" \
  LOG_TAG="$TAG" \
  bash "$CORE/eval_model.sh" > "$OUT/cf_${TAG}_eval_stdout.log" 2>&1
rc=$?
echo "[cf] eval rc=$rc"

# 参数量自检（MoE 是否生效）：< 18M ⇒ 残缺模型 ⇒ 结果作废
npm=$(grep -oE "number of params \(M\): [0-9.]+" "$OUT/eval_${TAG}_${DOMAIN}.log" 2>/dev/null \
      | head -1 | grep -oE "[0-9.]+$" || true)
echo "[cf] 参数量 = ${npm}M"

# 汇总成 markdown（**必须**是 .md/.json —— 自动关机守护只推 *.md/*.json/*.csv/*.png）
{
  echo "# 反事实评估：${TAG}"
  echo ""
  echo "- 模型目录：\`$DST\`（量化后的全局模型 + **干净 normalizer**）"
  echo "- 对照：\`$SRC\`（同一模型，normalizer 保持量化损伤）"
  echo "- 任务：$DOMAIN / $ENVNAME，${N_SEEDS} seeds × ${EPISODE_NUM} rollouts"
  echo "- 参数量：${npm}M（MoE 生效判据 > 18M）"
  echo ""
  echo '```'
  cat "$OUT/cf_${TAG}_eval_stdout.log" 2>/dev/null | tail -n 60
  echo '```'
} > "$OUT/cf_${TAG}_result.md"

echo "[cf] 结果摘要 → $OUT/cf_${TAG}_result.md"
grep -c '| avg' "$OUT/eval_${TAG}_${DOMAIN}.log" 2>/dev/null || echo 0
exit 0
