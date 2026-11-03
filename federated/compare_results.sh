#!/usr/bin/env bash
# FedVLA：把集中式基线与联邦全局模型的评估结果汇总成论文对比表。
#
# 用法：
#   bash compare_results.sh                      # 默认路径
#   CENTRAL_ROOT=/root/HPT/output/centralized \
#   FED_MODEL=/root/HPT/output/fedavg_hetero/global_r49 \
#   bash compare_results.sh
#
# 输出：每个任务上 集中式 vs 联邦 的均值 ± 95% CI，以及差值。
set -u
SCRIPT_DIR=/root/HPT/federated
HPT_PY=/root/miniconda3/envs/hpt/bin/python

CENTRAL_ROOT=${CENTRAL_ROOT:-/root/HPT/output/centralized}
FED_MODEL=${FED_MODEL:-/root/HPT/output/fedavg_hetero/global_r49}
OUT=/root/HPT/output/compare_results.txt

# task -> (domain, env)
TASKS=(
  "mujoco_metaworld_task1 reach-v2"
  "mujoco_metaworld_task2 push-v2"
  "mujoco_metaworld_task3 button-press-topdown-v2"
)

{
echo "================================================================"
echo "FedVLA 结果对比：集中式基线 vs 联邦全局模型"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo
echo "集中式根目录: $CENTRAL_ROOT"
echo "联邦模型:     $FED_MODEL"
echo

printf "%-26s %-24s %-24s %s\n" "任务" "集中式 (mean±95%CI)" "联邦 (mean±95%CI)" "差值"
printf "%s\n" "------------------------------------------------------------------------"

for pair in "${TASKS[@]}"; do
  set -- $pair
  dom="$1"; env="$2"; task="${dom#mujoco_metaworld_}"

  # 集中式：约定输出目录 output/centralized/<task>/，评估日志写在
  #   output/eval_centralized_<task>.log（由 run_centralized.sh + eval_model.sh 约定）
  #   若不存在，则找 output/eval_centralized_<dom>.log
  c_log=""
  for cand in \
      "/root/HPT/output/eval_centralized_${task}.log" \
      "/root/HPT/output/eval_centralized_${dom}.log" \
      "$CENTRAL_ROOT/${task}/eval.log"; do
    [ -f "$cand" ] && { c_log="$cand"; break; }
  done

  f_log="/root/HPT/output/eval_$(basename "$FED_MODEL")_${dom}.log"

  c_val="—"
  f_val="—"

  # summarize_eval.py 的输出行形如： "  reach-v2    n=3  0.983 ± 0.053   [0.930, 1.036]"
  # -> 成功率的 mean 在第 4 个字段，CI 半宽在第 6 个字段
  extract () {
    [ -f "$1" ] || return 1
    $HPT_PY "$SCRIPT_DIR/summarize_eval.py" "$1" 2>/dev/null \
      | awk -v e="$2" '$1==e {print $4" ± "$6; found=1} END{if(!found) print "—"}' \
      | head -1
  }

  [ -n "$c_log" ] && c_val=$(extract "$c_log" "$env")
  [ -f "$f_log" ] && f_val=$(extract "$f_log" "$env")

  printf "%-26s %-20s %-20s\n" "$env" "$c_val" "$f_val"
  echo "    集中式日志: ${c_log:-未找到}"
  echo "    联邦日志:   ${f_log:-未找到}"
done

echo
echo "================================================================"
echo "注：CI 为 t 分布 95% 置信区间（seed 数少时比正态更保守）"
echo "    统计口径：3 seeds × 20 rollouts"
echo "================================================================"
} | tee "$OUT"

echo
echo "已写入: $OUT"
echo "COMPARE_DONE"
