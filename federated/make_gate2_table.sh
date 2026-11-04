#!/usr/bin/env bash
# 门2 主对比表：集中式 vs 等预算集中式 vs IID 联邦 vs 异构 联邦 vs 异构本地模型。
#
# 为什么新写而不是改 compare_results.sh：
#   compare_results.sh 里联邦日志名按模型目录 basename 猜（global_r49），
#   而实际 LOG_TAG 是 hetero_r49 / iid_r49，等预算又是 equalbudget_task1，
#   三套命名对不上 -> 老脚本会全部显示"—"。
#   这里改成显式列出各列的日志候选（通配符），任何一套命名都能命中。
#
# 用法：bash make_gate2_table.sh
set -u
SCRIPT_DIR=/root/HPT/federated
HPT_PY=/root/miniconda3/envs/hpt/bin/python
OUT=/root/HPT/output/gate2_table.txt

# 列定义：  标签 | 日志 glob（按优先级）
COL_CENTRAL="集中式(6000步)"
COL_EQUAL="等预算集中式(2250步)"
COL_IID="IID联邦(50轮)"
COL_HETERO="异构联邦(50轮)"

pick_log() {  # $1 = glob（不带 eval_ 前缀），返回最新命中的完整路径
  ls -1t /root/HPT/output/eval_$1 2>/dev/null | head -1
}

extract() {  # $1=log $2=env -> "0.900 ± 0.092"
  [ -f "$1" ] || { echo "—"; return; }
  local v
  v=$("$HPT_PY" "$SCRIPT_DIR/summarize_eval.py" "$1" 2>/dev/null \
      | awk -v e="$2" '$1==e {print $4" ± "$6; f=1} END{if(!f) print "—"}' | head -1)
  [ -n "$v" ] && echo "$v" || echo "—"
}

row() {  # $1=env $2=central_glob $3=equal_glob $4=iid_glob $5=hetero_glob
  local env="$1"
  local lc le li lh
  lc=$(pick_log "$2"); le=$(pick_log "$3"); li=$(pick_log "$4"); lh=$(pick_log "$5")
  printf "%-24s | %-20s | %-22s | %-20s | %-20s\n" \
    "$env" \
    "$(extract "$lc" "$env")" \
    "$(extract "$le" "$env")" \
    "$(extract "$li" "$env")" \
    "$(extract "$lh" "$env")"
  printf "    日志: C=%s | E=%s | IID=%s | HET=%s\n" \
    "${lc:-无}" "${le:-无}" "${li:-无}" "${lh:-无}"
}

{
echo "================================================================"
echo "FedVLA 门2 主对比表"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "统计口径: 3 seeds × 20 rollouts/seed, 均值 ± 95% CI (t 分布)"
echo "================================================================"
echo
printf "%-24s | %-20s | %-22s | %-20s | %-20s\n" \
  "任务" "$COL_CENTRAL" "$COL_EQUAL" "$COL_IID" "$COL_HETERO"
printf "%s\n" "------------------------------------------------------------------------------------------------"

row "reach-v2" \
  "centralized_task1*.log" \
  "*equal_budget*task1*.log" \
  "iid_r49*task1*.log" \
  "hetero_r49*task1*.log"

row "push-v2" \
  "centralized_task2*.log" \
  "*equal_budget*task2*.log" \
  "iid_r49*task2*.log" \
  "hetero_r49*task2*.log"

row "button-press-topdown-v2" \
  "centralized_task3*.log" \
  "*equal_budget*task3*.log" \
  "iid_r49*task3*.log" \
  "hetero_r49*task3*.log"

echo
echo "---- 诊断项：异构联邦里 task1 客户端自己的本地模型（聚合前）----"
row "reach-v2 (client_0 local)" \
  "centralized_task1*.log" \
  "*equal_budget*task1*.log" \
  "client0_r49*task1*.log" \
  "hetero_r49*task1*.log"

echo
echo "================================================================"
echo "读法："
echo "  1) IID 联邦 ≈ 集中式  ->  联邦管线本身没问题（健全性对照）"
echo "  2) 异构联邦 << IID 联邦  ->  复现『异构有害』，门2 判定通过"
echo "  3) 等预算集中式 ≈ 集中式  ->  排除『训练量不足』这一替代解释"
echo "================================================================"
} | tee "$OUT"

echo
echo "已写入: $OUT"
echo "GATE2_TABLE_DONE"
