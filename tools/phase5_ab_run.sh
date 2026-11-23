#!/usr/bin/env bash
# ==============================================================================
# 第 5 阶段评估并行度 A/B（2026-11-23 重建版）
#
# 🔴 为什么重建（原版在 task1 上做 A/B ⇒ **无效**）：
#    task1（reach-v2）在异构联邦下**所有变体都是 0.000**（第 3 阶段表已证实：
#    异构+MoE+FedAvg 的 reach-v2 = 0.000 ± 0.000）⇒ nj3 与 nj1 的 avg 行
#    必然都是 "| avg | 0.00" ⇒ **0 = 0 恒成立，比对毫无分辨力**，
#    不能证明「并行度不影响数值」。
#    ⇒ 改用 **task3（button-press-topdown-v2）**：第 3 阶段 p3_moe_fedavg 在
#      task3 上 = 0.283 ± 0.587（非退化、有分辨力）。
#
# 🔴 为什么用第 3 阶段的模型而不是第 5 阶段的：
#    ① 第 3 阶段 p3_moe_fedavg 的 task3 结果是**已知非零**（0.283）⇒ 比对有分辨力；
#    ② 不碰第 5 阶段的目录 ⇒ 与随后的「清理作废产物 + 开跑」互不干扰；
#    ③ 顺带做一次**复现性校验**：nj1 的结果应当与第 3 阶段原始日志逐行一致
#       （同模型、同 domain、同 seed）⇒ 若一致，说明主表复用第 3 阶段那 3 行是安全的。
#
# 用法：
#    M=/root/HPT/output/p3_moe_fedavg/global_r49 \
#    D=mujoco_metaworld_task3 E=button-press-topdown-v2 \
#    bash /root/fedcore/phase5_ab_run.sh
#
# 产物：
#    /root/HPT/output/eval_ab_nj3_<domain>.log   ← 3 seed 并行的评估日志
#    /root/HPT/output/eval_ab_nj1_<domain>.log   ← 3 seed 串行的评估日志
#    /root/HPT/output/ab_status.log              ← 末尾写 AB_DONE（autostart 等它）
# ==============================================================================
set -u
AB=/root/HPT/output
M=${M:-/root/HPT/output/p3_moe_fedavg/global_r49}
D=${D:-mujoco_metaworld_task3}
E=${E:-button-press-topdown-v2}

echo "[ab] 开始 $(date '+%F %T')  model=$M  domain=$D  env=$E"
if [ ! -f "$M/model.pth" ]; then
  echo "[FATAL] 模型目录缺 model.pth：$M —— A/B 无法进行"
  exit 1
fi

rm -f "$AB/ab_status.log"

echo "[ab] --- nj3（3 seed 并行）开始 $(date '+%F %T') ---"
MODEL_DIR="$M" DOMAIN="$D" ENVNAME="$E" N_SEEDS=3 EPISODE_NUM=20 N_JOBS=3 \
  LOG_TAG=ab_nj3 bash /root/fedcore/eval_model.sh > "$AB/ab_nj3.log" 2>&1
rc3=$?
echo "[ab] nj3 rc=$rc3 完成 $(date '+%F %T')"

echo "[ab] --- nj1（3 seed 串行）开始 $(date '+%F %T') ---"
MODEL_DIR="$M" DOMAIN="$D" ENVNAME="$E" N_SEEDS=3 EPISODE_NUM=20 N_JOBS=1 \
  LOG_TAG=ab_nj1 bash /root/fedcore/eval_model.sh > "$AB/ab_nj1.log" 2>&1
rc1=$?
echo "[ab] nj1 rc=$rc1 完成 $(date '+%F %T')"

echo "nj3_rc=$rc3" >> "$AB/ab_status.log"
echo "nj1_rc=$rc1" >> "$AB/ab_status.log"
echo AB_DONE >> "$AB/ab_status.log"
echo "[ab] AB_DONE 已写入 $AB/ab_status.log"
