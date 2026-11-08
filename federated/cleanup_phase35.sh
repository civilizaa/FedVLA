#!/usr/bin/env bash
# 第 4 阶段开工前的磁盘回收（带严格保护）。
#
# 背景：第 3 阶段 4 个实验组的权重还占着 ~6G，加上第 4 阶段的 7 个配置，
#   30G 的系统盘会撑爆 —— 磁盘满会导致 relay agent 心跳停摆（历史头号故障源）。
#
# ⚠️ 铁律：**必须保留每组最后一轮的 global_r<N>/trunk.pth**
#   —— resume_all.sh 的幂等判据是 `global_r49/trunk.pth` 存在 ⇒ 第3阶段已完成。
#   删掉它会导致实例一重启，第3阶段的 4 组训练被**全部重跑**（4×96 分钟）。
#
# 删除范围（只对"已训完且不在训练中"的组）：
#   1) client_<c>_r<N>   —— 客户端中间产物，评估只用 global_r 的 model.pth
#   2) global_r<N> 中**非最后一轮**的
#   3) opt_state_c*.pt    —— Adam 状态，训练结束即无用
# 保留：global_r<最后>（含 eda_report.txt / eda_weights.pt / config.yaml）
set -u
OUT=/root/HPT/output
freed_total=0

# 正在训练的组（命令行里带该 tag）—— 一个都不能碰
busy_tags () {
  pgrep -af "hpt.run|run_comm_scan|run_fedavg" 2>/dev/null | \
    grep -oE "output/[A-Za-z0-9_]+" | sed 's|output/||' | sort -u
}
BUSY=$(busy_tags)
echo "正在运行的组（受保护）：${BUSY:-无}"
echo

for d in "$OUT"/*/; do
  tag=$(basename "$d")
  # 正在训练的组 → 跳过
  if echo "$BUSY" | grep -qx "$tag"; then
    echo "[SKIP] $tag 正在训练，全部保留"
    continue
  fi
  # 找最后一轮（global_r<N> 里 N 最大的）
  last=""
  for g in "$d"global_r*/; do
    [ -d "$g" ] || continue
    n=$(basename "$g" | sed 's/global_r//')
    if [ -z "$last" ] || [ "$n" -gt "$last" ]; then last="$n"; fi
  done
  if [ -z "$last" ]; then
    echo "[SKIP] $tag 没有 global_r* 目录，跳过"
    continue
  fi

  freed=0
  # 1) 客户端中间产物
  for c in "$d"client_*_r*/; do
    [ -d "$c" ] || continue
    sz=$(du -sm "$c" 2>/dev/null | cut -f1); [ -n "$sz" ] || sz=0
    rm -rf "$c"
    freed=$((freed + sz))
  done
  # 2) 非最后一轮的 global_r
  for g in "$d"global_r*/; do
    [ -d "$g" ] || continue
    n=$(basename "$g" | sed 's/global_r//')
    [ "$n" = "$last" ] && continue          # 最后一轮保留
    sz=$(du -sm "$g" 2>/dev/null | cut -f1); [ -n "$sz" ] || sz=0
    rm -rf "$g"
    freed=$((freed + sz))
  done
  # 3) Adam 状态
  for f in "$d"opt_state_c*.pt; do
    [ -f "$f" ] || continue
    sz=$(du -sm "$f" 2>/dev/null | cut -f1); [ -n "$sz" ] || sz=0
    rm -f "$f"
    freed=$((freed + sz))
  done

  if [ "$freed" -gt 0 ]; then
    echo "[CLEAN] $tag 释放 ${freed}MB（保留 global_r${last}）"
    freed_total=$((freed_total + freed))
  else
    echo "[KEEP] $tag 无可清理内容（保留 global_r${last}）"
  fi
done

echo
echo "总计释放 ${freed_total}MB"
df -BG /root | tail -1
