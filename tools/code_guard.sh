#!/usr/bin/env bash
# FedVLA 代码指纹守护（云端常驻）
#
# 背景（2026-11-08 实测事故）：
#   云端 /root/HPT/federated/ 下的 22 个脚本在 17:39~17:43 之间被**批量重写成旧版**
#   （属主 197609 = 推送通道的 UID，不是 root；内容完整、不是截断）。
#   后果链：
#     · aggregate.py 退回 `type=int` → `--drop-clients "0,1"` argparse 报错
#       → c_drop50/c_drop75 在 round 1 聚合失败（第一轮就白跑）
#     · run_fedavg.sh 退回旧版 → DROPPED_ARGS 修正丢失 → 同上
#     · run_comm_scan_v2.sh 若退回 v1（25 轮）→ 基线落在地板
#       → 通信曲线零区分度 → 又一次 silent failure
#   触发者尚未定位（relay 服务端无同步逻辑，本地无并发进程，automation 为一次性已过期）。
#   结论：**不能假设"推上去就一直在"**，必须有自愈。
#
# 做法：每 60 秒校验关键脚本的特征串；一旦发现被覆盖成旧版，
#   立刻从 /root/HPT/output/.guard_backup/ 恢复，并写日志。
#   守护脚本本身放在 output/ 下（不在 federated/），避免被同一机制覆盖。
set -u

DIR=/root/fedcore
BAK=/root/HPT/output/.guard_backup
LOG=/root/HPT/output/code_guard.log
LOCK=/root/HPT/output/.code_guard.lock

# ⚠️ 2026-11-08：守护对象是 **/root/fedcore**（隔离副本），不是 /root/HPT/federated。
#   原因：/root/HPT/federated/ 会被未知并发进程周期性还原成旧版
#   （relay 队列重放 + 目录级覆盖，实测 22 个文件被批量重写）。
#   /root/fedcore/ 是从 federated/ 拷出的**独立副本**，
#   所有第4阶段脚本都从 fedcore 跑，federated/ 被覆盖不影响实验。

mkdir -p "$BAK"

log () { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 抢占锁（pid 判活，死锁则清掉重抢 —— 实例重启会留下僵尸锁）
if mkdir "$LOCK" 2>/dev/null; then
  echo $$ > "$LOCK/pid"
else
  old=""
  [ -f "$LOCK/pid" ] && old=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$old" ] && ! kill -0 "$old" 2>/dev/null; then
    log "[锁] 死锁（pid=$old 已死），清理后重抢"
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null && echo $$ > "$LOCK/pid"
  else
    log "[锁] 已有守护在跑（pid=$old），退出"
    exit 0
  fi
fi

# 建立/刷新备份（只在当前文件"合格"时才刷新，避免把旧版存进备份）
backup_if_good () {
  local f="$1" pat="$2" name
  name=$(basename "$f")
  if [ -f "$f" ] && grep -q -- "$pat" "$f"; then
    cp -f "$f" "$BAK/$name"
    log "[备份] 已刷新 $name"
  else
    log "[备份] $name 当前不合格，**不覆盖备份**"
  fi
}

check_one () {
  local f="$1" pat="$2" name bak
  name=$(basename "$f")
  bak="$BAK/$name"
  if [ ! -f "$f" ]; then
    log "[!!] $name 缺失"
    if [ -f "$bak" ]; then cp -f "$bak" "$f"; log "[OK] 已从备份恢复 $name"; fi
    return 1
  fi
  if ! grep -q -- "$pat" "$f"; then
    log "[!!] $name 缺少特征 [$pat] —— 被覆盖成旧版"
    if [ -f "$bak" ]; then
      cp -f "$bak" "$f"
      if grep -q -- "$pat" "$f"; then
        log "[OK] 已从备份恢复 $name"
      else
        log "[!!] 恢复后仍不合格，备份本身可能是坏的"
      fi
    else
      log "[!!] 无备份，无法恢复 $name"
    fi
    return 1
  fi
  return 0
}

# 首次建立备份
backup_if_good "$DIR/aggregate.py" 'str(tok).replace'
backup_if_good "$DIR/run_fedavg.sh" 'DROPPED_ARGS'
backup_if_good "$DIR/run_comm_scan_v2.sh" 'eval_lock_acquire'
backup_if_good "$DIR/run_comm_scan_v2.sh" 'MAX_PAR'
backup_if_good "$DIR/privacy_experiment.py" 'allocate_by_sigma'
backup_if_good "$DIR/extract_privacy_results.py" 'modality_rev'

log "===== 代码指纹守护启动（每 60 秒校验）====="
n_bad=0
while true; do
  check_one "$DIR/aggregate.py" 'str(tok).replace' || n_bad=$((n_bad+1))
  check_one "$DIR/run_fedavg.sh" 'DROPPED_ARGS' || n_bad=$((n_bad+1))
  check_one "$DIR/run_comm_scan_v2.sh" 'eval_lock_acquire' || n_bad=$((n_bad+1))
  check_one "$DIR/run_comm_scan_v2.sh" 'MAX_PAR' || n_bad=$((n_bad+1))
  check_one "$DIR/privacy_experiment.py" 'allocate_by_sigma' || n_bad=$((n_bad+1))
  check_one "$DIR/extract_privacy_results.py" 'modality_rev' || n_bad=$((n_bad+1))
  if [ "$n_bad" -gt 0 ]; then
    log "[自愈] 本轮修复了 $n_bad 处覆盖；合格后刷新备份"
    n_bad=0
    backup_if_good "$DIR/aggregate.py" 'str(tok).replace'
    backup_if_good "$DIR/run_fedavg.sh" 'DROPPED_ARGS'
    backup_if_good "$DIR/run_comm_scan_v2.sh" 'eval_lock_acquire'
    backup_if_good "$DIR/run_comm_scan_v2.sh" 'MAX_PAR'
    backup_if_good "$DIR/privacy_experiment.py" 'allocate_by_sigma'
    backup_if_good "$DIR/extract_privacy_results.py" 'modality_rev'
  fi
  sleep 60
done
