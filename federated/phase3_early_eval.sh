#!/usr/bin/env bash
# 第 3 阶段「按需提前评估」守护（多 TAG 版）。
#
# 为什么需要它
# ------------
# phase3_eval_watch.sh 等 PHASE3_DONE（4 个组全训完）才开评。
# 但实测（2026-11-07 19:13）：P3-B/P3-C 在 17:46 就训完了，P3-D 还在 r3，
# 评估却要一直等到 P3-D 跑完 —— **白等 1~2 小时**。
#
# 本守护改为：**哪个组先训完就先评哪个**，评估与后续训练并行。
# 训练偏 GPU、评估偏 CPU/MuJoCo，两者可以并行 → 总时长 = max(剩余训练, 剩余评估)
# 而不是 剩余训练 + 剩余评估。
#
# 互斥（关键）
# -----------
# phase3_eval_watch.sh 也会跑 run_phase3_eval.sh。两者若同时对同一模型开跑，
# 会双写同一个日志（eval_model.sh 用 `> "$LOG"` 覆盖写）→ 结果错乱。
# 保护手段（两道）：
#   1) 目录锁 /root/HPT/output/.phase3_eval.lock —— 谁先拿到谁评估
#   2) run_phase3_eval.sh 内部每个 (log_tag, domain) 有独立锁
#      （.lock_eval_<tag>_<domain>），最细粒度防双写
#
# 幂等：run_phase3_eval.sh 按「日志里已有 N_SEEDS 张 | avg 表」跳过已完成项，
#       正式守护后续跑到同一模型会自动跳过 → 不会重复评估。
#
# 用法（云端）：
#   nohup setsid bash /root/HPT/federated/phase3_early_eval.sh \
#       > /root/HPT/output/phase3_early_eval.log 2>&1 < /dev/null &
set -u

SCRIPT_DIR=/root/HPT/federated
OUT=/root/HPT/output
LOCK="$OUT/.phase3_eval.lock"
LOG="$OUT/phase3_early_eval.log"
N_JOBS="${N_JOBS:-3}"
# 所有第 3 阶段的实验组（含 hetero 三组 + iid 一组）
TAGS="${TAGS:-p3_moe_fedavg p3_moe_eda p3_moe_eda_cluster p3_moe_iid}"
# 最后一个组的最后一轮
LAST_ROUND="${LAST_ROUND:-49}"

log() { echo "[$(date '+%F %T')] $*"; }

# ── 陈旧锁清理 + 抢锁 ────────────────────────────────────────────────────
# ⚠️ 陈旧锁处理（2026-11-07 补）：实例关机时持锁进程被杀，锁目录会**留在磁盘上**。
#    裸 mkdir 会失败 → 误判"已有守护持锁"→ 直接退出 → 没人评估，且零报错。
#    正确做法：先看锁里的 pid 是否还活着，死了就清掉再抢。
if [ -d "$LOCK" ]; then
  lockpid=$(cat "$LOCK/pid" 2>/dev/null || echo "")
  if [ -n "$lockpid" ] && kill -0 "$lockpid" 2>/dev/null; then
    log "已有评估守护持锁（pid=$lockpid 存活），本次提前评估退出（交给正式守护）"
    exit 0
  fi
  log "清理残留评估锁 $LOCK（pid=${lockpid:-?} 已死）"
  rm -rf "$LOCK"
fi
if ! mkdir "$LOCK" 2>/dev/null; then
  log "抢锁失败（$LOCK 可能被并发者抢走），退出"
  exit 0
fi
echo $$ > "$LOCK/pid"
log "拿到评估锁 $LOCK (pid=$$)"
trap 'rm -rf "$LOCK" 2>/dev/null || true' EXIT

# ── 轮询：哪个组先训完就先评哪个 ─────────────────────────────────────────
deadline=$(( $(date +%s) + ${MAX_WAIT:-43200} ))
done_tags=""
while :; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "等待超时，退出"; exit 1
  fi
  all_done=1
  for tag in $TAGS; do
    # 已处理过
    case " $done_tags " in *" $tag "*) continue ;; esac
    all_done=0
    MODEL="$OUT/$tag/global_r${LAST_ROUND}/trunk.pth"
    if [ ! -f "$MODEL" ]; then continue; fi
    log "$tag 训练完成（$MODEL），开始评估"
    STAGES="$tag" N_JOBS="$N_JOBS" bash "$SCRIPT_DIR/run_phase3_eval.sh"
    log "$tag 评估完成"
    done_tags="$done_tags $tag"
  done
  if [ "$all_done" = "1" ]; then
    log "所有组均已训练并评估完毕"
    break
  fi
  # 评估还在跑的时候不要空转太频繁
  sleep 60
done

log "PHASE3_EARLY_EVAL_DONE"
