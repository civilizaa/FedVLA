#!/usr/bin/env bash
# ==============================================================================
# 第4阶段「跑完自动关机」守护脚本
#
# 目的：用户最怕「实例空烧钱」。本脚本保证：
#   1) 正常路径 —— 所有实验跑完 → 云端提取 → 打包 → 标记完成 → 立即关机（停止计费）
#   2) 异常路径 —— DP 阶段中断 / GPU 卡死 / 超过绝对时限 → 同样先打包再关机
#   ⇒ 无论发生什么，实例都不会空跑到天亮。
#
# ⚠️ 关机方式（实测 2026-11-18）：
#    AutoDL 容器内 /usr/bin/shutdown 是官方放的小脚本，内容是 kill supervisord
#    ⇒ 容器退出 ⇒ AutoDL 判定实例停止 ⇒ 停止计费（关机后约 5~26 秒生效）。
#    /usr/sbin/poweroff → /bin/systemctl，而容器里 systemd 没跑（pid1=bash，
#    无 /run/systemd/system）⇒ systemctl 必定失败，**不能用 poweroff**。
#    ⚠️ 网上说「shutdown -h +120 是延时关机」是错的：该脚本不解析参数，
#       传任何参数都是**立即**关机。本脚本因此只在确认做完后才调用它。
#
# ⚠️ 关机 ⇒ relay agent 死亡 ⇒ 云端文件没法再拉。但 AutoDL 关机**不删系统盘**
#    ⇒ /root/HPT/output 里的产物全部留存，重开实例即可拉走。
#
# 用法：
#   nohup setsid bash /root/fedcore/phase4_autoshutdown.sh \
#         > /root/HPT/output/autoshutdown.log 2>&1 < /dev/null &
#
# 可调环境变量：
#   WAIT_DP=0     只等通信侧 7 个配置完成就收工（省掉 DP 效用扫描的 3.3 小时）
#   MAX_HOURS=10  绝对时限（小时），到点无论完成与否都先打包再关机
#   POLL=60       轮询间隔（秒）
# ==============================================================================

set -u

# ⚠️ OUT / CORE 可用环境变量覆盖 → 便于本地离线单测（造假目录跑分支逻辑）
OUT=${OUT:-/root/HPT/output}
CORE=${CORE:-/root/fedcore}
PY=${PY:-/root/miniconda3/envs/hpt/bin/python}
SHUTDOWN_BIN=${SHUTDOWN_BIN:-/usr/bin/shutdown}

COMM_TAGS="c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1"
DP_TAGS="dp_s0001 dp_s0003 dp_s0012"

WAIT_DP=${WAIT_DP:-1}
MAX_HOURS=${MAX_HOURS:-12}
POLL=${POLL:-60}
IDLE_MIN=${IDLE_MIN:-30}          # GPU 空闲多少分钟判定为卡死
FINAL_SLEEP=${FINAL_SLEEP:-60}    # 关机前等待（打包产物 flush）
GPU_UTIL_CMD=${GPU_UTIL_CMD:-"nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"}
PGREP_BIN=${PGREP_BIN:-pgrep}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

count_rc() {
    local n=0 t
    for t in $1; do
        [ -f "$OUT/.rc_$t" ] && n=$((n + 1))
    done
    echo "$n"
}

start_ts=$(date +%s)
idle_hits=0
last_sig=""
poll_n=0
REASON="未知"
log "===== 第4阶段自动关机守护启动 ====="
EXPECT_COMM=$(echo $COMM_TAGS | wc -w)
EXPECT_DP=$(echo $DP_TAGS | wc -w)
log "等待条件：通信 ${COMM_TAGS} (${EXPECT_COMM} 个) + DP ${DP_TAGS} (${EXPECT_DP} 个)"
log "WAIT_DP=$WAIT_DP  MAX_HOURS=$MAX_HOURS  POLL=${POLL}s"

# ------------------------------------------------------------------------------
# 主循环：等完成 / 等失败 / 等卡死 / 等超时
# ------------------------------------------------------------------------------
while : ; do
    ncomm=$(count_rc "$COMM_TAGS")
    ndp=$(count_rc "$DP_TAGS")
    chain_alive=$($PGREP_BIN -f 'chain_phase4_dp\.sh' | head -1 || true)

    # --- 正常完成 ---
    if [ "$ncomm" -eq "$EXPECT_COMM" ] && { [ "$WAIT_DP" -eq 0 ] || [ "$ndp" -eq "$EXPECT_DP" ]; }; then
        REASON="全部实验完成（通信 $ncomm/$EXPECT_COMM，DP $ndp/$EXPECT_DP）"
        break
    fi
    # --- 通信已完成但 DP 链路死了 ⇒ DP 失败，提前收工 ---
    if [ "$ncomm" -eq "$EXPECT_COMM" ] && [ "$WAIT_DP" -eq 1 ] && [ "$ndp" -lt "$EXPECT_DP" ] && [ -z "$chain_alive" ]; then
        REASON="通信已完成但 DP 接力链已退出（DP $ndp/$EXPECT_DP 不全）→ 提前收工"
        break
    fi
    # --- 通信本身跑挂了（主控进程死了且 .rc 不齐）⇒ 提前收工 ---
    if [ "$ncomm" -lt "$EXPECT_COMM" ] && ! $PGREP_BIN -f 'run_comm_scan_v2\.sh' > /dev/null 2>&1; then
        # 主控不在了，可能是跑完退出了（.rc 会齐）也可能崩了
        if [ "$ncomm" -eq "$EXPECT_COMM" ]; then :; else
            REASON="通信主控进程消失且 .rc 不全（$ncomm/$EXPECT_COMM）→ 提前收工"
            break
        fi
    fi
    # --- GPU 卡死保护：利用率低且轮次不变，持续 IDLE_MIN 分钟 ---
    util=$($GPU_UTIL_CMD 2>/dev/null | head -1)
    util=${util:-100}
    sig=$(ls -1dt "$OUT"/*/global_r* 2>/dev/null | head -1)
    if [ "$util" -lt 10 ] && [ "$sig" = "$last_sig" ]; then
        idle_hits=$((idle_hits + 1))
        if [ $((idle_hits * POLL)) -ge $((IDLE_MIN * 60)) ]; then
            REASON="GPU 空闲 ${IDLE_MIN} 分钟且轮次无变化（util=${util}%）→ 判定卡死"
            break
        fi
    else
        idle_hits=0
    fi
    last_sig="$sig"

    # --- 心跳（每 10 轮记一行，方便确认守护还活着）---
    poll_n=$((poll_n + 1))
    if [ $((poll_n % 10)) -eq 0 ]; then
        log "心跳：通信 .rc=$(count_rc "$COMM_TAGS")/$EXPECT_COMM  DP .rc=$(count_rc "$DP_TAGS")/$EXPECT_DP  util=${util}%  链=$([ -n "$chain_alive" ] && echo 活 || echo 死)"
    fi

    # --- 绝对时限 ---
    now_ts=$(date +%s)
    if [ $((now_ts - start_ts)) -ge $((MAX_HOURS * 3600)) ]; then
        REASON="达到绝对时限 ${MAX_HOURS} 小时 → 强制收工"
        break
    fi
    sleep "$POLL"
done

log "收工原因：$REASON"
log "通信 .rc = $(count_rc "$COMM_TAGS")/$EXPECT_COMM   DP .rc = $(count_rc "$DP_TAGS")/$EXPECT_DP"

# ------------------------------------------------------------------------------
# 提取结果（失败不阻断关机）
# ------------------------------------------------------------------------------
log "--- 云端提取通信侧结果 ---"
"$PY" "$CORE/extract_comm_results.py" >> "$OUT/extract_comm.log" 2>&1
log "extract_comm_results rc=$?"
log "--- 云端提取 DP 效用结果 ---"
"$PY" "$CORE/extract_dp_utility.py" >> "$OUT/extract_dp.log" 2>&1
log "extract_dp_utility rc=$?"

# ------------------------------------------------------------------------------
# 打包产物（关机不删盘，但先打好包，重开直接取）
# ------------------------------------------------------------------------------
log "--- 打包产物 ---"
cd "$OUT" || exit 0
FILES=$(ls -1 *.json *.md *.csv *.png 2>/dev/null | tr '\n' ' ')
# shellcheck disable=SC2086
tar czf "$CORE/final_out.tgz" $FILES 2>/dev/null
tar czf "$CORE/privacy_out.tgz" privacy/*.json privacy/*.md privacy/*.csv 2>/dev/null
ls -la "$CORE"/*.tgz 2>/dev/null
log "final_out.tgz = $(du -h "$CORE/final_out.tgz" 2>/dev/null | cut -f1)"

# ------------------------------------------------------------------------------
# 推进 relay 结果存储（🔴 最关键的一步）
#
# 为什么必须有：关机 ⇒ 容器退出 ⇒ **relay agent 一起死** ⇒
#   "下发命令→读 stdout"那条拉文件通道随之失效。
#   但 **relay 服务端跑在 WorkBuddy 托管的云主机上，不随容器死**。
#   ⇒ 只要在关机前把产物写进服务端的 /result 存储，
#     产物就留在云端，本地随时能取（tools/relay_fetch_out.py），
#     **不需要重开实例、不需要人工拉文件**。
#
# 附带好处：文本类文件以明文存放 ⇒ 报告自动填充可以直接读。
# ------------------------------------------------------------------------------
log "--- 推进 relay 结果存储 ---"
# ⚠️ extract_*.log 里存着**基线自检结论与数据表**（脚本把表打到 stdout），
#    绝不能漏掉；scanlog_*.log 是训练轨迹证据（LR 轨迹、续跑起点）。
PUSH_FILES=$(cd "$OUT" && ls -1 *.json *.md *.csv *.png \
             privacy/*.json privacy/*.md privacy/*.csv privacy/*.png \
             extract_comm.log extract_dp.log chain_phase4_dp.log \
             autoshutdown.log scanlog_*.log 2>/dev/null \
             | tr '\n' ' ')
"$PY" "$CORE/relay_push_out.py" --outdir "$OUT" --files "$PUSH_FILES" \
      --idbase 900000 >> "$OUT/relay_push.log" 2>&1
push_rc=$?
if [ "$push_rc" -ne 0 ]; then
    log "relay 推送失败 rc=$push_rc，等 20 秒重试一次"
    sleep 20
    "$PY" "$CORE/relay_push_out.py" --outdir "$OUT" --files "$PUSH_FILES" \
          --idbase 900000 >> "$OUT/relay_push.log" 2>&1
    push_rc=$?
fi
push_n=$(echo $PUSH_FILES | wc -w)
log "relay_push rc=$push_rc  文件数=$push_n"

# ------------------------------------------------------------------------------
# 完成标记
# ------------------------------------------------------------------------------
{
    echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "reason=$REASON"
    echo "comm_rc=$(count_rc "$COMM_TAGS")/$EXPECT_COMM"
    echo "dp_rc=$(count_rc "$DP_TAGS")/$EXPECT_DP"
    echo "relay_push_rc=$push_rc"
    echo "relay_push_files=$push_n"
    echo "relay_fetch_cmd=python tools/relay_fetch_out.py --since 900000 --outdir results/cloud_final"
} > "$OUT/PHASE4_ALL_DONE"

log "✓ 产物已打包（/root/fedcore/final_out.tgz）+ 已推进 relay + 完成标记已写"
log "60 秒后自动关机（AutoDL 将停止计费）"

# ------------------------------------------------------------------------------
# 关机
# ------------------------------------------------------------------------------
sleep "$FINAL_SLEEP"
log "执行 $SHUTDOWN_BIN → kill supervisord → 容器退出 → AutoDL 停止计费"
"$SHUTDOWN_BIN"
log "shutdown 已调用（若 30 秒后本进程仍存在，属正常：日志尚未 flush）"
