#!/usr/bin/env bash
# ==============================================================================
# 第4阶段「修正版重跑」**云端总控**（v3, 2026-11-19）。
#
# 为什么要有这个总控：
#   2026-11-18 那轮通信扫描的两组数字不可用（c_drop00/c_drop25 从 r13 续训；
#   c_int8/c_int4 成功率全 0 判定为故障）。修正版重跑需要：
#     1) **先清洗**再开跑（否则又接在旧权重上 → 再次不同构）；
#     2) 通信扫描与 DP 效用扫描**串行接力**（并跑会 OOM，崩掉 9 小时算力）；
#     3) 跑完自动**提取 → 打包 → 推进 relay → 回读自检 → 关机**（停止计费）。
#
# 本脚本把上述三件事串成一条命令，并用 nohup setsid 起（不占 agent 前台）。
#
# 用法（云端）：
#     cd /root/HPT && nohup setsid bash /root/fedcore/rerun_phase4_fix.sh \
#           > /root/HPT/output/rerun_phase4.log 2>&1 < /dev/null &
#
# 可调环境变量：
#     RERUN_TAGS   要重跑的配置（默认 5 个）
#     KEEP_TAGS    保留的配置（默认 c_drop50 c_drop75）
#     MAX_HOURS    绝对时限（默认 13 小时）
#     PUSH_IDBASE  推送 id 基址（默认 990000，必须比上一轮大！）
#     SKIP_PREP=1  跳过清洗（调试用，**别在正式重跑里用**）
# ==============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
CORE=${CORE:-/root/fedcore}
RERUN_TAGS=${RERUN_TAGS:-"c_drop00 c_drop25 c_int8 c_int4 c_bit1"}
KEEP_TAGS=${KEEP_TAGS:-"c_drop50 c_drop75"}
MAX_HOURS=${MAX_HOURS:-13}
PUSH_IDBASE=${PUSH_IDBASE:-990000}
SKIP_PREP=${SKIP_PREP:-0}

log () { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$OUT"

log "=============================================================="
log "FedVLA 第4阶段 修正版重跑 总控"
log "重跑配置：$RERUN_TAGS"
log "保留配置：$KEEP_TAGS"
log "推送 idbase=$PUSH_IDBASE   绝对时限=${MAX_HOURS}h"
log "=============================================================="

# ----------------------------------------------------------------------------
# 步骤 0：环境自检（import hpt 硬拦截 —— 铁律 G）
#   HPT 源码树在 /root/HPT，没 pip install ⇒ `python -m hpt.run` 依赖 CWD。
#   换个调用方就炸（DP 标定从 /root/fedcore 启动 → 直接失败）。
#   ⇒ 开训前必须显式 export PYTHONPATH 并验证 import 成功。
# ----------------------------------------------------------------------------
HPT_PY=/root/miniconda3/envs/hpt/bin/python
export HPT_ROOT="${HPT_DIR:-/root/HPT}"
export PYTHONPATH="$HPT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if ! "$HPT_PY" -c "import hpt" 2>/dev/null; then
    log "[FATAL] import hpt 失败（PYTHONPATH=$PYTHONPATH）—— 停止，不要开跑"
    log "        排查：HPT 源码树是否在 $HPT_ROOT"
    exit 1
fi
log "[OK] import hpt 自检通过（PYTHONPATH=$PYTHONPATH）"

if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 | grep -q .; then
    log "[FATAL] nvidia-smi 不可用 —— 实例可能还没真正起来"
    exit 1
fi
log "[OK] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ----------------------------------------------------------------------------
# 步骤 1：清洗（让 5 个配置真的从 r0 起训）
# ----------------------------------------------------------------------------
if [ "$SKIP_PREP" != "1" ]; then
    log "----- [1/5] 清洗重跑范围 -----"
    OUT="$OUT" RERUN_TAGS="$RERUN_TAGS" KEEP_TAGS="$KEEP_TAGS" \
        bash "$SCRIPT_DIR/prep_rerun_phase4.sh"
    rc=$?
    if [ "$rc" != "0" ]; then
        log "[FATAL] 清洗失败 rc=$rc —— 停止（继续跑下去必然又不同构）"
        exit 1
    fi
    log "[OK] 清洗完成"
else
    log "[WARN] SKIP_PREP=1 —— 跳过清洗（仅调试用）"
fi

# ----------------------------------------------------------------------------
# 步骤 2：启动通信扫描（后台）
#   传 COMM_TAGS = 7 个（含 2 个保留）—— 保留配置会被"评估齐全"判据 SKIP，
#   同时正常写出 .rc_<tag>，下游 chain 的"7 个 .rc 齐全"判据才不会卡死。
# ----------------------------------------------------------------------------
COMM_ALL="c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1"
log "----- [2/5] 启动通信扫描 -----"
cd "$HPT_ROOT" || exit 1
COMM_TAGS="$COMM_ALL" nohup setsid bash "$CORE/run_comm_scan_v2.sh" \
    > "$OUT/comm_scan_v2.log" 2>&1 < /dev/null &
sleep 45
if ! pgrep -f "run_comm_scan_v2" > /dev/null 2>&1; then
    log "[FATAL] 通信扫描没起来 —— 停止"
    tail -n 30 "$OUT/comm_scan_v2.log" 2>/dev/null
    exit 1
fi
log "[OK] 通信扫描已启动（pid $(pgrep -f run_comm_scan_v2 | head -1)）"

# ----------------------------------------------------------------------------
# 步骤 3：启动自动关机守护
#   ⚠️ 必须在通信扫描**之后**启动：守护的"通信主控进程消失且 .rc 不齐 ⇒ 提前收工"
#      判据在扫描进程还没起来时会被误触发。
# ----------------------------------------------------------------------------
log "----- [3/5] 启动自动关机守护 -----"
OUT="$OUT" CORE="$CORE" WAIT_DP=1 MAX_HOURS="$MAX_HOURS" PUSH_IDBASE="$PUSH_IDBASE" \
    nohup setsid bash "$CORE/phase4_autoshutdown.sh" \
    > "$OUT/autoshutdown.log" 2>&1 < /dev/null &
sleep 10
if ! pgrep -f "phase4_autoshutdown" > /dev/null 2>&1; then
    log "[!!] 守护没起来 —— 检查 $OUT/autoshutdown.log"
else
    log "[OK] 守护已启动（pid $(pgrep -f phase4_autoshutdown | head -1)）"
fi

# ----------------------------------------------------------------------------
# 步骤 4：启动 DP 接力链（等 7 个 .rc 齐全后自动开跑）
# ----------------------------------------------------------------------------
log "----- [4/5] 启动 DP 接力链 -----"
OUT="$OUT" nohup setsid bash "$CORE/chain_phase4_dp.sh" \
    > "$OUT/chain_phase4_dp.log" 2>&1 < /dev/null &
sleep 5
log "[OK] DP 接力链已排队（会在通信扫描完成后自动标定 C + 扫 σ）"

# ----------------------------------------------------------------------------
# 步骤 5：等待收工（守护会自己提取/推送/关机，本进程随之被容器退出带走）
# ----------------------------------------------------------------------------
log "----- [5/5] 等待收工 -----"
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
while : ; do
    if [ -f "$OUT/PHASE4_ALL_DONE" ]; then
        log "[OK] 完成标记已出现："
        cat "$OUT/PHASE4_ALL_DONE"
        break
    fi
    if ! pgrep -f "phase4_autoshutdown" > /dev/null 2>&1; then
        log "[!!] 守护进程已退出且没有完成标记 —— 查 $OUT/autoshutdown.log"
        break
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        log "[!!] 超过 ${MAX_HOURS}h 上限，总控退出（守护仍在跑，会自己收工）"
        break
    fi
    sleep 120
done

log "总控结束"
