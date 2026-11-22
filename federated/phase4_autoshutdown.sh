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

# 🔴 2026-11-19 新增：推送 id 基址可配置。
#   为什么必须能改：relay_push_out.py 从 idbase+1 开始**顺序分配** id，
#   所以用同一个 idbase 再推一次，会把上一轮的 id **整体覆盖**，
#   而且新 manifest 的 id 会**小于**上一轮 ⇒ 本地"取 id 最大的 manifest"
#   会取到**上一轮的旧 manifest** ⇒ 拉回过期数据（极隐蔽的静默失败）。
#   ⇒ 每一轮重跑都必须换一个**更大的** idbase（本轮用 990000）。
PUSH_IDBASE=${PUSH_IDBASE:-980000}

# 🔴 2026-11-18 修正：评估阶段会误触发「卡死保护」而**提前关机**。
#   旧判据：util < 10% 且 轮次文件 30 分钟未变 ⇒ 判定卡死。
#   问题：MuJoCo 评估（eval_model.sh / hpt.run_eval）**不产生新的 global_r***
#         且 GPU 负载极低（实测 util=3%/7%）⇒ 训练一结束进入评估，
#         两个条件同时成立 ⇒ 被误判卡死 ⇒ 丢掉 DP 阶段（3.3 小时）成果。
#   新判据：必须**同时**满足三条才判卡死——
#     ① GPU 空闲；② 没有任何训练/评估进程存活；③ 轮次文件未变。
#   其中 ② 是关键：只要还有 eval_model.sh / hpt.run / hpt.run_eval 在跑，
#   就说明系统正在干活，绝不能关机。
WORKER_PAT='run_comm_scan_v2|run_fedavg|run_dp_utility|chain_phase4_dp|calibrate_dp_clip|eval_model\.sh|hpt\.run|hpt\.run_eval|privacy_experiment'

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

count_rc() {
    local n=0 t
    for t in $1; do
        [ -f "$OUT/.rc_$t" ] && n=$((n + 1))
    done
    echo "$n"
}

# 是否有训练/评估进程活着（排除守护脚本自身与本次探测命令）
worker_alive() {
    # 用 ps 过滤，避免 pgrep 匹配到调用它的 shell 自身
    ps -eo pid,cmd 2>/dev/null \
      | grep -E "$WORKER_PAT" \
      | grep -v -E 'grep -E|phase4_autoshutdown\.sh' \
      | grep -q .
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
    # --- GPU 卡死保护（2026-11-18 加严）---
    #     必须同时满足：① GPU 空闲 ② 无任何训练/评估进程存活 ③ 轮次文件未变
    #     缺 ② 会误杀「评估阶段」（评估不产生 global_r* 且 GPU 负载极低）
    util=$($GPU_UTIL_CMD 2>/dev/null | head -1)
    util=${util:-100}
    sig=$(ls -1dt "$OUT"/*/global_r* 2>/dev/null | head -1)
    busy="否"
    if worker_alive; then busy="是"; fi
    if [ "$util" -lt 10 ] && [ "$sig" = "$last_sig" ] && [ "$busy" = "否" ]; then
        idle_hits=$((idle_hits + 1))
        if [ $((idle_hits * POLL)) -ge $((IDLE_MIN * 60)) ]; then
            REASON="GPU 空闲 ${IDLE_MIN} 分钟且无工作进程且轮次无变化（util=${util}%）→ 判定卡死"
            break
        fi
    else
        idle_hits=0
    fi
    last_sig="$sig"

    # --- 心跳（每 10 轮记一行，方便确认守护还活着）---
    poll_n=$((poll_n + 1))
    if [ $((poll_n % 10)) -eq 0 ]; then
        log "心跳：通信 .rc=$(count_rc "$COMM_TAGS")/$EXPECT_COMM  DP .rc=$(count_rc "$DP_TAGS")/$EXPECT_DP  util=${util}%  链=$([ -n "$chain_alive" ] && echo 活 || echo 死)  有任务=${busy}"
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
# 🔴 2026-11-22 修正：**必须包含 eval_*.log**（评估原文，含逐 seed 成功率与参数量）。
#    旧版只推 *.json/*.md/*.csv/*.png ⇒ 14 份评估日志全部漏推，
#    导致"DP 侧评估原文"缺失、三个消融变体结果无法审计（结论虽可从 relay 原始记录还原，
#    但审计链不完整）。⇒ 评估日志是**一级证据**，与 comm_report 同级。
PUSH_FILES=$(cd "$OUT" && ls -1 *.json *.md *.csv *.png \
             privacy/*.json privacy/*.md privacy/*.csv privacy/*.png \
             eval_*.log scanlog_*.log dplog_*.log \
             extract_comm.log extract_dp.log chain_phase4_dp.log \
             autoshutdown.log 2>/dev/null \
             | tr '\n' ' ')
"$PY" "$CORE/relay_push_out.py" --outdir "$OUT" --files "$PUSH_FILES" \
      --idbase "$PUSH_IDBASE" >> "$OUT/relay_push.log" 2>&1
push_rc=$?
if [ "$push_rc" -ne 0 ]; then
    log "relay 推送失败 rc=$push_rc，等 20 秒重试一次"
    sleep 20
    "$PY" "$CORE/relay_push_out.py" --outdir "$OUT" --files "$PUSH_FILES" \
          --idbase "$PUSH_IDBASE" >> "$OUT/relay_push.log" 2>&1
    push_rc=$?
fi
push_n=$(echo $PUSH_FILES | wc -w)
log "relay_push rc=$push_rc  文件数=$push_n"

# ------------------------------------------------------------------------------
# 🔴 2026-11-18 新增：推送后**回读自检**（关机前的最后一道闸）
#
# 为什么必须做：关机 ⇒ relay agent 一起死 ⇒ 再也无法从云端取产物。
#   若推送其实失败（网关 403 / 磁盘满 / token 失效），而脚本照样关机，
#   ⇒ 本轮 10+ 小时的算力成果**永久丢失**（AutoDL 关机不删盘，但要重开实例
#   手工捞文件，非常被动）。
#   ⇒ 推送失败时**不关机**：保留实例、保留磁盘产物、把失败原因写进标记文件，
#     让本地监控/automation 能看到并决定重推。
# ------------------------------------------------------------------------------
log "--- 回读自检：确认产物真的进了 relay 结果存储 ---"
VERIFY_RC=1
if [ "$push_rc" -eq 0 ]; then
    # 用 relay_push_out.py 里的凭据读法，直接查服务端最新 manifest 是否可读
    # ⚠️ heredoc 用 'PYEOF' 引号 ⇒ 不做 shell 展开 ⇒ idbase 通过环境变量传进去
    PUSH_IDBASE="$PUSH_IDBASE" "$PY" - <<'PYEOF' >> "$OUT/relay_verify.log" 2>&1
import json, os, re, sys, urllib.request

IDBASE = int(os.environ.get("PUSH_IDBASE", "980000"))
AGENT = "/root/HPT/agent.py"
txt = open(AGENT, encoding="utf-8", errors="ignore").read()
def grab(name, default=""):
    m = re.search(name + r'\s*=\s*"([^"]*)"', txt)
    return m.group(1) if m else default
tok, sc = grab("RELAY_TOKEN"), grab("RELAY_SHARECODE")
url = grab("RELAY_URL", "https://fedvla-relay.app.workbuddy.host").rstrip("/")
if not tok or not sc:
    print("[FATAL] 读不到 relay 凭据"); sys.exit(2)
req = urllib.request.Request(url + f"/results?since={IDBASE}&sharecode=" + sc,
                             headers={"X-Relay-Token": tok})
with urllib.request.urlopen(req, timeout=120) as r:
    data = json.loads(r.read().decode("utf-8"))
res = data.get("results") or []
mans = [x for x in res if (x.get("out") or "").startswith("MANIFEST:")]
print(f"取到 {len(res)} 条结果，其中 manifest {len(mans)} 条")
if not mans:
    print("[FATAL] 没找到 manifest —— 推送没生效"); sys.exit(3)
best = max(mans, key=lambda x: int(x["id"]))
mf = json.loads(best["out"][len("MANIFEST:"):])
files = mf.get("files") or []
print(f"manifest id={best['id']} 文件数={len(files)}")
if not files:
    print("[FATAL] manifest 里 0 个文件"); sys.exit(4)
# 逐文件核对：分块是否都真的在服务端、size 是否吻合
by_id = {int(x["id"]): (x.get("out") or "") for x in res}
missing, badsize = [], []
for f in files:
    if f.get("mode") == "text":
        ids = [int(f["id"])]
    else:
        ids = [int(i) for i in (f.get("ids") or [])]
    for i in ids:
        if i not in by_id:
            missing.append((f.get("name"), i)); break
    else:
        if int(f.get("size") or 0) > 0 and len(by_id.get(ids[0], "")) == 0:
            badsize.append(f.get("name"))
print(f"缺失分块文件数={len(missing)}  内容为空={len(badsize)}")
if missing:
    print("[FATAL] 缺失示例:", missing[:5]); sys.exit(5)
if badsize:
    print("[FATAL] 空内容示例:", badsize[:5]); sys.exit(6)
print("[OK] 所有文件分块均可在服务端取回 —— 产物安全")
sys.exit(0)
PYEOF
    VERIFY_RC=$?
fi
log "回读自检 rc=$VERIFY_RC"

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
    echo "relay_push_idbase=$PUSH_IDBASE"
    echo "relay_verify_rc=$VERIFY_RC"
    echo "relay_fetch_cmd=python tools/relay_fetch_out.py --since $PUSH_IDBASE --outdir results/cloud_final"
} > "$OUT/PHASE4_ALL_DONE"

# 🔴 闸门：产物没确认进 relay ⇒ **不关机**（保命 > 省电）
if [ "$VERIFY_RC" -ne 0 ]; then
    log "✗ 回读自检失败 rc=$VERIFY_RC —— 产物未能确认进入 relay 存储"
    log "  ⇒ **不关机**：保留实例与磁盘产物，避免成果永久丢失"
    log "  ⇒ 请查看 $OUT/relay_verify.log 与 $OUT/relay_push.log"
    log "  ⇒ 手动重推：$PY $CORE/relay_push_out.py --outdir $OUT --files '<文件列表>' --idbase 980000"
    exit 9
fi

log "✓ 产物已打包（/root/fedcore/final_out.tgz）+ 已推进 relay + 回读自检通过 + 完成标记已写"
log "60 秒后自动关机（AutoDL 将停止计费）"

# ------------------------------------------------------------------------------
# 关机
# ------------------------------------------------------------------------------
sleep "$FINAL_SLEEP"
log "执行 $SHUTDOWN_BIN → kill supervisord → 容器退出 → AutoDL 停止计费"
"$SHUTDOWN_BIN"
log "shutdown 已调用（若 30 秒后本进程仍存在，属正常：日志尚未 flush）"
