#!/usr/bin/env bash
# ==============================================================================
# 第 5 阶段「跑完自动关机」守护脚本
#
# 目的：用户最怕「实例空烧钱」。本脚本保证：
#   1) 正常路径 —— 主实验矩阵全部完成 → 云端提取 → 打包 → 推进 relay
#                  → **回读自检通过** → 立即关机（停止计费）
#   2) 异常路径 —— 主控崩了 / 超时 / 评估期卡住 ⇒ 同样先打包再关机
#   ⇒ 无论发生什么，实例都不会空跑到天亮。
#
# ⚠️ 继承自第 4 阶段的四条硬教训（不要在重写时丢掉）：
#   ① **回读自检不过 ⇒ 不关机**（exit 9）。关机 ⇒ relay agent 死 ⇒ 再也取不到产物；
#      推送其实失败而脚本照样关机 ⇒ 10+ 小时算力成果永久丢失。保命 > 省电。
#   ② **卡死保护必须三条件同时满足**：GPU 空闲 **且** 无训练/评估进程 **且** 轮次未变。
#      只看 GPU 占用会误触发 —— 评估期 util 只有 2~7% ⇒ 会提前关机丢成果。
#   ③ **推送 idbase 必须递增**（铁律 I）：relay_push_out.py 从 idbase+1 顺序分配 id，
#      同 idbase 会覆盖上一轮、且新 manifest 的 id 更小 ⇒ 本地"取 id 最大的 manifest"
#      会取到**过期数据**（极隐蔽的静默失败）。
#      本轮用 **1010000**（第 4 阶段最后用的是 1000000）。
#   ④ **推送清单必须含 eval_*.log**（一级证据：逐 seed 成功率 + 参数量）。
#      第 4 阶段旧版硬编码 *.json/*.md/*.csv/*.png ⇒ 14 份评估日志全漏推。
#
# ⚠️ 关机方式（实测 2026-11-18）：
#    AutoDL 容器内 /usr/bin/shutdown 是官方放的小脚本（内容是 kill supervisord）
#    ⇒ 容器退出 ⇒ AutoDL 判定实例停止 ⇒ 停止计费（关机后约 5~26 秒生效）。
#    /usr/sbin/poweroff 走 systemctl，而容器里 systemd 没跑 ⇒ **不能用 poweroff**。
#
# 用法：
#   nohup setsid bash /root/fedcore/phase5_autoshutdown.sh \
#         > /root/HPT/output/phase5/autoshutdown.log 2>&1 < /dev/null &
#
# 可调环境变量：
#   OUT          输出根目录（默认 /root/HPT/output/phase5）
#   MAX_HOURS    绝对时限（小时），到点无论完成与否都先打包再关机（默认 24）
#   POLL         轮询间隔（秒，默认 60）
#   PUSH_IDBASE  relay 推送 id 基址（**每轮必须递增**，默认 1010000）
# ==============================================================================
set -u

OUT=${OUT:-/root/HPT/output/phase5}
CORE=${CORE:-/root/fedcore}
PY=${PY:-/root/miniconda3/envs/hpt/bin/python}
SHUTDOWN_BIN=${SHUTDOWN_BIN:-/usr/bin/shutdown}
MAX_HOURS=${MAX_HOURS:-24}
POLL=${POLL:-60}
FINAL_SLEEP=${FINAL_SLEEP:-20}
PUSH_IDBASE=${PUSH_IDBASE:-1010000}

# 主实验矩阵的 7 个配置
CFG_TAGS="p5_bal_fedprox p5_bal_scaffold p5_iso_fedavg p5_iso_fedprox p5_iso_scaffold p5_iso_eda p5_iso_edaclu"

# 🔴 worker 存活判据：必须能识破 `pgrep -f` 自匹配（铁律 K：relay 用 /bin/sh -c
#    执行整段命令 ⇒ pgrep -f "run_phase5.sh" 恒返回 2 ⇒ 探测永远假活）。
#    正解：字符类 [r]un_phase5。
WORKER_PAT='[r]un_phase5\.sh|[r]un_fedavg|eval_model\.sh|hpt\.run|hpt\.run_eval|aggregate\.py|fed_baselines\.py'

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# rc_is_ok：只有内容为 0（成功）的 .rc 才算完成。
# ⚠️ 2026-11-23 修复：原实现只判文件是否存在 ⇒ 失败的配置写 .rc=1 也被计入
#    ⇒ 7 个 .rc 齐了就判"全部实验完成" ⇒ 推送一份混着残缺结果的主表。
rc_is_ok() {
    [ -f "$OUT/.rc_$1" ] || return 1
    [ "$(cat "$OUT/.rc_$1" 2>/dev/null | tr -d '[:space:]')" = "0" ]
}

count_rc() {
    local n=0 t
    for t in $1; do
        rc_is_ok "$t" && n=$((n + 1))
    done
    echo "$n"
}

# 是否有训练/评估进程活着（排除守护脚本自身与本次探测命令）
worker_alive() {
    ps -eo pid,cmd 2>/dev/null \
      | grep -E "$WORKER_PAT" \
      | grep -v -E 'grep -E|phase5_autoshutdown\.sh' \
      | grep -q .
}

gpu_busy() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | grep -qE '[0-9]+' && \
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | awk '{ if ($1 > 15) exit 0; else exit 1 }'
}

start_ts=$(date +%s)
idle_hits=0
REASON="未知"
log "===== 第5阶段自动关机守护启动 ====="
EXPECT=$(echo $CFG_TAGS | wc -w)
log "等待条件：主实验矩阵 ${CFG_TAGS}（${EXPECT} 个配置）"
log "MAX_HOURS=$MAX_HOURS  POLL=${POLL}s  PUSH_IDBASE=$PUSH_IDBASE"

while : ; do
    n=$(count_rc "$CFG_TAGS")

    # --- 正常完成 ---
    if [ "$n" -eq "$EXPECT" ]; then
        REASON="全部实验完成（主实验矩阵 $n/$EXPECT）"
        break
    fi

    # --- 主控进程死了但 .rc 不齐 ⇒ 提前收工（先打包再关机，不丢已有成果）---
    if ! worker_alive; then
        # 给一点点宽限：主控可能正在配置阶段
        idle_hits=$((idle_hits + 1))
        if [ "$idle_hits" -ge 3 ]; then
            REASON="无训练/评估进程存活且 .rc 不齐（$n/$EXPECT）⇒ 提前收工"
            break
        fi
    else
        idle_hits=0
    fi

    # --- 卡死保护：三条件同时满足（GPU 空闲 + 无进程 + 轮次未变）---
    # ⚠️ 评估期 GPU util 只有 2~7%，所以"无进程"这一条是关键 ——
    #    只要 eval_model.sh / hpt.run_eval 还在跑，就绝不能关机。
    if ! worker_alive && ! gpu_busy; then
        idle_hits=$((idle_hits + 1))
        if [ "$idle_hits" -ge 5 ]; then
            REASON="GPU 空闲 + 无训练/评估进程 ⇒ 判定卡死，提前收工"
            break
        fi
    fi

    # --- 绝对时限 ---
    now=$(date +%s)
    if [ $(( (now - start_ts) / 3600 )) -ge "$MAX_HOURS" ]; then
        REASON="超过绝对时限 ${MAX_HOURS} 小时（.rc $n/$EXPECT）"
        break
    fi

    sleep "$POLL"
done

log "收工原因：$REASON"
log "主实验矩阵 .rc = $(count_rc "$CFG_TAGS")/$EXPECT"

# ------------------------------------------------------------------------------
# 提取结果（失败不阻断关机，但要在日志里留痕）
# ------------------------------------------------------------------------------
log "--- 云端提取第5阶段结果 ---"
"$PY" "$CORE/extract_phase5_results.py" --out-root "$OUT" \
    --out "$OUT/phase5_raw.json" >> "$OUT/extract_phase5.log" 2>&1
log "extract_phase5_results rc=$?"

# ------------------------------------------------------------------------------
# 打包产物（关机不删盘，但先打好包，重开直接取）
# ------------------------------------------------------------------------------
log "--- 打包产物 ---"
cd "$OUT" || exit 0
FILES=$(ls -1 *.json *.md *.csv *.png *.log 2>/dev/null | tr '\n' ' ')
# shellcheck disable=SC2086
tar czf "$CORE/phase5_out.tgz" $FILES 2>/dev/null
ls -la "$CORE"/phase5_out.tgz 2>/dev/null
log "phase5_out.tgz = $(du -h "$CORE/phase5_out.tgz" 2>/dev/null | cut -f1)"

# ------------------------------------------------------------------------------
# 推进 relay 结果存储（🔴 最关键的一步）
#
# 为什么必须有：关机 ⇒ 容器退出 ⇒ **relay agent 一起死** ⇒
#   "下发命令→读 stdout"那条拉文件通道随之失效。
#   但 **relay 服务端跑在 WorkBuddy 托管的云主机上，不随容器死**。
#   ⇒ 只要在关机前把产物写进服务端的 /result 存储，
#     产物就留在云端，本地随时能取（tools/relay_fetch_out.py），
#     **不需要重开实例、不需要人工拉文件**。
# ------------------------------------------------------------------------------
log "--- 推进 relay 结果存储 ---"
# ⚠️ 清单必须包含 eval_*.log（一级证据：逐 seed 成功率 + 参数量）。
#    第 4 阶段旧版硬编码 *.json/*.md/*.csv/*.png ⇒ 14 份评估日志全漏推。
PUSH_FILES=$(cd "$OUT" && ls -1 *.json *.md *.csv *.png \
             eval_*.log autoshutdown.log extract_phase5.log 2>/dev/null \
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
# 🔴 回读自检（关机前的最后一道闸）
#
# 为什么必须做：关机 ⇒ relay agent 一起死 ⇒ 再也无法从云端取产物。
#   若推送其实失败（网关 403 / 磁盘满 / token 失效），而脚本照样关机，
#   ⇒ 本轮 10+ 小时的算力成果**永久丢失**。
#   ⇒ 推送失败时**不关机**：保留实例、保留磁盘产物、把失败原因写进标记文件。
# ------------------------------------------------------------------------------
log "--- 回读自检：确认产物真的进了 relay 结果存储 ---"
VERIFY_RC=1
if [ "$push_rc" -eq 0 ]; then
    PUSH_IDBASE="$PUSH_IDBASE" "$PY" - <<'PYEOF' >> "$OUT/relay_verify.log" 2>&1
import json, os, re, sys, urllib.request

IDBASE = int(os.environ.get("PUSH_IDBASE", "1010000"))
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
    echo "cfg_rc=$(count_rc "$CFG_TAGS")/$EXPECT"
    echo "relay_push_rc=$push_rc"
    echo "relay_push_files=$push_n"
    echo "relay_push_idbase=$PUSH_IDBASE"
    echo "relay_verify_rc=$VERIFY_RC"
    echo "relay_fetch_cmd=python tools/relay_fetch_out.py --since $PUSH_IDBASE --outdir results/cloud_final_phase5"
} > "$OUT/PHASE5_ALL_DONE"

# 🔴 闸门：产物没确认进 relay ⇒ **不关机**（保命 > 省电）
if [ "$VERIFY_RC" -ne 0 ]; then
    log "✗ 回读自检失败 rc=$VERIFY_RC —— 产物未能确认进入 relay 存储"
    log "  ⇒ **不关机**：保留实例与磁盘产物，避免成果永久丢失"
    log "  ⇒ 请查看 $OUT/relay_verify.log 与 $OUT/relay_push.log"
    log "  ⇒ 手动重推：$PY $CORE/relay_push_out.py --outdir $OUT --files '<文件列表>' --idbase $PUSH_IDBASE"
    exit 9
fi

log "✓ 产物已打包（/root/fedcore/phase5_out.tgz）+ 已推进 relay + 回读自检通过 + 完成标记已写"
log "${FINAL_SLEEP} 秒后自动关机（AutoDL 将停止计费）"

sleep "$FINAL_SLEEP"
log "执行 $SHUTDOWN_BIN → kill supervisord → 容器退出 → AutoDL 停止计费"
"$SHUTDOWN_BIN"
log "shutdown 已调用（若 30 秒后本进程仍存在，属正常：日志尚未 flush）"
