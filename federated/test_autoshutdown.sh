#!/usr/bin/env bash
# 离线单测：phase4_autoshutdown.sh 的收工分支
# 无需 GPU、无需云端。用假 pgrep / 假 nvidia-smi / 假 shutdown 驱动。
# ⚠️ 不用 mktemp + rm -rf（沙箱会拦），改用固定目录，测试完保留以便排查。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="$SCRIPT_DIR/phase4_autoshutdown.sh"
TMP="${AUTOSHUT_TEST_DIR:-/tmp/fedvla_autoshutdown_test}"

PASS=0
FAIL=0

mkdir -p "$TMP/bin"
cat > "$TMP/bin/fake_pgrep" <<'EOF'
#!/usr/bin/env bash
pat="$2"
case "$pat" in
  *chain_phase4_dp*)  [ -n "${FAKE_CHAIN:-}" ] && { echo 111; exit 0; }; exit 1 ;;
  *run_comm_scan_v2*) [ -n "${FAKE_COMM:-}"  ] && { echo 222; exit 0; }; exit 1 ;;
esac
exit 1
EOF
cat > "$TMP/bin/fake_gpu" <<'EOF'
#!/usr/bin/env bash
echo "${FAKE_UTIL:-100}"
EOF
chmod +x "$TMP/bin/fake_pgrep" "$TMP/bin/fake_gpu"

COMM_TAGS="c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1"
DP_TAGS="dp_s0001 dp_s0003 dp_s0012"

# $1=用例名 $2=期望关键词 $3=prep 模式: all|comm|none
run_case() {
    local name="$1" expect="$2" prep="$3"
    local out="$TMP/$name" core="$TMP/${name}_core" log="$TMP/$name.log"
    mkdir -p "$out" "$core"
    case "$prep" in
        all)  for t in $COMM_TAGS $DP_TAGS; do touch "$out/.rc_$t"; done ;;
        comm) for t in $COMM_TAGS; do touch "$out/.rc_$t"; done ;;
        none) : ;;
    esac
    ( OUT="$out" CORE="$core" PY=/bin/true SHUTDOWN_BIN=/bin/echo \
      PGREP_BIN="$TMP/bin/fake_pgrep" GPU_UTIL_CMD="$TMP/bin/fake_gpu" \
      FINAL_SLEEP=0 POLL=1 IDLE_MIN=0 MAX_HOURS=10 WAIT_DP=1 \
      bash "$SUT" ) > "$log" 2>&1
    local line; line=$(grep -o '收工原因：.*' "$log" | head -1)
    if echo "$line" | grep -q "$expect"; then
        echo "  [PASS] $name → ${line}"; PASS=$((PASS+1))
    else
        echo "  [FAIL] $name → 期望「$expect」实际：${line:-<无>}"
        tail -8 "$log" | sed 's/^/         /'; FAIL=$((FAIL+1))
    fi
}

echo "=== phase4_autoshutdown.sh 离线单测 ==="

export FAKE_CHAIN="" FAKE_COMM="yes" FAKE_UTIL="100"

# 用例 1：通信 7/7 + DP 3/3 ⇒ 正常完成
run_case case1 "全部实验完成" all

# 用例 2：通信 7/7，DP 0/3，DP 接力链已死 ⇒ 提前收工
export FAKE_CHAIN=""
run_case case2 "DP 接力链已退出" comm

# 用例 3：GPU 空闲 + 轮次不变 ⇒ 判定卡死
export FAKE_CHAIN="yes" FAKE_UTIL="0"
run_case case3 "判定卡死" none

# 用例 4：达到绝对时限 ⇒ 强制收工（用 case1 目录复用，只改 MAX_HOURS）
export FAKE_UTIL="100"
out="$TMP/case4"; core="$TMP/case4_core"; log="$TMP/case4.log"
mkdir -p "$out" "$core"; touch "$out/.rc_c_drop00"
( OUT="$out" CORE="$core" PY=/bin/true SHUTDOWN_BIN=/bin/echo \
  PGREP_BIN="$TMP/bin/fake_pgrep" GPU_UTIL_CMD="$TMP/bin/fake_gpu" \
  FINAL_SLEEP=0 POLL=1 IDLE_MIN=30 MAX_HOURS=0 WAIT_DP=1 \
  bash "$SUT" ) > "$log" 2>&1
line=$(grep -o '收工原因：.*' "$log" | head -1)
if echo "$line" | grep -q "达到绝对时限"; then
    echo "  [PASS] case4 → ${line}"; PASS=$((PASS+1))
else
    echo "  [FAIL] case4 → 期望「达到绝对时限」实际：${line:-<无>}"
    tail -8 "$log" | sed 's/^/         /'; FAIL=$((FAIL+1))
fi

# 用例 5：WAIT_DP=0 ⇒ 只要通信 7/7 就收工（可省 DP 的 3.3 小时）
out="$TMP/case5"; core="$TMP/case5_core"; log="$TMP/case5.log"
mkdir -p "$out" "$core"
for t in $COMM_TAGS; do touch "$out/.rc_$t"; done
( OUT="$out" CORE="$core" PY=/bin/true SHUTDOWN_BIN=/bin/echo \
  PGREP_BIN="$TMP/bin/fake_pgrep" GPU_UTIL_CMD="$TMP/bin/fake_gpu" \
  FINAL_SLEEP=0 POLL=1 IDLE_MIN=30 MAX_HOURS=10 WAIT_DP=0 \
  bash "$SUT" ) > "$log" 2>&1
line=$(grep -o '收工原因：.*' "$log" | head -1)
if echo "$line" | grep -q "全部实验完成"; then
    echo "  [PASS] case5 (WAIT_DP=0) → ${line}"; PASS=$((PASS+1))
else
    echo "  [FAIL] case5 (WAIT_DP=0) → 实际：${line:-<无>}"
    tail -8 "$log" | sed 's/^/         /'; FAIL=$((FAIL+1))
fi

# 用例 6：收工后必须留下完成标记
if [ -f "$TMP/case1/PHASE4_ALL_DONE" ] && grep -q "reason=" "$TMP/case1/PHASE4_ALL_DONE"; then
    echo "  [PASS] 完成标记已写入 → $(tr '\n' ' ' < "$TMP/case1/PHASE4_ALL_DONE")"
    PASS=$((PASS+1))
else
    echo "  [FAIL] 未写入 PHASE4_ALL_DONE"; FAIL=$((FAIL+1))
fi

# 用例 7：收工后确实调用了 shutdown（用 /bin/echo 代替，真机上是 kill supervisord）
if grep -q "shutdown 已调用" "$TMP/case1.log"; then
    echo "  [PASS] 已调用 shutdown"; PASS=$((PASS+1))
else
    echo "  [FAIL] 未调用 shutdown"; FAIL=$((FAIL+1))
fi

# 用例 8：🔴 收工前必须推进 relay 结果存储（关机后 agent 死，这是唯一取回通道）
if grep -q "relay_push rc=" "$TMP/case1.log"; then
    echo "  [PASS] 已执行 relay 推送步骤 → $(grep -o 'relay_push rc=.*' "$TMP/case1.log" | head -1)"
    PASS=$((PASS+1))
else
    echo "  [FAIL] 日志里没有 relay_push 步骤"; FAIL=$((FAIL+1))
fi

# 用例 9：完成标记必须写明 relay 推送结果 + 取回命令
if [ -f "$TMP/case1/PHASE4_ALL_DONE" ] \
   && grep -q "relay_push_rc=" "$TMP/case1/PHASE4_ALL_DONE" \
   && grep -q "relay_fetch_cmd=" "$TMP/case1/PHASE4_ALL_DONE"; then
    echo "  [PASS] 完成标记含 relay 推送结果与取回命令"
    PASS=$((PASS+1))
else
    echo "  [FAIL] 完成标记缺 relay_push_rc / relay_fetch_cmd"; FAIL=$((FAIL+1))
fi

echo "=== 结果：PASS=$PASS  FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ]
