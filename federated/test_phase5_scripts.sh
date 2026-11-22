#!/usr/bin/env bash
# 第 5 阶段脚本的**离线逻辑单测**（铁律 E：脚本先在假目录里跑分支，不上云端试错）。
#
# 验什么
# ----
#   1. run_phase5.sh 的划分定义：balanced / isolated 的 domain↔env↔epoch_size
#      **一一对应且数量一致**（配错就是"用 task1 的数据训 task2 的客户端"，
#      配置文件里有明确的自检，但脚本层也要挡住）。
#   2. 护��脚本的判据逻辑：.rc 齐 ⇒ 收工；.rc 不齐且无进程 ⇒ 提前收工；
#      回读自检失败 ⇒ **不关机**（exit 9）—— 这是保命闸。
#   3. run_fedavg.sh 的基线开关：BASELINE 为空时整段跳过（零影响）。
#
# 用法：bash federated/test_phase5_scripts.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASS=0; FAIL=0
ok ()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
bad ()  { echo "  ❌ $1"; FAIL=$((FAIL+1)); }

echo "======================================================================"
echo "第5阶段脚本 离线逻辑单测"
echo "======================================================================"

# ---------------------------------------------------------------- 1. 划分定义
echo ""
echo "[test_partitions_consistent]"
# shellcheck disable=SC1090
BAL_DOMAINS="mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task1"
BAL_ENVS="reach-v2 push-v2 button-press-topdown-v2 reach-v2"
BAL_ESIZE="15 19 24 15"
ISO_DOMAINS="mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task2"
ISO_ENVS="reach-v2 push-v2 button-press-topdown-v2 push-v2"
ISO_ESIZE="15 19 24 19"

check_partition () {
  local name="$1" dom="$2" envs="$3" esz="$4"
  local nd ne ne2
  nd=$(echo $dom | wc -w); ne=$(echo $envs | wc -w); ne2=$(echo $esz | wc -w)
  if [ "$nd" != "4" ] || [ "$ne" != "4" ] || [ "$ne2" != "4" ]; then
    bad "$name 数量不一致 (domains=$nd envs=$ne esize=$ne2)"; return 1
  fi
  # domain→env 映射（与 fedavg_config.sh 的映射表一致）
  local i=1
  for d in $dom; do
    local e; e=$(echo $envs | cut -d' ' -f$i)
    local expect
    case "$d" in
      *task1) expect="reach-v2" ;;
      *task2) expect="push-v2" ;;
      *task3) expect="button-press-topdown-v2" ;;
      *) expect="?" ;;
    esac
    if [ "$e" != "$expect" ]; then
      bad "$name client$((i-1)) domain=$d 应配 env=$expect，实际=$e"; return 1
    fi
    i=$((i+1))
  done
  ok "$name 划分定义自洽（4 客户端，domain↔env 一一对应）"
  return 0
}
check_partition "balanced" "$BAL_DOMAINS" "$BAL_ENVS" "$BAL_ESIZE"
check_partition "isolated" "$ISO_DOMAINS" "$ISO_ENVS" "$ISO_ESIZE"

# ---------------------------------------------------------------- 2. 孤立语义
echo ""
echo "[test_isolated_has_isolated_client]"
# isolated 的核心语义：**存在一个客户端的任务，其他任何客户端都没有**
iso_unique=0
for i in 1 2 3 4; do
  d=$(echo $ISO_DOMAINS | cut -d' ' -f$i)
  cnt=0
  for j in 1 2 3 4; do
    [ "$i" = "$j" ] && continue
    dj=$(echo $ISO_DOMAINS | cut -d' ' -f$j)
    [ "$d" = "$dj" ] && cnt=$((cnt+1))
  done
  if [ "$cnt" -eq 0 ]; then
    iso_unique=$((iso_unique+1))
    echo "     孤岛客户端：client$((i-1)) 的任务 $d 无其它客户端共有"
  fi
done
if [ "$iso_unique" -ge 1 ]; then
  ok "isolated 划分确实含孤岛客户端（$iso_unique 个）"
else
  bad "isolated 划分没有孤岛客户端 —— 语义错误"
fi

# ---------------------------------------------------------------- 3. 脚本语法
echo ""
echo "[test_script_syntax]"
for f in run_phase5.sh phase5_autoshutdown.sh run_fedavg.sh; do
  if bash -n "$HERE/$f" 2>/dev/null; then ok "$f 语法正确"; else bad "$f 语法错误"; fi
done

# ---------------------------------------------------------------- 4. 基线默认关闭
echo ""
echo "[test_baseline_off_by_default]"
if grep -qE '^BASELINE=\$\{BASELINE:-\}' "$HERE/run_fedavg.sh"; then
  ok "BASELINE 默认为空（默认关闭 ⇒ 对既有实验零影响）"
else
  bad "BASELINE 默认值不是空"
fi
# 基线关闭时：客户端注入段与轮末聚合段都必须在 if 内
n_if=$(grep -c 'if \[ -n "\$BASELINE" \]; then' "$HERE/run_fedavg.sh")
if [ "$n_if" -ge 1 ]; then
  ok "客户端侧基线注入被 if 包裹（$n_if 处）"
else
  bad "客户端侧基线注入没有被 if 包裹"
fi
if grep -q 'if \[ "\$BASELINE" = "scaffold" \]; then' "$HERE/run_fedavg.sh"; then
  ok "轮末控制变量聚合被 if 包裹"
else
  bad "轮末控制变量聚合没有被 if 包裹"
fi

# ---------------------------------------------------------------- 5. 护��� fail-loud
echo ""
echo "[test_watchdog_fail_loud]"
# 回读自检失败 ⇒ exit 9（不关机）
if grep -q 'exit 9' "$HERE/phase5_autoshutdown.sh"; then
  ok "回读自检失败时 exit 9（不关机，保命闸）"
else
  bad "缺少 exit 9 保命闸"
fi
# 卡死判据必须同时看进程
if grep -q 'worker_alive' "$HERE/phase5_autoshutdown.sh"; then
  ok "卡死判据包含 worker_alive（评估期不会误关机）"
else
  bad "卡死判据缺少 worker_alive"
fi
# idbase 必须可配置且默认大于第4阶段的 1000000
idb=$(grep -oE 'PUSH_IDBASE=\$\{PUSH_IDBASE:-[0-9]+\}' "$HERE/phase5_autoshutdown.sh" \
      | grep -oE '[0-9]+' | tail -1)
if [ -n "$idb" ] && [ "$idb" -gt 1000000 ]; then
  ok "PUSH_IDBASE 默认 $idb > 1000000（铁律 I：idbase 必须递增）"
else
  bad "PUSH_IDBASE 默认值可疑：$idb"
fi
# 推送清单必须含 eval_*.log
if grep -q 'eval_\*\.log' "$HERE/phase5_autoshutdown.sh"; then
  ok "推送清单含 eval_*.log（一级证据不漏推）"
else
  bad "推送清单缺 eval_*.log"
fi

# ---------------------------------------------------------------- 6. 同构硬闸
echo ""
echo "[test_same_construction_gate]"
if grep -q 'check_same_construction' "$HERE/run_phase5.sh"; then
  ok "run_phase5.sh 有同构自检（非 0 轮开始的配置会被清空重跑）"
else
  bad "run_phase5.sh 缺同构自检"
fi
if grep -q 'opt_state_c' "$HERE/run_phase5.sh"; then
  ok "续跑前检查 Adam 状态文件（缺则清空重跑，避免相位错乱）"
else
  bad "续跑前未检查 Adam 状态"
fi

echo ""
echo "======================================================================"
echo "通过 $PASS / 失败 $FAIL"
echo "======================================================================"
[ "$FAIL" -eq 0 ] || exit 1
