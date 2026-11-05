#!/usr/bin/env bash
# =============================================================================
# FedVLA 云端一键恢复脚本（实例重启后跑这个）
#
# 背景（2026-11-05）
# ----
# 云端 AutoDL 实例被关机（agent 心跳落后 13.5h），重启后需要：
#   1. 确认磁盘没满（满 → agent 心跳停摆，是历史头号故障源）
#   2. 确认 HPT 源码在、补丁能打上
#   3. 继续门2'（E1~E5 消融）+ 第 3 阶段（MoE/EDA）
#
# 本脚本是**幂等**的：每个阶段的产物存在就跳过，可以反复执行。
#
# 用法：
#   bash /root/HPT/federated/resume_all.sh            # 跑全部
#   STAGES="gate2b" bash /root/HPT/federated/resume_all.sh
#   STAGES="phase3" bash /root/HPT/federated/resume_all.sh
#   CHECK_ONLY=1 bash /root/HPT/federated/resume_all.sh   # 只体检不跑
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPT_PY=/root/miniconda3/envs/hpt/bin/python
OUT=/root/HPT/output
LOGDIR="$OUT/resume_logs"
STAGES=${STAGES:-"gate2b phase3"}
CHECK_ONLY=${CHECK_ONLY:-0}

mkdir -p "$LOGDIR"

export WANDB_MODE=offline
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

banner () {
  echo
  echo "=============================================================="
  echo ">>> $*"
  echo "=============================================================="
}

# ── 体检 ────────────────────────────────────────────────────────────────────
doctor () {
  banner "体检"
  echo "[1] 磁盘"
  df -h / /root 2>/dev/null | sed 's/^/    /'
  FREE_GB=$(df -BG /root 2>/dev/null | awk 'NR==2{print $4}' | tr -dc '0-9')
  echo "    可用: ${FREE_GB:-?} GB"
  if [ "${FREE_GB:-0}" -lt 6 ]; then
    echo "    !! 可用空间不足 6GB —— 先清理历史轮次权重（日志全部保留）"
    if [ "$CHECK_ONLY" = "1" ]; then
      echo "    [体检] 空间不足，检查模式到此为止"
      return 1
    fi
    # 清理联邦训练产出的历史轮次权重（global_r* / client_*_r*），
    # 这些是可重算的中间产物；评估日志、结果表一律保留。
    # 策略：每类只保留最新 2 轮（与 run_fedavg.sh 的 KEEP_ROUNDS=2 一致）。
    for d in "$OUT"/fedavg_hetero "$OUT"/fedavg_iid; do
      [ -d "$d" ] || continue
      ls -dt "$d"/global_r* 2>/dev/null | tail -n +3 | while read -r x; do rm -rf "$x"; done
      for c in 0 1 2 3; do
        ls -dt "$d"/client_${c}_r* 2>/dev/null | tail -n +3 | while read -r x; do rm -rf "$x"; done
      done
    done
    FREE_GB=$(df -BG /root 2>/dev/null | awk 'NR==2{print $4}' | tr -dc '0-9')
    echo "    清理后可用: ${FREE_GB} GB"
  fi

  echo "[2] HPT 环境"
  if [ -x "$HPT_PY" ]; then
    echo "    python: OK"
    "$HPT_PY" -c "import torch;print('    torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | sed 's/^/    /'
    "$HPT_PY" -c "import mujoco_py;print('    mujoco_py OK')" 2>&1 | tail -2 | sed 's/^/    /'
  else
    echo "    !! $HPT_PY 不存在 —— 环境没了，需要重装"
    return 1
  fi

  echo "[3] HPT 源码与补丁"
  for f in hpt/run.py hpt/policy.py hpt/transformer.py; do
    if [ -f "/root/HPT/$f" ]; then echo "    OK  $f"; else echo "    !! 缺 $f"; fi
  done
  for p in patch_hpt_load_trunk.py patch_hpt_global_lr.py patch_hpt_moe.py; do
    if [ -f "$SCRIPT_DIR/$p" ]; then echo "    OK  $p"; else echo "    !! 缺 $p（需重新推送）"; fi
  done

  echo "[4] 已有产物"
  for d in "$OUT"/fedavg_hetero "$OUT"/fedavg_iid "$OUT"/centralized_equal_budget "$OUT"/g2b "$OUT"/phase3; do
    if [ -d "$d" ]; then
      n=$(ls -1 "$d" 2>/dev/null | wc -l)
      echo "    OK  $d ($n 项)"
    else
      echo "    --  $d (未建)"
    fi
  done

  echo "[5] GPU"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>&1 | sed 's/^/    /'
  echo "[6] 跑着的进程"
  ps aux | grep -E "[hpt\.run|[c]ollect_data|[r]un_gate2b|[r]un_phase3" | awk '{print "    " $2, $11, $12, $13}' | head -8
  echo "    (空 = 无实验在跑)"
}

has_stage () { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# ── 拉起评估守护 ────────────────────────────────────────────────────────────
# 背景（2026-11-05 第三次实例重启）：评估守护（gate2b_eval_watch /
# phase3_eval_watch）是**独立进程**，实例关机时会被一起杀掉，而
# resume_all.sh 原本只管训练 → 重启后训练续跑了，评估却永远不会触发，
# 训练跑完也只有 loss 曲线、没有成功率。
# 这里在实验启动**之前**把它们拉起来，且做幂等检查（已在跑就不重复起）。
start_eval_watchers () {
  banner "拉起评估守护"
  for w in gate2b_eval_watch phase3_eval_watch; do
    script="$SCRIPT_DIR/$w.sh"
    [ -f "$script" ] || { echo "    -- $w.sh 不存在，跳过（需重新推送）"; continue; }
    if pgrep -f "$w.sh" > /dev/null 2>&1; then
      echo "    [OK] $w 已在运行 (pid $(pgrep -f "$w.sh" | head -1))"
      continue
    fi
    nohup setsid bash "$script" > "$OUT/${w}.log" 2>&1 < /dev/null &
    sleep 1
    echo "    [OK] $w 已启动 (pid $(pgrep -f "$w.sh" | head -1))"
  done
}

# ── 等评估完成 ──────────────────────────────────────────────────────────────
# 背景：评估由守护进程**异步**执行（各等自己的 DONE 标记，然后独占 GPU 评估）。
# 如果训练一结束就直接提取，成功率列会全是空 —— 这正是 2026-11-05 差点
# 让 8 小时训练白跑的那个坑。这里在提取前显式等两个评估守护收工。
wait_evals () {
  banner "等待评估完成"
  local timeout=${EVAL_WAIT:-21600}     # 默认最多等 6 小时
  local deadline=$(( $(date +%s) + timeout ))
  while :; do
    local pending=""
    for w_name in gate2b_eval_watch phase3_eval_watch; do
      case "$w_name" in
        gate2b_eval_watch) marker=GATE2B_EVAL_WATCH_DONE; logf=gate2b_eval_watch.log ;;
        phase3_eval_watch) marker=PHASE3_EVAL_WATCH_DONE; logf=phase3_eval_watch.log ;;
      esac
      if grep -q "$marker" "$OUT/$logf" 2>/dev/null; then
        continue                                   # 已完成
      fi
      if pgrep -f "$w_name.sh" > /dev/null 2>&1; then
        alive=1                                   # 还在跑，等它
        pending="$pending $w_name"
      else
        echo "    [WARN] $w_name 未运行且无完成标记 —— 跳过等待"
      fi
    done
    if [ -z "$pending" ]; then
      echo "    [OK] 评估全部完成（或无可等待项）"
      return 0
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      echo "    !! 等待评估超时，仍缺:$pending"
      echo "    !! 先按现有产物提取（可能缺成功率列），稍后可重跑提取脚本"
      return 1
    fi
    echo "    等待中:$pending  ($(date '+%H:%M:%S'))"
    sleep 120
  done
}

# ── 门2' ───────────────────────────────────────────────────────────────────
run_gate2b () {
  banner "门2'：联邦管线损耗归因消融（V1 V2 E3 E4 E5）"
  # run_gate2b.sh 自身是幂等的（每个阶段产物存在即跳过），这里不重复判断
  bash "$SCRIPT_DIR/run_gate2b.sh" 2>&1 | tee -a "$LOGDIR/gate2b.log"
  local rc=${PIPESTATUS[0]}
  echo "[gate2b] rc=$rc"
  return $rc
}

# ── 第 3 阶段 ───────────────────────────────────────────────────────────────
run_phase3 () {
  banner "第 3 阶段：MoE + EDA"
  # run_phase3.sh 自身幂等（产物存在即跳过）
  bash "$SCRIPT_DIR/run_phase3.sh" 2>&1 | tee -a "$LOGDIR/phase3.log"
  local rc=${PIPESTATUS[0]}
  echo "[phase3] rc=$rc"
  return $rc
}

# ── 结果提取 ────────────────────────────────────────────────────────────────
run_extract () {
  banner "提取结果"
  "$HPT_PY" "$SCRIPT_DIR/extract_phase3_results.py" 2>&1 | tee -a "$LOGDIR/extract.log"
  echo "[extract] rc=${PIPESTATUS[0]}"
}

# ── 主流程 ──────────────────────────────────────────────────────────────────
echo "FedVLA 恢复脚本启动于 $(date)"
doctor

if [ "$CHECK_ONLY" = "1" ]; then
  echo; echo "== CHECK_ONLY 模式：只体检，不跑实验 =="
  exit 0
fi

RC=0
# 守护必须先起：它们等的是各自的 DONE 标记，与训练并行不抢 GPU；
# 但若训练跑完才起，中间已经完成的阶段会被跳过 —— 所以越早起越安全。
start_eval_watchers
has_stage gate2b && { run_gate2b || RC=1; }
has_stage phase3 && { run_phase3 || RC=1; }

banner "结果提取"
# 先等评估守护收工，否则成功率列会是空的
wait_evals || true
run_extract

banner "汇总"
echo "结束于 $(date)  rc=$RC"
ls -la "$OUT" 2>/dev/null | head -20
exit $RC
