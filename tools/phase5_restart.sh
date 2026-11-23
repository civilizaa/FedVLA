#!/usr/bin/env bash
# ==============================================================================
# 第 5 阶段正式重跑编排（2026-11-23 修复后）
#
# 前置条件（硬闸，缺一不可）：
#   ① tools/phase5_smoke.sh 冒烟**通过**（钩子真的生效、无 skipped、ci_c 真落盘）
#   ② M1~M5 补丁已在云端生效（eval_model.sh 有 FEDVLA_MOE；train_test.py 有 optimizer 版钩子）
#   ③ 磁盘可用 ≥ 12G
#
# 本脚本做三件事：
#   1. 清理**已判定作废**的产物（不动还有效的）
#   2. 起 run_phase5.sh（后台）→ sleep 45s 验活（铁律 J）
#   3. 起自动关机守护（MAX_HOURS 可配）
#
# ⚠️ 作废判定（不可搞错）：
#   p5_bal_fedprox  —— **整目录作废**：此前 50 轮的近端项从未生效（钩子 NameError）
#                      ⇒ 那 50 轮 ≡ 纯 FedAvg ⇒ 必须清空从第 0 轮重跑（铁律 D）
#   p5_bal_scaffold —— 此前只跑到第 1 轮就聚合中止（目录已删）⇒ 从第 0 轮重跑
#   p5_iso_fedavg   —— **训练有效**（纯 FedAvg，不经钩子）⇒ 只删作废的评估日志，
#                      保留 global_r49，run_cfg 会走「补评估」分支
#   全部 eval_p5_*.log —— 旧的是 13.94M 残缺模型的评估 ⇒ 必须删（否则 eval_done 判已评估）
#   全部 .rc_*      —— 删掉让守护重新计数（且只认内容为 0）
# ==============================================================================
set -u
CORE=/root/fedcore
OUT=${OUT:-/root/HPT/output/phase5}
ROOT=/root/HPT/output
MAX_HOURS=${MAX_HOURS:-44}
POLL=${POLL:-60}
PUSH_IDBASE=${PUSH_IDBASE:-1010000}
NJOBS=${NJOBS:-1}          # 评估并行度（3 = 3 个 seed 并行）

echo "===== 第5阶段重跑编排 ====="
echo "OUT=$OUT  MAX_HOURS=$MAX_HOURS  PUSH_IDBASE=$PUSH_IDBASE  N_JOBS=$NJOBS"

# ---- 清理（**必须在磁盘闸之前**）----
# 🔴 2026-11-23 修正：闸门原本在清理之前 ⇒ 清理能释放的空间（2.4GB）没被计入
#    ⇒ 明明清理完就有 11.4GB，却因为清理前只有 9.0GB 而 FATAL 退出，永远起不来。
echo "===== 清理作废产物 ====="
rm -f  "$ROOT"/eval_p5_*.log                 # 13.94M 残缺评估（根目录；phase5/ 下是符号链接，保留）
rm -rf "$OUT/p5_bal_fedprox"                # 近端项从未生效 ⇒ 整目录作废
rm -f  "$OUT/logs/p5_bal_fedprox.log"
rm -f  "$OUT"/.rc_*                         # 让守护重新计数（且只认 0）
rm -rf "$OUT/smoke" "$OUT/smoke2"           # 冒烟产物
# p5_iso_fedavg：训练有效，但它的 client_*_r* 是聚合完就没人要的中间产物（~1.4GB）
#   ⚠️ opt_state_c*.pt **必须留** —— 它的评估还没做，后续要走「补评估」分支，
#      缺 opt_state 会被判成「Adam 相位错」⇒ 清空重跑 50 轮（铁律 D）。
if [ -d "$OUT/p5_iso_fedavg" ]; then
  rm -rf "$OUT/p5_iso_fedavg"/client_*_r* 2>/dev/null
fi
# 把第 3 阶段复用的 3 行评估日志**平铺复制**进 phase5/ —— 守护脚本只推 phase5/ 下的
#   `eval_*.log`，而第 3 阶段的日志在 output 根目录 ⇒ 不复制就推不上。
#   ⚠️ 必须**平铺**（不能放子目录）：守护的推送清单是 `ls -1 eval_*.log`，不递归。
#   ⚠️ 这 3 份原始日志**只存在于云端**（本地仓库没有）⇒ 推不上就等于证据永久丢失。
cp -f "$ROOT"/eval_p3_moe_fedavg_*.log "$ROOT"/eval_p3_moe_eda_*.log \
      "$ROOT"/eval_p3_moe_eda_cluster_*.log "$ROOT"/eval_p3_moe_iid_*.log \
      "$OUT/" 2>/dev/null
echo "  已平铺复制第3阶段评估日志到 phase5/（守护推送清单用 eval_*.log 匹配）"
echo "  清理完成；剩余 phase5 目录："; ls -1 "$OUT" 2>/dev/null | head -12

# ---- 硬闸 1：磁盘（**清理后**再判）----
AVAIL=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "[闸1] 清理后可用空间 ${AVAIL}GB（需 ≥ 8GB）"
# 阈值从 12G 降到 8G：加了「配置完成后回收中间产物」之后，
#   每个配置的稳态占用从 ~2.4GB 降到 ~0.35GB ⇒ 8GB 足够跑完 6 个新配置。
[ "${AVAIL:-0}" -lt 8 ] && { echo "[FATAL] 磁盘不足，禁止开跑"; exit 1; }

# ---- 硬闸 2：补丁到位 ----
echo "[闸2] 补丁锚点检查"
grep -q 'FEDVLA_MOE' "$CORE/eval_model.sh" || { echo "[FATAL] eval_model.sh 缺 FEDVLA_MOE（M1 未生效）"; exit 1; }
grep -q 'apply_baseline_hook(model, optimizer)' /root/HPT/hpt/train_test.py \
  || { echo "[FATAL] train_test.py 缺 optimizer 版钩子（M5 未生效）"; exit 1; }
grep -q 'cleanup_old_baseline' "$CORE/run_fedavg.sh" || { echo "[FATAL] run_fedavg.sh 缺 cleanup_old_baseline（M2 未生效）"; exit 1; }
grep -q 'rc_is_ok' "$CORE/phase5_autoshutdown.sh" || { echo "[FATAL] 守护缺 rc_is_ok（M3 未生效）"; exit 1; }
echo "  [OK] 4 项补丁锚点全部命中"

# ---- 硬闸 3：旧守护不得存在 ----
NP=$(ps -eo pid,cmd | grep -c '[p]hase5_autoshutdown.sh' || true)
echo "[闸3] 现存守护实例 $NP 个（必须为 0）"
[ "$NP" != "0" ] && { echo "[FATAL] 有残留守护，先清掉（避免重复推送/误收工）"; exit 1; }

# ---- 起训练 ----
echo "===== 起 run_phase5.sh ====="
cd /root/HPT
N_JOBS="$NJOBS" nohup setsid bash "$CORE/run_phase5.sh" > "$OUT/run_phase5.log" 2>&1 < /dev/null &
sleep 45
NP=$(ps -eo pid,cmd | grep -c '[r]un_phase5.sh' || true)
echo "[验活] run_phase5.sh 实例数 $NP（必须 ≥ 1）"
[ "$NP" -lt 1 ] && { echo "[FATAL] 训练没起来，日志尾："; tail -n 25 "$OUT/run_phase5.log"; exit 1; }
tail -n 12 "$OUT/run_phase5.log"

# ---- 起守护（必须在训练之后，铁律 J）----
echo "===== 起自动关机守护 ====="
cd /root/HPT
MAX_HOURS="$MAX_HOURS" POLL="$POLL" PUSH_IDBASE="$PUSH_IDBASE" \
  OUT="$OUT" CORE="$CORE" \
  nohup setsid bash "$CORE/phase5_autoshutdown.sh" >> "$OUT/autoshutdown.log" 2>&1 < /dev/null &
sleep 8
ND=$(ps -eo pid,cmd | grep -c '[p]hase5_autoshutdown.sh' || true)
echo "[验活] 守护实例数 $ND（必须 = 1）"
[ "$ND" != "1" ] && { echo "[FATAL] 守护实例数异常"; exit 1; }
tail -n 4 "$OUT/autoshutdown.log"
echo "===== 编排完成 ====="
