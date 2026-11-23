#!/usr/bin/env bash
# ==============================================================================
# 第 5 阶段**云端接线冒烟测试**（2026-11-23 新增）
#
# 🔴 为什么必须有这个（本次血案的直接教训）：
#   29 项离线单测全绿 ⇒ 但云端真实跑时，train_test.py 里注入的钩子调用
#   用了错误的优化器变量名 ⇒ 每个 step 都 NameError ⇒ 被 except 吞掉
#   ⇒ **FedProx 近端项 / SCAFFOLD 修正从未生效，且全程零报错**
#   ⇒ 7 个配置全部跑完也是 0/7 成功（还烧掉 20 小时算力）。
#   ⇒ 单测覆盖的是"模块数学"，**覆盖不到"train_test.py 里的接线"**。
#
# 冒烟判据（全部必须通过，否则禁止开跑）：
#   1. log_client_*.txt 里出现 "[FedVLA-BASE] FedProx 近端项已施加"（FedProx）
#   2. log_client_*.txt 里出现 "[FedVLA-BASE] SCAFFOLD 控制变量已应用"（SCAFFOLD）
#   3. SCAFFOLD 的 _baseline/ci_c{c}_r{r}.pt **真的落盘**（第 1 轮要能在第 2 轮被聚合读到）
#   4. log_client_*.txt 里 **没有** "hook skipped" / "finish skipped"
#
# 用法：bash phase5_smoke.sh            # 前台跑（2 基线 × 2 轮）
#      SMOKE_OUT=/root/HPT/output/smoke2 bash phase5_smoke.sh
# ==============================================================================
set -u
CORE=/root/fedcore
SMOKE_OUT=${SMOKE_OUT:-/root/HPT/output/smoke}
ROUNDS=${SMOKE_ROUNDS:-2}
LOCAL_EPOCHS=${SMOKE_LOCAL_EPOCHS:-1}

mkdir -p "$SMOKE_OUT"

run_one () {
  local base="$1" tag="$2"
  local out="$SMOKE_OUT/$base"
  echo "===== 冒烟: $base（tag=$tag，$ROUNDS 轮，LOCAL_EPOCHS=$LOCAL_EPOCHS）====="
  rm -rf "$out"
  OUT="$out" ONLY_TAGS="$tag" SKIP_EVAL=1 NUM_ROUNDS="$ROUNDS" \
    LOCAL_EPOCHS="$LOCAL_EPOCHS" \
    bash "$CORE/run_phase5.sh" > "$SMOKE_OUT/smoke_$base.log" 2>&1
  echo "[smoke] $base rc=$? （日志：$SMOKE_OUT/smoke_$base.log）"
}

run_one fedprox  p5_bal_fedprox
run_one scaffold p5_bal_scaffold

echo
echo "===================== 冒烟判据 ====================="
fail=0
for base in fedprox scaffold; do
  case "$base" in
    fedprox)  tag=p5_bal_fedprox ;;
    scaffold) tag=p5_bal_scaffold ;;
  esac
  d="$SMOKE_OUT/$base/$tag"
  echo "--------- $base ---------"
  # 🔴 2026-11-23：判据必须包含**配置自身的 rc**（.rc=0 才是成功），
  #    否则"钩子打印出现过"也会被误判为通过（第一版就犯了这个错）。
  rcf="$SMOKE_OUT/$base/.rc_$tag"
  rc=$(cat "$rcf" 2>/dev/null || echo "?")
  echo "[配置 rc] $rc （必须为 0）"
  [ "$rc" != "0" ] && fail=1
  echo "[hook 打印]"; grep -h 'FedVLA-BASE' "$d"/log_client_*.txt 2>/dev/null | sed 's/^ *//' | sort -u | head -6
  nskip=$(cat "$d"/log_client_*.txt 2>/dev/null | grep -c 'skipped' || true)
  echo "[skipped 计数] $nskip （必须为 0）"
  [ "$nskip" != "0" ] && fail=1
  if [ "$base" = "fedprox" ]; then
    n=$(cat "$d"/log_client_*.txt 2>/dev/null | grep -c 'FedProx 近端项已施加' || true)
    echo "[FedProx 近端项生效次数] $n （必须 > 0）"
    [ "$n" -le 0 ] && fail=1
  else
    n=$(cat "$d"/log_client_*.txt 2>/dev/null | grep -c 'SCAFFOLD 控制变量已应用' || true)
    echo "[SCAFFOLD 控制变量生效次数] $n （必须 > 0）"
    [ "$n" -le 0 ] && fail=1
    echo "[ci_c 落盘]"; ls -la "$d/_baseline"/ci_c*_r*.pt 2>/dev/null | head -8
    ncv=$(ls "$d/_baseline"/ci_c*_r*.pt 2>/dev/null | wc -l)
    echo "[ci_c 文件数] $ncv （必须 > 0）"
    [ "$ncv" -le 0 ] && fail=1
  fi
done
echo "===================================================="
if [ "$fail" = "0" ]; then
  echo "[结论] 冒烟通过 ✅ —— 钩子真的接线生效，可以开跑"
  exit 0
else
  echo "[结论] 冒烟失败 ❌ —— **禁止开跑**（否则又是 0/7 成功 + 白烧算力）"
  exit 1
fi
