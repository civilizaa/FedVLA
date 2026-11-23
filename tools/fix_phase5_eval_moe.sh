#!/usr/bin/env bash
# ==============================================================================
# 第 5 阶段修复补丁 + 续跑（**待用户批准后执行**；2026-11-23 13:36 尚未执行）
#
# 修三处：
#   M1  eval_model.sh          评估前 export FEDVLA_MOE* 5 个变量（否则评的是 13.94M 残缺模型）
#   M2  run_fedavg.sh          _baseline/ 参与轮次清理（否则磁盘必爆：SCAFFOLD 单配置要 10.1G）
#   M3  phase5_autoshutdown.sh count_rc 只认内容为 0 的 .rc（修 fail-open）
#   M4  extract_phase5_results.py  collect_stage 加 params_M >= 18 门槛（修 fail-open）
#
# 铁律：
#   B  推送后回读 md5 校验
#   D  跨批次续跑破坏 LR/Adam 相位 ⇒ 训练未完成的配置必须清空重跑；
#      p5_bal_fedprox 训练已完成 ⇒ 只重评，不重训（豁免）
#   Q  单测不全绿绝不开跑
# ==============================================================================
set -u
CLOUD_DIR=${CLOUD_DIR:-/root/fedcore}
HPT_DIR=${HPT_DIR:-/root/HPT}
OUT=/root/HPT/output/phase5
CLOUD_PY=/root/miniconda3/envs/hpt/bin/python
DRY_RUN=${DRY_RUN:-1}          # 1=只打印补丁不执行；0=真执行

echo "===== 第5阶段修复补丁（DRY_RUN=$DRY_RUN）====="

# ----------------------------------------------------------------------------
# M1：eval_model.sh —— 评估前 export MoE 环境变量
#     插入点：export WANDB_MODE=offline 之前
#     取值必须与训练侧 run_fedavg.sh / run_phase5.sh 的 MOE_* 默认值逐字一致
# ----------------------------------------------------------------------------
patch_m1 () {
  echo "--- [M1] eval_model.sh：注入 FEDVLA_MOE* ---"
  if grep -q 'FEDVLA_MOE' "$CLOUD_DIR/eval_model.sh" 2>/dev/null; then
    echo "    [SKIP] 已存在 FEDVLA_MOE，跳过（幂等）"
    return 0
  fi
  python3 - "$CLOUD_DIR/eval_model.sh" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
anchor = 'export WANDB_MODE=offline'
inject = '''# ===== [FedVLA-MOE] 评估侧必须开 MoE（2026-11-23 修复）=========================
# ⚠️ 这是第 3 阶段已踩过的致命缺陷（见 run_phase3_eval.sh 的 apply_moe_env）：
#    patch_hpt_moe.py 的判定是 `if os.environ.get("FEDVLA_MOE","0") == "1"`，
#    不设 ⇒ moe_cfg=None ⇒ policy 根本不创建 MoE 模块；
#    而 Policy.load_model 用 strict=False ⇒ 264 个 MoE 键被**静默丢弃**，
#    不报错、不告警 ⇒ 评估的是残缺模型（13.94M 而非 21.08M）。
#    5 个取值必须与训练侧逐字一致，否则结构对不上。
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half
# ==============================================================================
'''
if anchor not in s:
    print("[FATAL] 锚点 'export WANDB_MODE=offline' 不在 %s" % p); sys.exit(2)
s = s.replace(anchor, inject + anchor, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("[OK] 已注入 M1（%d 字节）" % len(s))
PYEOF
}

# ----------------------------------------------------------------------------
# M2：run_fedavg.sh —— _baseline/ 参与轮次清理
#     只删**已被消费完**的中间产物；ci_c{c}.pt 与 c_global_r{r+1}.pt 必须保留
# ----------------------------------------------------------------------------
patch_m2 () {
  echo "--- [M2] run_fedavg.sh：_baseline 历史轮次清理 ---"
  if grep -q 'cleanup_old_baseline' "$CLOUD_DIR/run_fedavg.sh" 2>/dev/null; then
    echo "    [SKIP] 已存在 cleanup_old_baseline，跳过（幂等）"
    return 0
  fi
  python3 - "$CLOUD_DIR/run_fedavg.sh" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
anchor = "echo \"===== FedVLA FedAvg =====\""
inject = '''# ===== _baseline 历史轮次清理（2026-11-23 修复）=================================
# 背景：FedProx 的 anchor_r{r}.pt（84MB）与 SCAFFOLD 的 ci_c{c}_r{r}.pt（50MB）
#       原本**从不被清理** ⇒ 单配置 4.2G / 10.1G ⇒ 30G 系统盘必爆。
#       2026-11-23 实测：p5_bal_scaffold 在第 1 轮就因磁盘水位触发
#       「可用空间仅 5GB < 6GB」而聚合中止。
# 口径：只删**已被消费完**的中间产物，保留最近 $KEEP 轮；
#       ci_c{c}.pt（当前控制变量）与 c_global_r{r}.pt（SCAFFOLD 下发源）永不删。
cleanup_old_baseline () {
  local cur="$1" keep="$2"
  local cutoff=$((cur - keep))
  [ "$cutoff" -lt 1 ] && return 0
  local n=0 f base r
  for f in "$BASELINE_DIR"/anchor_r*.pt "$BASELINE_DIR"/ci_c*_r*.pt \\
           "$BASELINE_DIR"/c_global_r*.pt "$BASELINE_DIR"/wglobal_r*.pt; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    case "$base" in
      ci_c[0-9].pt) continue ;;   # 当前控制变量的持久化副本（无轮次号，永不删）
    esac
    # 🔴 必须**按轮次**判断，只删 r < cutoff 的；
    #    原实现不判轮次 ⇒ 会把**当前轮**的 anchor_r{r}.pt / ci_c{c}_r{r}.pt 一起删掉
    #    ⇒ SCAFFOLD 的 c_k 下发链断裂 ⇒ 下一轮聚合中止。
    r=$(echo "$base" | grep -oE '_r[0-9]+\\.pt$' | grep -oE '[0-9]+')
    [ -n "$r" ] || continue
    if [ "$r" -lt "$cutoff" ]; then
      rm -f "$f" && n=$((n + 1))
    fi
  done
  [ "$n" -gt 0 ] && echo "[清理] 删除 $n 个 _baseline 历史轮次中间产物（保留最近 ${keep} 轮）"
  return 0
}

'''
if anchor not in s:
    print("[FATAL] 锚点不在 %s" % p); sys.exit(2)
s = s.replace(anchor, inject + anchor, 1)
# 在 cleanup_old_rounds 调用后追加清理
old = '  cleanup_old_rounds "$r" "$KEEP"'
new = ('  cleanup_old_rounds "$r" "$KEEP"\n'
       '  [ -n "${BASELINE_DIR:-}" ] && cleanup_old_baseline "$r" "$KEEP"')
if old not in s:
    print("[FATAL] 锚点 cleanup_old_rounds 调用不在 %s" % p); sys.exit(2)
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("[OK] 已注入 M2")
PYEOF
}

# ----------------------------------------------------------------------------
# M3：phase5_autoshutdown.sh —— count_rc 只认内容为 0 的 .rc
# ----------------------------------------------------------------------------
patch_m3 () {
  echo "--- [M3] phase5_autoshutdown.sh：count_rc 修 fail-open ---"
  if grep -q 'rc_is_ok' "$CLOUD_DIR/phase5_autoshutdown.sh" 2>/dev/null; then
    echo "    [SKIP] 已存在 rc_is_ok，跳过（幂等）"
    return 0
  fi
  python3 - "$CLOUD_DIR/phase5_autoshutdown.sh" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = '''count_rc() {
    local n=0 t
    for t in $1; do
        [ -f "$OUT/.rc_$t" ] && n=$((n + 1))
    done
    echo "$n"
}'''
new = '''# rc_is_ok：只有内容为 0（成功）的 .rc 才算完成。
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
}'''
if old not in s:
    print("[FATAL] count_rc 原实现不在 %s" % p); sys.exit(2)
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("[OK] 已注入 M3")
PYEOF
}

# ----------------------------------------------------------------------------
# M4：extract_phase5_results.py —— collect_stage 加 params_M 门槛
# ----------------------------------------------------------------------------
patch_m4 () {
  echo "--- [M4] extract_phase5_results.py：params_M >= 18 门槛 ---"
  if grep -q 'PARAMS_MIN_M' "$CLOUD_DIR/extract_phase5_results.py" 2>/dev/null; then
    echo "    [SKIP] 已存在 PARAMS_MIN_M，跳过（幂等）"
    return 0
  fi
  python3 - "$CLOUD_DIR/extract_phase5_results.py" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = 'DOMAINS = ["mujoco_metaworld_task1", "mujoco_metaworld_task2",'
new = ('# 🔴 2026-11-23：MoE 生效判据（参数量下限）。低于此值 = 评估的是残缺模型，\n'
       '#    该 domain 的结果一律作废并显式报错（修 fail-open：原来只记录不校验）。\n'
       'PARAMS_MIN_M = 18.0\n\n'
       'DOMAINS = ["mujoco_metaworld_task1", "mujoco_metaworld_task2",')
if old not in s:
    print("[FATAL] 锚点不在 %s" % p); sys.exit(2)
s = s.replace(old, new, 1)
old2 = '''        except Exception:
            pass
    return st'''
new2 = '''        except Exception:
            pass
        # 🔴 门槛：参数量不足 ⇒ MoE 未生效 ⇒ 该 domain 作废（显式报错，不静默）
        pm = (st["domains"][d] or {}).get("params_M")
        if pm is not None and pm < PARAMS_MIN_M:
            print("[FATAL] %s/%s 参数量仅 %.2fM < %.1fM ⇒ MoE 未生效，"
                  "评的是残缺模型，本项作废" % (tag, d, pm, PARAMS_MIN_M))
            st["domains"][d]["invalid_reason"] = (
                "params_M=%.2f < %.1f ⇒ MoE 未生效（残缺模型），结果作废"
                % (pm, PARAMS_MIN_M))
            st["domains"][d]["stats"] = None
            st["domains"][d]["seeds"] = []
    return st'''
if old2 not in s:
    print("[FATAL] 锚点2 不在 %s" % p); sys.exit(2)
s = s.replace(old2, new2, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("[OK] 已注入 M4")
PYEOF
}

# ----------------------------------------------------------------------------
# M5：还原 train_test.py 的旧钩子块（opt 错误版）→ 重打新补丁（optimizer + fail-loud）
#
# 🔴 2026-11-23 云端实测根因：patch 注入的是 apply_baseline_hook(model, opt)，
#    而 HPT 的 train() 里优化器变量名是 **optimizer** ⇒ 每个 step 都 NameError
#    ⇒ 被 except 吞掉 ⇒ **FedProx 近端项 / SCAFFOLD 修正从未生效且零报错**
#    （云端 log_client_*.txt 里满是 "hook skipped: NameError name 'opt'"）。
#    由于 _apply() 用锚点匹配，已打过旧补丁的文件**锚点不再命中** ⇒ 必须先还原。
# ----------------------------------------------------------------------------
patch_m5 () {
  echo "--- [M5] train_test.py：还原旧钩子块并重打新补丁 ---"
  if [ ! -f "$HPT_DIR/hpt/train_test.py" ]; then
    echo "  [--] 本地布局无 train_test.py，跳过 M5 的还原/重打"
    return 0
  fi
  HPT_DIR="$HPT_DIR" python3 - <<'PYEOF'
import os, re, sys
P = os.path.join(os.environ["HPT_DIR"], "hpt", "train_test.py")
s = open(P, encoding="utf-8").read()

# 1) 梯度修正钩子块：把 backward..optimizer.step() 整个区间规范成**精确锚点形态**
#    （单纯删块内容会残留多余空行 ⇒ 锚点不匹配 ⇒ 重打补丁失败）
pat1 = re.compile(r"(        domain_loss\.backward\(\)\n)(.*?)(\n        optimizer\.step\(\))", re.S)
m = pat1.search(s)
if not m:
    print("[FATAL] 找不到 domain_loss.backward() ... optimizer.step() 区间"); sys.exit(2)
had1 = "FedVLA-BASE" in m.group(2)
s = s[:m.start()] + "        domain_loss.backward()\n\n        optimizer.step()\n" + s[m.end():]
print("  [ok] 已还原梯度修正钩子块" if had1 else "  [--] 梯度修正钩子块本就不存在")

# 2) SCAFFOLD 导出块：存在则删（让 ANCHOR2 重新命中）；不存在**不是错误**
#    （上一次 M5 可能已经删掉了 ⇒ 再跑要幂等）
pat2 = re.compile(
    r"(    # \[FedVLA-BASE\] SCAFFOLD：本地训练结束后更新本地控制变量 c_k\n.*?)"
    r"(\n    return \{k: np\.mean\(v\) for k, v in info_log\.items\(\) if len\(v\) > 1\})", re.S)
m2 = pat2.search(s)
if m2:
    s = s[:m2.start(1)] + s[m2.end(1):]
    print("  [ok] 已还原 SCAFFOLD 导出块")
else:
    print("  [--] SCAFFOLD 导出块本就不存在，无需还原")

# 3) 校验 ANCHOR2 唯一存在（否则重打补丁会插错位置）
a2 = "    return {k: np.mean(v) for k, v in info_log.items() if len(v) > 1}"
if s.count(a2) != 1:
    print("[FATAL] ANCHOR2 出现 %d 次（应为 1）" % s.count(a2)); sys.exit(2)

open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  [ok] train_test.py 已回到可打补丁状态")
PYEOF
  # 3) 重打补丁（会写入新的 hook 模块 + 新的 BLOCK）
  "$CLOUD_PY" "$CLOUD_DIR/patch_hpt_fedbaseline.py" \
    || { echo "[FATAL] 重打补丁失败"; exit 1; }
  # 4) 硬断言：optimizer 版生效、opt 版消失（铁律 Q：判据必须 fail-loud）
  grep -q 'apply_baseline_hook(model, optimizer)' "$HPT_DIR/hpt/train_test.py" \
    || { echo "[FATAL] 断言失败：train_test.py 里没有 optimizer 版钩子"; exit 1; }
  if grep -q 'apply_baseline_hook(model, opt)' "$HPT_DIR/hpt/train_test.py"; then
    echo "[FATAL] 断言失败：train_test.py 里仍残留 opt 版钩子"; exit 1
  fi
  grep -q 'finish_baseline_round(model)' "$HPT_DIR/hpt/train_test.py" \
    || { echo "[FATAL] 断言失败：train_test.py 里没有 SCAFFOLD 导出钩子"; exit 1; }
  echo "  [OK] M5 通过（optimizer 版钩子已就位，opt 版已清除）"
}

apply_all () {
  patch_m1; patch_m2; patch_m3; patch_m4; patch_m5
  echo "--- 回读校验（铁律 B）---"
  md5sum "$CLOUD_DIR"/eval_model.sh "$CLOUD_DIR"/run_fedavg.sh \
         "$CLOUD_DIR"/phase5_autoshutdown.sh "$CLOUD_DIR"/extract_phase5_results.py 2>/dev/null
  echo "--- 语法自检 ---"
  for f in run_fedavg.sh phase5_autoshutdown.sh eval_model.sh; do
    bash -n "$CLOUD_DIR/$f" 2>&1 && echo "  [OK] $f 语法通过" || echo "  [FAIL] $f 语法错误"
  done
  "$CLOUD_PY" -c "import ast,sys; ast.parse(open('$CLOUD_DIR/extract_phase5_results.py',encoding='utf-8').read()); print('  [OK] extract_phase5_results.py 语法通过')" 2>&1
  echo "--- 补丁后 FEDVLA_MOE 命中数（应 > 0）---"
  echo "  eval_model.sh: $(grep -c 'FEDVLA_MOE' "$CLOUD_DIR/eval_model.sh" 2>/dev/null)"
}

if [ "$DRY_RUN" = "1" ]; then
  echo "（DRY_RUN=1，本轮不执行任何修改）"
  echo "确认后以 DRY_RUN=0 执行。"
else
  apply_all
fi

echo
echo "===== 修复补丁结束 ====="
