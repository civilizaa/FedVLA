#!/usr/bin/env bash
# FedVLA 第4阶段：一次性状态采集（只读，不改动任何东西）。
# 用途：回答"还要跑多久 / 现在什么成果"——一次调用拿齐所有状态，避免高频轮询。
OUT=${OUT:-/root/HPT/output}
PY=/root/miniconda3/envs/hpt/bin/python
TAGS="c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1"

echo "=== NOW ==="
date '+%F %T'
echo "=== DISK ==="
df -h /root | tail -1
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
echo "=== PROCS (训练/评估/隐私) ==="
ps -eo etime,cmd | grep -E "hpt\.run|privacy_experiment|eval_model|train_test|run_comm_scan|run_fedavg" | grep -v grep | cut -c1-190
echo "=== COMM CONFIGS ==="
for tag in $TAGS; do
  lf="$OUT/eval_${tag}_mujoco_metaworld_task3.log"
  n=0
  if [ -f "$lf" ]; then
    n=$(grep -c '| avg' "$lf" 2>/dev/null)
    [ -z "$n" ] && n=0
  fi
  rc="-"
  [ -f "$OUT/.rc_${tag}" ] && rc=$(cat "$OUT/.rc_${tag}")
  last=$(ls -d "$OUT/$tag"/global_r* 2>/dev/null | sed 's/.*global_r//' | sort -n | tail -1)
  [ -z "$last" ] && last="-"
  sl=$(tail -n 1 "$OUT/scanlog_${tag}.log" 2>/dev/null | tr -d '\r' | cut -c1-140)
  echo "[$tag] tables=$n/3 rc=$rc last_r=$last | $sl"
done
echo "=== PRIVACY ==="
"$PY" - <<'PYEOF'
import json, os, collections
p = '/root/HPT/output/privacy/privacy_results.json'
if not os.path.exists(p):
    print('NO privacy_results.json')
else:
    d = json.load(open(p))
    runs = d.get('runs', [])
    print('total_runs =', len(runs))
    c = collections.Counter(r.get('mode') for r in runs)
    print('by_mode =', dict(c))
    need = set()
    for s in (0, 1, 2):
        for sig in (1e-4, 1e-3, 1e-2, 1e-1, 1):
            for t in ('image', 'action', 'state'):
                need.add((s, 'modality_rev', sig, t))
    have = set((r.get('sample'), r.get('mode'), r.get('sigma_ref'), r.get('target')) for r in runs)
    print('modality_rev done %d/%d' % (len(need & have), len(need)))
    meta = d.get('meta', {})
    print('meta: dp_steps=%s modes=%s samples=%s targets=%s' %
          (meta.get('dp_steps'), meta.get('modes'), meta.get('samples'), meta.get('targets')))
    print('--- modality_rev 最近 6 条 ---')
    rev = [r for r in runs if r.get('mode') == 'modality_rev']
    for r in rev[-6:]:
        print('  s=%s sig=%s tgt=%-7s cos=%s corr=%s eps=%s' % (
            r.get('sample'), r.get('sigma_ref'), r.get('target'),
            r.get('cos'), r.get('corr_mean'), r.get('eps')))
PYEOF
echo "=== EVAL LOG FILES ==="
ls -la "$OUT"/eval_c_*.log 2>/dev/null | awk '{print $NF, $5"B"}'
echo "=== MARKERS ==="
ls -1 "$OUT"/.rc_* 2>/dev/null
ls -1 "$OUT"/privacy/*DONE 2>/dev/null
ls -1 "$OUT"/scanlog_*.log 2>/dev/null
echo "=== END ==="
