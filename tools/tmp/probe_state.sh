OUT=/root/HPT/output
echo "=== 身份 ==="
echo "hostname=$(hostname)"
echo "uptime_sec=$(cut -d' ' -f1 /proc/uptime)"
echo "date=$(date '+%F %T')"
echo "agent_pid_check: ppid chain"
ps -o pid,ppid,etime,cmd -p 1
echo "=== output 顶层（按时间）==="
ls -la "$OUT" | head -50
echo "=== dp_* 目录 ==="
ls -d "$OUT"/dp_* 2>/dev/null
for t in dp_s0001 dp_s0003 dp_s0012; do
  echo "[$t] exists=$([ -d "$OUT/$t" ] && echo yes || echo no) last=$(ls -1 "$OUT/$t" 2>/dev/null | grep -E '^global_r' | sort -V | tail -1) du=$(du -sh "$OUT/$t" 2>/dev/null | cut -f1)"
done
echo "=== scanlog 实际情况 ==="
ls -la "$OUT"/scanlog_*.log* 2>/dev/null | head -30
for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  F="$OUT/scanlog_$t.log"
  if [ -f "$F" ]; then
    printf '%-10s ' "$t"; head -1 "$F"
  else
    printf '%-10s ' "$t"; echo "MISSING (legacy: $(ls "$OUT"/scanlog_${t}.log* 2>/dev/null | tr '\n' ' '))"
  fi
done
echo "=== 磁盘 ==="
df -h /root | tail -1
du -sh "$OUT" 2>/dev/null
echo "=== 评估日志时间 ==="
ls -la "$OUT"/eval_c_drop00_mujoco_metaworld_task3.log* "$OUT"/eval_c_int8_mujoco_metaworld_task3.log* 2>/dev/null
echo "=== 关键日志尾部 ==="
echo "--- chain_phase4_dp.log"; tail -n 12 "$OUT/chain_phase4_dp.log" 2>/dev/null
echo "--- autoshutdown.log"; tail -n 12 "$OUT/autoshutdown.log" 2>/dev/null
echo "--- dplog_dp_s0001.log"; tail -n 8 "$OUT/dplog_dp_s0001.log" 2>/dev/null
echo "--- log_dp_s0001_train.txt"; tail -n 12 "$OUT/log_dp_s0001_train.txt" 2>/dev/null
echo "=== 是否有活跃训练进程 ==="
ps -eo pid,etime,cmd | grep -E '[h]pt\.run|[r]un_fedavg|[r]un_dp_utility|[c]hain_phase4|[p]hase4_autoshutdown|[e]val_model' | cut -c1-100
echo "=== GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
