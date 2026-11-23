cd /root/HPT
export FEDVLA_MOE=1 FEDVLA_MOE_EXPERTS=8 FEDVLA_MOE_TOPK=2 FEDVLA_MOE_EXPERT_RATIO=0.25 FEDVLA_MOE_LAYERS=half
export OMP_NUM_THREADS=2 PYTHONPATH=/root/HPT:$PYTHONPATH
cat > /root/fedcore/cf_eval_raw.sh <<'INNER'
#!/usr/bin/env bash
set -u
TAG=$1; SRC=$2
MODEL_DIR=$SRC DOMAIN=mujoco_metaworld_task3 ENVNAME=button-press-topdown-v2 \
 N_SEEDS=3 EPISODE_NUM=20 N_JOBS=2 LOG_TAG=$TAG \
 bash /root/fedcore/eval_model.sh > /root/HPT/output/${TAG}_stdout.log 2>&1
echo "rc=$? done $(date '+%F %T')"
INNER
chmod +x /root/fedcore/cf_eval_raw.sh
# 客户端步长变体（最关键）
nohup setsid bash /root/fedcore/cf_eval_raw.sh cf_p3c_int8_cs /root/fedcore/cf/cf_p3c_int8_cs > /root/HPT/output/cf_cs_launch.log 2>&1 < /dev/null &
sleep 3
# int8-trunk（补充，低优先级）
nohup nice -n 15 setsid bash /root/fedcore/cf_eval_raw.sh cf_p3c_int8_trunk /root/fedcore/cf/cf_p3c_int8_trunk > /root/HPT/output/cf_trunk_launch.log 2>&1 < /dev/null &
sleep 45
echo "=== 评估进程 ==="
ps -eo pid,ni,etime,cmd | grep -E '[h]pt.run_eval|[c]f_eval_raw' | cut -c1-95 | head -8
echo "=== DP ==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  d=/root/HPT/output/$t
  echo "  [$t] rounds=$(ls $d 2>/dev/null | grep -c '^global_r') last=$(ls $d 2>/dev/null | grep '^global_r' | sort -V | tail -1)"
done
