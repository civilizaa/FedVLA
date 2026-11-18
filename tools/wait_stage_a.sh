#!/usr/bin/env bash
# 等第4阶段阶段 A 收尾：.rc_c_bit1 出现即退出（它是最后一个配置）。
# 只读探针：不碰训练、不碰扫描、不重启任何东西。
#
# ⚠️ 判定只看 **stdout**：文件不存在时 `ls` 把报错写进 stderr，
#    若把 stderr 混进来会误判成 PRESENT（2026-11-18 踩过）。
cd "C:/Users/hp/WorkBuddy/2026-10-08-14-27-08/fedvla-repo" || exit 2
MAX=${1:-60}
i=0
while [ "$i" -lt "$MAX" ]; do
  i=$((i+1))
  out=$(MSYS_NO_PATHCONV=1 python tools/cloud.py "ls /root/HPT/output/.rc_c_bit1" \
        --cwd /root --timeout 60 --wait 150 2>/dev/null | tr -d '\r')
  if echo "$out" | grep -q "rc_c_bit1"; then
    echo "[$(date '+%H:%M:%S')] probe#$i: PRESENT -> .rc_c_bit1 已出现（阶段A完成）"
    echo "STAGE_A_RC_PRESENT"
    exit 0
  fi
  echo "[$(date '+%H:%M:%S')] probe#$i: missing"
  sleep 120
done
echo "[$(date '+%H:%M:%S')] 超过 $MAX 次探针仍未见 .rc_c_bit1"
echo "STAGE_A_TIMEOUT"
exit 1
