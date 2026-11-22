#!/usr/bin/env python3
"""低频监控云端第4阶段（DP 重跑）进度。

用法: python tools/dp_watch.py [轮次分钟] [间隔秒]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud  # noqa: E402

PROBE = r"""
cd /root/HPT
echo "T=$(date '+%F %T')  load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
for t in dp_s0001 dp_s0003 dp_s0012; do
  d=/root/HPT/output/$t
  n=$(ls $d 2>/dev/null | grep -c '^global_r')
  last=$(ls $d 2>/dev/null | grep '^global_r' | sort -V | tail -1)
  rc=$([ -f /root/HPT/output/.rc_$t ] && echo YES || echo no)
  ev=$(grep -c '| avg' /root/HPT/output/eval_${t}_mujoco_metaworld_task3.log 2>/dev/null || echo 0)
  echo "  [$t] rounds=$n last=$last rc=$rc evaltable=$ev"
done
n=0; for t in dp_s0001 dp_s0003 dp_s0012; do [ -f /root/HPT/output/.rc_$t ] && n=$((n+1)); done
echo "  dp_rc=$n/3"
tail -n 2 /root/HPT/output/autoshutdown.log 2>/dev/null
df -h /root | tail -1
"""


def main():
    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    deadline = time.time() + minutes * 60
    print(f"[watch] 监控 {minutes} 分钟，间隔 {interval}s", flush=True)
    while time.time() < deadline:
        try:
            r = cloud.run(PROBE, cwd="/root/HPT", timeout=150, wait=240)
            out = r.get("out") or r.get("stdout") or ""
            err = r.get("err") or r.get("stderr") or ""
            print(out.strip(), flush=True)
            if err.strip():
                print("  [stderr]", err.strip()[:300], flush=True)
            if "dp_rc=3/3" in out and "shutdown" in out.lower():
                print("[watch] ✅ DP 完成且已关机，退出监控", flush=True)
                return 0
        except Exception as e:  # noqa: BLE001
            print(f"[watch] err {type(e).__name__}: {e}", flush=True)
        time.sleep(interval)
    print("[watch] 监控窗口结束", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
