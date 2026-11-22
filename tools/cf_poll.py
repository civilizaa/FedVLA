#!/usr/bin/env python3
"""后台低频轮询云端反事实评估进度（省点数：每 5 分钟只发 1 次 relay 请求）。

用法：python tools/cf_poll.py <分钟数> <输出日志>
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cloud  # noqa: E402

CMD = r"""
F=/root/HPT/output/eval_cf_int4_restore_mujoco_metaworld_task3.log
echo "表数=$(grep -c '| avg' $F 2>/dev/null || echo 0)"
grep '| avg' $F 2>/dev/null | tr '\n' ' '
echo ""
ps -eo etime,cmd | grep '[h]pt.run_eval' | head -2 | cut -c1-60
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1
cat /proc/loadavg
"""


def main() -> int:
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "cf_poll.log")
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        ts = time.strftime("%H:%M:%S")
        try:
            r = cloud.run(CMD, cwd="/root/HPT", timeout=90, wait=150)
            body = r.get("out") or r.get("err") or "(空)"
        except Exception as e:  # noqa: BLE001
            body = f"查询失败: {type(e).__name__}"
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"===== {ts} =====\n{body}\n")
        if "表数=3" in body:
            with open(out, "a", encoding="utf-8") as f:
                f.write("[poll] 三张表齐 ⇒ 评估完成\n")
            break
        time.sleep(300)
    return 0


if __name__ == "__main__":
    sys.exit(main())
