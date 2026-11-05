"""门2'/阶段3 实验后台监控器。

每隔 --interval 秒拉一次云端状态，记录到 relay/monitor.log。
检测到阶段推进或异常时额外打印醒目标记。

用法：
    python relay/monitor.py --interval 300 --duration 28800
    python relay/monitor.py --once
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud  # noqa: E402

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor.log")

# 拉取云端状态的命令（一次调用拿齐，避免多次往返）
STATUS_CMD = r"""
echo "### STAGES"
grep -n '>>>' /root/HPT/output/resume_all.log 2>/dev/null | tail -12
echo "### TAIL"
tail -n 6 /root/HPT/output/resume_all.log 2>/dev/null
echo "### GPU"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
echo "### PROC"
ps aux | grep -c '[h]pt.run'
echo "### DISK"
df -BG /root 2>/dev/null | awk 'NR==2{print $4}'
echo "### MARKERS"
ls /root/HPT/output/g2b/DONE /root/HPT/output/phase3/DONE 2>/dev/null
"""


def log(msg, marker=False):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(("*** " if marker else "    ") + line, flush=True)


def parse(out):
    d = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("### "):
            cur = line[4:].strip()
            d[cur] = []
        elif cur:
            d[cur].append(line)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--duration", type=int, default=28800)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    last_stages = ""
    t0 = time.time()
    while True:
        try:
            res = cloud.run(STATUS_CMD, cwd="/root/HPT", timeout=120, wait=180)
            d = parse(res.get("out") or "")
            stages = " | ".join(x.strip() for x in d.get("STAGES", []) if ">>>" in x)
            gpu = (d.get("GPU") or ["?"])[0]
            proc = (d.get("PROC") or ["?"])[0].strip()
            disk = (d.get("DISK") or ["?"])[0].strip()
            markers = [x for x in (d.get("MARKERS") or []) if x.strip()]
            tail = (d.get("TAIL") or [])[-3:]

            new = stages != last_stages
            log(f"GPU[{gpu}] proc={proc} disk={disk} stages={len(stages.split('|'))} "
                f"markers={markers}", marker=new)
            for t in tail:
                log(f"  | {t[:150]}")
            if new:
                log(f"  >> 阶段列表: {stages[-300:]}", marker=True)
                last_stages = stages
        except Exception as e:  # noqa: BLE001
            log(f"!! 监控出错: {type(e).__name__}: {e}", marker=True)

        if args.once:
            break
        if time.time() - t0 > args.duration:
            log("监控时长到，退出")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
