"""第4阶段（通信损伤扫描）后台监控器。

每隔 --interval 秒拉一次云端状态，记录到 tools/monitor_phase4.log。
检测到配置推进、磁盘告急、进程异常时额外打印醒目标记。

用法：
    python tools/monitor_phase4.py --interval 300 --duration 28800
    python tools/monitor_phase4.py --once
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud  # noqa: E402

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_phase4.log")

# 一次调用拿齐，避免多次往返（agent 是串行执行，多次调用会排队）
STATUS_CMD = r"""
echo "### CONFIGS"
for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  if [ -d /root/HPT/output/$t ]; then
    n=$(ls /root/HPT/output/$t 2>/dev/null | grep -c '^global_r')
    last=$(ls /root/HPT/output/$t 2>/dev/null | grep '^global_r' | sed 's/global_r//' | sort -n | tail -1)
    ev=$(ls /root/HPT/output/eval_${t}_*.log 2>/dev/null | wc -l)
    echo "$t rounds=$n last=r$last evallogs=$ev"
  else
    echo "$t (未开始)"
  fi
done
echo "### WATCH"
tail -n 4 /root/HPT/output/phase4_watch.log 2>/dev/null
echo "### SCAN"
tail -n 4 /root/HPT/output/log_c_drop00_train.txt 2>/dev/null
echo "### PROC"
ps aux | grep -c '[h]pt.run'
echo "### DISK"
df -BG /root 2>/dev/null | awk 'NR==2{print $4" "$5}'
echo "### PRIV"
ls /root/HPT/output/privacy/PRIVACY_REV_DONE 2>/dev/null || echo "rev未完成"
tail -n 2 /root/HPT/output/privacy/privacy_rev3.log 2>/dev/null
echo "### DONE"
ls /root/HPT/output/PHASE4_DONE 2>/dev/null || echo "未完成"
"""

CONFIG_TAGS = ["c_drop00", "c_drop25", "c_drop50", "c_drop75",
               "c_int8", "c_int4", "c_bit1"]


def log(msg, marker=False):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
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

    last_sig = ""
    t0 = time.time()
    while True:
        try:
            res = cloud.run(STATUS_CMD, cwd="/root/HPT", timeout=120, wait=180)
            d = parse(res.get("out") or "")
            cfgs = [x.strip() for x in d.get("CONFIGS", []) if x.strip()]
            proc = (d.get("PROC") or ["?"])[0].strip()
            disk = (d.get("DISK") or ["?"])[0].strip()
            done = (d.get("DONE") or [""])[0].strip()
            watch = (d.get("WATCH") or [])[-2:]
            scan = (d.get("SCAN") or [])[-1:]

            sig = " | ".join(cfgs)
            new = sig != last_sig
            log(f"proc={proc} disk={disk} done={done}")
            for c in cfgs:
                log("  | " + c)
            if new:
                log("  >> 配置进度变化", marker=True)
                last_sig = sig
            for w in watch:
                log("  | W " + w[:140])

            # 告警：磁盘 < 6G 或 进程数为 0 但未完成
            try:
                free_g = int(disk.split()[0].rstrip("G"))
            except Exception:
                free_g = 99
            if free_g < 6 and done != "未完成":
                log(f"!! 磁盘仅剩 {free_g}G，低于 6G 红线", marker=True)
            if proc == "0" and done == "未完成":
                log("!! 无 hpt.run 进程但第4阶段未完成 —— 实例可能已关机", marker=True)
            if done != "未完成":
                log("  >> 第4阶段完成标记 PHASE4_DONE 已出现", marker=True)
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
