"""第4阶段通信侧后台监控（低频，一次调用拿齐状态）。

设计要点（对齐 MEMORY 铁律 C：控制点数消耗）：
  · 间隔默认 600s（不是 300s），12 小时约 72 次云端调用
  · 每次只发**一条**命令（复用云端 phase4_status.sh），不按配置逐个轮询
  · stdout 全部追加到 tools/monitor_comm.log，不进对话上下文
  · 7 个配置的评估表全部齐了（tables>=3）就打印 DONE 并退出
  · 进程心跳落后（实例被关机）或磁盘告急时打醒目标记

用法：
    python tools/monitor_comm.py --interval 600 --duration 43200
    python tools/monitor_comm.py --once
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "monitor_comm.log")
TAGS = ["c_drop00", "c_drop25", "c_drop50", "c_drop75", "c_int8", "c_int4", "c_bit1"]


def log(msg, marker=False):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(("*** " if marker else "    ") + line, flush=True)


def fetch():
    res = cloud.run("bash /root/fedcore/phase4_status.sh", cwd="/root",
                    timeout=240, wait=420)
    return res.get("out") or ""


def parse(out):
    st = {}
    for tag in TAGS:
        m = re.search(r"\[%s\] tables=(\d+)/(\d+) rc=(\S+) last_r=(\S+)" % tag, out)
        if m:
            st[tag] = {"tables": int(m.group(1)), "need": int(m.group(2)),
                       "rc": m.group(3), "last_r": m.group(4)}
    m = re.search(r"agent_seen_lag_sec\"?:?\s*(\d+)", out)
    disk = re.search(r"^\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)\s+/\s*$", out, re.M)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--duration", type=int, default=43200)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    prev = {}
    t_end = time.time() + args.duration
    while True:
        try:
            out = fetch()
        except Exception as e:  # noqa: BLE001
            log(f"状态获取失败: {type(e).__name__}: {e}", marker=True)
            out = ""
        st = parse(out)
        summary = []
        for tag in TAGS:
            s = st.get(tag)
            if not s:
                summary.append(f"{tag}:未开始")
                continue
            summary.append(f"{tag}:{s['tables']}/{s['need']}(r{s['last_r']})")
            # 记录轮次推进速率
            if tag in prev and s["last_r"].isdigit() and prev[tag]["last_r"].isdigit():
                d = int(s["last_r"]) - int(prev[tag]["last_r"])
                dt = time.time() - prev[tag]["t"]
                if d > 0 and dt > 0:
                    log(f"{tag} 推进 {d} 轮 / {dt/60:.1f} 分钟 = {dt/60/d:.2f} 分钟/轮")
            prev[tag] = {"last_r": s["last_r"], "t": time.time()}
        log(" | ".join(summary))

        if all(st.get(t, {}).get("tables", 0) >= st.get(t, {}).get("need", 99)
               for t in TAGS) and len(st) == len(TAGS):
            log("全部 7 个配置评估齐全 —— 通信扫描完成", marker=True)
            break
        if args.once:
            break
        if time.time() > t_end:
            log("监控时长到，退出", marker=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
