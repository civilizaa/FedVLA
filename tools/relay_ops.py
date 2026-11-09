"""relay 运维：清队列 + 快速探测 agent 活性。

用法:
    python tools/relay_ops.py clear           # 清空未完成队列（保留已完成结果）
    python tools/relay_ops.py ping            # 看 health：queued / agent_seen 落后
    python tools/relay_ops.py probe "<cmd>"   # 立刻执行一条短命令（绕过排队等待）
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud  # noqa: E402


def health():
    h = cloud.req("/health", timeout=15, retries=2)
    seen = int(h.get("agent_seen") or 0)
    lag = int(time.time() - seen) if seen else -1
    return h, lag


def do_clear():
    r = cloud.req("/clear", data={}, timeout=30, retries=3)
    print(json.dumps(r, ensure_ascii=False))


def do_ping():
    h, lag = health()
    print(json.dumps({"health": h, "agent_seen_lag_sec": lag,
                      "now": time.strftime("%Y-%m-%d %H:%M:%S")},
                     ensure_ascii=False, indent=2))


def do_probe(cmd):
    """enqueue 后不等结果轮询，而是短间隔多次探测 —— 用于快速判断 agent 是否在消费。"""
    r = cloud.req("/enqueue", data={"cmd": cmd, "cwd": "/root/HPT",
                                    "timeout": 120, "bg": False}, timeout=30, retries=3)
    cid = r["id"]
    print(f"enqueued cid={cid}")
    t0 = time.time()
    while time.time() - t0 < 100:
        try:
            res = cloud.req(f"/results?since={cid - 1}", timeout=15, retries=1)
        except Exception as e:  # noqa: BLE001
            print(f"  poll err {type(e).__name__}")
            time.sleep(2)
            continue
        for x in res.get("results", []):
            if int(x["id"]) == cid:
                print(f"--- stdout ---")
                print(x.get("stdout", "")[:6000])
                print(f"--- stderr ---")
                print((x.get("stderr") or "")[:2000])
                return
        time.sleep(2)
    print(f"!! cid={cid} 100s 内未返回")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ping"
    if cmd == "clear":
        do_clear()
    elif cmd == "ping":
        do_ping()
    elif cmd == "probe":
        do_probe(sys.argv[2])
    else:
        print(__doc__)
