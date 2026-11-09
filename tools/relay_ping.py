"""快速探测 relay 服务状态（不排队执行命令）。

用法: python tools/relay_ping.py
输出: 服务端 status JSON + 本地视角的 agent_seen 落后秒数
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud  # noqa: E402


def main():
    out = {"now": time.strftime("%Y-%m-%d %H:%M:%S")}
    for path in ("/status", "/health", "/queue"):
        try:
            t0 = time.time()
            r = cloud.req(path, timeout=15, retries=2)
            out[path] = {"ok": True, "rt": round(time.time() - t0, 2), "data": r}
        except Exception as e:  # noqa: BLE001
            out[path] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
