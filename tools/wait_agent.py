#!/usr/bin/env python3
"""等待云端 relay agent 复活（agent_seen 新鲜度 < 120s 即视为已接管）。

用法: python tools/wait_agent.py [最多等待分钟] [轮询间隔秒]
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud  # noqa: E402


def main():
    max_min = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    deadline = time.time() + max_min * 60
    print(f"[wait] 最多等 {max_min} 分钟，间隔 {interval}s", flush=True)
    while time.time() < deadline:
        try:
            h = cloud.req("/health", timeout=15, retries=1)
            seen = int(h.get("agent_seen") or 0)
            lag = int(time.time() - seen) if seen else -1
            q = h.get("queued")
            print(f"[wait] {time.strftime('%H:%M:%S')} lag={lag}s queued={q} "
                  f"alive={h.get('agent_alive')}", flush=True)
            if 0 <= lag < 120:
                print("[wait] ✅ agent 已接管", flush=True)
                return 0
        except Exception as e:  # noqa: BLE001
            print(f"[wait] {time.strftime('%H:%M:%S')} health err "
                  f"{type(e).__name__}", flush=True)
        time.sleep(interval)
    print("[wait] ❌ 超时，agent 未复活", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
