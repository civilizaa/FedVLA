#!/usr/bin/env python3
"""relay 后台看门狗：盯 queue 排空 + 抓回结果（铁律 C：后台任务 + 长间隔）。

用法:
    python tools/relay_watch.py <cids csv> [interval_sec] [max_minutes]

行为:
* 每 interval 秒打一次 health（queued / results / agent_seen 滞后）
* queued 排空后拉一次 /results，把指定 cid 的输出打出来
* 连续 3 次 queued=0 且 cid 都拿到结果 → 退出
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cloud  # noqa: E402


def main() -> int:
    cids = []
    if len(sys.argv) > 1 and sys.argv[1] not in ("-", "none"):
        cids = [int(x) for x in sys.argv[1].split(",") if x.strip()]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    max_min = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    t0 = time.time()
    deadline = t0 + max_min * 60
    idle = 0
    last_q = None
    while time.time() < deadline:
        try:
            h = cloud.req("/health", timeout=20, retries=3)
        except Exception as e:  # noqa: BLE001
            print(f"[{int(time.time()-t0):5d}s] health ERR {type(e).__name__}", flush=True)
            time.sleep(interval)
            continue
        q = h.get("queued")
        lag = int(time.time()) - int(h.get("agent_seen") or 0)
        done = []
        if cids and q == 0:
            try:
                r = cloud.req(f"/results?since={min(cids)-1}", timeout=40, retries=2)
                got = {int(x["id"]): x for x in r.get("results", [])}
                done = [c for c in cids if c in got]
            except Exception:  # noqa: BLE001
                pass
        print(f"[{int(time.time()-t0):5d}s] queued={q} results={h.get('results')} "
              f"next_id={h.get('next_id')} lag={lag}s 拿到结果={done}", flush=True)
        if q != last_q:
            print(f"        ↳ 队列变化 {last_q} → {q}", flush=True)
            last_q = q
        if cids and len(done) == len(cids):
            print("[OK] 全部 cid 结果已回传", flush=True)
            for c in done:
                x = {int(y["id"]): y for y in r.get("results", [])}[c]
                print(f"===== cid={c} rc={x.get('rc')} =====", flush=True)
                print(x.get("out") or x.get("stdout") or "", flush=True)
                e = x.get("err") or x.get("stderr") or ""
                if e.strip():
                    print("--- stderr ---", flush=True)
                    print(e[:3000], flush=True)
            return 0
        if q == 0 and not cids:
            idle += 1
            if idle >= 3:
                print("[OK] 队列已排空", flush=True)
                return 0
        else:
            idle = 0
        time.sleep(interval)
    print("[TIMEOUT] 监控超时，未满足条件", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
