"""一次性合并探针：health + 队列内容 + 最近结果（只发 3 次 HTTP，节省点数）。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud  # noqa: E402

h = cloud.req("/health", timeout=20, retries=3)
print("=== HEALTH ===")
print(json.dumps(h, ensure_ascii=False))

next_id = int(h.get("next_id") or 0)
since = max(0, next_id - 40)
res = cloud.req(f"/results?since={since}", timeout=45, retries=3)
rs = res.get("results", [])
print(f"\n=== RECENT RESULTS ({len(rs)}) ===")
for r in rs:
    cmd = (r.get("cmd") or "").replace("\n", " ")[:150]
    out = (r.get("out") or "")
    err = (r.get("err") or "")
    tail = (out.strip().splitlines() or [""])[-1][:160] if out.strip() else ""
    etail = err.strip().splitlines()[-1][:160] if err.strip() else ""
    print(f"[{r.get('id')}] rc={r.get('rc')} bg={r.get('bg')} cmd=<<{cmd}>>")
    if tail:
        print(f"    out_tail: {tail}")
    if etail:
        print(f"    err_tail: {etail}")

try:
    q = cloud.req("/poll", timeout=30, retries=2)
    cmds = q.get("cmds", [])
    print(f"\n=== QUEUE ({len(cmds)}) ===")
    for c in cmds:
        print(f"[{c.get('id')}] timeout={c.get('timeout')} bg={c.get('bg')} cmd=<<{(c.get('cmd') or '').replace(chr(10), ' ')[:220]}>>")
except Exception as e:  # noqa: BLE001
    print(f"\n!! /poll failed: {type(e).__name__}: {e}")

print(f"\nnow={time.strftime('%Y-%m-%d %H:%M:%S')}")
