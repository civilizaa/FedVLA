"""验证 agent 在"超大输出"场景下不会死锁。

复现 2026-11-03 事故：一条命令打印 200KB 字符，
agent 用 stdout=PIPE + communicate() 收集 -> 管道写满 -> 子进程阻塞
-> communicate 超时后 kill 的是 shell，python 子进程仍持有写端
-> communicate() 第二次调用永远等不到 EOF -> agent 永久卡死。

修复方案：stdout/stderr 重定向到临时文件。

本测试用真实 relay 服务端 + 真实 agent 跑一遍：
  1) 启动 agent
  2) 下发一条打印 500KB 的命令
  3) 验证 agent 仍能响应后续命令（不死锁）
  4) 验证心跳持续更新
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8899"
TOK = sys.argv[2] if len(sys.argv) > 2 else "testtoken1234567890123456789012"
SC = sys.argv[3] if len(sys.argv) > 3 else "testshare"
TPL = sys.argv[4] if len(sys.argv) > 4 else "relay/agent_template.py"

for k in list(os.environ):
    if "proxy" in k.lower():
        os.environ.pop(k, None)

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path, payload=None):
    sep = "&" if "?" in path else "?"
    url = BASE + path + f"{sep}sharecode={SC}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data is not None else "GET",
        headers={"X-Relay-Token": TOK, "Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=40) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8", "replace") or "{}")
        except Exception:
            return {}


# 渲染 agent
tpl = open(TPL, encoding="utf-8").read()
src = (tpl.replace("__RELAY_URL__", BASE)
          .replace("__RELAY_TOKEN__", TOK)
          .replace("__RELAY_SHARECODE__", SC))
agent_path = os.path.join(tempfile.gettempdir(), "_agent_stress.py")
with open(agent_path, "w", encoding="utf-8") as f:
    f.write(src)
print(f"[setup] agent 渲染到 {agent_path}")

# 启动 agent
proc = subprocess.Popen([sys.executable, agent_path],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        start_new_session=True)
print(f"[setup] agent pid = {proc.pid}")
time.sleep(4)

try:
    # 1) 心跳基线
    h = call("/health")
    base_ts = int(h.get("agent_seen") or 0)
    print(f"[1] agent_seen = {base_ts}, queued={h.get('queued')}")

    # 2) 下发超大输出命令（500KB）
    big = "python3 -c \"print('X'*500000)\""
    cid = call("/enqueue", {"cmd": big, "timeout": 60})["id"]
    print(f"[2] 入队超大输出命令 id={cid} (500KB 输出)")

    # 3) 等待结果（关键：不能卡死）
    ok = False
    t0 = time.time()
    while time.time() - t0 < 150:
        time.sleep(3)
        res = call(f"/results?since={cid-1}")["results"]
        if any(r["id"] == cid for r in res):
            r = [x for x in res if x["id"] == cid][0]
            out = r.get("out") or ""
            print(f"[3] 大命令完成 rc={r.get('rc')}, 用时 {time.time()-t0:.0f}s, "
                  f"回传 {len(out)} 字符")
            print(f"    输出尾部: ...{out[-40:].strip()}")
            ok = True
            break
    if not ok:
        print("[3] 大命令超时未完成 ❌")

    # 4) 关键验证：agent 还能响应新命令（不死锁）
    cid2 = call("/enqueue", {"cmd": "echo STILL_ALIVE", "timeout": 30})["id"]
    t0 = time.time()
    alive = False
    while time.time() - t0 < 90:
        time.sleep(2)
        res = call(f"/results?since={cid2-1}")["results"]
        if any(r["id"] == cid2 for r in res):
            r = [x for x in res if x["id"] == cid2][0]
            print(f"[4] 后续命令 rc={r.get('rc')}, out={(r.get('out') or '').strip()[:40]}")
            alive = "STILL_ALIVE" in (r.get("out") or "")
            break
    if not alive:
        print("[4] agent 卡死 ❌")

    # 5) 心跳是否持续更新
    h2 = call("/health")
    new_ts = int(h2.get("agent_seen") or 0)
    print(f"[5] agent_seen: {base_ts} -> {new_ts} (更新: {new_ts > base_ts})")

    print()
    verdict = ok and alive and new_ts > base_ts
    print("=" * 60)
    print("结论:", "PASS —— agent 在超大输出下不死锁，心跳持续" if verdict else "FAIL")
    print("=" * 60)
finally:
    try:
        proc.kill()
    except Exception:
        pass
