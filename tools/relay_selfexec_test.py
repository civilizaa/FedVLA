#!/usr/bin/env python3
"""relay 自推链路验证：云端命令自己把输出 POST 到高 id 段。

## 为什么需要这个脚本（2026-11-23 实测）

relay 服务端 `POST /result` 只保留**按 int(id) 排序后的末尾 500 条**：

```python
if len(_state["results"]) > 500:
    for k in sorted(_state["results"], key=lambda x: int(x))[:-500]:
        _state["results"].pop(k, None)
```

第 4 阶段用 idbase 段（980278~1000374）把存储占满后，`/enqueue` 分配到的
**普通小 id**（如 3639）结果**一写入就被淘汰** ⇒ `/poll` 的
`done = set(results.keys())` 永远不含它 ⇒ **每轮 poll 重新派发** ⇒
agent 重复执行、串行通道被堵死（实测：queued 恒为 1，命令永不返回）。

⇒ 解法：**让云端自己用高 id 段写结果**（第 4 阶段 `relay_push_out.py` 同款机制），
本地改从 `/results?since=<idbase>` 读，彻底绕开小 id 被淘汰的缺陷。

用法:
    python tools/relay_selfexec_test.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cloud  # noqa: E402

CID = 1010000
GUARD = "/root/HPT/.probe_once"

# 云端执行的命令：
#   1) 一次性守卫（服务端会重派未完成的命令，靠守卫文件保证只真正执行一次）
#   2) 采集环境快照
#   3) 自己 POST /result（id=1010000，高 id 段 ⇒ 不会被淘汰）
CLOUD_CMD = f'''
if [ -f {GUARD} ]; then echo '[skip] already executed'; exit 0; fi
touch {GUARD}
python3 - <<'PYX'
import json, re, subprocess, urllib.request

src = open('/root/HPT/agent.py', encoding='utf-8', errors='ignore').read()

def grab(name, default=''):
    m = re.search(name + r'\\s*=\\s*"([^"]*)"', src)
    return m.group(1) if m else default

tok = grab('RELAY_TOKEN')
sc  = grab('RELAY_SHARECODE')
url = grab('RELAY_URL', 'https://fedvla-relay.app.workbuddy.host').rstrip('/')

probe = r"""
echo '=== [1] agent.py 位置（铁律 P）==='
ls -la /root/HPT/agent.py /root/fedvla_relay/agent.py 2>&1
echo '=== [2] agent 进程 ==='
ps -eo pid,etime,cmd | grep '[a]gent.py' | head -5
echo '=== [3] /root/HPT 顶层 ==='
ls -la /root/HPT 2>&1 | head -30
echo '=== [4] /root/HPT/federated ==='
ls -la /root/HPT/federated 2>&1 | head -40
echo '=== [5] GPU ==='
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | head -5
echo '=== [6] 磁盘 ==='
df -h / 2>&1 | tail -2
echo '=== [7] conda 环境 ==='
ls /root/miniconda3/envs/ 2>&1
echo '=== [8] 遗留训练/评估进程 ==='
ps -eo pid,etime,cmd | grep -E '[r]un_comm_scan|[r]un_fedavg|[h]pt\\.run|[e]val_model|[r]un_phase5' | head -10
echo '=== [9] output 目录 ==='
ls -la /root/HPT/output 2>&1 | head -25
echo '=== [10] 第3阶段产物（要复用）==='
ls -d /root/HPT/output/*/ 2>/dev/null | head -25
echo '=== [11] HPT 源码 ==='
ls /root/HPT/hpt/ 2>&1 | head -20
echo '=== [12] train_test.py 锚点现状 ==='
grep -n 'domain_loss.backward()' /root/HPT/hpt/train_test.py 2>&1 | head -5
grep -c 'apply_baseline_hook' /root/HPT/hpt/train_test.py 2>&1 | head -2
echo '=== [13] import hpt 硬拦截 ==='
cd /root/HPT && /root/miniconda3/envs/hpt/bin/python -c "import hpt; print('[OK] import hpt', hpt.__file__)" 2>&1 | tail -3
echo '=== [14] 凭据自检（只报长度，不打印明文）==='
echo "token_len={{len(tok)}} sharecode_len={{len(sc)}} url={{url}}"
echo '=== [PROBE-END] ==='
"""
try:
    p = subprocess.run(['bash', '-lc', probe], capture_output=True, text=True, timeout=180)
    out, err, rc = p.stdout, p.stderr, p.returncode
except Exception as e:
    out, err, rc = '', repr(e), -1

if not tok or not sc:
    err += f'\n[FAIL] 没读到凭据 token_len={{len(tok)}} sc_len={{len(sc)}}'
    rc = rc or -2

body = json.dumps({{'id': {CID}, 'rc': rc, 'out': out[-190000:], 'err': err[-20000:]}},
                 ensure_ascii=False).encode('utf-8')
req = urllib.request.Request(url + '/result?sharecode=' + sc, data=body, method='POST')
req.add_header('X-Relay-Token', tok)
req.add_header('Content-Type', 'application/json; charset=utf-8')
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        print('POSTED', r.status, r.read().decode('utf-8')[:200])
except Exception as e:
    print('POST FAILED', repr(e))
PYX
'''


def main() -> int:
    print("== enqueue 自推验证命令 ==")
    cid = None
    for attempt in range(8):
        try:
            r = cloud.req("/enqueue", data={"cmd": CLOUD_CMD, "cwd": "/root/HPT",
                                            "timeout": 300, "bg": False},
                          timeout=120, retries=1)
            cid = r["id"]
            print(f"  入队成功 cid={cid} (attempt {attempt+1})")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  入队失败(403 是网关限流) attempt {attempt+1}: {type(e).__name__}")
            time.sleep(min(2 ** attempt, 8))
    if cid is None:
        print("[FATAL] 入队失败")
        return 1

    print("== 轮询高 id 段结果 ==")
    t0 = time.time()
    while time.time() - t0 < 420:
        try:
            r = cloud.req(f"/results?since={CID - 1}", timeout=40, retries=2)
        except Exception:  # noqa: BLE001
            time.sleep(4)
            continue
        for x in r.get("results", []):
            if int(x["id"]) == CID:
                print(f"[OK] 拿到自推结果 rc={x.get('rc')}")
                print("--- out ---")
                print(x.get("out") or "")
                e = x.get("err") or ""
                if e.strip():
                    print("--- err ---")
                    print(e[:3000])
                # 清掉这条命令（服务端会因结果被小 id 淘汰而重派）
                try:
                    cloud.req("/clearall", data={"before": 0}, timeout=40, retries=2)
                    print("[清理] 已 clearall")
                except Exception:  # noqa: BLE001
                    pass
                return 0
        print(f"  [{int(time.time()-t0)}s] 还没到，继续等 ...")
        time.sleep(6)
    print("[TIMEOUT] 420s 内没拿到自推结果")
    return 2


if __name__ == "__main__":
    sys.exit(main())
