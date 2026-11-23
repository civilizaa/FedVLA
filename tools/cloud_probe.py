#!/usr/bin/env python3
"""云端只读探查：一次拿齐环境状态（铁律 C：轮询必须合并，禁止每配置一次调用）。

只发**只读**命令，不改任何东西。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import push_files as pf  # noqa: E402

PROBE = r"""
echo '=== [1] agent.py 位置（铁律 P：必须在 /root/HPT/agent.py）==='
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
echo '=== [8] 遗留训练/评估进程（铁律 Q：不该有残留）==='
ps -eo pid,etime,cmd | grep -E '[r]un_comm_scan|[r]un_fedavg|[h]pt\.run|[e]val_model|[r]un_phase5' | head -10
echo '=== [9] output 目录 ==='
ls -la /root/HPT/output 2>&1 | head -20
echo '=== [10] 第3阶段产物是否还在（要复用）==='
ls -d /root/HPT/output/*/ 2>/dev/null | head -20
echo '=== [11] HPT 源码 ==='
ls /root/HPT/hpt/ 2>&1 | head -20
echo '=== [12] train_test.py 锚点现状 ==='
grep -n 'domain_loss.backward()' /root/HPT/hpt/train_test.py 2>&1 | head -5
grep -c 'apply_baseline_hook' /root/HPT/hpt/train_test.py 2>&1 | head -2
echo '=== [13] hpt python 可用性（铁律 F/G：import hpt 硬拦截）==='
cd /root/HPT && /root/miniconda3/envs/hpt/bin/python -c "import hpt, sys; print('[OK] import hpt', hpt.__file__)" 2>&1 | tail -3
echo '=== [PROBE-END] ==='
"""


def main() -> int:
    print("relay health:", pf.health())
    # 铁律 B：查状态前先清残留命令
    try:
        print("clearall:", pf.req("/clearall", data={"n": 0}, timeout=30, retries=3))
    except Exception as e:  # noqa: BLE001
        print("clearall 失败（不致命）:", e)

    cid, st = pf.enqueue(PROBE, timeout=300)
    print("enqueue:", cid, st)
    if cid is None:
        return 1
    res = pf.wait_results([cid], timeout=600).get(cid, {})
    print("rc =", res.get("rc"))
    print("--- out ---")
    print(res.get("out") or "")
    err = res.get("err") or ""
    if err.strip():
        print("--- err ---")
        print(err[-2000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
