#!/usr/bin/env python3
"""第 5 阶段自举：一条命令完成「写入 relay_bootstrap.py + 立刻用它执行探查」。

用法:
    python tools/phase5_bootstrap.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import phase5_remote as pr  # noqa: E402

BOOT = os.path.join(ROOT, "federated", "relay_bootstrap.py")

# 探查命令（尽量短；长逻辑应写成文件再执行）
PROBE = r"""
echo '=== [1] agent.py 位置（铁律 P：必须 /root/HPT/agent.py）==='
ls -la /root/HPT/agent.py /root/fedvla_relay/agent.py 2>&1 | head -5
echo '=== [2] agent 进程 ==='
ps -eo pid,etime,cmd | grep '[a]gent.py' | head -5
echo '=== [3] /root/fedcore 内容（第 4 阶段遗留）==='
ls /root/fedcore/ 2>&1 | head -40
echo '=== [4] GPU ==='
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | head -3
echo '=== [5] 磁盘 ==='
df -h / 2>&1 | tail -2
echo '=== [6] conda 环境 ==='
ls /root/miniconda3/envs/ 2>&1
echo '=== [7] 遗留训练/评估进程（不该有）==='
ps -eo pid,etime,cmd | grep -E '[h]pt\.run|[r]un_|[e]val_model' | head -8
echo '=== [8] output 目录（第3/4阶段产物是否还在）==='
ls /root/HPT/output/ 2>&1 | head -25
echo '=== [9] phase4 产物 ==='
ls /root/HPT/output/phase4/ 2>&1 | head -15
echo '=== [10] HPT 源码 ==='
ls /root/HPT/hpt/ 2>&1 | head -15
echo '=== [11] train_test.py 锚点现状（0 = 未打基线补丁）==='
grep -c 'apply_baseline_hook' /root/HPT/hpt/train_test.py 2>&1 | head -2
echo '=== [12] import hpt 硬拦截（铁律 F/G）==='
cd /root/HPT && /root/miniconda3/envs/hpt/bin/python -c "import hpt; print('[OK] import hpt')" 2>&1 | tail -2
echo '=== [13] relay 凭据自检（只报长度）==='
python3 - <<'PYX'
import re
s = open('/root/HPT/agent.py', encoding='utf-8', errors='ignore').read()
g = lambda n: (re.search(n + r'\s*=\s*"([^"]*)"', s) or [None,''])[1]
print('token_len=%d sharecode_len=%d' % (len(g('RELAY_TOKEN')), len(g('RELAY_SHARECODE'))))
PYX
echo '=== [PROBE-END] ==='
"""


def main() -> int:
    print("relay health:", pr.cloud.req("/health", timeout=20, retries=3))
    remote = pr.Remote()
    print(f"自举：写入 {pr.CLOUD_DIR}/relay_bootstrap.py 并立刻执行探查 …")
    rc, out, err = remote.bootstrap(BOOT, PROBE)
    print(f"--- rc={rc} ---")
    print(out)
    if (err or "").strip():
        print("--- err ---")
        print(err[:2000])
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
