#!/usr/bin/env python3
"""relay WAF 阶梯测试：定位 /enqueue 的命令长度阈值（只看 HTTP 码，不等结果）。

背景（2026-11-23）：2.7KB 的命令 8 次重试全 403，而 70 字节的命令能入队。
记忆里的历史结论：`REFERENCE.md` 记着「WAF 拦 1KB~12KB 纯文本命令；带 sharecode 后
100KB body 放行；命令里出现裸 `base64 -d` 必 403（`b64decode` 不触发）」。

本脚本对同一外壳、仅改变 b64 载荷长度的若干档位各试 N 次：
  全部失败 ⇒ 该档被 WAF 拦；偶尔成功 ⇒ 只是随机限流。
"""
from __future__ import annotations

import base64
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cloud  # noqa: E402

SHELL = ('python3 -c "import base64;open(\'/tmp/waf_probe\',\'ab\')'
         '.write(base64.b64decode(\'{B64}\'))"')

TRIES = 6


def trial(payload_len: int, label: str) -> tuple[int, int]:
    """返回 (成功次数, 尝试次数)。"""
    blob = base64.b64encode(b"A" * payload_len).decode("ascii")
    cmd = SHELL.replace("{B64}", blob)
    ok = 0
    for i in range(TRIES):
        try:
            r = cloud.req("/enqueue",
                          data={"cmd": cmd, "cwd": "/root/HPT",
                                "timeout": 60, "bg": False},
                          timeout=60, retries=1)
            ok += 1
            print(f"    [OK] {label} payload={payload_len}B cmd={len(cmd)}B cid={r['id']}")
            break
        except Exception as e:  # noqa: BLE001
            last = type(e).__name__
        time.sleep(min(2 ** i, 6))
    if ok == 0:
        print(f"    [403x{TRIES}] {label} payload={payload_len}B cmd={len(cmd)}B 全被拦")
    return ok, TRIES


def main() -> int:
    print("relay health:", cloud.req("/health", timeout=20, retries=3))
    # 干净起点
    try:
        cloud.req("/clearall", data={"before": 0}, timeout=40, retries=2)
    except Exception:  # noqa: BLE001
        pass

    ladder = [
        (0, "L0 纯短命令"),
        (200, "L1 b64 200B"),
        (800, "L2 b64 800B"),
        (2000, "L3 b64 2KB"),
        (6000, "L4 b64 6KB"),
        (16000, "L5 b64 16KB"),
    ]
    results = []
    for n, label in ladder:
        print(f"\n--- {label} ---")
        ok, tot = trial(n, label)
        results.append((label, n, ok, tot))
        time.sleep(1)

    print("\n===== 阶梯结论 =====")
    for label, n, ok, tot in results:
        verdict = "放行" if ok else "WAF 拦截"
        print(f"  {label:<18} b64载荷={n:>6}B  成功{ok}/{tot}  ⇒ {verdict}")

    try:
        cloud.req("/clearall", data={"before": 0}, timeout=40, retries=2)
        print("\n[清理] 已 clearall")
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
