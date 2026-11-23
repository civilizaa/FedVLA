#!/usr/bin/env python3
"""推送第 5 阶段「磁盘管理 + A/B 重建 + 重跑编排」补丁到云端并回读 md5（铁律 B）。

本轮推送 4 个文件：
  1. federated/run_phase5.sh   ← 新增 cleanup_completed_config（配置成功后回收中间产物）
  2. tools/phase5_restart.sh   ← 清理挪到磁盘闸之前 + 阈值 12G→8G + 复制第3阶段评估日志
  3. tools/phase5_autostart.sh ← A/B 比对改用 task3 + 新增「非退化性检查」
  4. tools/phase5_ab_run.sh    ← 新建：A/B 驱动（p3_moe_fedavg + task3，非退化）

刻意**不推** eval_model.sh（云端那份是第 4 阶段实战版，OMP_NUM_THREADS=8）。
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import phase5_remote as pr  # noqa: E402

FILES = [
    ("federated/run_phase5.sh", "/root/fedcore/run_phase5.sh"),
    ("tools/phase5_restart.sh", "/root/fedcore/phase5_restart.sh"),
    ("tools/phase5_autostart.sh", "/root/fedcore/phase5_autostart.sh"),
    ("tools/phase5_ab_run.sh", "/root/fedcore/phase5_ab_run.sh"),
]


def local_md5_lf(path: str) -> str:
    raw = open(path, "rb").read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.md5(raw).hexdigest()


def main() -> int:
    r = pr.Remote()
    expect = {}
    for rel, remote in FILES:
        path = os.path.join(ROOT, rel)
        content = open(path, encoding="utf-8").read()
        r.push_text(content, remote, label=rel)
        expect[remote] = local_md5_lf(path)
        print("[push] %s -> %s" % (rel, remote))
    print("--- 回读 md5 校验（铁律 B：推送后必须回读）---")
    got = r.md5_remote([remote for _, remote in FILES], clear_first=False)
    bad = 0
    for remote, exp in expect.items():
        g = got.get(remote, "(缺失)")
        flag = "OK " if g == exp else "!! "
        if g != exp:
            bad += 1
        print("  %s%-42s %s (期望 %s)" % (flag, remote, g, exp))
    print("[结果] %d/%d 一致" % (len(expect) - bad, len(expect)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
