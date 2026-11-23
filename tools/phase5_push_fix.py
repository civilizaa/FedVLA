#!/usr/bin/env python3
"""推送第 5 阶段修复文件到云端并回读 md5 校验（铁律 B）。

刻意**不推** eval_model.sh：云端那份是第 4 阶段实战版（OMP_NUM_THREADS=8），
本地那份差一行；让云端原地打 M1 补丁，既补上 MoE 环境变量，又保住与
第 3/4 阶段评估口径一致的线程设置。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import phase5_remote as pr  # noqa: E402

FILES = [
    ("federated/patch_hpt_fedbaseline.py", "/root/fedcore/patch_hpt_fedbaseline.py"),
    ("federated/fedvla_baseline_hook.py", "/root/fedcore/fedvla_baseline_hook.py"),
    ("federated/run_fedavg.sh", "/root/fedcore/run_fedavg.sh"),
    ("federated/phase5_autoshutdown.sh", "/root/fedcore/phase5_autoshutdown.sh"),
    ("federated/extract_phase5_results.py", "/root/fedcore/extract_phase5_results.py"),
    ("tools/fix_phase5_eval_moe.sh", "/root/fedcore/fix_phase5_eval_moe.sh"),
]


def main() -> int:
    r = pr.Remote()
    for rel, remote in FILES:
        path = os.path.join(ROOT, rel)
        content = open(path, encoding="utf-8").read()
        r.push_text(content, remote, label=rel)
        print("[push] %s -> %s" % (rel, remote))
    print("--- 回读 md5 校验（铁律 B）---")
    ok = r.md5_remote([remote for _, remote in FILES], clear_first=False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
