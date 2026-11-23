#!/usr/bin/env python3
"""探查 2：确认第 3 阶段可复用产物 + 第 5 阶段依赖脚本是否齐全。"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import phase5_remote as pr  # noqa: E402

PROBE = r"""
echo '=== [A] agent.py 两处位置 ==='
ls -la /root/HPT/agent.py /root/fedvla_relay/agent.py 2>&1 | head -5
echo '=== [B] agent 进程 ==='
ps -eo pid,etime,cmd | grep '[a]gent.py' | head -3
echo '=== [C] /root/fedcore 依赖脚本检查 ==='
for f in run_fedavg.sh eval_model.sh fedavg_config.sh patch_inline.sh \
         patch_hpt_load_trunk.py patch_hpt_global_lr.py patch_hpt_moe.py \
         relay_push_out.py aggregate.py partition.py run_comm_scan.sh \
         resume_all.sh; do
  if [ -e "/root/fedcore/$f" ]; then echo "OK   $f"; else echo "MISS $f"; fi
done
echo '=== [D] /root/fedcore 全部 .sh/.py（排除临时 _*.txt）==='
ls /root/fedcore/*.sh /root/fedcore/*.py 2>/dev/null | head -40
echo '=== [E] 第3阶段产物目录（要复用）==='
ls /root/HPT/output/ 2>/dev/null | grep -i -E 'p3|moe|fedavg|eda' | head -30
echo '=== [F] output 下所有目录（前 40）==='
ls -d /root/HPT/output/*/ 2>/dev/null | head -40
echo '=== [G] 第3阶段各配置目录内容（找 trunk.pth + eval 日志）==='
for d in /root/HPT/output/*moe* /root/HPT/output/*p3*; do
  [ -d "$d" ] || continue
  echo "--- $d"; ls "$d" 2>/dev/null | head -6
done
echo '=== [H] HPT utils 已打补丁的文件 ==='
ls /root/HPT/hpt/utils/ 2>/dev/null | head -25
echo '=== [I] HPT models 补丁状态 ==='
ls /root/HPT/hpt/models/ 2>/dev/null | head -15
echo '=== [J] 输出根占用 ==='
du -sh /root/HPT/output 2>/dev/null | tail -2
echo '=== [PROBE2-END] ==='
"""


def main() -> int:
    remote = pr.Remote()
    rc, out, err = remote.run(PROBE, timeout=300)
    print(f"--- rc={rc} ---")
    print(out)
    if (err or "").strip():
        print("--- err ---")
        print(err[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
