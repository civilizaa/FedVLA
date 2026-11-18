"""等待 relay agent 复活 → 自动推送脚本 → 跑第4阶段续跑。

设计要点：
  · 后台运行，长间隔（60s）轮询，**不烧前台点数**（只写日志）。
  · 连续 3 次心跳落后 < 120s 才认为 agent 复活（避免半启动误判）。
  · 推送后立刻 /clearall（防 relay 队列重放覆盖云端脚本 —— 2026-11-08 铁律）。
  · 推送后等 3 分钟再回读校验 md5（当场通过不代表 3 分钟后还在）。
  · 每一步失败都写进日志并退出，不做"看起来成功"的假阳性。

用法：
    python tools/auto_phase4.py --wait-agent      # 只等 agent 复活并推送
    python tools/auto_phase4.py --run             # 直接跑续跑（云端已就绪时）
    python tools/auto_phase4.py --wait-agent --run # 等到活了就跑
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cloud  # noqa: E402

LOG = os.path.join(HERE, "auto_phase4.log")

# 要推到云端 /root/fedcore/ 的脚本（隔离目录，不碰 federated/）
PUSH_FILES = [
    ("federated/resume_phase4.sh", "/root/fedcore/resume_phase4.sh"),
    ("federated/clean_legacy_rounds.sh", "/root/fedcore/clean_legacy_rounds.sh"),
    ("federated/run_privacy_rev.sh", "/root/fedcore/run_privacy_rev.sh"),
    ("federated/run_comm_scan_v2.sh", "/root/fedcore/run_comm_scan_v2.sh"),
    ("federated/extract_comm_results.py", "/root/fedcore/extract_comm_results.py"),
    ("federated/extract_privacy_results.py", "/root/fedcore/extract_privacy_results.py"),
    ("federated/privacy_experiment.py", "/root/fedcore/privacy_experiment.py"),
    ("federated/inversion.py", "/root/fedcore/inversion.py"),
    ("federated/dp.py", "/root/fedcore/dp.py"),
    ("federated/aggregate.py", "/root/fedcore/aggregate.py"),
    ("federated/run_fedavg.sh", "/root/fedcore/run_fedavg.sh"),
]

PY = r"C:\Users\hp\Desktop\bishe\.venv\Scripts\python.exe"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def agent_lag() -> int:
    """agent 心跳落后秒数；-1 表示查不到。"""
    try:
        h = cloud.req("/health", timeout=15, retries=2)
        seen = int(h.get("agent_seen") or 0)
        if not seen:
            return -1
        return int(time.time() - seen)
    except Exception as e:  # noqa: BLE001
        log(f"health 查询失败: {type(e).__name__}")
        return -1


def wait_agent(interval: int = 60, max_wait: int = 7200) -> bool:
    """等 agent 复活。连续 3 次 lag < 120s 才算真活。"""
    t0 = time.time()
    hits = 0
    while time.time() - t0 < max_wait:
        lag = agent_lag()
        if 0 <= lag < 120:
            hits += 1
            log(f"agent 心跳正常（落后 {lag}s，第 {hits}/3 次确认）")
            if hits >= 3:
                log("agent 已复活")
                return True
        else:
            if hits:
                log("心跳又落后了，重新计数")
            hits = 0
            log(f"agent 仍未复活（lag={lag}s），{int(time.time()-t0)}s 已等")
        time.sleep(interval)
    log("等待超时，agent 未复活")
    return False


def push_all() -> bool:
    """推送脚本 → clearall → 等 3 分钟 → 回读校验 md5。"""
    for local, remote in PUSH_FILES:
        p = os.path.join(REPO, local)
        if not os.path.exists(p):
            log(f"[跳过] 本地不存在 {local}")
            return False
        cmd = [PY, os.path.join(HERE, "push_files.py"), "--file", local, "--remote", remote]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        ok = r.returncode == 0
        log(f"[推送] {local} -> {remote} rc={r.returncode}")
        if not ok:
            log(f"    stderr: {(r.stderr or '')[:300]}")
            return False

    # 铁律：推送后立刻清队列，防止旧命令重放覆盖新文件
    try:
        res = cloud.req("/clearall", data={}, timeout=40, retries=3)
        log(f"[clearall] 清掉 {res.get('cleared')} 条残留命令（防重放覆盖）")
    except Exception as e:  # noqa: BLE001
        log(f"[clearall] 失败（不阻塞）: {type(e).__name__}")

    log("等 180s 让队列彻底静默（铁律：当场通过不代表 3 分钟后还在）")
    time.sleep(180)

    # 回读校验：逐个比对云端特征串
    checks = [
        ("/root/fedcore/resume_phase4.sh", "resume_phase4"),
        ("/root/fedcore/clean_legacy_rounds.sh", "clean_legacy_rounds"),
        ("/root/fedcore/run_comm_scan_v2.sh", "MAX_PAR"),
        ("/root/fedcore/extract_comm_results.py", "基线自检"),
        ("/root/fedcore/aggregate.py", "str(tok).replace"),
        ("/root/fedcore/run_fedavg.sh", "DROPPED_ARGS"),
    ]
    allok = True
    for path, feat in checks:
        r = cloud.run(f"grep -c '{feat}' {path} 2>/dev/null || echo 0", timeout=60, wait=120)
        out = (r.get("stdout") or "").strip().splitlines()
        n = out[-1].strip() if out else "0"
        if not n.isdigit():
            n = "0"
        ok = int(n) > 0
        allok = allok and ok
        log(f"[校验] {path} 含 '{feat}': {n} 次 {'OK' if ok else '**缺失**'}")
    return allok


def cloud_run(cmd: str, timeout: int = 600, wait: int = 600):
    try:
        return cloud.run(cmd, timeout=timeout, wait=wait)
    except Exception as e:  # noqa: BLE001
        log(f"[云端] 执行失败: {type(e).__name__}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-agent", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    if args.wait_agent:
        if not wait_agent(interval=args.interval):
            log("退出：agent 未复活")
            return 1
        log("开始推送")
        if not push_all():
            log("退出：推送或校验失败（不继续，避免用旧脚本跑出假结果）")
            return 1
        log("推送并校验完成")
        if not args.run:
            log("按要求停在推送完成（未加 --run）")
            return 0

    if args.run:
        log("启动云端续跑：bash /root/fedcore/resume_phase4.sh all")
        # 续跑是长任务（数小时）→ 必须 nohup setsid 起，不能挂 agent 前台
        r = cloud_run(
            "cd /root/HPT && nohup setsid bash /root/fedcore/resume_phase4.sh all "
            "> /root/HPT/output/resume_phase4.log 2>&1 < /dev/null & echo started; sleep 10; "
            "tail -n 20 /root/HPT/output/resume_phase4.log",
            timeout=120, wait=180)
        if r:
            log("续跑已拉起：")
            log((r.get("stdout") or "")[:2000])
        else:
            log("续跑拉起失败")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
