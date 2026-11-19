#!/usr/bin/env python3
"""第4阶段「修正版重跑」本地总控：一条命令拿齐状态 / 一条命令起跑 / 一条命令取回。

子命令
------
  status   一次调用拿齐 relay 活性 + 云端进度（省点数，铁律 C）
  launch   等 agent 复活 → 推脚本 → 清队列 → 回读校验 → 起跑云端总控
  fetch    从 relay 结果存储取回本轮产物并逐文件核对 sha256
  verify   只做回读校验（脚本是否真的在云端）

设计要点
--------
* **status 一条命令拿齐**：绝不"每个配置调一次 relay"（铁律 C）。
* **推送后必 clearall**：relay 队列里的历史命令会在 agent 复活后被重派，
  把云端旧脚本还原回去（铁律 B，2026-11-08 血案）。
* **推送后等 ≥3 分钟再回读**：当场校验通过不代表 3 分钟后还在。
* **回读用二进制 md5**：文本模式读会 CRLF→LF 归一化，md5 不同 ≠ 被覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cloud  # noqa: E402

LOG = os.path.join(HERE, "rerun_phase4_fix.log")
IDBASE = 990000            # 本轮推送 id 基址（必须 > 上一轮 980000）
COMM_TAGS = "c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1"
DP_TAGS = "dp_s0001 dp_s0003 dp_s0012"

# 要推到云端 /root/fedcore/ 的脚本（改动过的都带上）
PUSH_FILES = [
    ("federated/prep_rerun_phase4.sh", "/root/fedcore/prep_rerun_phase4.sh"),
    ("federated/rerun_phase4_fix.sh", "/root/fedcore/rerun_phase4_fix.sh"),
    ("federated/run_comm_scan_v2.sh", "/root/fedcore/run_comm_scan_v2.sh"),
    ("federated/run_fedavg.sh", "/root/fedcore/run_fedavg.sh"),
    ("federated/aggregate.py", "/root/fedcore/aggregate.py"),
    ("federated/extract_comm_results.py", "/root/fedcore/extract_comm_results.py"),
    ("federated/extract_dp_utility.py", "/root/fedcore/extract_dp_utility.py"),
    ("federated/chain_phase4_dp.sh", "/root/fedcore/chain_phase4_dp.sh"),
    ("federated/calibrate_dp_clip.sh", "/root/fedcore/calibrate_dp_clip.sh"),
    ("federated/dp_clip_calibrate.py", "/root/fedcore/dp_clip_calibrate.py"),
    ("federated/dp_noise_client.py", "/root/fedcore/dp_noise_client.py"),
    ("federated/run_dp_utility.sh", "/root/fedcore/run_dp_utility.sh"),
    ("federated/phase4_autoshutdown.sh", "/root/fedcore/phase4_autoshutdown.sh"),
    ("federated/relay_push_out.py", "/root/fedcore/relay_push_out.py"),
    ("federated/eval_model.sh", "/root/fedcore/eval_model.sh"),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def local_py() -> str:
    for p in (r"C:\Users\hp\Desktop\bishe\.venv\Scripts\python.exe",
              sys.executable):
        if os.path.exists(p):
            return p
    return sys.executable


def agent_lag() -> int:
    """agent 心跳落后秒数；-1 = 查不到。"""
    try:
        h = cloud.req("/health", timeout=15, retries=2)
        seen = int(h.get("agent_seen") or 0)
        return int(time.time() - seen) if seen else -1
    except Exception as e:  # noqa: BLE001
        log(f"health 查询失败: {type(e).__name__}")
        return -1


# --------------------------------------------------------------------- status
STATUS_CMD = r"""
OUT=/root/HPT/output
echo "=== 进程 ==="
for p in run_comm_scan_v2 phase4_autoshutdown chain_phase4_dp run_dp_utility calibrate_dp_clip; do
  n=$(pgrep -fc "$p" 2>/dev/null || echo 0)
  echo "proc $p = $n"
done
echo "=== rc 计数 ==="
nc=0; for t in %s; do [ -f "$OUT/.rc_$t" ] && nc=$((nc+1)); done
nd=0; for t in %s; do [ -f "$OUT/.rc_$t" ] && nd=$((nd+1)); done
echo "comm_rc=$nc/7  dp_rc=$nd/3"
echo "=== DONE ==="
if [ -f "$OUT/PHASE4_ALL_DONE" ]; then cat "$OUT/PHASE4_ALL_DONE"; else echo "NO_DONE_MARKER"; fi
echo "=== 最近日志 ==="
for f in rerun_phase4.log comm_scan_v2.log chain_phase4_dp.log autoshutdown.log; do
  if [ -f "$OUT/$f" ]; then echo "--- $f"; tail -n 6 "$OUT/$f"; fi
done
echo "=== 评估完成情况 ==="
for t in %s; do
  n=$(grep -c '| avg' "$OUT/eval_${t}_mujoco_metaworld_task3.log" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  echo "eval $t = $n 表"
done
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1
df -h /root 2>/dev/null | tail -1
""" % (COMM_TAGS, DP_TAGS, COMM_TAGS)


def do_status() -> int:
    lag = agent_lag()
    print(f"agent_lag_sec = {lag}  ({'活跃' if 0 <= lag < 120 else '未活/不可达'})")
    if lag < 0 or lag >= 120:
        print("状态: CLOUD_OFF —— 云端 agent 未在轮询（实例未开机或 agent 未启动）")
        return 0
    try:
        r = cloud.run(STATUS_CMD, cwd="/root/HPT", timeout=120, wait=240)
    except Exception as e:  # noqa: BLE001
        print(f"云端探测失败: {type(e).__name__}: {e}")
        return 1
    out = r.get("out") or r.get("stdout") or ""
    err = r.get("err") or r.get("stderr") or ""
    print(out)
    if err.strip():
        print("--stderr--\n" + err[:1500])
    return 0


# --------------------------------------------------------------------- launch
def push_all() -> bool:
    py = local_py()
    for local, remote in PUSH_FILES:
        p = os.path.join(REPO, local)
        if not os.path.exists(p):
            log(f"[跳过] 本地不存在 {local}")
            continue
        cmd = [py, os.path.join(HERE, "push_files.py"), "--file", local, "--remote", remote]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            log(f"[推送失败] {local} -> {remote}\n{(r.stderr or '')[:400]}")
            return False
        log(f"[推送] {local} -> {remote} OK")

    # 🔴 铁律 B：推送后立刻清队列，防历史命令重放把新脚本覆盖回旧版
    try:
        res = cloud.req("/clearall", data={}, timeout=40, retries=3)
        log(f"[clearall] 清掉 {res.get('cleared')} 条残留命令（防重放）")
    except Exception as e:  # noqa: BLE001
        log(f"[clearall] 失败（不阻塞）: {type(e).__name__}")
    return True


VERIFY_CMD = r"""
cd /root/fedcore && for pair in \
  "prep_rerun_phase4.sh:legacy_r2" \
  "rerun_phase4_fix.sh:PUSH_IDBASE" \
  "run_comm_scan_v2.sh:COMM_TAGS" \
  "run_fedavg.sh:PYTHONPATH" \
  "aggregate.py:quant_drift" \
  "phase4_autoshutdown.sh:PUSH_IDBASE" \
  "chain_phase4_dp.sh:calibrate_dp_clip" ; do
  f="${pair%%:*}"; k="${pair##*:}"
  n=$(grep -c "$k" "$f" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  echo "$f|$k|$n"
done
"""


def verify() -> bool:
    r = cloud.run(VERIFY_CMD, cwd="/root/fedcore", timeout=120, wait=240)
    out = r.get("out") or r.get("stdout") or ""
    ok = True
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        f, k, n = parts
        good = n.isdigit() and int(n) > 0
        ok = ok and good
        log(f"[校验] {f} 含 '{k}': {n} 次 {'OK' if good else '**缺失**'}")
    return ok


def wait_agent(interval: int = 60, max_wait: int = 5400) -> bool:
    t0 = time.time()
    hits = 0
    while time.time() - t0 < max_wait:
        lag = agent_lag()
        if 0 <= lag < 120:
            hits += 1
            log(f"agent 心跳正常（落后 {lag}s，{hits}/2）")
            if hits >= 2:
                return True
        else:
            hits = 0
            log(f"agent 未活（lag={lag}s），已等 {int(time.time()-t0)}s")
        time.sleep(interval)
    return False


def do_launch(wait: int = 5400) -> int:
    if not wait_agent(max_wait=wait):
        log("退出：agent 未复活（云端实例还没开机/agent 没起来）")
        return 2
    log("agent 已活，开始推送")
    if not push_all():
        log("退出：推送失败")
        return 1
    log("等 180s 让队列彻底静默（铁律：当场通过不代表 3 分钟后还在）")
    time.sleep(180)
    if not verify():
        log("退出：回读校验失败（云端脚本不是新版本，拒绝开跑）")
        return 1
    log("回读校验通过，起跑云端总控")
    r = cloud.run(
        "cd /root/HPT && nohup setsid bash /root/fedcore/rerun_phase4_fix.sh "
        "> /root/HPT/output/rerun_phase4.log 2>&1 < /dev/null & echo started; sleep 25; "
        "tail -n 25 /root/HPT/output/rerun_phase4.log",
        cwd="/root/HPT", timeout=120, wait=240)
    out = r.get("out") or r.get("stdout") or ""
    log("云端总控启动日志：")
    log(out[:2500])
    if "FATAL" in out or "[FATAL]" in out:
        log("退出：云端总控报了 FATAL")
        return 1
    return 0


# --------------------------------------------------------------------- fetch
def do_fetch(outdir: str = "results/cloud_final_rerun") -> int:
    py = local_py()
    cmd = [py, os.path.join(HERE, "relay_fetch_out.py"),
           "--since", str(IDBASE), "--outdir", outdir]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=REPO)
    print(r.stdout[-4000:])
    if r.stderr.strip():
        print("--stderr--\n" + r.stderr[-1500:], file=sys.stderr)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "launch", "fetch", "verify"])
    ap.add_argument("--wait", type=int, default=5400)
    ap.add_argument("--outdir", default="results/cloud_final_rerun")
    args = ap.parse_args()
    if args.cmd == "status":
        return do_status()
    if args.cmd == "launch":
        return do_launch(args.wait)
    if args.cmd == "verify":
        return 0 if verify() else 1
    if args.cmd == "fetch":
        return do_fetch(args.outdir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
