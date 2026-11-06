#!/usr/bin/env python3
"""云端 agent 复活后自动接管：推文件 → 校验 → 启动实验。

## 为什么需要这个
2026-11-05 实测：云端 AutoDL 实例被关机（agent 心跳落后 13.5h）。
用户在 AutoDL 终端粘贴 bootstrap 命令后，agent 会复活 —— 但此时：
- 云端 `federated/` 目录里的脚本可能是旧版本（甚至残缺的，
  11-04 的分块推送漏了 `eda.py.part002`）
- 门2' 消融与第 3 阶段都还没启动

本脚本在本地**轮询等待 agent 复活**，一旦探针成功就自动完成剩余全部工作，
用户不需要再回到对话里。

## 流程
1. 轮询 `/health` 直到 `agent_seen` 落后 < 3 分钟（或发探针拿到结果）
2. `push_files.py --dir federated`（整目录 base64 tar 包推送，一条命令）
3. 逐个文件 md5 校验，不一致的重推
4. 下发 `bash /root/HPT/federated/resume_all.sh`（后台跑，日志落盘）
5. 定期轮询日志，汇报进度

## 用法
    python relay/auto_resume.py            # 一直等到成功
    python relay/auto_resume.py --timeout 7200
    python relay/auto_resume.py --status   # 只看当前状态
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)


def _find_federated():
    """定位 federated/ 目录：兼容两种布局
    (a) 仓库内 tools/ -> ../federated
    (b) 工作区 relay/ -> ../fedvla-repo/federated
    """
    for cand in (os.path.join(WS, "federated"),
                 os.path.join(WS, "fedvla-repo", "federated"),
                 os.path.join(os.path.dirname(WS), "fedvla-repo", "federated")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(WS, "federated")


FED = _find_federated()

_spec = importlib.util.spec_from_file_location("pf", os.path.join(HERE, "push_files.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def agent_lag_min():
    """agent 心跳落后本机多少分钟。返回 None 表示拿不到。"""
    try:
        h = pf.health()
    except Exception:
        return None
    seen = h.get("agent_seen")
    if not seen:
        return None
    return (time.time() - int(seen)) / 60.0


def probe_alive(timeout=100):
    """发一条探针命令看 agent 是否真的在执行。

    ⚠️ 必须靠**探针回传内容**判定，不能只看心跳：
    2026-11-05 本机测试 bootstrap 时短暂起过一个 agent，它刷新了心跳时间戳
    （agent_seen 变新鲜），但进程随即退出。只看心跳会误判"云端已复活"。

    探针让 agent 执行 `hostname`，回传里应含云端实例特征串。
    """
    try:
        cid, st = pf.enqueue(
            "echo PROBE_HOST=$(hostname) PROBE_PWD=$(pwd) PROBE_MARK",
            timeout=30)
    except Exception as e:
        log(f"  探针入队失败: {e}")
        return False, ""
    if cid is None:
        return False, ""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            res = pf.req(f"/results?since={cid - 1}", timeout=40)
        except Exception:
            time.sleep(4)
            continue
        for r in res.get("results", []):
            if int(r["id"]) == cid:
                out = r.get("out") or ""
                return True, out
        time.sleep(4)
    return False, ""


def wait_agent(max_wait=7200, poll=30):
    """等 agent 复活。返回 (ok, probe_output)。"""
    log(f"等待云端 agent 复活（最多 {max_wait/60:.0f} 分钟）...")
    t0 = time.time()
    last_lag = None
    while time.time() - t0 < max_wait:
        lag = agent_lag_min()
        if lag is not None and lag != last_lag:
            log(f"  agent_seen 落后 {lag:.1f} 分钟")
            last_lag = lag
        # 心跳新鲜 < 3 分钟 -> 再用探针确认它真的在执行命令
        if lag is not None and lag < 3:
            log("  心跳已新鲜，发探针确认...")
            ok, out = probe_alive()
            if ok:
                log(f"  ✓ agent 已复活并能执行命令: {out.strip()[:120]}")
                return True, out
            log("  探针未回传，继续等")
        time.sleep(poll)
    log("!! 等待超时")
    return False, ""


def push_all():
    """逐文件推送 + md5 校验。返回 True/False。

    刻意**不再用整目录 tar 推送**（2026-11-05 教训）：
    整目录打包失败时会在云端留下 60+ 个 *.part000.h0.h1... 残骸，
    残骸会**掩盖真实文件的缺失**（ls 看起来文件很多，关键的可能不在），
    而且 tar 一次性覆盖，无法逐个回读校验。
    逐文件推送每个都有独立回读校验，失败能定位到具体文件。
    """
    log("推送 federated/ 关键文件（逐文件，带回读校验）...")
    pf.wait_queue_drain(max_wait=120)
    pf.cleanup_parts("/root/HPT/federated")
    n_ok = n_bad = 0
    for local, remote in pf.DEFAULT_FILES:
        cands = [os.path.join(FED, local), local]
        path = next((p for p in cands if os.path.exists(p)), None)
        if not path:
            log(f"  [MISS] {local}（本地不存在，跳过）")
            continue
        if pf.push_file(path, remote):
            n_ok += 1
        else:
            n_bad += 1
            log(f"  [FAIL] {local}")
    log(f"  推送完成：成功 {n_ok}，失败 {n_bad}")
    return n_bad == 0


def verify():
    """逐文件比对云端 md5，不一致的重推一次。返回**重推后**的真实结果。

    （旧版返回的是重推**前**的 bad 列表 → 即使重推成功也返回 False，
      是个假阳性，会让调用方误以为推送失败。）
    """
    log("校验云端文件 md5 ...")
    bad = []
    for local, remote in pf.DEFAULT_FILES:
        cands = [os.path.join(FED, local), local]
        path = next((p for p in cands if os.path.exists(p)), None)
        if not path:
            continue
        lm = pf.md5_local(path)
        rm = pf.md5_remote(pf.fix_remote(remote))
        if lm != rm:
            log(f"  [DIFF] {local}")
            bad.append((path, remote))
        else:
            log(f"  [OK]   {local}  {lm[:10]}")
    if bad:
        log(f"重推 {len(bad)} 个不一致的文件 ...")
        for path, remote in bad:
            pf.push_file(path, remote)
        log("重推完成，再次校验")
        still_bad = []
        for path, remote in bad:
            lm = pf.md5_local(path)
            rm = pf.md5_remote(pf.fix_remote(remote))
            ok = (lm == rm)
            log(f"  {'[OK]' if ok else '[FAIL]'} {os.path.basename(path)}")
            if not ok:
                still_bad.append(path)
        return not still_bad
    return True


def start_experiment():
    """下发 resume_all.sh（后台跑，日志落盘）。"""
    log("启动 resume_all.sh ...")
    cmd = ("nohup setsid bash /root/HPT/federated/resume_all.sh "
           "> /root/HPT/output/resume_all.log 2>&1 < /dev/null & "
           "echo RESUME_STARTED")
    cid, st = pf.enqueue(cmd, timeout=120)
    if cid is None:
        log(f"  入队失败 http={st}")
        return False
    res = pf.wait_results([cid], timeout=180).get(cid, {})
    if "RESUME_STARTED" in (res.get("out") or ""):
        log("  ✓ resume_all.sh 已启动（日志 /root/HPT/output/resume_all.log）")
        return True
    log(f"  启动确认失败 rc={res.get('rc')} out={(res.get('out') or '')[:150]}")
    return False


def tail_log(n=40):
    """取云端日志末尾。"""
    cmd = f"tail -n {n} /root/HPT/output/resume_all.log 2>/dev/null"
    cid, _ = pf.enqueue(cmd, timeout=60)
    if cid is None:
        return ""
    res = pf.wait_results([cid], timeout=180).get(cid, {})
    return res.get("out") or ""


def status():
    h = pf.health()
    lag = agent_lag_min()
    print(json.dumps(h, ensure_ascii=False, indent=2))
    print(f"agent_seen 落后 {lag:.1f} 分钟" if lag is not None else "agent_seen: 拿不到")
    print(f"判定: {'实例已关机，需用户在 AutoDL 重启并粘贴 bootstrap 命令' if (lag or 999) > 3 else 'agent 存活'}")


def main():
    ap = argparse.ArgumentParser(description="agent 复活后自动接管")
    ap.add_argument("--timeout", type=int, default=7200, help="等待秒数")
    ap.add_argument("--status", action="store_true", help="只看状态")
    ap.add_argument("--no-wait", action="store_true", help="不等 agent，立刻推")
    args = ap.parse_args()

    if args.status:
        status()
        return 0

    if not args.no_wait:
        ok, probe_out = wait_agent(args.timeout)
        if not ok:
            log("agent 未复活，退出")
            return 1
        # 确认 agent 跑在云端 AutoDL（不是本机）——
        # 云端工作目录是 /root/HPT，本机是 Windows 路径。
        if "/root/HPT" not in probe_out and "root" not in probe_out.lower():
            log(f"!! 探针显示 agent 不在预期的云端环境: {probe_out.strip()[:120]}")
            log("   若是本机误起的 agent，请先 kill 掉再继续")
            return 1

    push_all()
    ok = verify()
    if not ok:
        # 用**没同步成功的代码**跑 8 小时实验 = 白烧 8 小时 GPU 且产出一份
        # 不可信的表格。宁可停下来让人看一眼。
        log("!! md5 校验仍有不一致 —— 拒绝启动实验")
        log("!! 请手动检查云端文件，或重跑本脚本")
        return 2
    start_experiment()
    log("=== 接管完成，实验已启动 ===")
    log("日志: /root/HPT/output/resume_all.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
