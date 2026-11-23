#!/usr/bin/env python3
"""第 5 阶段上云编排：一次跑完「推代码 → 打补丁 → 验单测 → 起训练 → 起守护」。

## 为什么要有这个脚本
第 5 阶段要跨 6 个步骤、跨越 20+ 分钟，其中每一步都可能静默失败。
逐条手敲命令 = 每一步都是一次「以为成功了其实没有」的机会。
本脚本把每一步都做成 **可判定** 的动作，失败即 `exit 1` 并打印现场。

## 继承的硬教训（不要在改这个脚本时丢掉）
* **铁律 B（队列重放）**：推送前 `clearall`；推送后**等 ≥3 分钟**再回读校验；
  判「被覆盖」用**二进制 md5**（不是看文件在不在）。
* **铁律 I（idbase 递增）**：守护推送用 `PUSH_IDBASE=1010000`（第 4 阶段是 1000000）。
* **铁律 J（守护顺序）**：守护必须在**被等的进程之后**启动 ⇒
  先起训练 → sleep 45s 确认存活 → 再起守护。否则「主控消失且 .rc 不齐 ⇒ 提前收工」会误触发。
* **铁律 Q（fail-loud）**：单测不全绿 ⇒ 绝不开跑；预检不过 ⇒ 绝不起守护。
* **铁律 K（pgrep 自匹配）**：探活必须用字符类 `[r]un_phase5`，否则恒返回假活。

## 用法
    python tools/phase5_go.py push     # 清队列 + 推 10 个文件 + 等 3 分钟 + 回读 md5 校验
    python tools/phase5_go.py patch    # 打 HPT 联邦基线补丁（幂等）+ --verify
    python tools/phase5_go.py test     # 云端跑 29 项离线单测（必须全绿）
    python tools/phase5_go.py launch   # 起训练 → sleep45 验活 → 起自动关机守护
    python tools/phase5_go.py all      # 以上全部按顺序执行
    python tools/phase5_go.py status   # 只看当前状态（不改动任何东西）
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import push_files as pf  # noqa: E402  复用 enqueue / wait_results / req / 凭据

# ---------------------------------------------------------------- 云端常量
CLOUD_DIR = "/root/HPT/federated"
HPT_DIR = "/root/HPT"
CLOUD_PY = "/root/miniconda3/envs/hpt/bin/python"
OUT_DIR = "/root/HPT/output/phase5"
PUSH_IDBASE = 1010000          # 🔴 铁律 I：必须比第 4 阶段（1000000）大
MAX_HOURS = 24

# ---------------------------------------------------------------- 待推文件
# (本地相对路径, 云端路径)
FILES = [
    ("federated/fed_baselines.py", f"{CLOUD_DIR}/fed_baselines.py"),
    ("federated/fedvla_baseline_hook.py", f"{CLOUD_DIR}/fedvla_baseline_hook.py"),
    ("federated/patch_hpt_fedbaseline.py", f"{CLOUD_DIR}/patch_hpt_fedbaseline.py"),
    ("federated/run_phase5.sh", f"{CLOUD_DIR}/run_phase5.sh"),
    ("federated/phase5_autoshutdown.sh", f"{CLOUD_DIR}/phase5_autoshutdown.sh"),
    ("federated/extract_phase5_results.py", f"{CLOUD_DIR}/extract_phase5_results.py"),
    ("federated/test_fed_baselines.py", f"{CLOUD_DIR}/test_fed_baselines.py"),
    ("federated/test_baseline_hook.py", f"{CLOUD_DIR}/test_baseline_hook.py"),
    ("federated/test_phase5_scripts.sh", f"{CLOUD_DIR}/test_phase5_scripts.sh"),
    ("federated/relay_push_out.py", f"{CLOUD_DIR}/relay_push_out.py"),
    ("federated/run_fedavg.sh", f"{CLOUD_DIR}/run_fedavg.sh"),   # 已改造：BASELINE 三态开关
]

WORKER_PAT = r"[r]un_phase5\.sh|[r]un_fedavg|eval_model\.sh|hpt\.run|hpt\.run_eval"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- 云端执行
def run(cmd: str, timeout: int = 600, label: str = "") -> tuple[int, str, str]:
    """在云端跑一条命令并拿回 stdout/stderr。返回 (rc, out, err)。"""
    if label:
        log(f"→ 云端执行：{label}")
    cid, st = pf.enqueue(cmd, timeout=180)
    if cid is None:
        log(f"[FATAL] 命令入队失败（st={st}）：{cmd[:120]}")
        return 99, "", "enqueue failed"
    res = pf.wait_results([cid], timeout=timeout).get(cid, {})
    rc = res.get("rc", 99)
    out = res.get("out") or ""
    err = res.get("err") or ""
    if rc is None:
        rc = 99
    return rc, out, err


def clear_queue():
    """🔴 铁律 B：推送前必须清空残留命令（agent 死后队列会卡着旧命令，复活后会执行）。"""
    try:
        r = pf.req("/clearall", data={"n": 0}, timeout=30, retries=3)
        log(f"[清队列] {r}")
    except Exception as e:  # noqa: BLE001
        log(f"[清队列] 失败（不致命，继续）：{e}")


# ---------------------------------------------------------------- 步骤 1：推代码
def step_push_impl() -> bool:
    """真正的推送实现（把入队 cid 收集起来统一等待）。"""
    log("=" * 64)
    log("步骤 1/4：推送第 5 阶段代码")
    log("=" * 64)

    missing = [p for p, _ in FILES if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        log(f"[FATAL] 本地缺文件：{missing}")
        return False
    oversize = [p for p, _ in FILES
                if os.path.getsize(os.path.join(ROOT, p)) > 200 * 1024]
    if oversize:
        log(f"[FATAL] 文件超 200KB（WAF 会拦）：{oversize}")
        return False

    clear_queue()

    pairs = [(os.path.join(ROOT, lp), pf.fix_remote(rp)) for lp, rp in FILES]
    cids = []
    for local, remote in pairs:
        text = open(local, encoding="utf-8").read()
        cid, st = pf.enqueue(pf.b64_put_cmd(text, remote), timeout=180)
        log(f"  [入队] {os.path.basename(local):<28} cid={cid} st={st}")
        if cid is not None:
            cids.append(cid)

    log(f"[等待] {len(cids)} 条写入命令回传 ...")
    pf.wait_results(cids, timeout=900)
    log("[等待] 写入阶段结束")

    # 🔴 铁律 B：推送后**等 ≥3 分钟**再回读校验（relay 会重派历史命令，可能把文件还原成旧版）
    wait_sec = 180
    log(f"[铁律 B] 等待 {wait_sec}s 后再回读校验（防队列重放覆盖）...")
    for i in range(0, wait_sec, 30):
        time.sleep(min(30, wait_sec - i))
        log(f"  ...还剩 {wait_sec - i - 30 if wait_sec - i > 30 else 0}s")

    # 独立回读 md5（一条命令校验全部）
    remotes = " ".join(r for _, r in pairs)
    rc, out, err = run(f"md5sum {remotes} 2>/dev/null", timeout=600, label="回读 md5 校验")
    got = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            got[parts[1]] = parts[0]

    bad = []
    for local, remote in pairs:
        want = pf.md5_local(local)
        have = got.get(remote, "")
        ok = (want == have)
        log(f"  [{'OK ' if ok else 'BAD'}] {os.path.basename(local):<28} "
            f"want={want[:10]} remote={have[:10] or '缺文件'}")
        if not ok:
            bad.append(os.path.basename(local))

    if bad:
        log(f"[FAIL] {len(bad)} 个文件未落地或被覆盖：{bad}")
        return False

    # chmod（一次搞定）
    run(f"chmod +x {' '.join(r for _, r in pairs)} 2>/dev/null || true", timeout=240)

    log(f"[OK] {len(pairs)} 个文件推送 + 回读校验全部通过")
    return True


# ---------------------------------------------------------------- 步骤 2：打补丁
def step_patch() -> bool:
    log("=" * 64)
    log("步骤 2/4：给 HPT 打联邦基线补丁（FedProx / SCAFFOLD）")
    log("=" * 64)

    rc, out, err = run(
        f"cd {HPT_DIR} && {CLOUD_PY} {CLOUD_DIR}/patch_hpt_fedbaseline.py "
        f"--hpt-dir {HPT_DIR} 2>&1",
        timeout=300, label="打补丁")
    log(out.strip()[-1500:])
    if rc != 0:
        log(f"[FAIL] 打补丁 rc={rc}\n{err[-800:]}")
        return False

    rc, out, err = run(
        f"cd {HPT_DIR} && {CLOUD_PY} {CLOUD_DIR}/patch_hpt_fedbaseline.py "
        f"--hpt-dir {HPT_DIR} --verify 2>&1",
        timeout=300, label="补丁 verify")
    log(out.strip()[-1500:])
    if rc != 0:
        log(f"[FAIL] 补丁 verify rc={rc}\n{err[-800:]}")
        return False

    # 再确认锚点真的在 train_test.py 里（铁律 A：环境变量没生效 = 静默退化）
    rc, out, _ = run(
        f"grep -c 'apply_baseline_hook' {HPT_DIR}/hpt/train_test.py || true",
        timeout=180, label="确认锚点存在")
    n = (out or "").strip().splitlines()[0].strip() if (out or "").strip() else "0"
    if not n.isdigit():
        n = "0"
    if int(n) < 1:
        log("[FAIL] train_test.py 里没有 apply_baseline_hook ⇒ 钩子未生效（会静默退化成 FedAvg）")
        return False
    log(f"[OK] 钩子锚点命中 {n} 处")
    return True


# ---------------------------------------------------------------- 步骤 3：云端单测
def step_test() -> bool:
    log("=" * 64)
    log("步骤 3/4：云端跑 29 项离线单测（不全绿绝不开跑）")
    log("=" * 64)

    cmds = [
        (f"cd {CLOUD_DIR} && {CLOUD_PY} test_fed_baselines.py 2>&1", "test_fed_baselines.py"),
        (f"cd {CLOUD_DIR} && {CLOUD_PY} test_baseline_hook.py 2>&1", "test_baseline_hook.py"),
        (f"cd {CLOUD_DIR} && bash test_phase5_scripts.sh 2>&1", "test_phase5_scripts.sh"),
    ]
    all_ok = True
    for cmd, name in cmds:
        rc, out, err = run(cmd, timeout=600, label=name)
        tail = (out or "")[-1200:]
        log(tail)
        ok = rc == 0 and ("全部" in (out or "") or "通过" in (out or ""))
        if not ok:
            log(f"[FAIL] {name} rc={rc}")
            all_ok = False
    if not all_ok:
        log("[FAIL] 单测未全绿 ⇒ 拒绝开跑（fail-loud）")
        return False
    log("[OK] 29 项离线单测全绿")
    return True


# ---------------------------------------------------------------- 步骤 4：起跑 + 起守护
def step_launch() -> bool:
    log("=" * 64)
    log("步骤 4/4：启动主实验矩阵 + 自动关机守护")
    log("=" * 64)

    clear_queue()

    # ---- 4a. 起训练（后台，nohup setsid；不能挂 agent 前台，MAX_CMD_WAIT=600s 会 SIGKILL）----
    cmd = (
        f"mkdir -p {OUT_DIR} && cd {HPT_DIR} && "
        f"(nohup setsid bash {CLOUD_DIR}/run_phase5.sh > {OUT_DIR}/run_phase5.log 2>&1 < /dev/null &) "
        f"&& sleep 10 && echo '[OK] 训练已拉起'"
    )
    rc, out, err = run(cmd, timeout=300, label="启动 run_phase5.sh")
    log((out or "").strip()[-800:])
    if rc != 0:
        log(f"[FAIL] 训练启动失败 rc={rc}\n{err[-800:]}")
        return False

    # 🔴 铁律 J：守护必须在被等的进程之后启动 ⇒ 先确认训练真的活着
    log("[铁律 J] sleep 45s 确认训练进程存活 ...")
    time.sleep(45)
    rc, out, _ = run(
        f"ps -eo pid,cmd | grep -E '{WORKER_PAT}' | grep -v grep | head -5; "
        f"echo '---轮次---'; ls -1 {OUT_DIR}/logs 2>/dev/null | head",
        timeout=180, label="确认训练进程存活")
    log((out or "").strip()[:800])
    alive = bool(re.search(r"run_phase5|run_fedavg|hpt", out or ""))
    if not alive:
        log("[FAIL] 45s 后仍没看到训练进程 ⇒ 不起守护（否则会误触发「提前收工」关机）")
        run(f"tail -n 30 {OUT_DIR}/run_phase5.log 2>/dev/null", timeout=180)
        return False
    log("[OK] 训练进程存活")

    # ---- 4b. 起守护 ----
    cmd = (
        f"mkdir -p {OUT_DIR} && cd {CLOUD_DIR} && "
        f"(nohup setsid env OUT={OUT_DIR} CORE={CLOUD_DIR} MAX_HOURS={MAX_HOURS} "
        f"POLL=60 PUSH_IDBASE={PUSH_IDBASE} "
        f"bash {CLOUD_DIR}/phase5_autoshutdown.sh > {OUT_DIR}/autoshutdown.log 2>&1 < /dev/null &) "
        f"&& sleep 8 && tail -n 6 {OUT_DIR}/autoshutdown.log"
    )
    rc, out, err = run(cmd, timeout=300, label="启动自动关机守护")
    log((out or "").strip()[-800:])
    if rc != 0:
        log(f"[FAIL] 守护启动失败 rc={rc}\n{err[-800:]}")
        return False

    # 确认守护只有一个实例（铁律 F/G：重启守护确认只剩 1 个实例）
    rc, out, _ = run(
        f"ps -eo pid,cmd | grep '[p]hase5_autoshutdown.sh' | wc -l",
        timeout=180, label="确认守护实例数")
    n = (out or "").strip().splitlines()[0].strip() if (out or "").strip() else "0"
    log(f"[守护实例数] {n}（应为 1）")
    if n.isdigit() and int(n) != 1:
        log("[WARN] 守护实例数异常，请人工确认")

    log("[OK] 主实验矩阵已启动，守护已挂上（到点自动关机 + 推送 + 回读自检）")
    return True


# ---------------------------------------------------------------- 状态
def step_status() -> None:
    log("=" * 64)
    log("当前状态")
    log("=" * 64)
    rc, out, err = run(
        f"echo '--- 进程 ---'; ps -eo pid,etime,cmd | grep -E '{WORKER_PAT}' | grep -v grep | head -8; "
        f"echo '--- .rc ---'; ls -1 {OUT_DIR}/.rc_* 2>/dev/null; "
        f"echo '--- 产物 ---'; ls -1 {OUT_DIR} 2>/dev/null | head -20; "
        f"echo '--- 训练日志尾 ---'; for t in p5_bal_fedprox p5_bal_scaffold p5_iso_fedavg "
        f"p5_iso_fedprox p5_iso_scaffold p5_iso_eda p5_iso_edaclu; do "
        f"f={OUT_DIR}/logs/$t.log; [ -f $f ] && echo \"[$t] $(tail -n 1 $f)\"; done; "
        f"echo '--- 守护日志尾 ---'; tail -n 8 {OUT_DIR}/autoshutdown.log 2>/dev/null; "
        f"echo '--- 磁盘 ---'; df -h / | tail -1",
        timeout=300, label="拉状态")
    print(out or err or "(无输出)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "patch", "test", "launch", "all", "status"])
    args = ap.parse_args()

    log(f"relay 健康：{pf.health()}")

    if args.cmd == "push":
        return 0 if step_push_impl() else 1
    if args.cmd == "patch":
        return 0 if step_patch() else 1
    if args.cmd == "test":
        return 0 if step_test() else 1
    if args.cmd == "launch":
        return 0 if step_launch() else 1
    if args.cmd == "status":
        step_status()
        return 0
    if args.cmd == "all":
        for fn in (step_push_impl, step_patch, step_test, step_launch):
            if not fn():
                log(f"[ABORT] {fn.__name__} 失败 ⇒ 中止，后续步骤不执行")
                return 1
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
