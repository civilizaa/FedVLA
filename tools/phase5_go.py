#!/usr/bin/env python3
"""第 5 阶段上云编排：推代码 → 打补丁 → 验单测 → 起训练+守护。

    python tools/phase5_go.py push     # 推 11 个文件 + 回读 md5 校验
    python tools/phase5_go.py patch    # 打 HPT 联邦基线补丁（幂等）+ verify
    python tools/phase5_go.py test     # 云端跑 29 项离线单测（必须全绿）
    python tools/phase5_go.py launch   # 起训练 → sleep45 验活 → 起自动关机守护
    python tools/phase5_go.py status   # 只看当前状态（不改动任何东西）

## 为什么不用 agent 上报的结果（2026-11-23 实测）

relay 服务端 `/result` 只保留按 `int(id)` 排序后的**末尾 500 条**。第 4 阶段用
idbase 段（980278~1000374）占满后，`/enqueue` 分配到的普通小 id 结果**一写入
就被淘汰** ⇒ `/poll` 判定未完成 ⇒ **每轮重新派发** ⇒ agent 重复执行、串行通道
堵死（实测 queued 恒为 1）。⇒ 全部改走 `relay_bootstrap.py` 的**高 id 段**通道。

## 继承的硬教训（改这个脚本时不要丢）
* **铁律 B（队列重放）**：每步前后 `clearall`；推送后回读 **二进制 md5** 校验。
* **铁律 I（idbase 递增）**：过程性命令用 1005xxx 段；守护产物用 **1010000**（最大）。
* **铁律 J（守护顺序）**：先起训练 → sleep 45s 确认存活 → **才**起守护。
* **铁律 Q（fail-loud）**：单测不全绿 ⇒ 绝不开跑；预检不过 ⇒ 绝不起守护。
* **铁律 K（pgrep 自匹配）**：探活用字符类 `[r]un_phase5`。
* **铁律 B 的 WAF 教训**：长命令一律**写成文件推送**，执行时只发极短 base64 外壳。
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

import phase5_remote as pr  # noqa: E402

CLOUD_DIR = "/root/fedcore"
HPT_DIR = "/root/HPT"
CLOUD_PY = "/root/miniconda3/envs/hpt/bin/python"
OUT_DIR = "/root/HPT/output/phase5"
PUSH_IDBASE = pr.IDBASE_FINAL      # 1010000
MAX_HOURS = 24
TAGS = ["p5_bal_fedprox", "p5_bal_scaffold", "p5_iso_fedavg", "p5_iso_fedprox",
        "p5_iso_scaffold", "p5_iso_eda", "p5_iso_edaclu"]
WORKER_PAT = r"[r]un_phase5\.sh|[r]un_fedavg|eval_model\.sh|hpt\.run|hpt\.run_eval"

# 待推文件（本地相对路径 → 云端路径）
FILES = [
    "federated/fed_baselines.py",
    "federated/fedvla_baseline_hook.py",
    "federated/patch_hpt_fedbaseline.py",
    "federated/run_phase5.sh",
    "federated/phase5_autoshutdown.sh",
    "federated/extract_phase5_results.py",
    "federated/test_fed_baselines.py",
    "federated/test_baseline_hook.py",
    "federated/test_phase5_scripts.sh",
    "federated/relay_bootstrap.py",
    "federated/run_fedavg.sh",      # 改造版：BASELINE 三态开关（空 ⇒ 逐字节等价）
]
# ⚠️ 刻意【不推】：fedavg_config.sh / aggregate.py / relay_push_out.py / eval_model.sh
#    —— 已逐一核对过 md5：前三者本地与云端完全一致；eval_model.sh 仅 OMP 线程数差异，
#    沿用云端第 4 阶段实战版本以保持评估同构。


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- 步骤 1：推代码
def step_push() -> bool:
    log("=" * 64)
    log("步骤 1/4：推送第 5 阶段代码")
    log("=" * 64)

    missing = [p for p in FILES if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        log(f"[FATAL] 本地缺文件：{missing}")
        return False

    remote = pr.Remote()
    remote.clear()

    def push_all(targets):
        for lp in targets:
            remote.push_file(os.path.join(ROOT, lp),
                             f"{CLOUD_DIR}/{os.path.basename(lp)}")

    def verify(targets):
        import hashlib
        rpaths = [f"{CLOUD_DIR}/{os.path.basename(p)}" for p in targets]
        # ⚠️ clear_first=False：推送命令可能还在队列里排队，clear 掉就永远推不上；
        #    agent 串行 FIFO ⇒ 校验命令排在推送之后 ⇒ 校验必然反映推送后的状态。
        got = remote.md5_remote(rpaths, clear_first=False)
        bad = []
        for lp, rp in zip(targets, rpaths):
            want = hashlib.md5(
                pr.normalize_lf(open(os.path.join(ROOT, lp), encoding="utf-8").read())
                .encode("utf-8")).hexdigest()
            have = got.get(rp, "")
            ok = (want == have)
            log(f"  [{'OK ' if ok else 'BAD'}] {os.path.basename(lp):<26} "
                f"want={want[:10]} remote={have[:10] or '缺文件'}")
            if not ok:
                bad.append(lp)
        return bad

    push_all(FILES)
    bad = verify(FILES)
    for attempt in range(1, 4):
        if not bad:
            break
        log(f"[重试 {attempt}] {len(bad)} 个文件未落地，重新推送："
            f"{[os.path.basename(b) for b in bad]}")
        push_all(bad)
        bad = verify(bad)
    if bad:
        log(f"[FAIL] {len(bad)} 个文件推送失败：{[os.path.basename(b) for b in bad]}")
        return False

    remote.run(f"chmod +x {' '.join(f'{CLOUD_DIR}/{os.path.basename(p)}' for p in FILES)} "
               f"2>/dev/null || true", timeout=180, clear_first=False)
    log(f"[OK] {len(FILES)} 个文件推送 + md5 校验全部通过")
    return True


# ---------------------------------------------------------------- 步骤 2：打补丁
def step_patch() -> bool:
    log("=" * 64)
    log("步骤 2/4：给 HPT 打联邦基线补丁（FedProx / SCAFFOLD）")
    log("=" * 64)
    remote = pr.Remote()
    rc, out, err = remote.run(
        f"cd {HPT_DIR} && {CLOUD_PY} {CLOUD_DIR}/patch_hpt_fedbaseline.py "
        f"--hpt-dir {HPT_DIR} 2>&1", timeout=300)
    log(out.strip()[-1200:])
    if rc != 0:
        log(f"[FAIL] 打补丁 rc={rc}\n{err[-800:]}")
        return False

    rc, out, err = remote.run(
        f"cd {HPT_DIR} && {CLOUD_PY} {CLOUD_DIR}/patch_hpt_fedbaseline.py "
        f"--hpt-dir {HPT_DIR} --verify 2>&1", timeout=300)
    log(out.strip()[-1200:])
    if rc != 0:
        log(f"[FAIL] 补丁 verify rc={rc}\n{err[-800:]}")
        return False

    # 🔴 铁律 A：环境变量没生效 = 静默退化成 FedAvg ⇒ 必须断言锚点真的在
    rc, out, _ = remote.run(
        f"n=$(grep -c 'apply_baseline_hook' {HPT_DIR}/hpt/train_test.py); "
        f"[ -n \"$n\" ] || n=0; echo $n", timeout=180)
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
    remote = pr.Remote()
    cmds = [
        (f"cd {CLOUD_DIR} && {CLOUD_PY} test_fed_baselines.py 2>&1", "test_fed_baselines.py"),
        (f"cd {CLOUD_DIR} && {CLOUD_PY} test_baseline_hook.py 2>&1", "test_baseline_hook.py"),
        (f"cd {CLOUD_DIR} && bash test_phase5_scripts.sh 2>&1", "test_phase5_scripts.sh"),
    ]
    all_ok = True
    for cmd, name in cmds:
        rc, out, err = remote.run(cmd, timeout=600)
        log((out or "")[-900:])
        ok = rc == 0 and ("全部" in (out or "") or "通过" in (out or ""))
        if not ok:
            log(f"[FAIL] {name} rc={rc}")
            all_ok = False
    if not all_ok:
        log("[FAIL] 单测未全绿 ⇒ 拒绝开跑（fail-loud）")
        return False
    log("[OK] 29 项离线单测全绿")
    return True


# ---------------------------------------------------------------- 步骤 4：起跑 + 守护
def step_launch() -> bool:
    log("=" * 64)
    log("步骤 4/4：启动主实验矩阵 + 自动关机守护")
    log("=" * 64)
    remote = pr.Remote()
    remote.clear()

    cmd = (f"mkdir -p {OUT_DIR} && cd {HPT_DIR} && "
           f"(nohup setsid bash {CLOUD_DIR}/run_phase5.sh "
           f"> {OUT_DIR}/run_phase5.log 2>&1 < /dev/null &) && "
           f"sleep 12 && echo '[OK] 训练已拉起'")
    rc, out, err = remote.run(cmd, timeout=300)
    log((out or "").strip()[-600:])
    if rc != 0:
        log(f"[FAIL] 训练启动失败 rc={rc}\n{err[-800:]}")
        return False

    # 🔴 铁律 J：守护必须在被等的进程之后启动
    log("[铁律 J] sleep 45s 确认训练进程存活 …")
    time.sleep(45)
    rc, out, _ = remote.run(
        f"ps -eo pid,cmd | grep -E '{WORKER_PAT}' | grep -v grep | head -5; "
        f"echo '---日志尾---'; tail -n 12 {OUT_DIR}/run_phase5.log 2>/dev/null",
        timeout=180)
    log((out or "").strip()[:900])
    if not re.search(r"run_phase5|run_fedavg|hpt", out or ""):
        log("[FAIL] 45s 后仍没看到训练进程 ⇒ 不起守护（否则会误触发「提前收工」关机）")
        return False
    log("[OK] 训练进程存活")

    cmd = (f"mkdir -p {OUT_DIR} && cd {CLOUD_DIR} && "
           f"(nohup setsid env OUT={OUT_DIR} CORE={CLOUD_DIR} MAX_HOURS={MAX_HOURS} "
           f"POLL=60 PUSH_IDBASE={PUSH_IDBASE} "
           f"bash {CLOUD_DIR}/phase5_autoshutdown.sh > {OUT_DIR}/autoshutdown.log 2>&1 < /dev/null &) "
           f"&& sleep 8 && tail -n 6 {OUT_DIR}/autoshutdown.log")
    rc, out, err = remote.run(cmd, timeout=300)
    log((out or "").strip()[-600:])
    if rc != 0:
        log(f"[FAIL] 守护启动失败 rc={rc}\n{err[-800:]}")
        return False

    rc, out, _ = remote.run(
        "ps -eo pid,cmd | grep '[p]hase5_autoshutdown.sh' | wc -l", timeout=180)
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
    remote = pr.Remote()
    cmd = (f"echo '--- 进程 ---'; ps -eo pid,etime,cmd | grep -E '{WORKER_PAT}' "
           f"| grep -v grep | head -8; "
           f"echo '--- .rc ---'; ls -1 {OUT_DIR}/.rc_* 2>/dev/null; "
           f"echo '--- 产物 ---'; ls -1 {OUT_DIR} 2>/dev/null | head -20; "
           f"echo '--- 各配置日志尾 ---'; "
           + " ".join(f"echo '[{t}] $(tail -n 1 {OUT_DIR}/logs/{t}.log 2>/dev/null)';" for t in TAGS)
           + f" echo '--- 守护日志尾 ---'; tail -n 10 {OUT_DIR}/autoshutdown.log 2>/dev/null; "
           f"echo '--- GPU ---'; nvidia-smi --query-gpu=memory.used,utilization.gpu "
           f"--format=csv,noheader 2>/dev/null | head -2; "
           f"echo '--- 磁盘 ---'; df -h / | tail -1")
    rc, out, err = remote.run(cmd, timeout=300)
    print(out or err or "(无输出)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "patch", "test", "launch", "all", "status"])
    args = ap.parse_args()
    log(f"relay 健康：{pr.cloud.req('/health', timeout=20, retries=3)}")

    fns = {"push": step_push, "patch": step_patch, "test": step_test, "launch": step_launch}
    if args.cmd == "status":
        step_status()
        return 0
    if args.cmd == "all":
        for name, fn in fns.items():
            if not fn():
                log(f"[ABORT] 步骤 {name} 失败 ⇒ 中止，后续步骤不执行")
                return 1
        return 0
    return 0 if fns[args.cmd]() else 1


if __name__ == "__main__":
    sys.exit(main())
