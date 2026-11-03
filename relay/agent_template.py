#!/usr/bin/env python3
"""FedVLA 远程指挥通道 —— 云端 agent（在 AutoDL 实例里跑）。

只依赖标准库（urllib / subprocess），不需要 pip 装任何东西。
通过轮询从 relay 服务端领取命令，执行后回传结果。

由服务端 /agent.py 下发，__RELAY_URL__ / __RELAY_TOKEN__ / __RELAY_SHARECODE__ 会被自动替换。

【健壮性要点 —— 2026-11-01 事故修复】
  事故：agent 执行一条含 '&' 的命令时被 SIGTERM 杀死，整条通道中断，
        队列积压 67 条命令无人领取。
  根因：subprocess 派生的子进程与 agent 同处一个进程组/会话，
        任何发给进程组的信号会同时打到 agent 和子进程。
  修复：
    1) 所有子进程用 start_new_session=True -> 独立会话，不受 agent 信号影响
       （长训练即使 agent 挂了也会继续跑完）
    2) agent 自身忽略 SIGTERM/SIGHUP/SIGINT -> 平台或父进程的信号打不死它
    3) 每轮循环上报心跳 -> 客户端能区分"agent 死了"和"agent 在忙"
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

RELAY_URL = "__RELAY_URL__".rstrip("/")
RELAY_TOKEN = "__RELAY_TOKEN__"
RELAY_SHARECODE = "__RELAY_SHARECODE__"
POLL_INTERVAL = 3
MAX_OUTPUT = 200_000
BG_LOG = "/root/relay_bg.log"

# 已执行但结果尚未成功回传的命令：{id: result}
# 放在队列里重试上报，避免"服务端重派 -> agent 重复执行长训练命令"
_pending_report = {}

# 忽略终止信号：宁可 agent 变僵尸，也不让通道断掉
# 用 getattr 保护 —— Windows 上没有 SIGHUP
for _name in ("SIGTERM", "SIGHUP", "SIGINT"):
    _sig = getattr(signal, _name, None)
    if _sig is None:
        continue
    try:
        signal.signal(_sig, signal.SIG_IGN)
    except Exception:  # noqa: BLE001
        pass


def _req(path, payload=None, method=None, timeout=60):
    url = RELAY_URL + path
    # 平台网关要求 sharecode
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}sharecode={RELAY_SHARECODE}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"X-Relay-Token": RELAY_TOKEN, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dec(b):
    if isinstance(b, bytes):
        return b.decode("utf-8", errors="replace")
    return b or ""


def _spawn(cmd, cwd, **kw):
    """所有子进程统一走这里：独立会话 + 独立进程组。

    start_new_session=True 之后，子进程成为新 session 的首进程，
    不会被发给 agent 进程组的信号（SIGTERM 等）波及。
    """
    return subprocess.Popen(
        cmd, shell=True, cwd=cwd or None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True, **kw)


def _run_bg(cmd, cwd):
    """后台执行：把命令写进临时脚本再 nohup 起来，避免多层引号被吃掉。"""
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="relay_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n" + cmd + "\n")
    os.chmod(path, 0o755)
    shell_cmd = f"nohup setsid bash {path} >> {BG_LOG} 2>&1 < /dev/null & echo STARTED_PID=$!"
    try:
        p = _spawn(shell_cmd, cwd)
        out, _ = p.communicate(timeout=60)
        return {"rc": 0, "out": _dec(out), "err": f"(后台执行，日志: {BG_LOG})"}
    except Exception as e:  # noqa: BLE001
        return {"rc": -1, "out": "", "err": f"{type(e).__name__}: {e}"}


def _run_fg(cmd, cwd, timeout):
    """前台执行：同样用独立会话，避免长训练被外部信号打断。"""
    try:
        p = _spawn(cmd, cwd)
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            return {"rc": -9, "out": _dec(out), "err": f"TIMEOUT after {timeout}s"}
        return {"rc": p.returncode, "out": _dec(out), "err": _dec(err)}
    except Exception as e:  # noqa: BLE001
        return {"rc": -1, "out": "", "err": f"{type(e).__name__}: {e}"}


def _flush_pending():
    """重试上报之前没报成功的结果。"""
    for cid in list(_pending_report.keys()):
        r = _pending_report[cid]
        try:
            _req("/result", r)
            _pending_report.pop(cid, None)
        except Exception as e:  # noqa: BLE001
            _log(f"report #{cid} retry failed: {e}")


def main():
    started_at = int(time.time())
    _log(f"agent start -> {RELAY_URL}  pid={os.getpid()}")
    # 启动自检：确认自己确实是完整的 agent 源码、且凭据已注入。
    # 若下发的是错误页（如 401 JSON），这里会明确报错而不是静默跑不通。
    try:
        _n = len(open(os.path.abspath(__file__), encoding="utf-8").read())
        _log(f"self-check: source ok ({_n} bytes)")
    except Exception as e:  # noqa: BLE001
        _log(f"self-check FAILED: {e}")
        raise
    if not RELAY_TOKEN or RELAY_TOKEN.startswith("__RELAY"):
        _log("self-check FAILED: token 未注入，拒绝启动")
        raise SystemExit(2)
    # 服务端会重派"超时未回传"的命令；agent 侧靠 _pending_report 去重，
    # 避免同一条命令被执行两次（例如长训练被重复启动）。
    #
    # 首次轮询带 fresh_after=启动时间戳：丢弃云端实例重启前遗留的陈旧命令。
    # 场景：AutoDL 实例关机 -> agent 进程死亡 -> 队列里堆着几十条历史命令
    #       （其中可能包含训练任务）。若不丢弃，agent 复活后会逐条重跑，
    #       既烧 GPU 又堵死串行通道。
    first_poll = True
    last_beat = 0.0
    # 启动即清空服务端队列里"启动前"的陈旧命令。
    # 传 before=started_at -> 只清更早的，不会误删"启动后刚入队"的新命令。
    # 老版本服务端不支持 before 参数，会全清；此时 fresh_after 仍然兜底。
    try:
        _req("/clearall", {"before": started_at}, timeout=20)
        _log("startup: cleared stale queue")
    except urllib.error.HTTPError:
        pass
    except Exception as e:  # noqa: BLE001
        _log(f"startup clearall skipped: {e}")
    while True:
        try:
            _flush_pending()
            # 心跳：让客户端能区分"agent 死了"和"agent 正在跑长命令"
            if time.time() - last_beat > 10:
                last_beat = time.time()
                try:
                    _req("/heartbeat", {"pid": os.getpid(), "ts": int(time.time())})
                except Exception:  # noqa: BLE001
                    pass
            if first_poll:
                data = _req(f"/poll?fresh_after={started_at}", timeout=30)
                first_poll = False
            else:
                data = _req("/poll", timeout=30)
            cmds = data.get("cmds") or []
            for c in cmds:
                cid = c["id"]
                if cid in _pending_report:
                    continue
                cmd, cwd = c["cmd"], c.get("cwd") or ""
                timeout = int(c.get("timeout") or 600)
                _log(f"run #{cid}: {cmd[:90]}")
                try:
                    if c.get("bg"):
                        r = _run_bg(cmd, cwd)
                    else:
                        r = _run_fg(cmd, cwd, timeout)
                except Exception as e:  # noqa: BLE001
                    r = {"rc": -1, "out": "", "err": f"{type(e).__name__}: {e}"}
                r["id"] = cid
                r["out"] = (r.get("out") or "")[:MAX_OUTPUT]
                r["err"] = (r.get("err") or "")[:MAX_OUTPUT]
                _pending_report[cid] = r
                try:
                    _req("/result", r)
                    _pending_report.pop(cid, None)
                except Exception as e:  # noqa: BLE001
                    _log(f"report #{cid} failed: {e} (will retry)")
                _log(f"done #{cid} rc={r.get('rc')}")
        except urllib.error.URLError as e:
            _log(f"network: {e} (retry)")
        except Exception as e:  # noqa: BLE001
            _log(f"error: {type(e).__name__}: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
