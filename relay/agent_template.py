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
import threading
import time
import urllib.error
import urllib.request

RELAY_URL = "__RELAY_URL__".rstrip("/")
RELAY_TOKEN = "__RELAY_TOKEN__"
RELAY_SHARECODE = "__RELAY_SHARECODE__"
POLL_INTERVAL = 3

# ⚠️ 结果回传的体积上限【必须远小于平台 WAF 的限制】！
# 实测（2026-11-03）：sites 平台的 WAF 会拦截【请求体超过约 40KB】的 POST，
# 拦截后 agent 的 /result 上报被 403 拒绝；旧实现会无限重试，
# 而 WAF 在大 body 上还可能挂住连接 —— 结果是 agent 主线程卡死，
# 整条通道既不执行新命令也收不到任何结果（心跳还在，极具迷惑性）。
# 现在：上报前把输出截到 REPORT_LIMIT，并给上报加短超时 + 重试上限，
#       且把上报动作放到后台线程，永远不阻塞"取命令/执行命令"的主循环。
MAX_OUTPUT = 60_000      # 本地临时文件里保留的输出上限
REPORT_LIMIT = 12_000    # 实际 POST 出去的每段输出上限（out/err 各一份）

# 单条命令的最长等待（秒）。原实现直接用命令自带的 timeout（客户端默认 900），
# 一条卡住的命令会把 agent 堵住 15 分钟。超过这个时间就杀掉进程组并继续。
MAX_CMD_WAIT = 600

# 结果上报的重试上限
MAX_REPORT_ATTEMPTS = 6
BG_LOG = "/root/relay_bg.log"

# 已执行但结果尚未成功回传的命令：{id: result}
# 放在队列里重试上报，避免"服务端重派 -> agent 重复执行长训练命令"
_pending_report = {}

# 上报重试计数 + 被 WAF 永久放弃的命令 id（避免服务端反复重派同一条命令）
_pending_report_attempts = {}
_poisoned = set()

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
    """前台执行：独立会话 + 【临时文件收集输出】。

    【2026-11-03 事故修复 —— 管道死锁导致 agent 永久卡死】
      现象：一条命令打印了 200KB 字符后，agent 整个通道卡死，
            心跳停了几十分钟，进程还在但什么都不做。
      根因：用 stdout=PIPE + communicate() 收集输出。子进程写入超过
            管道缓冲区（64KB）后阻塞在 write()，而 communicate() 在
            超时后调 p.kill() 杀的是 shell 包装进程，真正的 python
            子进程仍持有管道写端 -> communicate() 第二次调用永远等不到 EOF。
      修复：stdout/stderr 重定向到临时文件，进程结束后读文件。
            磁盘缓冲没有容量上限，不会死锁；且顺带支持"输出超大也不炸"。
    """
    out_fd, out_path = tempfile.mkstemp(suffix=".out", prefix="relay_out_")
    err_fd, err_path = tempfile.mkstemp(suffix=".err", prefix="relay_err_")
    os.close(out_fd)
    os.close(err_fd)
    try:
        with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
            p = subprocess.Popen(
                cmd, shell=True, cwd=cwd or None,
                stdout=fo, stderr=fe,
                start_new_session=True)
            timed_out = False
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # 杀整个进程组（start_new_session 后 pgid == pid）
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    try:
                        p.kill()
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    p.wait(timeout=15)
                except Exception:  # noqa: BLE001
                    pass
        out = _read_file(out_path)
        err = _read_file(err_path)
        if timed_out:
            return {"rc": -9, "out": out, "err": (err + f"\n[TIMEOUT after {timeout}s]").strip()}
        return {"rc": p.returncode, "out": out, "err": err}
    except Exception as e:  # noqa: BLE001
        return {"rc": -1, "out": "", "err": f"{type(e).__name__}: {e}"}
    finally:
        for path in (out_path, err_path):
            try:
                os.unlink(path)
            except Exception:  # noqa: BLE001
                pass


def _read_file(path, limit=MAX_OUTPUT):
    """读临时输出文件；只保留尾部 limit 字节（超长时保留最有信息量的部分）。"""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > limit:
                f.seek(size - limit)
                data = f.read()
                return "[...输出过长，仅保留尾部...]\n" + data.decode("utf-8", errors="replace")
            return f.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"(读取输出失败: {e})"


def _flush_pending():
    """重试上报之前没报成功的结果（由后台线程调用，不阻塞主循环）。

    [2026-11-03 修复] 原实现在主循环里无限重试：一旦结果被 WAF 拦下（HTTP 403），
    就每轮都重试、永不放弃；而服务端因为没收到结果会重新派发同一条命令，
    agent 又会再执行一遍 —— 形成"命令被反复重跑"的死循环。
    现在：
      - 上报用短超时（10s），失败不会长时间挂住；
      - 403 时把输出逐级截断再试（保留尾部信息），最后退化成占位文本；
      - 最多重试 MAX_REPORT_ATTEMPTS 次，仍失败则放弃并把该 id 记入 _poisoned，
        之后 poll 到同 id 直接跳过，避免服务端重派导致同一条命令被反复执行。
    """
    for cid in list(_pending_report.keys()):
        r = _pending_report[cid]
        attempts = _pending_report_attempts.get(cid, 0)
        if attempts >= MAX_REPORT_ATTEMPTS:
            _log(f"report #{cid} abandoned after {attempts} attempts (WAF blocked?)")
            _pending_report.pop(cid, None)
            _pending_report_attempts.pop(cid, None)
            _poisoned.add(cid)
            continue
        try:
            _req("/result", r, timeout=10)
            _pending_report.pop(cid, None)
            _pending_report_attempts.pop(cid, None)
        except urllib.error.HTTPError as e:
            attempts += 1
            _pending_report_attempts[cid] = attempts
            if e.code == 403:
                cur = r.get("out") or ""
                keep = max(200, len(cur) // 2)
                r["out"] = cur[-keep:] if keep < len(cur) else cur
                r["err"] = (r.get("err") or "")[:300]
                if attempts >= 3:
                    r["out"] = "[WAF blocked: output withheld] " + (cur[-200:] if cur else "")
                _log(f"report #{cid} 403 -> truncate to {len(r['out'])} chars, retry {attempts}")
                continue
            _log(f"report #{cid} retry failed ({attempts}): {e}")
        except Exception as e:  # noqa: BLE001
            attempts += 1
            _pending_report_attempts[cid] = attempts
            _log(f"report #{cid} retry failed ({attempts}): {e}")


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

    # 心跳放到独立线程：主线程在跑长命令（训练可达数十分钟）时，
    # 心跳仍要持续上报，否则客户端会误判"agent 死了"并重启它。
    _stop_beat = threading.Event()

    def _beat_loop():
        # 心跳 + 结果上报重试都放在这个后台线程里：
        # 上报可能因为 WAF 拦截而变慢/失败，绝不能让它堵住主循环（取命令/执行命令）。
        while not _stop_beat.is_set():
            try:
                _flush_pending()
            except Exception as e:  # noqa: BLE001
                _log(f"flush pending error: {e}")
            try:
                _req("/heartbeat", {"pid": os.getpid(), "ts": int(time.time())}, timeout=15)
            except Exception:  # noqa: BLE001
                pass
            _stop_beat.wait(10)

    _beat_thread = threading.Thread(target=_beat_loop, daemon=True)
    _beat_thread.start()

    while True:
        try:
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
                if cid in _poisoned:
                    # [FedVLA patch 2026-11-03] 该命令的结果被 WAF 拦下且已放弃，
                    # 服务端会一直重派；直接跳过，避免反复执行同一条命令。
                    continue
                cmd, cwd = c["cmd"], c.get("cwd") or ""
                timeout = min(int(c.get("timeout") or 600), MAX_CMD_WAIT)
                _log(f"run #{cid}: {cmd[:90]}")
                try:
                    if c.get("bg"):
                        r = _run_bg(cmd, cwd)
                    else:
                        r = _run_fg(cmd, cwd, timeout)
                except Exception as e:  # noqa: BLE001
                    r = {"rc": -1, "out": "", "err": f"{type(e).__name__}: {e}"}
                r["id"] = cid
                r["out"] = (r.get("out") or "")[:REPORT_LIMIT]
                r["err"] = (r.get("err") or "")[:REPORT_LIMIT]
                _pending_report[cid] = r
                _pending_report_attempts[cid] = 0
                try:
                    _req("/result", r, timeout=15)
                    _pending_report.pop(cid, None)
                    _pending_report_attempts.pop(cid, None)
                except Exception as e:  # noqa: BLE001
                    _log(f"report #{cid} failed: {e} (will retry in background)")
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
