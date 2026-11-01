#!/usr/bin/env python3
"""FedVLA 远程指挥通道 —— 云端 agent（在 AutoDL 实例里跑）。

只依赖标准库（urllib / subprocess），不需要 pip 装任何东西。
通过轮询从 relay 服务端领取命令，执行后回传结果。

由服务端 /agent.py 下发，__RELAY_URL__ / __RELAY_TOKEN__ 会被自动替换。
"""
import json
import os
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

# 已执行但结果尚未成功回传的命令：{id: result}
# 放在队列里重试上报，避免"服务端重派 -> agent 重复执行长训练命令"
_pending_report = {}


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


def _run_bg(cmd, cwd):
    """后台执行：把命令写进临时脚本再 nohup 起来，避免多层引号被吃掉。"""
    log = "/root/relay_bg.log"
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="relay_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n" + cmd + "\n")
    os.chmod(path, 0o755)
    shell_cmd = f"nohup bash {path} >> {log} 2>&1 & echo STARTED_PID=$!"
    p = subprocess.run(shell_cmd, shell=True, cwd=cwd or None,
                       capture_output=True, timeout=60)
    return {"rc": 0, "out": _dec(p.stdout), "err": f"(后台执行，日志: {log})"}


def _run_fg(cmd, cwd, timeout):
    try:
        p = subprocess.run(["bash", "-lc", cmd], cwd=cwd or None,
                           capture_output=True, timeout=timeout)
        return {"rc": p.returncode, "out": _dec(p.stdout), "err": _dec(p.stderr)}
    except subprocess.TimeoutExpired as e:
        return {"rc": -9, "out": _dec(e.stdout), "err": f"TIMEOUT after {timeout}s"}


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
    _log(f"agent start -> {RELAY_URL}")
    # 服务端会重派"超时未回传"的命令；agent 侧靠 _pending_report 去重，
    # 避免同一条命令被执行两次（例如长训练被重复启动）。
    while True:
        try:
            _flush_pending()
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
