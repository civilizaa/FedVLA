#!/usr/bin/env python3
"""FedVLA 远程指挥通道 —— 服务端。

设计目标：让 AutoDL 实例（只能出网、连不上本机）把"我这边要执行什么"取走，
执行完再把结果回传，从而实现"本机 Agent ↔ 云端实例"的自动化闭环。

协议（全部需要 header: X-Relay-Token: <SECRET>）
  GET  /health                      -> {"ok": true}
  POST /enqueue                     body {"cmd","cwd","timeout","bg"} -> {"id": N}
  GET  /poll?since=<last_id>        -> {"cmds":[{id,cmd,cwd,timeout,bg}, ...]}
  POST /result                      body {"id","rc","out","err"} -> {"ok":true}
  GET  /results?since=<last_id>     -> {"results":[{id,rc,out,err,ts}, ...]}
  POST /run                         body 同 enqueue，同步等待结果并返回（带 timeout）

状态持久化到 relay_state.json（服务重启不丢队列与结果）。
"""
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "relay_state.json")
MAX_OUT = 200_000  # 单条输出最大保留字符


def _read_secret_file():
    """从 relay_secret.txt 读凭据（该文件已 gitignore，不入库）。

    仓库是 public，密钥绝不能硬编码在代码里。
    文件格式（每行一条，'#' 开头为注释）：
        第1行 = RELAY_TOKEN
        第2行 = RELAY_SHARECODE
        第3行 = RELAY_PUBLIC_URL
    """
    path = os.path.join(HERE, "relay_secret.txt")
    if not os.path.exists(path):
        return "", "", ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f
                     if l.strip() and not l.strip().startswith("#")]
    except Exception:  # noqa: BLE001
        return "", "", ""
    tok = lines[0] if len(lines) > 0 else ""
    sc = lines[1] if len(lines) > 1 else ""
    url = lines[2] if len(lines) > 2 else ""
    return tok, sc, url


_f_tok, _f_sc, _f_url = _read_secret_file()
SECRET = os.environ.get("RELAY_SECRET", "") or _f_tok
# 平台网关要求 sharecode，代理层会自动补上；agent 侧仍需自己带
SHARECODE = os.environ.get("RELAY_SHARECODE", "") or _f_sc
# 下发给 agent 的公网地址。不能用 Host 头 —— 反向代理传进来的是沙箱内网域名，
# AutoDL 那边解析不了（2026-11-01 实测踩过）。
PUBLIC_URL = (os.environ.get("RELAY_PUBLIC_URL", "") or _f_url
               or "https://fedvla-relay.app.workbuddy.host").rstrip("/")

_lock = threading.Lock()
_state = {"next_id": 1, "queue": [], "results": {}, "heartbeat": {}}


def _load():
    global _state
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                _state = json.load(f)
            _state.setdefault("next_id", 1)
            _state.setdefault("queue", [])
            _state.setdefault("results", {})
        except Exception:
            pass


def _save():
    try:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_state, f, ensure_ascii=False)
        os.replace(tmp, STATE_PATH)
    except Exception:
        pass


_load()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # 静音默认日志
        pass

    # ---------- helpers ----------
    def _auth_ok(self):
        got = self.headers.get("X-Relay-Token", "")
        if not got:
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            for kv in q.split("&"):
                if kv.startswith("k="):
                    got = kv[2:]
        return got == SECRET

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _q(self, key, default=None):
        if "?" not in self.path:
            return default
        for kv in self.path.split("?", 1)[1].split("&"):
            if kv.startswith(key + "="):
                return kv[len(key) + 1:]
        return default

    def _agent_source(self):
        """读 agent 模板并注入本次部署的 URL / token / sharecode。"""
        tpl_path = os.path.join(HERE, "agent_template.py")
        if not os.path.exists(tpl_path):
            return None
        with open(tpl_path, "r", encoding="utf-8") as f:
            src = f.read()
        src = src.replace("__RELAY_URL__", PUBLIC_URL)
        src = src.replace("__RELAY_TOKEN__", SECRET)
        src = src.replace("__RELAY_SHARECODE__", SHARECODE)
        return src

    # ---------- routes ----------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        # /bootstrap 用一次性码鉴权（不用长期 token）——
        # 这样"在 AutoDL 终端粘贴启动 agent"这件事永远不需要暴露长期凭据。
        if path == "/bootstrap":
            code = self._q("code", "") or ""
            with _lock:
                otc = _state.setdefault("otc", {})
                rec = otc.get(code)
                if not rec:
                    return self._send(401, {"ok": False, "error": "bad or used code"})
                if rec.get("used"):
                    return self._send(401, {"ok": False, "error": "code already used"})
                if int(time.time()) > int(rec.get("expires", 0)):
                    otc.pop(code, None)
                    _save()
                    return self._send(401, {"ok": False, "error": "code expired"})
                rec["used"] = True
                _save()
            src = self._agent_source()
            if src is None:
                return self._send(500, {"ok": False, "error": "agent_template.py missing"})
            body = src.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/x-python; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        if not self._auth_ok():
            return self._send(401, {"ok": False, "error": "bad token"})
        if path == "/health":
            with _lock:
                hb = _state.get("heartbeat", {}) or {}
                return self._send(200, {"ok": True, "queued": len(_state["queue"]),
                                        "results": len(_state["results"]),
                                        "next_id": _state["next_id"],
                                        "agent_alive": bool(hb),
                                        "agent_pid": hb.get("pid"),
                                        "agent_seen": hb.get("ts")})
        if path == "/poll":
            # 返回所有"尚未完成"的命令；不删除队列，靠 results 里的 id 判定完成，
            # 避免因 agent 取走后崩溃导致命令丢失。超时未回传的会被重新派发。
            now = int(time.time())
            with _lock:
                done = set(_state["results"].keys())
                cmds, requeue = [], []
                for c in _state["queue"]:
                    if str(c["id"]) in done:
                        continue
                    claim = c.get("claimed_at", 0)
                    if claim and now - claim < int(c.get("timeout", 600)) + 180:
                        continue  # 正在执行中
                    if claim:
                        requeue.append(c["id"])
                    c["claimed_at"] = now
                    cmds.append({"id": c["id"], "cmd": c["cmd"], "cwd": c.get("cwd", ""),
                                 "timeout": c.get("timeout", 600), "bg": c.get("bg", False)})
                if cmds:
                    _save()
            if requeue:
                print(f"[relay] re-dispatch timed-out cmds: {requeue}", flush=True)
            return self._send(200, {"cmds": cmds})
        if path == "/results":
            since = int(self._q("since", "0") or 0)
            with _lock:
                res = [r for rid, r in sorted(_state["results"].items(), key=lambda kv: int(kv[0]))
                       if int(rid) > since]
            return self._send(200, {"results": res})
        if path == "/agent.py":
            # 下发 agent 脚本（自动填入本次部署的 URL 与密钥）
            src = self._agent_source()
            if src is None:
                return self._send(500, {"ok": False, "error": "agent_template.py missing"})
            body = src.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/x-python; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        return self._send(404, {"ok": False, "error": "no route"})

    def do_POST(self):
        if not self._auth_ok():
            return self._send(401, {"ok": False, "error": "bad token"})
        path = self.path.split("?", 1)[0]
        body = self._read_json()

        if path == "/heartbeat":
            # agent 存活心跳：客户端据此区分"agent 死了"和"agent 在跑长命令"
            with _lock:
                _state.setdefault("heartbeat", {})["pid"] = body.get("pid")
                _state.setdefault("heartbeat", {})["ts"] = int(time.time())
            return self._send(200, {"ok": True})

        if path == "/clear":
            # 清空未完成的队列（agent 挂掉后积压的命令会被重派，先清掉）
            with _lock:
                done = set(_state["results"].keys())
                n = len(_state["queue"])
                _state["queue"] = [c for c in _state["queue"] if str(c["id"]) in done]
                _save()
            return self._send(200, {"ok": True, "cleared": n - len(_state["queue"]),
                                    "queue": len(_state["queue"])})

        if path == "/newcode":
            # 签发一次性 bootstrap 码：单次使用、默认 15 分钟失效。
            # 用途：让"在 AutoDL 终端粘贴命令启动 agent"这件事
            #      不需要把长期 token 写进任何脚本/历史/剪贴板。
            ttl = int(body.get("ttl", 900))
            code = secrets.token_urlsafe(18)
            with _lock:
                otc = _state.setdefault("otc", {})
                now = int(time.time())
                for k in [k for k, v in otc.items()
                          if int(v.get("expires", 0)) < now or v.get("used")]:
                    otc.pop(k, None)
                otc[code] = {"expires": now + ttl, "used": False}
                _save()
            return self._send(200, {"ok": True, "code": code, "ttl": ttl,
                                    "url": f"{PUBLIC_URL}/bootstrap?code={code}"})

        if path == "/enqueue":
            cmd = (body.get("cmd") or "").strip()
            if not cmd:
                return self._send(400, {"ok": False, "error": "empty cmd"})
            with _lock:
                cid = _state["next_id"]
                _state["next_id"] = cid + 1
                _state["queue"].append({
                    "id": cid, "cmd": cmd, "cwd": body.get("cwd", ""),
                    "timeout": int(body.get("timeout", 600)),
                    "bg": bool(body.get("bg", False)),
                })
                _save()
            return self._send(200, {"ok": True, "id": cid})

        if path == "/result":
            cid = body.get("id")
            if cid is None:
                return self._send(400, {"ok": False, "error": "no id"})
            out = str(body.get("out", ""))[:MAX_OUT]
            err = str(body.get("err", ""))[:MAX_OUT]
            with _lock:
                _state["results"][str(cid)] = {
                    "id": cid, "rc": body.get("rc"), "out": out, "err": err,
                    "ts": int(time.time()),
                }
                # 只保留最近 500 条
                if len(_state["results"]) > 500:
                    for k in sorted(_state["results"], key=lambda x: int(x))[:-500]:
                        _state["results"].pop(k, None)
                _save()
            return self._send(200, {"ok": True})

        if path == "/run":
            cmd = (body.get("cmd") or "").strip()
            if not cmd:
                return self._send(400, {"ok": False, "error": "empty cmd"})
            wait = int(body.get("wait", 900))
            with _lock:
                cid = _state["next_id"]
                _state["next_id"] = cid + 1
                _state["queue"].append({
                    "id": cid, "cmd": cmd, "cwd": body.get("cwd", ""),
                    "timeout": int(body.get("timeout", wait)), "bg": False,
                })
                _save()
            deadline = time.time() + wait
            while time.time() < deadline:
                time.sleep(2)
                with _lock:
                    r = _state["results"].get(str(cid))
                if r is not None:
                    return self._send(200, {"ok": True, "result": r})
            return self._send(504, {"ok": False, "error": "timeout waiting for agent",
                                    "id": cid})

        return self._send(404, {"ok": False, "error": "no route"})


def main():
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[relay] listening on 0.0.0.0:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
