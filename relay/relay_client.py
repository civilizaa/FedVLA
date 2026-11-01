#!/usr/bin/env python3
"""FedVLA relay —— 本机客户端。

用法：
  python relay_client.py exec "命令"            # 执行并等结果（默认 15 分钟）
  python relay_client.py exec -b "长任务"        # 后台执行，立刻返回
  python relay_client.py put 本地文件 远端路径   # 上传文件（base64 内联）
  python relay_client.py get 远端路径 本地文件   # 下载文件
  python relay_client.py status                 # 查看队列与历史结果
  python relay_client.py results [since]        # 拉取结果
  python relay_client.py watch                  # 持续监视新结果

环境变量（也可写 relay_client.json）：
  RELAY_URL / RELAY_TOKEN / RELAY_SHARECODE
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "relay_client.json")
SECRET_PATH = os.path.join(HERE, "relay_secret.txt")


def _from_secret_file():
    """读 relay_secret.txt（已 gitignore）：第1行 token，第2行 sharecode，第3行 url。"""
    if not os.path.exists(SECRET_PATH):
        return {}
    try:
        with open(SECRET_PATH, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f
                     if l.strip() and not l.strip().startswith("#")]
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    if len(lines) > 0:
        out["token"] = lines[0]
    if len(lines) > 1:
        out["sharecode"] = lines[1]
    if len(lines) > 2:
        out["url"] = lines[2]
    return out


def _cfg():
    """凭据来源优先级：环境变量 > relay_client.json > relay_secret.txt。

    注意：仓库是 public，密钥绝不能硬编码在代码里。
    """
    c = {
        "url": os.environ.get("RELAY_URL", ""),
        "token": os.environ.get("RELAY_TOKEN", ""),
        "sharecode": os.environ.get("RELAY_SHARECODE", ""),
    }
    c.update(_from_secret_file())
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH, encoding="utf-8") as f:
                c.update(json.load(f))
        except Exception:  # noqa: BLE001
            pass
    missing = [k for k, v in c.items() if not v]
    if missing:
        raise SystemExit(
            "缺少凭据: " + ", ".join(missing) +
            f"\n请写 {SECRET_PATH}（第1行 token / 第2行 sharecode / 第3行 url）"
            " 或设同名环境变量。"
        )
    return c


C = _cfg()


def call(path, payload=None, method=None, timeout=60, raw=False):
    url = C["url"].rstrip("/") + path
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}sharecode={C['sharecode']}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"X-Relay-Token": C["token"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read().decode("utf-8", errors="replace")
        return b if raw else json.loads(b)


def enqueue(cmd, cwd="", timeout=600, bg=False):
    return call("/enqueue", {"cmd": cmd, "cwd": cwd, "timeout": timeout, "bg": bg})["id"]


def wait_result(cid, wait=900):
    deadline = time.time() + wait
    while time.time() < deadline:
        time.sleep(3)
        res = call(f"/results?since={cid - 1}")["results"]
        for r in res:
            if r["id"] == cid:
                return r
    return None


def show(r):
    print(f"--- rc={r.get('rc')} ---")
    if r.get("out"):
        print(r["out"].rstrip())
    if r.get("err"):
        print("[stderr]", r["err"].rstrip())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("exec")
    p.add_argument("command")
    p.add_argument("-b", "--background", action="store_true")
    p.add_argument("-c", "--cwd", default="")
    p.add_argument("-t", "--timeout", type=int, default=900)

    p = sub.add_parser("put")
    p.add_argument("local")
    p.add_argument("remote")

    p = sub.add_parser("get")
    p.add_argument("remote")
    p.add_argument("local")

    sub.add_parser("status")
    p = sub.add_parser("results")
    p.add_argument("since", nargs="?", default="0")
    sub.add_parser("watch")

    a = ap.parse_args()

    try:
        if a.cmd == "exec":
            cid = enqueue(a.command, a.cwd, a.timeout, a.background)
            if a.background:
                print(f"后台已派发 cmd#{cid}")
                return 0
            r = wait_result(cid, a.timeout)
            if r is None:
                print(f"等待超时（{a.timeout}s），命令 id={cid} 仍在队列中")
                return 3
            show(r)
            return int(r.get("rc") or 0)

        if a.cmd == "put":
            with open(a.local, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            cmd = (f"mkdir -p $(dirname {a.remote}) && "
                   f"echo {b64} | base64 -d > {a.remote} && "
                   f"chmod +x {a.remote} 2>/dev/null; ls -l {a.remote}")
            cid = enqueue(cmd, "", 300)
            r = wait_result(cid, 300)
            if r:
                show(r)
                return int(r.get("rc") or 0)
            print("上传超时")
            return 3

        if a.cmd == "get":
            cmd = f"base64 -w0 {a.remote} 2>/dev/null || base64 {a.remote}"
            cid = enqueue(cmd, "", 300)
            r = wait_result(cid, 300)
            if not r or not r.get("out"):
                print("读取失败:", (r or {}).get("err"))
                return 3
            data = base64.b64decode(r["out"].strip())
            with open(a.local, "wb") as f:
                f.write(data)
            print(f"已保存 {a.local} ({len(data)} bytes)")
            return 0

        if a.cmd == "status":
            print(json.dumps(call("/health"), ensure_ascii=False))
            return 0

        if a.cmd == "results":
            for r in call(f"/results?since={a.since}")["results"]:
                print(f"===== cmd#{r['id']} rc={r.get('rc')} =====")
                if r.get("out"):
                    print(r["out"].rstrip())
                if r.get("err"):
                    print("[stderr]", r["err"].rstrip())
            return 0

        if a.cmd == "watch":
            seen = 0
            while True:
                for r in call(f"/results?since={seen}")["results"]:
                    seen = max(seen, r["id"])
                    print(f"===== cmd#{r['id']} rc={r.get('rc')} =====")
                    if r.get("out"):
                        print(r["out"].rstrip())
                    if r.get("err"):
                        print("[stderr]", r["err"].rstrip())
                    sys.stdout.flush()
                time.sleep(5)
    except KeyboardInterrupt:
        return 0
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
