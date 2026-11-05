"""云端命令执行器（可靠版）。

背景（2026-11-05 实测）：sites 边缘网关对 URL 里的 **第二个 `?`** 直接返回 403。
`/results?since=1100` + `?sharecode=XXX` 拼在一起就成了 `?since=1100?sharecode=XXX`
→ 全程 403。正确写法必须用 `&` 连接。

另一个坑：`curl` 对 HTTP 403/404 返回**退出码 0**，会静默把网关的 HTML 错误页
当成结果。所以本工具用 urllib + 显式 HTTPError 处理 + 指数退避重试。

用法：
    python relay/cloud.py "<命令>"              # 执行并打印 stdout/stderr
    python relay/cloud.py "<命令>" --timeout 600 # 放宽等待上限
    python relay/cloud.py "<命令>" --cwd /root  # 指定工作目录
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def load_creds():
    token = os.environ.get("RELAY_SECRET", "")
    sc = os.environ.get("RELAY_SHARECODE", "")
    url = os.environ.get("RELAY_PUBLIC_URL", "").rstrip("/")
    path = os.environ.get("RELAY_SECRET_FILE") or os.path.join(HERE, "relay_secret.txt")
    if not (token and sc and url) and os.path.exists(path):
        lines = [l.strip() for l in open(path, encoding="utf-8")
                 if l.strip() and not l.strip().startswith("#")]
        if len(lines) >= 3:
            token = token or lines[0]
            sc = sc or lines[1]
            url = url or lines[2].rstrip("/")
    if not (token and sc and url):
        raise SystemExit("!! 缺少 relay 凭据")
    return token, sc, url


TOKEN, SC, URL = load_creds()
_op = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _url(path):
    """关键：URL 里只能有一个 '?'。已有 query 就用 '&' 接 sharecode。"""
    sep = "&" if "?" in path else "?"
    return URL + f"{path}{sep}sharecode={SC}"


def req(path, data=None, timeout=90, retries=8):
    """带重试的请求。403 会被网关限流，必须重试。"""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
    last = None
    for attempt in range(retries):
        r = urllib.request.Request(_url(path), data=body,
                                   method="POST" if data is not None else "GET")
        r.add_header("X-Relay-Token", TOKEN)
        r.add_header("Content-Type", "application/json")
        try:
            with _op.open(r, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code not in (403, 429, 502, 503):
                raise
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"relay 请求失败（重试 {retries} 次）: {last}")


def run(cmd, cwd="/root/HPT", timeout=600, wait=600):
    r = req("/enqueue", data={"cmd": cmd, "cwd": cwd,
                               "timeout": timeout, "bg": False}, timeout=120)
    cid = r["id"]
    t0 = time.time()
    while time.time() - t0 < wait:
        try:
            res = req(f"/results?since={cid - 1}", timeout=45, retries=3)
        except Exception:  # noqa: BLE001
            continue
        for x in res.get("results", []):
            if int(x["id"]) == cid:
                return x
        time.sleep(3)
    raise SystemExit(f"!! {wait}s 内没有拿到 cid={cid} 的结果")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--cwd", default="/root/HPT")
    ap.add_argument("--timeout", type=int, default=600, help="云端命令超时上限")
    ap.add_argument("--wait", type=int, default=600, help="本地等待回传上限")
    ap.add_argument("--raw", action="store_true", help="只打印 stdout")
    args = ap.parse_args()

    res = run(args.cmd, cwd=args.cwd, timeout=args.timeout, wait=args.wait)
    out = res.get("out") or ""
    err = res.get("err") or ""
    if args.raw:
        print(out)
    else:
        print(out)
        if err.strip():
            print("--stderr--", file=sys.stderr)
            print(err[:3000], file=sys.stderr)
    sys.exit(0 if res.get("rc") in (0, None) else 1)


if __name__ == "__main__":
    main()
