#!/usr/bin/env python3
"""探测 AutoDL API：用本地 token 文件列出实例（不打印任何明文凭据）。"""
from __future__ import annotations
import json
import os
import sys
import urllib.request
import urllib.error

API = "https://www.autodl.com/api/v1/instance"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(ROOT)


def load_token():
    for p in (os.path.join(WS, ".autodl_token.tmp"),
              os.path.join(ROOT, ".autodl_token.tmp"),
              os.path.join(os.path.expanduser("~"), ".autodl_token.tmp")):
        if os.path.exists(p):
            t = open(p, encoding="utf-8", errors="ignore").read().strip()
            if t:
                return t, p
    return "", ""


def api(method, token, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        API, data=data,
        headers={"Authorization": token, "Content-Type": "application/json"},
        method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    tok, path = load_token()
    if not tok:
        print("NO_TOKEN_FILE")
        return 2
    print(f"token_file={path} len={len(tok)}")
    for method in ("POST", "GET"):
        try:
            r = api(method, tok)
        except urllib.error.HTTPError as e:
            print(f"{method} HTTP {e.code}: {e.read()[:200].decode('utf-8','ignore')}")
            continue
        except Exception as e:
            print(f"{method} ERR {e!r}"[:200])
            continue
        print(f"{method} code={r.get('code')} msg={r.get('msg')}")
        data = r.get("data", r)
        items = None
        if isinstance(data, dict):
            for k in ("list", "instances", "items"):
                if isinstance(data.get(k), list):
                    items = data[k]
                    break
        elif isinstance(data, list):
            items = data
        if items is None:
            print("  no list; keys=" + str(list(data.keys()) if isinstance(data, dict) else type(data)))
            continue
        for it in items:
            def g(*names):
                for n in names:
                    if it.get(n) not in (None, ""):
                        return it[n]
                return None
            print(json.dumps({
                "uuid": g("uuid", "instance_uuid", "id"),
                "name": g("name", "instance_name"),
                "status": g("status", "state"),
                "gpu": g("gpu_type", "gpuName", "gpu"),
                "region": g("region", "area"),
                "ssh_cmd_present": bool(g("ssh_command", "sshCommand")),
            }, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
