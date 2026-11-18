#!/usr/bin/env python3
"""诊断 relay POST /result 403 的成因（区分：token / sharecode / WAF / 请求体大小）。"""
import json
import os
import re
import urllib.error
import urllib.request

AGENT = "/root/HPT/agent.py"
txt = open(AGENT, encoding="utf-8", errors="ignore").read()


def grab(name, default=""):
    m = re.search(name + r'\s*=\s*"([^"]*)"', txt)
    return m.group(1) if m else default


TOKEN = grab("RELAY_TOKEN")
SC = grab("RELAY_SHARECODE")
URL = grab("RELAY_URL", "https://fedvla-relay.app.workbuddy.host").rstrip("/")

print(f"token 长度={len(TOKEN)}  sharecode 长度={len(SC)}  url={URL}")
print(f"token 前缀形态={TOKEN[:6]!r}  sharecode 前缀形态={SC[:6]!r}")


def hit(label, path, body=None, with_sc=False, extra=None):
    url = URL + path
    if with_sc:
        url += ("&" if "?" in url else "?") + "sharecode=" + SC
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    req.add_header("X-Relay-Token", TOKEN)
    req.add_header("Content-Type", "application/json")
    for k, v in (extra or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body_txt = r.read().decode("utf-8", "replace")[:120]
            print(f"  [{label}] HTTP {r.status}  {body_txt}")
            return r.status
    except urllib.error.HTTPError as e:
        print(f"  [{label}] HTTP {e.code} {e.reason}  "
              f"{e.read().decode('utf-8','replace')[:120]}")
        return e.code
    except Exception as e:
        print(f"  [{label}] ERR {e!r}")
        return -1


print("--- 1) GET /health 变体 ---")
hit("health 无sc", "/health")
hit("health 带sc", "/health", with_sc=True)

print("--- 2) POST /result 极小 body 变体 ---")
tiny = {"id": 960001, "out": "ping", "err": "", "rc": 0}
hit("result 无sc 小body", "/result", body=tiny)
hit("result 带sc 小body", "/result", body=tiny, with_sc=True)

print("--- 3) POST /result 大 body 变体（模拟 100KB base64）---")
big = {"id": 960002, "out": "A" * 100000, "err": "", "rc": 0}
hit("result 带sc 大body", "/result", body=big, with_sc=True)
