#!/usr/bin/env python3
"""一次调用拿齐 relay 状态：健康 / 队列 / 最近结果。

用法: python3 tools/relay_status_probe.py [--since 980000]
输出尽量短，便于塞进 automation prompt。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "https://fedvla-relay.app.workbuddy.host"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read_creds(paths):
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8", errors="ignore").read()

        def grab(name, default=""):
            m = re.search(name + r'\s*=\s*"([^"]*)"', txt)
            return m.group(1) if m else default

        tok = grab("RELAY_TOKEN")
        sc = grab("RELAY_SHARECODE")
        url = grab("RELAY_URL", DEFAULT_URL).rstrip("/")
        if tok and not tok.startswith("__RELAY"):
            return tok, sc, url
    return "", "", DEFAULT_URL


def get(url, token, sc, timeout=30):
    full = url + ("&" if "?" in url else "?") + f"sharecode={sc}"
    req = urllib.request.Request(full, headers={"X-Relay-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=980000)
    ap.add_argument("--maxresults", type=int, default=5)
    args = ap.parse_args()

    tok, sc, url = read_creds([
        os.path.join(ROOT, "relay", "agent_orig.py"),
        os.path.join(ROOT, "relay", "agent_test.py"),
        os.path.join(ROOT, "relay", "agent.py"),
        os.path.join(os.path.dirname(ROOT), "relay", "agent_orig.py"),
        os.path.join(os.path.dirname(ROOT), "relay", "agent_test.py"),
        "/root/fedvla_relay/agent.py",
        "/root/HPT/agent.py",
    ])
    if not tok:
        print("NO_CREDS")
        return 2
    print(f"url={url}")
    try:
        h = get(url + "/health", tok, sc)
        print("health=" + h[:600])
    except Exception as e:
        print("health_ERR=" + repr(e)[:200])
    try:
        r = json.loads(get(url + f"/results?since={args.since}", tok, sc))
        items = r if isinstance(r, list) else r.get("results", [])
        print(f"results_count={len(items)}")
        for it in items[-args.maxresults:]:
            out = (it.get("out") or "")[:400].replace("\n", " | ")
            err = (it.get("err") or "")[:200].replace("\n", " | ")
            print(f"  id={it.get('id')} rc={it.get('rc')} out={out} err={err}")
    except Exception as e:
        print("results_ERR=" + repr(e)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
