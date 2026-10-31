"""AutoDL 自动化通道：API 列实例 → paramiko SSH → 上传/执行/下载。

用法（Windows 本机执行）：
    python autodl_auto.py list                       # 列实例（看状态/SSH信息）
    python autodl_auto.py exec "nvidia-smi"          # 在实例上执行命令
    python autodl_auto.py put local.txt /root/HPT/   # 上传文件
    python autodl_auto.py get /root/HPT/out.log ./   # 下载文件
    python autodl_auto.py putx a.sh /root/HPT/a.sh   # 上传并 chmod +x
    python autodl_auto.py script a.sh               # 上传并执行（可传 env/参数）

凭据（任一）：
    环境变量 AUTODL_TOKEN  —— 控制台 → 设置 → 开发者 Token
    环境变量 AUTODL_SSH_HOST / AUTODL_SSH_PORT / AUTODL_SSH_PASSWORD —— 手填 SSH 信息

Token 获取：https://www.autodl.com/console/homepage/personal 设置 → 开发者 Token
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
import time

API = "https://www.autodl.com/api/v1/instance"

# 实例 SSH 主机名形如 connect.xxxx.seetacloud.com，需要解析出 host 与 port
SSH_RE = None


def _api_headers(token: str) -> dict:
    return {"Authorization": token, "Content-Type": "application/json"}


def api_list_instances(token: str) -> list:
    """调 AutoDL API 列出实例。

    文档给的写法是 POST + JSON body {}；用 requests 的话 GET 也常被接受。
    这里两个都试，兼容不同版本。
    """
    import urllib.request
    import urllib.error

    def _do(method: str):
        data = json.dumps({}).encode()
        req = urllib.request.Request(
            API, data=data, headers=_api_headers(token), method=method
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    last = None
    for method in ("POST", "GET"):
        try:
            r = _do(method)
            if isinstance(r, dict) and r.get("code") not in (None, "Success"):
                last = r
                continue
            return r.get("data", r) if isinstance(r, dict) else r
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"API 调用失败：{last}")


def parse_instances(payload) -> list:
    """从 API 返回里提取实例列表（兼容多种嵌套结构）。

    实际返回可能是：
        {"code":"Success","data":{"list":[...], "total":n}}   ← 官方文档形态
        {"code":"Success","data":[...]}                        ← 扁平形态
        [...]                                                  ← 裸数组
    """
    def _walk(node, depth=0):
        if depth > 6:
            return None
        if isinstance(node, list):
            # 列表：若元素都是 dict 且含实例特征字段，则认为命中
            if node and all(isinstance(x, dict) for x in node):
                return node
            return None
        if isinstance(node, dict):
            for k in ("list", "instances", "items", "rows"):
                v = node.get(k)
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    return v
            for v in node.values():
                r = _walk(v, depth + 1)
                if r:
                    return r
        return None

    r = _walk(payload)
    if r is not None:
        return r
    if isinstance(payload, dict):
        return [payload]
    return []


def pick_instance(instances: list, want: str | None) -> dict:
    if not instances:
        raise RuntimeError("API 没返回任何实例")
    if want:
        for it in instances:
            if want in json.dumps(it, ensure_ascii=False):
                return it
        raise RuntimeError(f"没找到匹配 {want!r} 的实例")
    # 默认取第一个 running 的，否则第一个
    for it in instances:
        st = str(it.get("status", "")).lower()
        if st in ("running", "started", "online"):
            return it
    return instances[0]


def _first(d: dict, *names, default=None):
    for n in names:
        if isinstance(d, dict) and d.get(n) not in (None, ""):
            return d[n]
    return default


def ssh_info(inst: dict) -> tuple[str, int, str]:
    """从实例信息里抽出 (host, port, password)。"""
    cmd = _first(inst, "ssh_command", "sshCommand", default="") or ""
    pwd = _first(inst, "root_password", "rootPassword", "password", default="") or ""
    host = _first(inst, "ssh_host", "sshHost", "host", default="") or ""
    port = _first(inst, "ssh_port", "sshPort", "port", default=None)

    if cmd:
        # 形如: ssh -p 12345 root@connect.xxx.seetacloud.com
        import re
        m = re.search(r"-p\s+(\d+)", cmd)
        if m and not port:
            port = int(m.group(1))
        m = re.search(r"root@([\w\.\-]+)", cmd)
        if m and not host:
            host = m.group(1)
        if not host:
            m = re.search(r"([\w\.\-]+\.seetacloud\.com)", cmd)
            if m:
                host = m.group(1)

    if not host:
        raise RuntimeError(f"实例信息里找不到 SSH 主机名：{json.dumps(inst, ensure_ascii=False)[:400]}")
    port = int(port or 22)
    return host, port, pwd


def connect(host: str, port: int, password: str, timeout: int = 30):
    import paramiko

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        host,
        port=port,
        username="root",
        password=password or None,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
        allow_agent=False,
        look_for_keys=False,
    )
    return c


def _client(args):
    """建立连接：优先环境变量手填，其次用 token 查 API。"""
    host = os.environ.get("AUTODL_SSH_HOST")
    port = os.environ.get("AUTODL_SSH_PORT")
    pwd = os.environ.get("AUTODL_SSH_PASSWORD")
    if host:
        return connect(host, int(port or 22), pwd or "")

    token = os.environ.get("AUTODL_TOKEN", "")
    if not token:
        raise RuntimeError(
            "缺少凭据：请设环境变量 AUTODL_TOKEN（控制台→设置→开发者Token），"
            "或设 AUTODL_SSH_HOST/AUTODL_SSH_PORT/AUTODL_SSH_PASSWORD"
        )
    insts = parse_instances(api_list_instances(token))
    inst = pick_instance(insts, args.instance)
    h, p, pw = ssh_info(inst)
    print(f"[auto] 选中实例 status={inst.get('status')} -> {h}:{p}", file=sys.stderr)
    return connect(h, p, pw)


def cmd_list(args):
    token = os.environ.get("AUTODL_TOKEN", "")
    if not token:
        print("没有 AUTODL_TOKEN，无法列实例。可改用 AUTODL_SSH_HOST 手填模式。")
        return 1
    payload = api_list_instances(token)
    insts = parse_instances(payload)
    print(f"共 {len(insts)} 个实例：")
    for it in insts:
        h = _first(it, "ssh_host", "host", default="")
        p = _first(it, "ssh_port", "port", default="")
        print(json.dumps({
            "uuid": _first(it, "uuid", "id", "instance_uuid", default=""),
            "name": _first(it, "name", "instance_name", default=""),
            "status": _first(it, "status", default=""),
            "sub_status": _first(it, "sub_status", default=""),
            "ssh": f"{h}:{p}" if h else "",
            "has_password": bool(_first(it, "root_password", "password", default="")),
        }, ensure_ascii=False))
    return 0


def cmd_exec(args):
    c = _client(args)
    try:
        chan = c.get_transport().open_session()
        chan.get_pty(width=200, height=50)
        chan.exec_command(args.command)
        out = []
        while True:
            if chan.recv_ready():
                data = chan.recv(65536).decode("utf-8", "replace")
                out.append(data)
                sys.stdout.write(data)
                sys.stdout.flush()
            elif chan.recv_stderr_ready():
                data = chan.recv_stderr(65536).decode("utf-8", "replace")
                out.append(data)
                sys.stderr.write(data)
                sys.stderr.flush()
            elif chan.exit_status_ready():
                while chan.recv_ready():
                    d = chan.recv(65536).decode("utf-8", "replace")
                    out.append(d)
                    sys.stdout.write(d)
                break
            else:
                time.sleep(0.15)
        rc = chan.recv_exit_status()
        print(f"\n[auto] exit code = {rc}")
        return rc
    finally:
        c.close()


def _sftp(c):
    return c.open_sftp()


def cmd_put(args):
    c = _client(args)
    try:
        s = _sftp(c)
        remote = args.remote
        d = posixpath.dirname(remote)
        if d and d not in ("/", ""):
            try:
                s.mkdir(d)
            except IOError:
                pass
        s.put(args.local, remote)
        if args.chmod:
            s.chmod(remote, 0o755)
        print(f"[auto] 已上传 {args.local} -> {remote}")
        return 0
    finally:
        c.close()


def cmd_get(args):
    c = _client(args)
    try:
        s = _sftp(c)
        s.get(args.remote, args.local)
        print(f"[auto] 已下载 {args.remote} -> {args.local}")
        return 0
    finally:
        c.close()


def cmd_script(args):
    """上传脚本 → chmod → 执行 → 打印输出。"""
    c = _client(args)
    try:
        s = _sftp(c)
        s.put(args.local, args.remote)
        s.chmod(args.remote, 0o755)
        chan = c.get_transport().open_session()
        chan.get_pty(width=200, height=50)
        chan.exec_command(args.remote)
        out = []
        while True:
            if chan.recv_ready():
                d = chan.recv(65536).decode("utf-8", "replace")
                out.append(d)
                sys.stdout.write(d)
                sys.stdout.flush()
            elif chan.recv_stderr_ready():
                d = chan.recv_stderr(65536).decode("utf-8", "replace")
                out.append(d)
                sys.stderr.write(d)
                sys.stderr.flush()
            elif chan.exit_status_ready():
                while chan.recv_ready():
                    d = chan.recv(65536).decode("utf-8", "replace")
                    out.append(d)
                    sys.stdout.write(d)
                break
            else:
                time.sleep(0.15)
        rc = chan.recv_exit_status()
        print(f"\n[auto] exit code = {rc}")
        return rc
    finally:
        c.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="AutoDL 自动化通道")
    ap.add_argument("--instance", default=None, help="实例名/uuid 关键字，默认取第一个 running")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出实例")

    p = sub.add_parser("exec", help="执行远程命令")
    p.add_argument("command")

    p = sub.add_parser("put", help="上传文件")
    p.add_argument("local")
    p.add_argument("remote")
    p.add_argument("--chmod", action="store_true")

    p = sub.add_parser("get", help="下载文件")
    p.add_argument("remote")
    p.add_argument("local")

    p = sub.add_parser("script", help="上传并执行脚本")
    p.add_argument("local")
    p.add_argument("remote")

    args = ap.parse_args(argv)
    fn = {
        "list": cmd_list, "exec": cmd_exec, "put": cmd_put,
        "get": cmd_get, "script": cmd_script,
    }[args.cmd]
    try:
        return fn(args)
    except Exception as e:  # noqa: BLE001
        print(f"[auto][ERR] {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
