"""把 tools/tmp/<name>.sh 全文下发到云端执行并回显。

为什么需要它（2026-11-22 实测踩坑）：
    `python -c "..."` 会被本地 Bash 的双引号提前展开 `$OUT` / `$t`
    => 云端实际执行的是 `F=/scanlog_.log`，读出来全是空（假象）。
    写文件 + 整体下发可彻底绕开所有转义问题。

用法：
    python tools/tmp/send.py dp3.sh
"""
import os
import pathlib
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tools/

import cloud  # noqa: E402


def main():
    name = sys.argv[1]
    if not name.endswith(".sh"):
        name += ".sh"
    path = pathlib.Path(HERE) / name
    cmd = path.read_text(encoding="utf-8")
    res = cloud.run(cmd, cwd="/root/HPT", timeout=240, wait=360)
    out = res.get("out") or ""
    err = res.get("err") or ""
    sys.stdout.write(out)
    if err.strip():
        sys.stderr.write("\n--stderr--\n" + err[:4000])
    print(f"\n--rc={res.get('rc')} cid={res.get('id')}")


if __name__ == "__main__":
    main()
