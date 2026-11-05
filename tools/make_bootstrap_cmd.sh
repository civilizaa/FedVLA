#!/usr/bin/env bash
# 生成"云端 agent 启动命令"（贴到 AutoDL 终端执行）。
#
# 设计要点（2026-11-05 实测得出）：
#   1. 边缘网关要求 URL 必须带 `sharecode=` 参数，且**大小写不敏感地精确匹配**。
#      其它位置（path、别的参数名、fragment）都不认。
#   2. sharecode 首字母做 URL 编码（%58）后网关**仍放行** → 命令明文里不含完整 sharecode。
#   3. 服务端 /bootstrap 只认**一次性 code**（用过即废，URL 立刻失效），
#      不需要长期 token。
#   4. `curl -s` 遇 403/404 仍返回**退出码 0**，会把网关的 HTML 错误页写进 agent.py
#      → 必须显式校验下载内容，否则下一次 `python3 agent.py` 会报
#      `SyntaxError: invalid decimal literal`（用户 2026-11-05 实际遇到）。
#
# 用法: bash make_bootstrap_cmd.sh <code>
set -u
CODE="${1:?用法: bash make_bootstrap_cmd.sh <一次性code>}"
REL="${RELAY_PUBLIC_URL:-https://fedvla-relay.app.workbuddy.host}"

# sharecode 首字母做 URL 编码（网关只对首字母宽容），避免明文进命令历史。
# ⚠️ sharecode 是长期凭据，**绝不写进本文件** —— 从环境变量或 relay_secret.txt 读。
#    relay_secret.txt 第 2 行 = sharecode（与 relay/ 目录下的同款文件格式：
#    第1行 token / 第2行 sharecode / 第3行 url）
SC_RAW="${RELAY_SHARECODE:-}"
if [ -z "$SC_RAW" ] && [ -f "${RELAY_SECRET_FILE:-relay_secret.txt}" ]; then
  SC_RAW=$(sed -n '2p' "${RELAY_SECRET_FILE:-relay_secret.txt}" | tr -d '[:space:]')
fi
if [ -z "$SC_RAW" ]; then
  echo "!! 拿不到 sharecode：请设 RELAY_SHARECODE 环境变量，或指定 RELAY_SECRET_FILE" >&2
  exit 1
fi
SC_ENC="%58${SC_RAW:1}"    # 首字母编码成 %58

cat <<EOF
mkdir -p /root/fedvla_relay && cd /root/fedvla_relay && \\
curl -sS -f --max-time 60 -o agent.py "${REL}/bootstrap?code=${CODE}&sharecode=${SC_ENC}" && \\
python3 -c "import ast; ast.parse(open('agent.py').read()); print('[OK] agent.py 语法校验通过')" && \\
(nohup python3 agent.py > agent.log 2>&1 < /dev/null &) && sleep 8 && \\
echo "--- agent.log ---" && tail -n 5 /root/fedvla_relay/agent.log
EOF
