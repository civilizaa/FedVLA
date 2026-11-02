#!/usr/bin/env bash
# FedVLA 云端 agent 一键启动（无凭据版）
#
# 与旧版 START_AGENT2.sh 的区别：
#   旧版把含明文 token 的 agent 源码 base64 内嵌在脚本里 → 脚本一进 git 就等于公开密钥。
#   本版只带一个【一次性 bootstrap 码】：单次使用、默认 15 分钟失效，
#   agent 源码由 relay 服务端在取用瞬间注入凭据并下发。
#   → 无论这个脚本怎么传播，都不会泄露可复用的长期凭据。
#
# 用法：
#   1) 在本机执行：python relay/relay_client.py bootstrap
#      它会打印一行完整命令（内含当次有效的一次性码）
#   2) 把那一行粘贴到 AutoDL 终端回车
#
# 脚本本体（供本机生成命令用，也可直接粘贴）：
set -e
D=/root/fedvla_relay
mkdir -p "$D" && cd "$D"
# __BOOTSTRAP_URL__ 会被本机脚本替换成 https://<relay>/bootstrap?code=<一次性码>&sharecode=<码>
curl -sSL "__BOOTSTRAP_URL__" -o agent.py
python3 - <<'PYEOF'
import ast
ast.parse(open("agent.py").read())
print("agent.py 语法 OK")
PYEOF
grep -c start_new_session agent.py
nohup python3 agent.py > "$D/agent.log" 2>&1 < /dev/null &
sleep 6
echo "--- agent.log ---"
tail -n 5 "$D/agent.log"
