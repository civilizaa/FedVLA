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
# 【2026-11-03 事故修复】
#   现象：粘贴命令后 agent 死活连不上（心跳停在 40+ 小时前）。
#   根因：一次性 bootstrap 码是【单次使用】的，若在签发后被本地自测消费掉，
#         用户粘贴时 curl 下载到的是 401 的 JSON 错误页而非 Python 源码，
#         `python3 agent.py` 立刻 SyntaxError 退出 —— 且静默失败，极难察觉。
#   修复：下载后立刻做三重校验（是源码 / 语法合法 / 含关键标记），
#         任一不过就重新下载；仍失败则明确报错，绝不"看起来启动了其实没启动"。
set -u
D=/root/fedvla_relay
mkdir -p "$D" && cd "$D"

# __BOOTSTRAP_URL__ 会被本机脚本替换成 https://<relay>/bootstrap?code=<一次性码>&sharecode=<码>
BOOT_URL="__BOOTSTRAP_URL__"

fetch_agent() {
    # 下载并校验；成功返回 0，失败返回非 0
    for attempt in 1 2 3; do
        echo "[fetch] 尝试 $attempt/3 ..."
        curl -sSL --max-time 60 "$BOOT_URL" -o agent.py || { echo "[fetch] curl 失败"; sleep 3; continue; }
        # 校验 1：不能是错误页（JSON / HTML）
        if head -c 1 agent.py | grep -q '{'; then
            echo "[fetch] 下载到的是 JSON 错误页：$(head -c 120 agent.py)"
            sleep 3; continue
        fi
        if head -c 1 agent.py | grep -qi '<'; then
            echo "[fetch] 下载到的是 HTML 页面：$(head -c 120 agent.py)"
            sleep 3; continue
        fi
        # 校验 2：语法合法
        if ! python3 -c "import ast,sys; ast.parse(open('agent.py').read())" 2>/dev/null; then
            echo "[fetch] 语法校验失败"
            sleep 3; continue
        fi
        # 校验 3：含关键标记
        if ! grep -q "start_new_session" agent.py; then
            echo "[fetch] 缺少 start_new_session 标记，不是 agent 源码"
            sleep 3; continue
        fi
        echo "[fetch] agent.py 校验通过 ($(wc -c < agent.py) bytes)"
        return 0
    done
    return 1
}

if ! fetch_agent; then
    echo "!!! agent.py 获取失败（一次性码可能已被消费或已过期）"
    echo "!!! 请在本机重新执行: python relay/relay_client.py bootstrap"
    exit 1
fi

echo "--- 关键标记检查 ---"
grep -c start_new_session agent.py   # 子进程独立会话（防被信号连带杀死）
grep -c fresh_after agent.py         # 丢弃陈旧命令
grep -c clearall agent.py             # 启动清理

# 已有 agent 在跑就不要再起一个
if pgrep -f "python3 agent.py" > /dev/null 2>&1; then
    echo "!!! 检测到 agent 已在运行，先杀掉旧的"
    pkill -f "python3 agent.py" || true
    sleep 2
fi

nohup python3 agent.py > "$D/agent.log" 2>&1 < /dev/null &
AGENT_PID=$!
echo "agent 启动 pid=$AGENT_PID"

sleep 8
echo "--- agent.log ---"
tail -n 10 "$D/agent.log"
echo "--- 存活检查 ---"
if kill -0 "$AGENT_PID" 2>/dev/null; then
    echo "AGENT_ALIVE pid=$AGENT_PID"
else
    echo "!!! AGENT_DEAD —— 进程已退出，日志见上方 agent.log"
    exit 1
fi
