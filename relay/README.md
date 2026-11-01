# FedVLA 远程指挥通道（relay）

## 是什么

AutoDL 实例**能出公网**（下过 HuggingFace 权重、pip 装过包），只是 **GitHub 被封**、
且控制台详情页里的 SSH 密码会被截断。因此不走 SSH，改用一条 HTTP 回路：

```
本机 Agent  ──HTTP──▶  relay 服务（公网）  ◀──轮询──  AutoDL 实例内的 agent
     ▲                                                    │
     └──────────── 执行结果（stdout/stderr/rc）───────────┘
```

## 三个文件

| 文件 | 位置 | 作用 |
|---|---|---|
| `server.py` | 部署到公网 | 命令队列 + 结果存储 + `/agent.py` 下发 |
| `agent_template.py` | 模板 | 云端 agent 源码（下发时注入 URL/密钥/sharecode） |
| `relay_client.py` | 本机 | `exec` / `put` / `get` / `status` / `results` / `watch` |

## 使用

### 1. 云端（AutoDL 终端，整块粘贴 `START_AGENT.sh`）

脚本分 6 块把 base64 的 agent 写到 `/root/fedvla_relay/agent.py`，
校验语法后 `nohup` 后台启动。日志在 `/root/fedvla_relay/agent.log`。

### 2. 本机

```powershell
cd C:/Users/hp/WorkBuddy/2026-10-08-14-27-08/relay
python relay_client.py exec "nvidia-smi"          # 同步执行，等结果
python relay_client.py exec -b "bash train.sh"     # 后台执行，立刻返回
python relay_client.py put local.py /root/HPT/x.py # 上传（base64 内联）
python relay_client.py get /root/HPT/out.log ./out.log
python relay_client.py watch                      # 持续监视
```

## 安全

- 双重鉴权：网关 `sharecode` + 应用层 `X-Relay-Token`，缺一返回 401/403。
- 凭据只出现在 agent 脚本与 `relay_client.py` 里，**不入 Git**（见 .gitignore）。

## 已验证

- 本地全链路自测：鉴权拦截、密钥注入、命令下发/回传、后台执行、同步 `/run`
- **公网端到端**：`enqueue → poll → 执行 → result` 真实跑通（rc=0）
- `START_AGENT.sh` 在 bash 下实测：字节数正确、语法 OK、agent 连上 relay
- 去重：同一条命令只执行 1 次（agent 侧 `_pending_report` + 服务端 claimed_at 双重保险）
