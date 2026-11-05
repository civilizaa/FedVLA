# tools/ —— 云端运维工具

把云端 AutoDL 实例连回本机的 relay 通道用。**不参与论文实验本身**，但是复现的关键零件。

## 文件

| 文件 | 作用 |
|---|---|
| `push_files.py` | 把本地文件/目录推到云端 `/root/HPT/` |
| `make_bootstrap_cmd.sh` | 生成"启动云端 agent"的安全命令 |
| `auto_resume.py` | agent 复活后自动接管：推文件 → 校验 → 启动实验 |
| `cloud.py` | **在云端执行一条命令并取回输出**（可靠版，日常运维主力） |
| `monitor.py` | 后台监控云端实验进度（GPU/进程/磁盘/阶段） |

## 凭据

**凭据绝不进 git。** 全部工具按以下优先级读凭据：

1. 环境变量 `RELAY_SECRET` / `RELAY_SHARECODE` / `RELAY_PUBLIC_URL`
2. `relay_secret.txt`（**必须自己创建，不入库**），格式三行：
   ```
   <token>
   <sharecode>
   <public_url>
   ```
   或用 `RELAY_SECRET_FILE` 指定路径

`tools/` 下的代码已确认不含任何明文凭据（`git grep` 可验证）。

## 用法

```bash
# ★ 在云端执行命令并看输出（最常用）
python tools/cloud.py "nvidia-smi; ps aux | grep hpt.run"
python tools/cloud.py "tail -n 40 /root/HPT/output/resume_all.log" --cwd /root/HPT

# 后台监控实验进度（每 5 分钟一次，写 monitor.log）
python tools/monitor.py --interval 300 --duration 28800
python tools/monitor.py --once          # 只拉一次当前状态

# 推送整个 federated 目录（一条命令，base64 tar 包）
python tools/push_files.py --dir federated --remote /root/HPT/federated

# 逐个文件推送（默认清单）
python tools/push_files.py

# 只比对云端 md5
python tools/push_files.py --check

# 生成 agent 启动命令
bash tools/make_bootstrap_cmd.sh <一次性code>

# 等 agent 复活后自动接管（推送 + 校验 + 启动实验）
python tools/auto_resume.py --timeout 7200
```

## 三个必须知道的平台坑（2026-11-05 实测）

1. **网关只认 URL 里的 `sharecode=` 参数**（大小写不敏感，只对首字母宽容 → 可用 `%58` 编码首字母）。
   放 path / fragment / 别的参数名 / header 里全部 403。
   服务端 `/bootstrap` 本身只认一次性 code，sharecode 纯粹是网关的门槛。
2. **URL 里出现第二个 `?` → 403**。`/results?since=1100` 再手拼 `?sharecode=X` 会变成
   `?since=1100?sharecode=X`，**全程 403**，症状像"agent 卡死"其实是脚本取不到结果。
   → 所有请求走 `cloud.py` 的 `_url()` 封装（`sep = "&" if "?" in path else "?"`）。
3. **`curl -s` 遇 403/404 返回退出码 0**，会把网关的 HTML 错误页写进文件。
   → 必须 `curl -sS -f` + 下载后 `ast.parse()` 校验。
   `make_bootstrap_cmd.sh` 已内置这两道保护。

