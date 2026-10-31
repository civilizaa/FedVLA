# AutoDL 自动化通道 —— 使用说明

## 你只需要做一件事：给我凭据

二选一：

### 方式 A（推荐，一次配好长期有效）：开发者 Token

1. 打开 https://www.autodl.com/console/homepage/personal
2. 找到「开发者 Token」→ 新建 → 复制
3. 在本机 PowerShell 里设环境变量（**只对当前窗口有效**，够用了）：

```powershell
$env:AUTODL_TOKEN = "你的token"
```

Token 权限 = 能读实例列表即可（读 SSH 信息需要实例管理权限）。

### 方式 B（不用 Token，从控制台抄 SSH 信息）

打开 AutoDL 控制台 → 你的实例详情页，会看到：

```
SSH 命令：ssh -p 45678 root@connect.abc123.seetacloud.com
密码：xxxxxxxx
```

在 PowerShell 里设：

```powershell
$env:AUTODL_SSH_HOST = "connect.abc123.seetacloud.com"
$env:AUTODL_SSH_PORT = "45678"
$env:AUTODL_SSH_PASSWORD = "xxxxxxxx"
```

> ⚠️ **网页上的密码可能被截断**（AutoDL 的已知问题）。如果方式 B 连不上，
> 改用方式 A —— API 返回的是完整密码。

---

## 凭据到位后我能干什么

```powershell
cd C:/Users/hp/WorkBuddy/2026-10-08-14-27-08/fedvla-repo/federated

# 列出实例，看状态（有卡/无卡/关机）和 SSH 信息
python autodl_auto.py list

# 在实例上执行任意命令（流式输出）
python autodl_auto.py exec "nvidia-smi; ls /root/HPT"

# 上传文件
python autodl_auto.py put patch_and_verify_inline.sh /root/HPT/patch_and_verify_inline.sh --chmod

# 上传并直接执行（最常用）
python autodl_auto.py script patch_and_verify_inline.sh /root/HPT/patch_and_verify_inline.sh

# 下载结果
python autodl_auto.py get /root/HPT/gate1.log ./gate1.log
```

多实例时用 `--instance <名字>` 指定：

```powershell
python autodl_auto.py --instance fedvla exec "nvidia-smi"
```

---

## 我会跑的自动化流程

| 阶段 | 命令 | 判据 |
|---|---|---|
| 门1 | `script patch_and_verify_inline.sh` | `LOADED_TRUNK_MATCHES_SOURCE = True` |
| 校准 | `exec` 跑单客户端 → grep `Epoch size` | `LOCAL_TOTAL_ITERS = 3 × Epoch size` |
| 门2 | `exec` 跑 `run_fedavg.sh`（NUM_ROUNDS=1，异构） | 4 客户端都有 model.pth + 聚合产出 trunk.pth/extras.pth + 续训自检通过 |
| IID 健全性 | 同上，CLIENT_DOMAINS 全 task1 | FedAvg ≈ 0.98 |
| 正式 50 轮 | `script` + nohup 后台跑 + 定期取日志 | `S_fedavg` → 判门 2 |

每一阶段我都会先读日志判断，通过了才进下一阶段；**失败会停下来把日志给你看**，
不会闷头往下跑。

---

## 边界（说清楚，避免误会）

1. **开机/关机、充值、换卡** —— 我能做（API 支持），但**会花钱**，所以每次开关机前我会先跟你确认。
2. **实例若已关机**，门1（纯 CPU）我会用无卡模式跑，¥0.1/时。
3. **我改不了 AutoDL 控制台的 UI 操作**（如设置开发者 Token 本身），这一步必须你手动。
4. **密码/token 不会写进任何文件或提交到 Git** —— 只通过环境变量传递。

---

## 常见问题

**Q: 报 `AuthorizeFailed: 登录超时，请重新登录`**
A: token 无效或已过期，重新去控制台生成。注意**不要带 `Bearer` 前缀**，直接 `Authorization: <token>`。

**Q: 报 `Unable to connect to ... port ...`**
A: 实例关机了，或 host/port 抄错。先 `python autodl_auto.py list` 看状态。

**Q: 连上了但命令秒返回空**
A: AutoDL 镜像的 conda 在 `/root/miniconda3/bin`，非交互 SSH 不加载 login profile。
   所以脚本里**一律用绝对路径 python**（`/root/miniconda3/envs/hpt/bin/python`），不要用 `conda activate`。
