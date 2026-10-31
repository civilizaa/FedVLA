# AutoDL 自动化通道 —— 使用说明

## 你只需要做一件事：给我凭据

### 方式 A（推荐）：写一个凭据文件

在 `C:\Users\hp\.workbuddy\2026-10-08-14-27-08\` 下新建文件 **`.autodl_creds.json`**，内容：

```json
{
  "host": "connect.westb.seetacloud.com",
  "port": 45678,
  "password": "你的实例密码"
}
```

（`host`/`port` 就是控制台实例详情页那行 `ssh -p 45678 root@connect.westb.seetacloud.com` 里的部分。）

### 方式 B：环境变量

```powershell
$env:AUTODL_SSH_HOST = "connect.westb.seetacloud.com"
$env:AUTODL_SSH_PORT = "45678"
$env:AUTODL_SSH_PASSWORD = "密码"
```

### 方式 C：开发者 Token（走 API）

```powershell
$env:AUTODL_TOKEN = "你的token"
```

> ⚠️ **实测（2026-10-31）**：开发者 Token 被 API 拒绝，所有端点/所有鉴权头组合都返回
> 同一个 `AuthorizeFailed: 登录超时`。最可能是**账号未实名认证**导致 API 未激活。
> 所以**优先用方式 A**。
>
> ⚠️ **另一个坑**：不要把 token 写在命令行里（`export AUTODL_TOKEN=...`）——
> 某些执行环境会对命令行凭据做保护性替换，实际发出的与你写的不是同一个，
> 会误判成"token 无效"。**写文件最稳。**

---

## 凭据到位后先体检

```powershell
cd C:/Users/hp/WorkBuddy/2026-10-08-14-27-08/fedvla-repo/federated
python autodl_auto.py doctor
```

会依次检查：凭据 → TCP 连通 → SSH 认证 → 远端环境（GPU / HPT 目录 / hpt 环境 /
集中式模型 / 联邦目录 / 磁盘）。**任何一环有问题会立刻指出**，不用等跑实验才发现。

---

## 常用操作

```powershell
# 列出实例（需 token）
python autodl_auto.py list

# 执行任意命令（流式输出）
python autodl_auto.py exec "nvidia-smi"

# 上传文件
python autodl_auto.py put patch_and_verify_inline.sh /root/HPT/patch_and_verify_inline.sh --chmod

# 上传并直接执行（最常用）
python autodl_auto.py script patch_and_verify_inline.sh /root/HPT/patch_and_verify_inline.sh

# 下载结果
python autodl_auto.py get /root/HPT/gate1.log ./gate1.log
```

> `python` 指 `C:\Users\hp\.workbuddy\binaries\python\versions\3.13.12\python.exe`（已装 paramiko）。

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
