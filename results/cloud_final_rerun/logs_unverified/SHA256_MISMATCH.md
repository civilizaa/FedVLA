# 未通过 sha256 校验的抢救件（logs_unverified/）

这三个文件在 relay 结果存储中对应 id 的内容**已被后续写入覆盖**（relay 的 id 由服务端统一分配，
推送脚本写入的文件 id 与后续命令结果发生冲突 ⇒ 存储里留下的是**旧版本**）。
fetch 脚本对它们判定 sha256 不符并**拒绝写盘**，因此此处是从 relay 存储手工取回并**明确标记为未校验**。

| 文件 | MANIFEST 记录 size | 存储实际内容 | sha256 是否匹配 | 说明 |
|---|---|---|---|---|
| `autoshutdown.log` | 2911B | 849 字符 | ❌ 不符 | 存储版本 ≠ 推送版本 |
| `extract_comm.log` | 4246B | 5520 字符 | ❌ 不符 | 存储版本 ≠ 推送版本 |
| `extract_dp.log` | 3001B | 3824 字符 | ❌ 不符 | 存储版本 ≠ 推送版本 |

## 逐文件 sha256（本地实际）

- `autoshutdown.log`: cd6de92998e41f7907cbcb181882bd0dc5cf06ee122e239658388897ddf73339
- `extract_comm.log`: 8dce5f6cc6768c2b280e7c35481325dde2bb2b8689fc40ee5754f1ad88121ed7
- `extract_dp.log`: d6f69eb7e200ba80ced73089d3316181f914f01436073efd70941e26dedaf74b

## 可用性判定

- `autoshutdown.log`：**内容可用但被截断**（守护写日志到一半就被打包推送 ⇒ 只有前 849 字符）。
  其中包含最关键的判据行「收工原因：全部实验完成（通信 7/7，DP 3/3）」⇒ 可作收工依据。
- `extract_comm.log` / `extract_dp.log`：**旧版本**（11-19 的 extract 输出，基线仍是 0.750），
  **不可用于引用今日结果**。今日 DP 结果以 `dp_utility_raw.json`（sha256 已校验）为准。

## 今日结果的权威来源（均已通过 sha256 校验）

- `dp_utility_raw.json` —— DP 三个 σ 的真实成功率；其 `rows[*].n = 3, df = 2` 证明
  三份评估日志**确实存在且各含 3 张表**（`mean_ci([])` 返回 `None`，不可能产出 n=3），
  故 0.000 是**真实评估结果**而非「无评估日志」占位。
- `comm_raw.json` / `scanlog_c_*.log` / `c_*/comm_report_r49.json` —— 通信侧 7/7 同构证据与实测流量。
