# 自动化执行记录 — 第4阶段「修正版重跑」看护

自动化 ID：`9a401ba4-133a-4e16-a184-71a647254645`
调度：每轮唤醒只允许跑一条 `tools/rerun_phase4_fix.py status`，不并发轮询、不逐配置调 relay。

---

## 2026-11-19 14:10 — 运行中（分支 3）

- 云端在线，relay agent 活跃（心跳滞后 5s）。
- 进度：通信扫描 `comm_rc=4/7`，DP 标定 `dp_rc=0/3`，尚无 `PHASE4_ALL_DONE` 标记。
- 进程态：`run_comm_scan_v2=4`、`train_fedavg=3`、自动关机守护与 DP 接力链均在跑，DP 接力仍处于「等通信扫描 7/7 收尾」的等待态。
- 最近动作：`c_int4`（14:06）与 `c_bit1`（14:07）先后启动；`c_drop00/c_drop25/c_drop50/c_drop75` 的评估表已齐（各 3 表），`c_int8/c_int4/c_bit1` 评估表仍为 0（尚在训练）。
- GPU：RTX 3090，显存 8.7G/24G，util 67~98%；根分区 30G 用 20G（67%）。
- 本轮动作：仅执行一次 status，未做任何其它干预。取回与报告回填留待 DONE 后再执行。

### 待办（下次唤醒）
- 若仍为「运行中」→ 继续只汇报进度一行。
- 若出现 `NO_DONE_MARKER` 且 `run_comm_scan_v2=0` → 先复跑一次 status 看 `comm_rc` 是否接近 7，再决定是收工还是向用户如实汇报报错。
- 若出现 DONE（含 `finished_at=`）→ 执行 `fetch --outdir results/cloud_final_rerun`，核对 MANIFEST sha256、回填 `results\第4阶段结果与结论.md` 的「修正版重跑」小节、核对 7 份 scanlog 同构性与 c_int8/c_int4 量化是否全 0（必要时查 `output/quant_drift.jsonl`），完成后把本自动化置为 PAUSED。

---

## 2026-11-19 17:13 — 运行中（分支 3）

- 云端在线，relay agent 活跃（心跳滞后 4s）。
- 进度：通信扫描 `comm_rc=6/7`，DP 标定 `dp_rc=0/3`，仍无 `PHASE4_ALL_DONE` 标记。
- 进程态：`run_comm_scan_v2=2`、`eval_model=2`、`train_fedavg=0`；自动关机守护与 DP 接力链均在跑，DP 接力仍处于「等通信扫描 7/7 收尾」的等待态。
- 最近一步：第 7 个配置 `c_int4` 评估表 2/3（唯一未齐），其余 6 配置（c_drop00/c_drop25/c_drop50/c_drop75/c_int8/c_bit1）评估表均已 3/3 齐。
- GPU：RTX 3090，利用率 4~27%，显存 506MiB/24G；根分区 30G 用 18G（58%）。
- 遗留标记：`r25_backup_flag = YES`（上一轮 c_drop25 备份旗标，预期行为）。
- 踩坑：Windows Git Bash 下 `tools\xxx.py` 的反斜杠会被吞（`can't open file '...toolsrerun_phase4_fix.py'`），命令须写正斜杠 `tools/xxx.py`。
- 本轮动作：仅执行一次 status（首条因路径写法报错后以正斜杠重试成功），未做任何其它干预。取回与报告回填留待 DONE 后再执行。

---

## 2026-11-22 12:45 — 混合态：通信侧已收工，DP 侧重跑中（分支 3 变体）

- **通信侧已完全完成**：`comm_rc=7/7`，7 个配置评估表全 3/3，`comm_scan_v2.log` 尾部「全部处理完毕」。
- **DP 侧正在重跑**：11-19 那次训练到 `global_r27/27/28`（约 28/50 轮）时**云端实例被硬中断** ⇒
  `.rc_dp_*` 未写出 ⇒ 守护未收工 ⇒ **990000 批次从未推送**（`fetch` 报 `[FATAL] 没找到 MANIFEST`，
  这**不是** idbase 配错，而是根本没推）。
- **11-22 12:38:18 已从第 0 轮重跑三路 σ**（先清空 `output/dp_s0001|dp_s0003|dp_s0012` 旧目录，
  保留 `dp_calib/calib.json` ⇒ C=1284.2807639417144 幂等复用，不重标）。
  **12:40:35 守护同步重启**：`WAIT_DP=1 MAX_HOURS=8 POLL=60s PUSH_IDBASE=1000000`。
- 当前态：`agent_lag_sec=15 活跃`、`comm_rc=7/7`、**`dp_rc=0/3`**、`NO_DONE_MARKER`、
  `train_fedavg=3`、`chain_phase4_dp=1`、`phase4_autoshutdown=1`、GPU 99% / 12.9GB/24GB。
- **同构性已逐个核对通过**（7/7）：5 个重跑配置 scanlog 首行「从第 0 轮开始」、
  c_drop50/c_drop75 首行「[SKIP] 已完成（评估齐全）」且 `.legacy_r2` 备份本身即 r0 起训。
- **量化非故障已查实**（按"先算误差量级再下结论"铁律）：
  `quant_rel_err_max` int8=0.0044 / int4=0.0252；`quant_drift` 逐轮**下降**（r0 0.00354 → r49 0.000205，无累积）；
  客户端侧自比对清零率 int8 8.168% / int4 8.189%（非零元素口径 7.694%）；
  清零最集中于 **14 个** `blocks.*.attn.in_proj_weight`（每张量 15.3~16.5%，张量级 rel err 仅 0.56%）；
  失明度量 task3 state scale 被清零 23/39 维（干净对照 39/39）；
  反事实 `cf_int4_restore`（只换回未量化 normalizer）评估 **0.00/0.00/0.00** ⇒ 权重侧清零同样致命。
- **本轮已做**：报告 `results/第4阶段结果与结论.md` 回填（4.6 节重写 + 局限第 1/9/10/11 条 + 第九节改执行记录
  + 图注与产物清单更新 + 「16 个→14 个」修正），共 246+/174-；新增 `tools/tmp/send.py`（绕开 Bash 转义的下发通道）。

### 下轮判定（重要：勿重复踩坑）
- **DP 预计 16:00~16:30 收工**（实测 4.25 min/轮：r1→r2 用时 4 分 15 秒；12:38 起跑 ⇒ 50 轮约 16:05，加评估约 16:40）⇒
  守护随后 extract → 打包 → 推 **1000000** 批次 → 回读自检 → 写 `PHASE4_ALL_DONE` → 关机。**关机后 relay 服务端仍存活，产物照样能取。**
- ✅ **取回链路已端到端验证**（12:5x）：990000 批次**确实推送过**（id 990001–990037，末尾即 MANIFEST），
  `fetch --outdir results/cloud_final_rerun` ⇒ **36/36 文件 sha256 一致**。
  ⚠️ 此前那次 `fetch` 报 `[FATAL] 没找到 MANIFEST` 是 **relay 服务端瞬时故障**（只返回了 1 条 probe），
  **不是 idbase 配错、也不是"没推送"** —— 下轮若再遇 FATAL，**先直接查 `/results?since=N` 看 id 连续性再下结论**。
- ✅ **同构性已用本地文件逐个核对通过**（7/7，`results/cloud_final_rerun/scanlog_*.log` 首行）：
  5 个重跑组「从第 0 轮开始」+ c_drop50/c_drop75「[SKIP] 已完成（评估齐全）」且其 `.legacy_r2` 备份本身即 r0 起训。
- ✅ **成功率与流量已用本地 `comm_report_r49.json` 逐位核对**：与报告 3.1 / 3.2 完全一致
  （fp32 321.7 / int8 80.4 / int4 40.2 / bit1 10.1 MiB/轮；`quant_rel_err_max` int8 0.00443 / int4 0.02519 / bit1 0.77047）。
- 若 `dp_rc=3/3` 且出现 `PHASE4_ALL_DONE`（含 `finished_at=`）⇒ 执行
  `python tools/rerun_phase4_fix.py fetch --outdir results/cloud_final_rerun`
  （`--since 990000` 覆盖 1000000 批次；`relay_fetch_out.py` 取 id 最大的 MANIFEST），
  然后：逐文件 sha256 校验 → 回填 4.6 节「真实成功率」列与判定标准 → 同步第六节局限第 1/10 条 → PAUSED。
- 若 `dp_rc` 仍为 0/3 但 `train_fedavg > 0` ⇒ 仍在跑，只汇报进度一行，**不要重启**。
- 若 `dp_rc=0/3` 且 `train_fedavg=0` 且仍 `NO_DONE_MARKER` ⇒ 又被中断，**先把报错原文如实汇报，等用户决定**。
- ⚠️ **`agent_seen` 会给出陈旧读数导致误报 CLOUD_OFF**：本轮第一次 status 返回 lag=234804 秒 + CLOUD_OFF，
  但 `/health` 原始字段与 `/results` 新时间戳证明云端是活的 ⇒ **遇到 CLOUD_OFF 先用只读方式交叉验证再下结论**。
- 📌 低频监控工具已就绪：`python tools/dp_watch.py [分钟] [间隔秒]`（默认 240 分钟 / 600 秒一次，只读）。

---

## 2026-11-22 15:51 — DP 侧推进中（分支 3 变体：run_dp_utility>0）

- 云端在线：agent_lag_sec=17 活跃；GPU RTX 3090 util 41~97% 波动，显存 2010MiB/24G；根分区 30G 用 21G（70%）。
- 通信侧仍 `comm_rc=7/7`（11-19 17:19 已「全部处理完毕」，7 配置评估表全 3/3）。
- **DP 侧 `dp_rc=1/3`**（autoshutdown 心跳 15:29:51 由 0/3 变 1/3），`NO_DONE_MARKER`。
- 进程态：`run_comm_scan_v2=0`、`train_fedavg=0`、**`run_dp_utility=3`（DP 效用扫描在跑）**、`eval_model=4`、
  `phase4_autoshutdown=1`、`chain_phase4_dp=1`；无 `.eval_lock`；`r25_backup_flag=YES`（预期）。
- 判读：**不属于分支 2 的「扫描意外消失」**——通信扫描是 7/7 正常收工，DP 侧进程活着、GPU 有负载，
  故仅汇报进度，未做任何干预（未 restart、未 fetch、未重复起跑）。
- 时间账：首路 σ 于 12:38 起跑、约 15:29 收（≈171 分钟/50 轮），余 2 路若并行预计 18:00 前后收工，
  之后守护 extract→打包→推 1000000 批次→写 `PHASE4_ALL_DONE`→自动关机。

### 下轮判定
- 若 `dp_rc=3/3` 且见 `PHASE4_ALL_DONE`（含 `finished_at=`）⇒ 走 `fetch --outdir results/cloud_final_rerun`
  → 逐文件 sha256 校验 → 回填 `results/第4阶段结果与结论.md` 4.6 节「真实成功率」列与判定标准 →
  同步第六节局限第 1/10 条 → 置 automation 为 PAUSED。
- 若仍 `dp_rc<3/3` 且 `run_dp_utility>0` ⇒ 只汇报一行进度。
- 若 `dp_rc<3/3` 且 `run_dp_utility=0` 且 `train_fedavg=0` 且仍 NO_DONE_MARKER ⇒ 又是中断，
  先只读交叉验证再如实汇报报错，等用户决定。
- ⚠️ 沿用上轮教训：CLOUD_OFF 先交叉验证再下结论；Bash 下路径必须用正斜杠 `tools/xxx.py`。

---

## 2026-11-22 22:00 — ✅ 任务闭环（分支 4 全部执行完毕，automation 置 PAUSED）

### 一、状态判定（重要：CLOUD_OFF 是关机后的正常表现，不是任务中断）

- 单条 `status` 返回 `agent_lag_sec=9556` + **CLOUD_OFF** ⇒ 按分支 1 本该只回一句话。
- **但先做了只读交叉验证**（沿用上轮教训）：
  - `/health` 原始字段：`{"ok":true,"queued":46,"results":500,"next_id":3615,"agent_alive":true,"agent_pid":1376,"agent_seen":1795335282}`
  - `/results?since=1000000`：**374 条结果，id 1000001..1000374，MANIFEST id=1000374（base=1000000，49 文件）**
- ⇒ 判定：**守护已按设计收工推送成功，实例随后自动关机**（relay 服务端独立于容器存活）。
  **未重复触发起跑，未重启任何东西。**

### 二、收工时间线（`autoshutdown.log` 实测）

- `16:13:06` 守护判定「收工原因：全部实验完成（通信 7/7，DP 3/3）」→ `16:13:07` `extract_comm_results rc=0`
  → `16:13:10` `extract_dp_utility rc=0` → 打包 `final_out.tgz`=164K → 推进 relay 结果存储 → 自动关机。

### 三、取回与校验

- `python tools/rerun_phase4_fix.py fetch --outdir results/cloud_final_rerun`
  ⇒ 取回 **415 条结果**（id > 990000）⇒ **成功 46 / 失败 3**。
- **46 个数据产物（JSON/PNG/scanlog/隐私 JSON 等）sha256 逐个一致**。
- **3 个失败项全是追加型日志**（`relay_fetch_out.py` 的铁律是"不符即拒写"，所以这 3 个文件**没落盘**）：
  | 文件 | 清单 size | 存储 size | 前缀快照命中 | 判定 |
  |---|---|---|---|---|
  | `extract_comm.log` | 4246 | 6367 | ✅ | 算哈希后被**继续追加** |
  | `extract_dp.log` | 3001 | 4609 | ✅ | 同上 |
  | `autoshutdown.log` | 2911 | 979 | ❌ | 被**截断后重写**（清单哈希对应截断前版本） |
  ⇒ **推送窗口内并发写入竞态，非损坏非缺失**；内容本身完整（尾部有 `*_EXTRACT_DONE`）。
- ✅ 新增 `tools/relay_salvage_logs.py`：把这 3 个文件的**实际存储内容**原样抢救到
  `results/cloud_final_rerun/logs_unverified/` + `SHA256_MISMATCH.md` 逐项说明（不冒充"校验通过"）。

### 四、同构性（硬规则 c）—— 7/7 通过

- 5 个重跑组（c_drop00 / c_drop25 / c_int8 / c_int4 / c_bit1）scanlog 首行「从第 0 轮开始」
- c_drop50 / c_drop75（保留组）首行「[SKIP] 已完成（评估齐全）」

### 五、量化非全 0 的判定（硬规则 d）—— 真实信道效应，非故障

- `quant_rel_err_max` int8=0.00443 / int4=0.02519 / bit1=0.77047；`quant_drift` 逐轮**下降**（无累积）。
- `cf_forensics_int4.json`：`q1_global_killed_pct=7.7136%`，集中在 `trunk.blocks.*.attn.in_proj_weight`
  （每张量清零 15.4~16.5%，但张量级 rel_err 仅 0.00035~0.0006）；`q3_normalizer`：int4 下 state.scale 丢 31/39 维、offset 丢 21 维。
- `cf_blind_int4.json`：task3 state 有效 **16/39**（干净 39/39）⇒ 量化确实把状态通道打瞎。
- **决定性证据**：后验消融 `cf_p3c_int8_all` = **0.817 ≈ 基线 0.717** ⇒ 量化算子本身几乎无损，
  **伤害来自"训练闭环里逐轮施加"** ⇒ 3.4.2 的旧机制归因（"归一化统计量被打坏"）必须撤回。

### 六、DP 侧最终结果（回填核心）

- C = **1284.2807639417144**（‖Δ‖₂ 中位数，n=4）；基线 **0.717 ± 0.502**（seeds 0.95/0.60/0.60）。
- **σ=1e-4 / 1e-3 / 1e-2 三个点真实成功率全部 0.000 ± 0.000（各 3 seeds 全 0）** ⇒ 相对基线 **−71.7 pp**。
- 命中事先写定的判据「σ=1e-4 就已归零」⇒ **"DP 可用窗口"在本实验台上不存在**（σ≤1e-4 区间无可用点）。
- **代理列是脚本 bug 不是数据缺失**：`extract_dp_utility.py::load_privacy_proxy` 读 `records`，
  而 `privacy_results.json` 顶层是 `runs` ⇒ 脚本输出代理列为「—」、fig7 上无代理曲线。
  已手工复算：保真度 0.9240/0.2350/0.0244、动作 cos 0.9709/0.9742/0.9562，**与 4.5 表逐位一致**。

### 七、本轮新发现（已写入报告 4.6.3）

**代理指标双向失效**：σ=1e-4 时真实成功率已 0.000，而代理保真度仍 0.924、动作 cos 仍 0.971
⇒ 代理指标在低 σ 端**过于乐观、漏报效用崩塌**（方向与预写判据第 2 行**相反**）；
σ=1e-2 时保真度已崩 0.024 而动作 cos 仍 0.956 ⇒ 高 σ 端**过于悲观**。
机制：保真度测的是"攻击者能否看清信号"，**与隐私同向、并不与效用同向**。
⇒ 与 3.4.4（"离线权重保真度 ↔ 闭环任务效用无可靠映射"）是同一结构性事实的两个实例。

### 八、遗留缺口（如实记录，未解决）

1. **DP 侧逐 seed 原始评估日志（`eval_dp_s0001_*.log` 等）不在收工批清单里**，990000 取证批也没有
   ⇒ DP 真实成功率只有 `dp_utility_raw.json` + `fig7_dp_utility.png` + `extract_dp.log` 三个来源。
   通信侧 7 份 `eval_c_*.log` 全部留存 ⇒ 两侧不对称，需重开实例补取。
2. **3.4.4 后验消融 3 个变体未出结果**：`cf_p3c_int8_trunk`（0/3 seed）、`cf_p3c_int8b_norm`（1/3 seed）、
   `cf_p3c_int4b_all`（无评估日志）、`cf_p3c_int8_cs`（既无元数据也无日志）
   ⇒ 3.4.4 主结论（后验 int8 ≈ 0.817 ⇒ 量化本身不致命）仍成立，但
   "int4 是否与位宽无关"、"客户端侧步长是否有独立贡献"两问**未答**，论文不得当已有结论写。

### 九、本轮产出

- `results/第4阶段结果与结论.md` 回填完成（472+ / 193-）：头部状态块、第一节方法论段、
  3.4.4 变体状态、**4.6.1 时间线 / 4.6.2 实测表 + 预写判据落点 / 4.6.3 代理指标双向失效（新增）/ 4.7 取回与校验记录（新增）**、
  第五节叙事第 1/4 条、局限第 1/10 条、产物清单 2 行、第九节 9.1/9.3/9.4。
- 新增 `tools/relay_salvage_logs.py`（抢救归档工具）。
- 数值校验脚本逐位比对通过：DP 三行 mean/ci、通信侧 7 行 mean/ci/流量 与原始 JSON 完全一致。
- ✅ **本 automation 已置为 PAUSED。**
