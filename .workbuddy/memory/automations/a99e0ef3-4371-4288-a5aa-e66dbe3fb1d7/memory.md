# automation a99e0ef3 — 第5阶段主实验矩阵进度检查与收尾

## 2026-11-23 13:19~13:40 首次执行

### 结论
**未完成，且按当前轨迹必定 0/7 成功** ⇒ 未取产物、未生成 `results/第5阶段结果与结论.md`。
云端零改动（未重启训练/守护、未改/删任何云端文件、未 `git add -A`）。

### 云端实测状态（13:36）
- `.rc` 2/7，且**两个都是失败**（`.rc_p5_bal_fedprox=1`、`.rc_p5_bal_scaffold=1`）
- 正在跑 `p5_iso_fedavg`（第 4/50 轮），其余 4 个未开始
- 守护 pid 17309 存活；**实测 MAX_HOURS=60**（任务书写的 40 不准 —— 11:14 起过 40、11:26 重启成 60），
  绝对时限 2026-11-26 03:26
- relay agent 存活（滞后 3~5s）；**relay id>1010000 = 0 条结果、无 MANIFEST** ⇒ 无产物可取
- 磁盘 30G，28G 已用，**2.2G 可用（93%）**；`output/phase5` 11G

### 两个致命阻塞点（下次执行直接照抄结论，别重查）
1. **评估期 MoE 被丢掉**（13.94M 而非 21.08M）
   - `run_phase5.sh` / `eval_model.sh` 里 `FEDVLA_MOE` 命中数 = **0**；
     `run_phase3_eval.sh` = 13（第3阶段已有 `apply_moe_env()` 修法，**第5阶段没继承**）
   - 机制：`patch_hpt_moe.py` 注入 policy.py 的 `moe_cfg` 依赖 `os.environ["FEDVLA_MOE"]=="1"`；
     不设 ⇒ policy 不建 MoE 模块 ⇒ `load_state_dict(strict=False)` **静默丢弃 264 个 MoE 键**
   - 训练侧是对的：`trunk.pth` 有 264 个 expert/gate 键、21.08M、专家权重非零；
     且 `p5_bal_fedprox/trunk.pth` 与 `p3_moe_fedavg/trunk.pth` **字节数完全相同**（84,449,610 B）
   - 判别速查：无 MoE = **13.94M**（第3阶段 centralized/g2b/iid 全是这个值），MoE = **22.38M**
   - ⇒ 7 个配置的每个 domain 都会撞 18M 自检 ⇒ **必然全失败**
2. **磁盘必爆**：`_baseline/` 从不被清理（`run_fedavg.sh` 全脚本仅 1 处 `rm -rf`，只删 global_r/client_*_r*）
   - FedProx `anchor_r{r}.pt` 84MB×50 = 4.2G/配置；SCAFFOLD `ci_c{c}_r{r}.pt` 50MB×4×50 = 10.1G/配置
   - 剩余 4 配置需 ≈17G，可用 2.2G ⇒ 缺口 ≈15G
   - 已爆过一次：`p5_bal_scaffold` 第 1 轮就因「可用空间仅 5GB < 6GB」→ 缺 `ci_c0_r1.pt` → 聚合中止

### 守护的两个 fail-open（会让失败被当成完成）
- `count_rc()` 只判 `.rc` **文件存在**、不看内容 ⇒ `.rc=1`（失败）也计数
- `extract_phase5_results.py` 的 `collect_stage()` **不校验参数量** ⇒ 13.94M 残缺评估会被写进主表

### 产出物
- `results/第5阶段阻塞诊断与续跑方案_2026-11-23.md` —— 完整证据链 + 续跑方案
- `tools/fix_phase5_eval_moe.sh` —— M1~M4 四处补丁（**DRY_RUN=1，默认不执行**，待用户批准）
- `tools/tmp/p5status.sh` + `p5diag*.sh` + `p5snap.sh` —— 本轮只读探查脚本（可复用）

### 下次执行注意
- **先看 `results/第5阶段阻塞诊断与续跑方案_2026-11-23.md`**，不要从零重查
- 待用户拍板两件事：① 是否清第4阶段遗留产物（`dp_calib` 686M 等，`p3_moe_*` 190M×3 **不能删**）
  ② 是否接受 `p5_bal_fedprox` 只重评不重训（省 1.5h）
- **重评前必须先删** `eval_p5_bal_fedprox_mujoco_metaworld_task1.log`（13.94M 残缺评估），
  否则 `eval_done()` 会因已有 3 张 avg 表而直接 SKIP
- 铁律 D：训练未完成的配置（`p5_bal_scaffold`/`p5_iso_fedprox`/`p5_iso_scaffold`）必须清空重跑
- 探查注意：`phase5/eval_*.log` 是**软链**指向 `output/eval_*.log`；
  `ls phase5/eval_*.log` 会列出 21 个但多数悬空 ⇒ 用 `ls -la` 看清链接目标

---

## 2026-11-23 15:39~16:10 第二次执行

### 结论：**仍未完成（0/7 成功），且当前没有训练在跑 —— 但并发会话已接手续跑**
本轮未取产物（relay id>1010000 无 MANIFEST）、未写 `results/第5阶段结果与结论.md`（评估未完成 ⇒
写了就是"判定实验完成"）。云端零改动：未重启训练/守护、未改删任何云端文件、未 `git add -A`。

### 状态快照（16:03~16:10 云端实测）
- `.rc` 存在 3/7，**内容全是 1（失败）⇒ 成功 0/7**（新判据 `rc_is_ok` 只认内容为 0）
- 轮次：`p5_bal_fedprox`=50、`p5_bal_scaffold`=1（目录已删）、`p5_iso_fedavg`=50，其余 4 个未开始
- **守护 `phase5_autoshutdown.sh` 已不存在（GUARD_COUNT=0）**；主控 `run_phase5.sh` 也不存在
- 存活的是并发会话的：`phase5_ab_run.sh`（A/B 评估并行度）+ `phase5_autostart.sh`（等 A/B 后自动重跑）
- 磁盘 9.0G 可用（比 13:36 的 2.2G 好转）；relay agent 滞后 11s；id>1010000 = 0 条、无 MANIFEST
- `p3_reuse/` 仍为空；云端 `eval_p3_moe_*.log` 有 10 个

### 13:36 的两个洞已修好，M1~M5 云端全部生效（实测）
M1 `eval_model.sh` FEDVLA_MOE=6 / M2 `cleanup_old_baseline`=2 / M3 `rc_is_ok`=3 / M4 `PARAMS_MIN_M`=4

### 🔴 新阻塞点 M5（比 M1/M2 更致命，因为**静默**）—— 13:36 的结论被它推翻
`patch_hpt_fedbaseline.py` 注入 `apply_baseline_hook(model, opt)`，而 HPT `train()` 真实变量名是
**`optimizer`** ⇒ 每 step NameError ⇒ 被 except 吞掉 ⇒ **FedProx 近端项 / SCAFFOLD 修正从未施加、
零告警**。证据：`p5_bal_fedprox/log_client_0_r0.txt` 里 45 次/文件的
`"[FedVLA-BASE] hook skipped: NameError name 'opt' is not defined"`。
⇒ **那 50 轮 ≡ 纯 FedAvg，该格作废必须清空重跑（铁律 D）** ——
13:36 说的"p5_bal_fedprox 只重评不重训"**是错的**。
修复：`opt`→`optimizer`；联邦基线模式不再吞异常（raise，铁律 Q）；键空间错配/控制变量零命中 ⇒ 显式抛异常；
FedProx 失效判据改为"整个 epoch 一次都没非零"（首个 step 必然为 0 是正确行为）。

### 并发会话的续跑编排（已在云端就位，本轮未干预）
`phase5_ab_run.sh` → `phase5_autostart.sh` → `phase5_restart.sh`（清作废产物 → 三道硬闸 → 起训练 → 挂守护）
- **A/B 关键修正**：第一版在 task1 上做**无效**（异构下 task1 全 0.000 ⇒ 0=0 恒成立）；
  改 **task3**（p3_moe_fedavg task3 = 0.283±0.587，非退化）+ 新增非退化性检查
- `phase5_restart.sh` 清理口径：删 `p5_bal_fedprox` 整目录（M5 作废）、删 `p5_bal_scaffold`、
  **保留 `p5_iso_fedavg` 的 `global_r49` + `opt_state_c*.pt`**、删 `eval_p5_*.log`、删 `.rc_*`
- 三道硬闸：清理后磁盘 ≥8G / 4 个补丁锚点全命中 / 残留守护实例数 = 0

### ⚠️ 唯一要盯的：A/B 90 分钟超时
`phase5_autostart.sh` 等 `ab_status.log` 出现 `AB_DONE`，最多等 90 min（16:00:20 起 ⇒ 最晚 ~17:30）；
超时则 `exit 1` **永不开跑**，而守护已不在 ⇒ 容器空转到 AutoDL 到期。

### 下次执行必做（**别从零重查**）
1. **先看 `ab_status.log` 有没有 `AB_DONE`** —— 没有就看 `eval_ab_nj3_mujoco_metaworld_task3.log`
   的大小与 avg 行数判断进度
2. **`ab_decision.txt` 有没有 `decision: N_JOBS=N`** —— 有 ⇒ 重跑已启动，去看 `run_phase5.sh` 轮次
3. `.rc` 齐全性用**新判据**（内容为 0 才算成功），别只看文件存在
4. **探查一律用 `tools/tmp/p5probe_safe*.py`（独占 id 段 1002000~1002999），不要用 `cloud_exec.py` 默认段**

### 本轮新踩的 4 个坑（复用价值高，下次直接照抄）
1. **relay 的 id 段会被并发会话抢占 ⇒ 读到别人的结果**
   `phase5_remote.py` 默认段 `IDBASE_PROC=1005000`，`_alloc_id()` 取"段内 max+1"。
   本轮首个探查用 `cloud_exec.py`（默认段）**id 被并发会话占用**，读回来的是别的会话的输出
   （`--- A/B 已起 ---` / 云端单测通过），差点误判。
   ⇒ 解法：**独占 id 段** `1002000~1002999`（低于对方用的 `1005000~1009999`，
   对方按"段内 max+1"永远够不到本段 ⇒ 结构性不冲突）+ 输出带自己的标记（`PROBE-END`）校验。
   探查脚本已落盘 `tools/tmp/p5probe_safe.py` ~ `p5probe_safe12.py`（13 个）。
2. **relay 的 `out` 字段按字节截断 ⇒ 多字节中文被切一半 ⇒ 客户端 `UnicodeDecodeError`**
   ⇒ 探查输出尽量全 ASCII；必须带中文就把结果 base64 后再取。
3. **heredoc 在 relay 的 shell 里被当字面量回显**（`<<'__EOF__'` 不生效）⇒ 长逻辑一律写文件再执行。
4. **`ps | grep` 统计进程数会把自己算进去**（`grep -c '[p]hase5_autoshutdown.sh'` 报假 `GUARD=2`）
   ⇒ 遍历 `/proc/[0-9]*/cmdline` + 不自匹配正则（`shutdo[w]n`）。

### 产出物
- `results/第5阶段进度检查与收尾报告_2026-11-23-1610.md` —— 完整状态 + M5 证据链 + 3 个风险 + 4 个新坑

### 遗留风险（下次盯）
- 风险 1：A/B 超时 ⇒ 永不开跑（见上）　→ **已解决**（17:03 判决完成，见第三轮）
- 风险 2：`p5_iso_fedavg` 补评估依赖 `opt_state_c*.pt` 齐全，否则清空重跑 50 轮（restart.sh 已保留 ✔）
- 风险 3：**复用的第 3 阶段原始日志不在守护推送清单里**
  （`PUSH_FILES` 用 `ls -1` 不带 `-R` ⇒ `p3_reuse/` 子目录不会推）——
  主表能出（extract 的 `eval_roots` 含父目录 `/root/HPT/output`），
  但"第 3 阶段复用三行"的一级证据日志拿不回来（本地 `results/` 里 `eval_p3_moe*` = 0 个）
  → **已解决**：10 份 p3 日志已平铺到 `$OUT` 且为实体文件（`P3_REUSE_in_OUT=10`，子目录=0）

---

## 第三轮：2026-11-23 18:15~18:49（健康核验轮）

### 结论：✅ **正常运行中，0/7 失败，0 个隐藏缺陷（不是完成态）**
`.rc` 0/7（未跑到，非失败）；训练与守护均健在；**未执行任何关机/重启/删除/改码操作**。

### 状态快照（18:45~18:49 云端实测）
- `p5_bal_fedprox` **50/50 轮训完**，18:45:02 进入评估（task1）
- 进程树唯一：ROOT_MASTER 恰好 1 个（127355 → 127368 → 127369），守护恰好 1 个（127867）
- 磁盘 **9.2~9.6 GB 可用**（> 8GB 报警线）
- relay agent 存活，`agent_seen` 滞后 **0~3s**
- `.rc` 存在 0/7

### 五项隐患闭环验证（全部实测通过）
1. **A/B 判据正确**：`ab_decision.txt` = `decision_basis: NJ3_MATCHES_REFERENCE` / `decision: N_JOBS=3`
   / `reproduce_vs_phase3: nj3_match=1 nj1_match=0`。传导链逐环实测：
   `/proc/127355/environ` 含 `N_JOBS=3` → `run_phase5.sh:61` → `:197` → `eval_model.sh:27` → `:75` `hydra.launcher.n_jobs`
2. **M5 真正生效**：新跑 `p5_bal_fedprox` 的 `hook skipped` 命中 **0 个文件**；
   日志实测 `FedProx 近端项已施加 mu=0.01 键数=488 ‖g_prox‖₂=0→1.0123e-03`、`近端项统计：非零 44/45 个 step`
   🔴 **关键辨析**：旧 `p5_iso_fedavg` 的 200 个 `hook skipped` **不影响其有效性** ——
   该配置 `BASELINE=""`（纯 FedAvg），NameError 发生在**调用点参数求值**、钩子体未执行
   ⇒ 梯度零修改 ≡ no-op ⇒ 那 50 轮是合法 FedAvg（只需补评估，不需重训）
3. **M1 真正生效**：评估日志实测 `params=22.38`（非 13.94）；
   训练侧 `trunk.pth` 488 keys / **264 MoE 键** / 21.082M，专家权重非零；
   `p5_iso_fedavg/global_r49/trunk.pth` 同样 264 MoE / 21.082M ⇒ 补评估也会报 22.38M
4. **M2 磁盘回收到位**：`cleanup_old_baseline` 按轮次删 `anchor_r*/ci_c*/c_global_r*` 且跳过 `ci_c[0-9].pt`；
   `cleanup_completed_config` 调用点在 `run_phase5.sh:327`（3 个 eval 之后）
5. **守护脚本 md5 `3c2713ed57a050f05a19b7eb18d9b802` = 本地逐字节一致**；
   `rc_is_ok`（只认内容 0）/ `WORKER_PAT` 含 `eval_model.sh` / `/usr/bin/shutdown` /
   推送含 `eval_*.log` / `exit 9` 全部在位

### phase3_raw.json 口径差异分析（本地 vs 云端 md5 不同）
diff 后确认**只差 2 处**：`generated_at` 与 `domains[*].ci`（云端 df=3 / 本地 df=n−1=2）。
验算：`4.303×0.136423=0.58701`（本地）vs `3.182×0.136423=0.43409`（云端）
⇒ **本地版才是任务书口径**（t_crit=4.303, df=2），且 `phase5_table.py` 从 `per_seed` **独立复算 CI** ⇒ 主表不受影响。
差异文件已落盘 `tools/tmp/cloud_phase3_raw.json`（19399 B）。

### ETA / 费用重算（按实测）
101.5 min/配置 + ~18 min/评估 ⇒ 6 配置 + 21 评估 ⇒ 明早 ≈ **09:00~09:45**，GPU 费 ≈ **41 元**，跑完自动关机 ⇒ 之后 0 元。
守护 `MAX_HOURS=44`（至 2026-11-25 13:04）远在 ETA 之后。

### 本轮新踩的 3 个坑
- **坑 5**：relay 队列会累积「已执行但结果被淘汰」的命令 ⇒ 被服务端重派（queued 1→14）。
  全为只读探查 ⇒ 无害。🔴 **不要 `POST /clearall`**（`before=0` 清空整个队列，可能删掉并发会话正在用的命令）
  —— 本轮刻意没清。守护推送走 `POST /result` 直连、不经队列 ⇒ 不阻塞。
- **坑 6**：探查输出含中文会被 relay 按字节截断 ⇒ 客户端 `UnicodeDecodeError`；
  heredoc 在 relay shell 里被当字面量回显 ⇒ 改 **ASCII-only + 云端 base64 + 行数组拼接**。
- **坑 7**：`check_same_construction` 匹配 `"从第 [0-9]+ 轮开始"` 而 `run_fedavg.sh` 打印 `"从轮 N 开始"`
  ⇒ 该铁律 D 自检**恒返回 0**（不触发清空）。好处：`p5_iso_fedavg` 保留产物不会被误清空；
  副作用：该自检目前形同虚设（本次无害，清理由 `restart.sh` 手工完成）。

### 产出物
- `results/第5阶段进度检查与健康核验报告_2026-11-23-1849.md` —— 9 节完整报告

### 下次执行必做（按顺序）
1. **先看 `.rc` 数量**（内容为 0 才算成功）：7/7 ⇒ 走收尾流程（分支 A）
2. 7/7 但 relay 无 id≥1010000 的 manifest ⇒ 守护推送未成功 ⇒ **不要重跑**，手动触发推送或查守护日志
3. `.rc` 里有内容为 1 ⇒ 部分失败 ⇒ **不要盲目重跑**（铁律 D：跨批次续跑破坏 LR/Adam 相位），先报告失败配置与原因
4. 守护不在 + 训练在 ⇒ **不要关机**，重新挂起守护（见任务书分支 B）
5. 守护不在 + 训练不在 + `.rc` 不全 ⇒ 空转烧钱 ⇒ 先取产物再立即关机（分支 C）
6. 探查一律用 `tools/tmp/p5probe_r*.py`（**独占 id 段 1002000~1002999**），
   **不要**用 `cloud_exec.py` 默认段（会被并发会话抢占），**不要** `POST /clearall`
7. 磁盘 < 8GB ⇒ 立即报警并评估是否需要手动清理
8. 评估期（GPU util 2~7%）不要误判成卡死 —— `WORKER_PAT` 已含 `eval_model.sh`

### 18:51 补充复核（r28/r29 一次合并命令拿齐 8 项）
| 项 | 实测 |
|---|---|
| `.rc` | 7 个全 `NONE`（0/7，未跑到 ≠ 失败） |
| `p5_bal_fedprox` | `done_gm=50`（**50 轮训完**）、`glob=3`(r47/48/49)、client_*_r47~49、evlogs=1 |
| `p5_iso_fedavg` | `done_gm=50`（早已训完）、`glob=2`(r48/49) ⇒ 只待补评估 |
| 其余 5 配置 | `NOT_STARTED` |
| `trunk.pth` | 在 `global_r*/trunk.pth`（**不是配置根目录** —— r28 查根目录得 0 属查错位置，非故障） |
| 守护 | PID 127867 存活（17:04:19 起，MAX_HOURS=44 / POLL=60s / PUSH_IDBASE=1010000） |
| 训练/评估 | `run_phase5.sh` PID 127355、`hpt.run_eval` PID 182529 ⇒ 各恰好 1 个 |
| GPU | RTX 3090，37% util，1508/24576 MiB（评估并行中） |
| 磁盘 | **9.6G 可用**（30G 用 21G，69%）> 8GB 报警线 |
| 评估参数量 | `eval_p5_bal_fedprox_..._task1.log` **p=22.38** ✔ M1 生效（avg=0 仍在跑，mtime 18:51:28 活跃） |
| relay agent | `agent_alive=true, pid=1841, seen_lag=6.0s` ✔；`queued=5`（全只读探查）；`/results?since=1010000` **count=0** ⇒ 尚无产物落地，符合预期 |

**ETA 精算（18:51 基准）**：cfg1 剩 3 评估 47 min ⇒ 19:38；cfg2 155 min ⇒ 21:53；
cfg3（仅评估）54 min ⇒ 22:47；cfg4~7 各 155 min ⇒ **≈09:00~09:10**（明早）。
守护时限 2026-11-25 13:04，远在之后。GPU 费 ≈ 40 元，跑完自动关机 ⇒ 之后 0 元。

### 🔴 坑 8（本轮新发现并已修）：`/health` 403 的根因是凭据读错文件
`tools/relay_status_probe.py` 的 `read_creds()` 只找 `relay/agent*.py` 里的 `RELAY_TOKEN` 常量，
而仓库里那份是 `agent_template.py` 的**占位符**（`__RELAY_TOKEN__`）⇒ 拿到空 token ⇒ `/health` 403。
**真正的凭据在 `tools/relay_secret.txt`**（第 1 行 token / 第 2 行 sharecode / 第 3 行 url）。
**已修**：`relay_status_probe.py` 的 `read_creds()` 优先读 `relay_secret.txt`
⇒ 现在 `/health` 正常返回 `agent_alive / agent_pid / agent_seen`。
⇒ **后续轮次判断「relay agent 是否活着」直接用 `tools/relay_status_probe.py`**，
不要自己拼凭据。

### 🔴 坑 9：`/proc/<pid>/cmdline` 里 relay 自身的 bash -lc 整段命令极长
⇒ 若直接把 cmdline 打进输出，会把 relay 的 `out` 挤爆（r28 就是这样被截断，`=== E/H2` 两段丢失）。
**正解**：①`head -c 300` 截断 ②`cut -c1-60` 再截 ③排除自身 pid（`$$`）
④**最稳做法：用「命令行关键词」分类只打印需要的几类**（guard / master / fedavg / eval），
而不是全量打印。

---

## 第四轮：2026-11-23 20:53~21:07（健康核验轮 · 磁盘告警排查）

### 结论：**正常运行中，1/7 成功，零错误零隐藏缺陷；本轮未做任何关机/重启/删除/改码**
分支 = **A-pending**。`.rc`：仅 `p5_bal_fedprox=0`，其余 6 个 NONE（未跑到 ≠ 失败）。
守护 PID 127867（17:04:19 起，state=S 正常轮询）、训练树唯一（127355→188778→188779）、
relay agent 滞后 6.9s、错误扫描 0 命中、评估参数量 p=22.38 ✔。

### 状态快照（21:04~21:07）
- `p5_bal_scaffold` **round 40/50 训练中**（轮速 ~2.0 min/轮），du 5.6G
- `p5_iso_fedavg` 50 轮训完待补评估；其余 4 配置未开始
- GPU 58% util / 4296 MiB；磁盘 **6.0G 可用（81%）< 8GB 报警线**
- relay id≥1010000 count=0（未到推送时机，符合预期）

### 🔴 磁盘告警已查清：**稳态振荡，不是泄漏**
90 秒连续采样：5.9G(37 轮) → 6.0G(38 轮) → **6.3G(39 轮)**，随轮次推进在**回升**。
`cleanup_old_baseline` 按 `cutoff=cur-KEEP_ROUNDS(2)` 回收 anchor_r*/ci_c*_r*/c_global_r*
（`ci_c{c}.pt` 与 `c_global_r{r}.pt` 永不删）；`cleanup_completed_config` 只删
client_*/opt_state_c*/早期 global_r，**不删 `_baseline`（2.1G）** ⇒ SCAFFOLD 类配置有 3.2G 永久残留。
🔴 **`MIN_FREE_GB=6` 硬闸是自愈式**（`run_fedavg.sh:583` → 强制只留最近 1 轮），**不会中止实验** ——
13:36 那次「可用空间仅 5GB < 6GB 而聚合中止」发生在该修复**之前**。
剩余风险：`p5_iso_scaffold` 阶段峰值可能压到 4~5G ⇒ 盯 `.rc` 是否变 1。

### 🔴 推送链路终检（决定最终数据能否取回）——通过
`relay_push_out.py`：`os.path.isfile()` **跟随软链**（正常软链 OK），但**悬空软链静默 skip ⇒ manifest 缺文件**。
守护清单 `cd $OUT && ls -1 *.json *.md *.csv *.png eval_*.log autoshutdown.log extract_phase5.log`。
实测 `$OUT` 下 31 个 `eval_*.log`：**13 可解析 / 18 悬空**，而 18 = 恰好「尚未评估的 6 配置 × 3 domain」。
软链目标 `/root/HPT/output/eval_<tag>_<domain>.log` 正是 `eval_model.sh:55-56` 写入的路径 ⇒ 跑完自动变可解析。
**实证**：`p5_bal_fedprox` 3 条软链（11:04 预建）现已 RESOLVE → 76373/76340/76868 B 实体文件。
`run_phase5.sh`/`eval_model.sh` 里**无 `ln -s`** ⇒ 软链是 11:04 一次性预建的 21 条，已覆盖全部 7×3。
`extract_phase5_results.py` 的 `eval_roots=[phase5, output]` 双根搜索 + `collect_baseline` 取
`/root/HPT/output/phase3_raw.json` ⇒ 云端提取链路完整。**推送不会静默丢一级证据。**

### ETA / 费用（实测重算）
scaffold 训练 ≈21:25 → 3 评估 ≈22:20 → iso_fedavg 补评估 ≈23:15 → 4 配置 ×154 min
⇒ **≈09:30（11-24 明早）**。守护时限 2026-11-25 13:04（MAX_HOURS=44）远在之后。
GPU 费 ≈ **40 元**，跑完自动关机 ⇒ 之后 0 元。

### 本轮新踩的 4 个坑（**下次直接照抄**）
- **坑 10**：任务书给的 marker `完成，全局模型` **恒为 0** —— 云端日志实际只有 `全局模型`。
  正确：`LC_ALL=C grep -c '全局模型' $OUT/logs/<tag>.log`，且**逐配置 grep 再求和**
  （`grep -rc ... $d/` 会把子目录重复计数）。
- **坑 11**：`glob=$(ls -d $d/global_r*|wc -l)` **严重低估进度**（cleanup 会回收，稳态只剩 2~4 个）
  ⇒ r30 读到 glob=4 误判「只跑 4 轮」，实际已 36 轮。**轮次只能用日志 marker 判。**
- **坑 12**：relay `out` 有长度上限，探查输出会被截断（r32 丢了 S3/S4 后半）
  ⇒ 长代码一律 base64 传输 + 本地解码；`===` 标题后紧跟一行 b64。
- **坑 13**：`du -sh` 对软链报 0，**不能用它判断 eval 日志是否存在**
  ⇒ 用 `for f in ...; do [ -f "$f" ] && RESOLVE || BROKEN; done`，且要 `[ -e ] || [ -L ]` 兜底悬空。

### 产出物
- `results/第5阶段进度检查与健康核验报告_2026-11-23-2107.md` —— 8 节完整报告
- 探查脚本 `tools/tmp/p5probe_r30.py ~ p5probe_r35.py`（独占 id 段 1002000~1002999）

### 下次执行必做（按顺序）
1. `.rc` 数量**与内容**（内容 0 才算成功）：7/7 ⇒ 走收尾流程（分支 A）
2. `relay /results?since=1010000` 找 `MANIFEST:` ⇒ 有则 `python tools/relay_fetch_out.py --since 1010000 --outdir results/phase5`（逐文件 sha256 校验）
3. 云端 `python /root/fedcore/extract_phase5_results.py` 生成 phase5_raw.json/md
4. `python tools/phase5_table.py results/phase5/phase5_raw.json --phase3 results/phase3_raw.json` 出主表 + 配对 t 检验（**不要手工抄数字**）
5. 写 `results/第5阶段结果与结论.md`（t_crit=4.303 / df=2；每 domain 参数量 ≥21.0 的证据；H2 用单变量消融对照）
6. `.rc` 有内容 1 ⇒ 部分失败 ⇒ **不要盲目重跑**（铁律 D）
7. 守护不在 + 训练在 ⇒ 不关机，重挂守护（分支 B 裸串命令，不用 pkill）
8. 守护不在 + 训练不在 + `.rc` 不全 ⇒ 先取产物再 `/usr/bin/shutdown`（分支 C）
9. 磁盘 < 8GB 继续报警，重点看 iso_scaffold 阶段峰值
10. automation 每 2 小时一轮（22:53 / 00:53 / 02:53 / 04:53 / 06:53 / 08:53 / **10:53**）⇒ **10:53 那轮应命中完成态**

---

## 第五轮：2026-11-23 23:10~23:20（健康核验轮 · 3/7 已完成）

### 结论：✅ **正常运行中，3/7 成功，0 失败，0 隐藏缺陷 ⇒ A-pending，本轮零写操作**
分支判定：A-pending（`.rc` 3/7、relay count=0、守护健在、训练健在）
⇒ **未关机、未重启、未删除、未改码、未 `git add -A`**。

### 状态快照（23:16~23:18 云端实测）
- `.rc`：`p5_bal_fedprox=0`、`p5_bal_scaffold=0`、`p5_iso_fedavg=0`（**3 个全 0 = 成功**），
  `p5_iso_fedprox/iso_scaffold/iso_eda/iso_edaclu` = NONE（未跑到 ≠ 失败）
- 轮次（日志 marker `全局模型`）：bal_fedprox=51、bal_scaffold=51、iso_fedavg=1（复用段）、**iso_fedprox=9（第 9 轮在跑）**、其余 0
- 进程树：`run_phase5.sh` pid 127355（ppid=1）→ fork 子 shell 280089 → `run_fedavg.sh` 280090 → 3× `hpt.run` 子进程；
  守护 `phase5_autoshutdown.sh` pid 127867（ppid=1，et 06:12:55，MAX_HOURS=44 / POLL=60 / PUSH_IDBASE=1010000）
  🔴 **`MASTER pid=280089 ppid=127355` 是 fork 子 shell，不是第二实例**（判重复要看 ppid=1 的根）
- GPU RTX 3090：3 次采样 47% / 52% / 0%（4346MiB）；磁盘 30G 用 23G，**7.6~7.9G 可用（75%）**
- relay agent 存活 `agent_alive=True pid=1841 seen_lag=2.3s`；`/results?since=1010000` **count=0**（未到推送时机）

### 🔴 坑 14（差点误判）：**GPU 0% / 6MiB 不等于训练挂了**
r40 单点采样读到 GPU 0% + 6MiB + mem 1MiB，疑似训练进程已死。r41 用
①3 次连续采样（47/52/0%）②`run_fedavg.sh` + 3× `hpt.run` 子进程健在 ③日志 mtime 23:17:32 仍在写
⇒ **是轮次间隙的瞬时 0%**（1~10 秒），训练正常。
⇒ **判据必须是「worker 进程是否存活 + 日志 marker 是否推进」，不能只看 nvidia-smi 单点采样。**

### 🔴 坑 15：磁盘 7.6~7.9G（低于 8GB 线）不是泄漏，是稳态振荡
22:58:50 iso_fedavg 完成 ⇒ `cleanup_completed_config` 回收 ⇒ 磁盘回到 11G ⇒ iso_fedprox 训练按轮堆积
`_baseline/anchor_r*.pt` ⇒ 回落到 7.6~7.9G ⇒ 每轮 `cleanup_old_baseline` 释放 ~520MB。
`MIN_FREE_GB=6` 硬闸是**自愈式**（`run_fedavg.sh:583`），实测 `run_phase5.log` 中磁盘闸命中 **0 次**。
⇒ 只需盯 `p5_iso_scaffold` 阶段峰值（`ci_c*.pt` 更多）+ `.rc` 是否变 1。

### M5 / M1 复验（今天修的三个致命静默缺陷，仍全部生效）
- `hook skipped` 文件数：**bal_fedprox=0 / bal_scaffold=0 / iso_fedprox=0**；iso_fedavg=200 文件 11550 行（无害：BASELINE="" 纯 FedAvg，NameError 在调用点参数求值、钩子体未执行 ⇒ 梯度零修改 ≡ no-op）
- FedProx 生效证据：`近端项已施加 mu=0.01 键数=488 ‖g_prox‖₂=3.0047e-03`（iso_fedprox）/ `4.1773e-04`（bal_fedprox）＋`非零 44/45 个 step`
- SCAFFOLD 生效证据：`控制变量已应用 键数=444` ＋ `本地控制变量已更新：键数=488 K=45 eta=0.0001 → ci_c0_r9.pt`
- 评估参数量：已完成 3 配置 × 3 domain 全部 **params=22.38**（≥21.0，无 13.94M 残缺）
- 错误扫描：`run_phase5.log` + 4 份配置日志，`Traceback/NameError/OOM/No space/失败/错误` 全部 **0 命中**

### 守护源码复核（读本地同 md5 副本 `federated/phase5_autoshutdown.sh`，md5 3c2713ed57a050f05a19b7eb18d9b802 = 云端）
- `WORKER_PAT='[r]un_phase5\.sh|[r]un_fedavg|eval_model\.sh|hpt\.run|hpt\.run_eval|aggregate\.py|fed_baselines\.py'`
  ⇒ **训练期 `run_fedavg.sh`+`hpt.run` 必命中** ⇒ 轮次间隙 GPU 0% **不会**触发「提前收工」(idle≥3) 或「卡死」(idle≥5)
  ⇒ 评估期 `eval_model.sh` 命中 ⇒ GPU 2~7% 也不会误判。**这是「跑完必关机」能安全依赖的关键。**
- `rc_is_ok` 只认内容 0 ✔；`exit 9` 自检不过不关机 ✔；`/usr/bin/shutdown` ✔；MAX_HOURS=44 → 2026-11-25 13:04 ✔

### 推送链路完整性
`$OUT/eval_*.log` **31 个：19 可解析 / 12 悬空** ⇒ 12 = 恰好「4 个未评估配置 × 3 domain」，
评估跑完自动变可解析（bal_fedprox 的 3 条已由悬空转实体，印证该机制）⇒ 推送不会静默丢一级证据。

### ETA / 费用（实测速率）
实测 2.08 min/轮（iso_fedprox 22:58:50→23:17:32 完成 8 轮）、**实测 ~9 min/评估**（iso_fedavg task2 22:49:57→task3 22:58:50）
⇒ iso_fedprox ~01:10~01:40 → iso_scaffold ~03:30~04:15 → iso_eda ~05:45~06:30 → **iso_edaclu ~08:00~08:45**
**总 ETA 11-24 早 08:00~09:00**；剩余 ~9.4h ⇒ **GPU 费 ≈ 20~25 元**，跑完自动关机 ⇒ 之后 0 元。

### 产出物
- `results/第5阶段进度检查与健康核验报告_2026-11-23-2320.md` —— 10 节完整报告
- 探查脚本 `tools/tmp/p5probe_r40.py / r41.py / r42.py`（独占 id 段 1002000~1002999）

### 下次执行必做（按顺序）
1. **`.rc` 数量与内容**（内容 0 才算成功）：7/7 ⇒ 走收尾流程（分支 A）
2. 7/7 但 relay 无 id≥1010000 manifest ⇒ 守护推送未成功 ⇒ **不要重跑**，查守护日志 / 手动触发推送
3. `.rc` 有内容 1 ⇒ 部分失败 ⇒ **不要盲目重跑**（铁律 D：跨批次续跑破坏 LR/Adam 相位）
4. 守护不在 + 训练在 ⇒ **不要关机**，重挂守护（分支 B 裸串命令，不用 pkill）
5. 守护不在 + 训练不在 + `.rc` 不全 ⇒ 先取产物再 `/usr/bin/shutdown`（分支 C）
6. 探查用 `tools/tmp/p5probe_r4*.py`（**独占 id 段 1002000~1002999**），不用 `cloud_exec.py` 默认段、不 `POST /clearall`
7. 磁盘 < 8GB 报警但先分清是振荡还是泄漏（看 `run_phase5.log` 磁盘闸命中数）
8. **判断训练是否活着必须看「worker 进程 + 日志 marker」，不能只看 nvidia-smi 单点采样**
9. 收尾时：`relay_fetch_out.py --since 1010000` → 云端 `extract_phase5_results.py` → **`tools/phase5_table.py` 出主表（不手工抄数）** → 写 `results/第5阶段结果与结论.md`（t_crit=4.303 / df=2；每 domain 参数量 ≥21.0 证据；**H2 用单变量消融对照**）
10. 🔴 下一轮 00:53 / 02:53 / 04:53 / 06:53 / 08:53 ⇒ **08:53 那轮应命中完成态，10:53 兜底**

---

## 2026-11-24 05:30~05:35 第六轮（分支 A-pending，零写操作）

### 结论
**5/7 `.rc=0`**（bal_fedprox / bal_scaffold / iso_fedavg / iso_fedprox / iso_scaffold 全 0），
`p5_iso_eda` 第 **47/51** 轮训练中（05:34:14），`p5_iso_edaclu` 未开始。
守护 pid 127867（ppid=1，et 12:29:57）**GUARD_POLLING=YES**（70s vcs 采样递增）。
⇒ **不需要关机 / 重启 / 删除 / 改码**。云端写操作 = 零。

### 🔴 ETA 修正（务必读）：上轮「08:00~08:30」偏早约 45 分钟
- eval 日志进度条实测 `success: 0.0: 100%|…| 20/20 [19:22<00:00, 66.71s/it]`
  ⇒ **单 seed 20 episodes ≈ 19~20 min**（上轮误把「~9 min/评估」当单 domain 墙钟，用的是 task3 的 8.9 min 偏小样本）。
- master log 实测各段墙钟：iso_fedavg task1 10.2 / task2 18.1 / task3 8.9 min ⇒ **单 domain 保守取 18 min，3 domain ≈ 54 min**。
- 训练速率 **1.83 min/轮**（iso_eda 04:08:26 → 05:34:14 跑 47 轮）。
- **重算**：iso_eda 训练 ~05:42 完 → 评估 ~06:36 完 → iso_edaclu 训练 ~08:10 完 → 评估 **~09:05 完**
  ⇒ **全矩阵 ETA ≈ 09:00~09:45**。守护绝对时限 2026-11-25 13:04，余量 ~27h，安全。
- ⇒ **08:53 那轮很可能只到「最后一个配置评估中」，10:53 那轮才是兜底必查**；09:00 前不得判定「未完成⇒需干预」。

### 🔴🔴 H2 判决口径已独立复核（用本地 phase3_raw.json 复算，与任务书逐位吻合）

| 对照 | A(task3) | B(task3) | Δ均值 | 复算 t | 任务书 |
|---|---|---|---|---|---|
| 簇感知 vs 中心性EDA | [0.9,0.6,0.5] | [0.5,0.2,0.05] | +0.4167 | **25.000** | t 25.0 ✓ |
| 簇感知 vs MoE+FedAvg | [0.9,0.6,0.5] | [0.55,0.2,0.1] | +0.3833 | **23.000** | t 23.0 ✓ |

⇒ **这两组数字 = 第3阶段 `p3_moe_eda_cluster` 在 task3 上的配对 t**。

**澄清**（任务书表述易误读）：

* 「p3 原始 = nj3 = **0.55/0.20/0.10**」是 `p3_moe_fedavg` 的 **task3 三 seed 值**（不是三个 domain！）；nj1=0.50/0.10/0.20 同理。
* p3 的 MoE 臂里 **task1(reach-v2) 与 task2(push-v2) 三 seed 全 0**，**task3(button-press) 才有分**
  （fedavg 0.283 / 中心性 0.250 / 簇感知 0.667）。
* ⇒ **H2 判决战场 = task3**；主表口径「哪一组与第3阶段原始日志逐行相同」= 比 **task3 的 per-seed 三元组**。
* 第5阶段要复现的同一对照（isolated 臂单变量消融）：`p5_iso_edaclu` vs `p5_iso_eda` vs `p5_iso_fedavg`，**同样主战场 task3**。

### 🔴 三域分辨力诊断（已实测 15/30 格）

| domain | env | p3 MoE臂 | p5 balanced | p5 isolated | 分辨力 |
|---|---|---|---|---|---|
| task1 | reach-v2 | 0.000 全0 | 0.000 / 0.133 | **0.600 / 0.567 / 0.000** | ✅ isolated **救回** task1 |
| task2 | push-v2 | 0.000 全0 | 0.000 / 0.000 | **0.000 / 0.000 / 0.000** | ❌ 5/5 全0 **零分辨力** |
| task3 | button-press | 0.283/0.250/0.667 | 0.350 / 0.000 | **0.783 / 0.683 / 0.000** | ✅ 主战场 |

已填 15 格（3 seeds × 20 rollouts，per-seed 原值）：

- bal_fedprox 0.000 / 0.000 / 0.350
- bal_scaffold 0.133 / 0.000 / 0.000
- iso_fedavg **0.600 / 0.000 / 0.783**
- iso_fedprox **0.567 / 0.000 / 0.683**
- iso_scaffold **0.000 / 0.000 / 0.000**

⇒ **task1 不是恒 0**（前几轮记忆有误，isolated 下有分）；**真正恒 0 的是 task2/push-v2**（p3+p5 全 0）。
⇒ 主表 task2 一列必然全是 0.000，**不能承载任何结论**，出表时须显式标注。

### 🔴 `p5_iso_scaffold` 三域全 0（本轮新发现，待解释，非故障）

- iso_fedavg 0.600/0/0.783、iso_fedprox 0.567/0/0.683、**iso_scaffold 0/0/0**；bal_scaffold 0.133/0/0 也近乎全 0
  ⇒ **SCAFFOLD 两条臂都全线退化**。
- **已排除**：M5 失效（客户日志 ctrl_applied=11550 skipped=0、键数=444 ✔）；模型残缺（params=22.38 ✔）；
  NaN/Inf（全部 15 份 eval 日志 **NaN=0、inf=0** ✔）；评估跑错任务（env/domains/seed/n_jobs 全对 ✔）；
  fail-open（rc=0 ✔）。
- ⇒ 只剩两种可能：(a) SCAFFOLD 在此设置下的真实行为；(b) 实现细节缺陷。
  **收尾时作为独立发现写清，并声明 H2 三对照（iso_edaclu/iso_eda/iso_fedavg）都不含 SCAFFOLD 行 ⇒ H2 不受影响。**
- 本轮**未**改动云端、未重启。

### `p5_iso_fedavg` 来源与有效性论证（H2 基准行，必须有效）

- master log：`[22:21:41] [WARN] 训练产物在但评估不齐 —— 补评估` + `[续跑] 从第 50 轮继续`
  ⇒ 其 50 轮训练发生在 **11-23 13:28~14:24**（comm_report_r0..r31 时间戳），**早于 17:03 正式开跑**。
- 有效性三条：① M1 只作用**评估侧**（训练侧 trunk.pth 本就 21.08M），iso_fedavg 评估 22:21~22:58 跑、params=22.38 ⇒ M1 生效；
  ② M5 不影响该配置（BASELINE="" 纯 FedAvg，skipped=11550 属预期：NameError 在调用点参数求值，钩子体未执行 ⇒ 梯度零修改 ≡ no-op）；
  ③ M2/M3/M4（磁盘预算/fail-open/钩子变量名）不改数值。
- **划分同构性双向确认**：`run_phase5.sh` 第 13~18 行（balanced 复用第3阶段）+ 第 138 行
  「孤立客户端：全部方法重跑（划分与第3阶段不同 ⇒ 不可复用）」；
  `extract_phase5_results.py` STAGES：`p3_moe_fedavg/eda/eda_cluster`=balanced/第3阶段复用，
  `p5_iso_fedavg`=isolated/第5阶段新跑。
  ⇒ **H2「同划分」前提成立**（isolated 臂三行全是 isolated+新跑）。
- STAGES 已核：7 个 p5 配置 + 3 个 p3 复用行 **齐全** ⇒ 主表 10 行 × 3 domain = 30 格。

### M1/M5/N_JOBS 三项复验（全通过）

- 15 份 eval_p5_*.log 全 `params=22.38` + avg 表 3 行；OUT 内 10 份 p3 日志也全 22.38。
- M5 客户日志计数：bal_fedprox prox=10950 skipped=0｜bal_scaffold ctrl=10950 skipped=0｜
  iso_fedprox prox=11550 skipped=0｜iso_scaffold ctrl=11550 skipped=0｜iso_fedavg skipped=11550（预期）｜iso_eda 全0（预期）。
  样本：`[FedVLA-BASE] FedProx 近端项已施加 mu=0.01 键数=488 ‖g_prox‖₂=3.4095e-04`、
  `近端项统计：非零 44/45 个 step`、`[FedVLA-BASE] SCAFFOLD 控制变量已应用 键数=444`。
- N_JOBS：master 127355 与 466787 均 `N_JOBS=3`；`eval_model.sh` 含 `n_jobs=$N_JOBS`；
  eval 头部 `[HYDRA] Joblib.Parallel(n_jobs=3,...) is launching 3 jobs`、pretrained_dir=`global_r49`、seed=0/1/2、episode_num=20。

### 磁盘（5.7 GiB，低但可控）

`overlay 30G 25G 5.7G 82%`；`可用空间仅` 命中 **0**。
⚠️ 低于任务书 8GB 报警线（上轮 6.1G → 本轮 5.7G）；**仍高于 MIN_FREE_GB=6**（5.7 GiB = 6.12 GB 十进制），距闸 ~0.12 GB。
稳态成分：bal_scaffold 2044MB + iso_scaffold 1561MB = SCAFFOLD `_baseline`（`ci_c*.pt`/`c_global_r*.pt` 按设计永不删）；
iso_eda 2251MB = 训练中占用（评估完成后 cleanup_completed_config 回收 13~18 个中间产物）。
**风险定级：中等可控且不影响数值** —— 将来的 iso_eda/iso_edaclu 都是 EDA 聚合、**不使用 anchor/ci**，
磁盘闸即使触发回收历史锚点也不改这两个配置的数值。**不做任何手工删除。**

### 🔴 坑 17（探查超时 11m44s 的根因）：relay **双 id 错配 ⇒ 僵尸队列饿死**

一条 `/enqueue` 产生**两份** result：云端 `relay_bootstrap.py --idbase N` 报 **N**（大整数，存活）；
云端 relay agent 报 **队列 id**（小整数，如 3960）。服务端 `/result` 只保留按 `int(id)` 排序后的**末尾 500 条**
⇒ 小整数先出局 ⇒ `/poll` 的 done 集合找不到队列 id ⇒ 命令被判「未完成」⇒ **每轮重新派发** ⇒ 无限僵尸循环
（实测 25 条历史探查每轮被串行重跑，新探查永远排队尾 ⇒ r52 超时 11m44s 零结果）。

**修复**：`tools/tmp/p5probe_v2.py` **入队前** `POST /clearall {"before": int(time.time())}` 丢弃陈旧命令。
实测 `cleared=25 → queue_after=0` 后探查立刻成功。

**为什么这次可以安全清**（与前几轮「绝不 clearall」相反但成立）：① 队列 25 条 idbase 全在本 automation 自己的探查段
（1002038~1004000），无一条属并发会话（并发用 1005000~1009999 段）；② 并发会话 relay 最后活动 11-23 18:46（12h 前）；
③ 其它 3 个 automation 最后活动也在 11-23 18:46 前；④ **守护关机路径不走队列**（`phase5_autoshutdown.sh` 里
`enqueue|/poll` 命中 **0**，推送走 `relay_push_out.py` → 直接 `POST /result`）⇒ 清队列不影响最高优先级链路；
⑤ 已逐条审计 `tools/tmp/p5probe_r*.py` 全部只读。

⇒ **本轮起 `tools/tmp/p5probe_v2.py` 是标准探查器**，已加 `--seg` 参数（坑 16：每轮换新独占段）。

### 🔴 坑 18：M5 钩子日志**不在主日志**里，在**客户日志**里（假阴性）

r54 在 `$OUT/logs/<tag>.log` grep `prox_applied|ctrl_applied|skipped` 得 0/0/0 ⇒ 一度误判 M5 失效。
真相：钩子日志在 `$OUT/<tag>/log_client_<c>_r<r>.txt`（每客户端一份）。改到客户日志内 grep 立即拿到真实证据。
⇒ **以后查联邦基线钩子一律 grep `$OUT/<tag>/log_client_*.txt`**。

### 🔴 坑 19：主表汇总在 **master log**（不在 eval_*.log）；`nan` 检测会误报

- `eval_*.log` 里 grep `逐任务汇总`/`OVERALL` **全空** ⇒ 汇总段在 **`$OUT/run_phase5.log`**（每 domain 一段：
  `output: [per-seed]` → 逐任务汇总 → `SUMMARY_DONE` → `完整日志: <path>`）。
- r57 的 `nan=3` 是**误报**：`grep -ciE 'nan|inf'` 里 `inf` ⊂ `info`（3 次 "info"）⇒ 精确检测须用
  `\b(NaN|nan)\b` / `\binf\b` ⇒ 实测全 **0**。
- ⇒ **提取主表数据源 = `run_phase5.log` 汇总段**；`eval_*.log` 只给 hydra 头部（n_jobs/pretrained_dir/env_names/seed/episode_num）+ 进度条。

### 产出物

- `results/第5阶段进度检查与健康核验报告_2026-11-24-0535.md` —— 10 节完整报告（含 15 格分数预览 + 三域分辨力诊断 + ETA 修正）
- `tools/tmp/p5probe_v2.py`（新增 `--seg`）、`tools/tmp/p5probe_r57.py`（分数结构）、`p5probe_r58.py`（master log 汇总段 + 复算）、
  `p5probe_r59.py`（iso_fedavg 来源）

### 下次执行必做（按顺序）

1. **`.rc` 数量与内容**（内容 0 才算成功）：7/7 ⇒ 走收尾流程（分支 A）
2. 7/7 但 relay 无 id≥1010000 manifest ⇒ 守护推送未成功 ⇒ **不要重跑**，查守护日志 / 手动触发推送
3. `.rc` 有内容 1 ⇒ 部分失败 ⇒ **不要盲目重跑**（铁律 D：跨批次续跑破坏 LR/Adam 相位）
4. 守护不在 + 训练在 ⇒ **不要关机**，重挂守护（分支 B 裸串命令，不用 pkill）
5. 守护不在 + 训练不在 + `.rc` 不全 ⇒ 先取产物再 `/usr/bin/shutdown`（分支 C）
6. 探查用 `tools/tmp/p5probe_v2.py --seg <新段>`（**每轮换新独占段**）；必要时追加 r6x 专项脚本（只用 ls/stat/grep/awk/head/tail/ps/nvidia-smi/df）
7. 磁盘 < 8GB 报警但先分清是振荡还是泄漏（看 `可用空间仅` 命中数）；**绝不手工删云端文件**
8. 判训练是否活着必须看「worker 进程 + 日志 marker 推进」，不能只看 nvidia-smi 单点采样（坑 14）
9. 收尾时：`relay_fetch_out.py --since 1010000` → 云端 `extract_phase5_results.py` →
   **`tools/phase5_table.py` 出主表（不手工抄数）** → 写 `results/第5阶段结果与结论.md`
   （t_crit=4.303 / df=2；每 domain params_M ≥ 21.0 证据；**H2 用单变量消融对照且落在 task3**；
   task2 零分辨力要标注；task1 被 isolated 救回要写；iso_scaffold 全 0 要写）
10. 🔴 **06:53 / 08:53 / 10:53 三轮**；**08:53 大概率还没完**（ETA 09:00~09:45），**10:53 是兜底必查**

---

## 第七轮：2026-11-24 07:36~08:47 —— ✅ **任务终结：实验全完成 + 实例已关机 + 主表已交付**

### 结论（终结态，不再需要后续监控轮次）
**7/7 配置全部成功**（`.rc` 内容全 0）→ 守护自动走完「提取 → 打包 → 推进 relay → 回读自检 → `/usr/bin/shutdown`」→
**实例已停止计费**。产物已取回本地（35 文件 sha256 全一致），主表 + 结论文档已交付。
**后续轮次若再触发本 automation：直接确认实例已停 + 产物已在本地 `results/phase5/`，无需任何云端动作。**

### 关键时间线
07:37（6/7，edaclu 训练中 48/51）→ 07:42:43 训练完 → 07:53:26 task1=0.550 → 08:11:26 task2 完成 → 08:28:14 task3=0.300
→ **08:28:37 守护检测 7/7 收工**（extract rc=0，打包 52K，manifest id=1010036 / 35 文件）→ 08:32 本地取回完毕
→ 08:38~08:47 relay `agent_seen` lag 578→651→697→778 s 持续增长 + 探查命令入队超时 ⇒ **容器已退出**

### 🔴 本阶段最重要的科学发现：**H2 在两个划分上结论相反**（不可只报一边）
| 划分 | 对照 | t | p | 同向 | 判决 |
|---|---|---|---|---|---|
| balanced（复用第3阶段） | 簇感知 vs 中心性EDA | **25.000** | 0.002 | 3/3 | ✅ 显著 |
| balanced | 簇感知 vs MoE+FedAvg | **23.000** | 0.002 | 3/3 | ✅ 显著 |
| isolated（第5阶段新跑） | 簇感知 vs 中心性EDA | **−15.588** | 0.004 | 0/3 | ❌ **反向显著** |
| isolated | 簇感知 vs MoE+FedAvg | **−8.043** | 0.015 | 0/3 | ❌ **反向显著** |

机制（实测权重，非推测）：`eda_cluster_weights` 是**簇间等权**（每簇 1/C）。
- balanced：clusters=[0,0,1,0]，**task3 独占客户端 c2 单独成簇**拿 0.5000 ⇒ task3=0.667
- isolated：clusters=[0,1,1,1]，**task1 独占客户端 c0 单独成簇**拿 0.5000，
  而 **task3 独占客户端 c2 被并进 3 客户端簇、权重仅 0.1629**（低于中心性 EDA 的 0.2438）
  ⇒ task3 从 FedAvg 0.783 塌到 0.300，3/3 seed 全线塌陷
⇒ **簇感知收益来自"把 1/C 权重给到正确那个簇"，不是"按簇聚合"本身**。
两条根因（聚类质量 / 簇间等权公式对簇大小敏感）**尚未区分** ⇒ P0 修复方向：簇间改按簇内任务分布或独占度加权。

### 三域分辨力（最终定论）
- **task2 (push-v2)：10 行配置全部 0.000（30/30 格）⇒ 零分辨力，主表该列不承载任何结论**
- **task1 在 isolated 划分下被救回**（0.550~0.617）⇒ isolated 臂多承载一个可用域
- **task3 (button-press) 是主战场**，两划分皆有分辨力

### SCAFFOLD 两条臂全线退化（独立发现，已排除 5 种成因）
bal_scaffold=0.133/0/0；iso_scaffold=0/0/0（vs FedAvg task3 t=−11.750 **p=0.007 显著更差**）。
已排除：M5 失效（hook skipped=0、ctrl 键数=444 ✔）、模型残缺（22.38M ✔）、NaN/Inf（31 份日志 0 命中 ✔）、
评估跑错任务 ✔、fail-open（rc=0 ✔）。**H2 三行对照不含 SCAFFOLD ⇒ H2 判决不受影响。**

### 数据完整性
- **30/30 格独立交叉验证零不一致**（云端 extract vs 本地重解析 eval 日志）
- **全部 30 格 params_M = 22.38**（集合仅 {22.38}），≥21.0 ⇒ 无残缺模型
- 35 文件 sha256 逐个一致；N_JOBS=3 全链一致

### 🔴 坑 20（本轮新踩，下轮务必照抄）：**云端长轮询会堵死 relay agent，且 id 段会被并发会话抢占**
- r63 用「云端 900 秒 while 循环」→ 21 分钟后本地零结果：relay agent **串行执行**，一条超长命令把后续全堵住
- 实测 **1009100~1009147 全是并发会话的内容** ⇒ 我的 cid 被抢占
⇒ 正解：**每条云端命令都短（<30s）+ 本地循环**（`tools/tmp/p5watch.py`），独占段用 **1007xxx/1008xxx**，
  **绝不碰 1009xxx**（并发会话的地盘）；且 **`p5probe_v2.py` 的 `clearall` 在并发会话活跃时会清掉别人的在队命令**
  ⇒ 等待阶段改用**不调 clearall** 的 `p5watch.py`。

### 产出物
- `results/第5阶段结果与结论.md`（10 节：主表 / H2 判决 / 机制 / 三域分辨力 / 数据证据 / 局限 / 产物清单 / 复现命令 / 关机费用）
- `results/phase5/`（35 个云端产物，含 21 份 p5 + 10 份 p3 eval 日志、phase5_raw.json/md、MANIFEST.json）
- `results/phase5/phase5_table.md`（主表 + 配对 t 检验，脚本生成未手抄）、`phase5_table.csv`、`phase5_h2_tests.csv`
- 探查脚本 `tools/tmp/p5probe_r60.py`~`r64.py`、`tools/tmp/p5watch.py`

### 费用
实例 2026-11-23 11:03 ~ 2026-11-24 08:28 ≈ **21.4 h**，RTX 3090 ≈2.2 元/h ⇒ **≈47 元（估算，以 AutoDL 账单为准）**；
**关机后 0 元**。

---

## 第八轮：2026-11-24 10:56~11:05 —— 终局复核轮（无新动作）

### 结论
**任务已终结，本轮为只读复核，未做任何云端写操作。** 分支 A 已成立且**本轮独立复核通过**：
- 实例**确已停止计费**（下文两条独立证据）
- 产物**完整无缺**（35/35 文件 sha256 与 manifest 逐个一致）
- 主表**可复现**（重跑 `phase5_table.py` 与既有 `phase5_table.md` **逐字节相同**）
- 结论文档**已交付**（`results/第5阶段结果与结论.md`，10 节）

### 本轮做的独立复核（不依赖上轮结论，全部现测）
| 项 | 手段 | 结果 |
|---|---|---|
| 实例已停 | relay `agent_seen` 时间戳 vs 当前时间 | last_seen=2026-11-24 08:29:05，lag **8926s ≈ 2.48h**（>120s 阈值 ⇒ 失联） |
| 实例已停 | 新下发一条只读探针（独占 id 段 1008600）并等 120s | **无任何结果返回** ⇒ relay agent 已不执行命令 ⇒ 容器确已退出 |
| 产物完整 | 本地 35 文件重算 sha256 vs `MANIFEST.json` | **OK 35 / mismatch 0 / missing 0** |
| 主表可复现 | 重跑 `python tools/phase5_table.py results/phase5/phase5_raw.json --phase3 results/phase3_raw.json` | 与既有 `phase5_table.md` **逐字节相同**（diff 无输出） |
| MoE 完整性 | 31 份 eval 日志 grep `number of params (M):` | 取值集合 = **{22.38}** ⇒ 30/30 格全部 ≥21.0，**无 13.94M 残缺** |
| NaN/Inf | 31 份日志精确匹配 `\b(NaN|nan)\b` / `\binf\b` | **0 命中** |
| 阶段完整 | `phase5_raw.json` | 10 stage × 3 domain × 3 seeds = 30 格齐全 |

### 🔴 坑 21（本轮新踩，已修）：**参数量正则漏了 `(M)` 单位 ⇒ 误报「params 缺失」**
第一次用 `number of params[:\s]*([0-9.]+)` 在 31 份 eval 日志上**全部匹配失败**（集合为空），
差点误判「模型残缺 / 产物损坏」。实际日志格式是 `number of params (M): 22.38`
⇒ 括号里的 `(M)` 打断了正则。
⇒ 正解：`r"number of params \(M\):\s*([0-9.]+)"`。
**教训：探查正则必须先 `grep -m1` 看一眼真实格式再写正则；「匹配不到」≠「数据不存在」，
先区分「正则错」还是「数据缺」再下结论。**

### 🔴 坑 22：relay `/health` 的 `agent_alive: true` **不可信**
`/health` 仍返回 `agent_alive: true, agent_pid: 1841`（服务端缓存的旧值），
但 `agent_seen` 已 2.48h 未更新 ⇒ **该字段是陈旧缓存，不代表进程存活**。
⇒ 判活**必须用 `agent_seen` 的滞后时间**（阈值 120s），不能信 `agent_alive` 布尔值。

### 后续
🔴 **本 automation 已无监控对象**（实例已关机、实验已完成、产物已落地、结论文档已交付）。
若后续仍被触发，**只需确认实例仍停 + `results/phase5/` 产物仍在**，无需任何云端动作。
**下一步实验需求已写进 `results/第5阶段结果与结论.md` §8「后续优先级」**（P0 = 固定 C=3 重跑两臂 + 补 task4 域），
需用户批准后另起新任务，**不要在已关机的实例上做任何动作**。

---

## 第九轮：2026-11-24 13:02~13:12 —— 终局复核轮 + **automation 已暂停**

### 结论
**任务终结，本轮为只读复核 + 收尾。** 云端写操作 = 零（未关机/未重启/未删除/未改码/未 `git add -A`）。
**分支 A 独立复核通过**：实例确已停止计费、产物完整无缺、主表可复现、结论文档已交付。
**本轮已把本 automation 置为 PAUSED**（与已完成的「第4阶段收尾监控」一致），避免无意义的空转唤醒。

### 本轮独立复核（全部现测，不依赖上轮结论）

| 项 | 手段 | 结果 |
|---|---|---|
| 实例已停 | relay `/health` 的 `agent_seen` 时间戳 vs 当前时间 | last_seen=2026-11-24 **08:29:05**，lag **16427 s ≈ 4.56 h**（>120 s 阈值 ⇒ 失联） |
| 实例已停 | 新下发一条**只读**探针（`echo`+`date`+`nvidia-smi`）到**全新独占段 1020000~1020999**，等 150 s | **NO_RESPONSE_AFTER_150s** ⇒ relay agent 已不执行任何命令 ⇒ 容器确已退出 |
| 产物完整 | 本地 35 文件重算 sha256 vs `MANIFEST.json` | **OK 35 / mismatch 0 / missing 0** |
| 主表可复现 | 重跑 `python tools/phase5_table.py results/phase5/phase5_raw.json --phase3 results/phase3_raw.json` | 与既有 `phase5_table.md` **逐字节相同**（diff 无输出） |
| MoE 完整性 | 31 份 eval 日志 grep `number of params \(M\):` | 取值集合 = **{22.38}** ⇒ 30/30 格全部 ≥21.0，**无 13.94M 残缺** |
| NaN/Inf | 31 份日志精确匹配 `\b(NaN|nan)\b` / `\binf\b` | **0 命中** |
| 阶段完整 | `phase5_raw.json` | 10 stage × 3 domain × 3 seeds = **90 个 seed 值 / 30 格**齐全 |
| 收工原因 | `results/phase5/autoshutdown.log` | `[2026-11-24 08:28:37] 收工原因：全部实验完成（主实验矩阵 7/7）` → extract rc=0 → 打包 52K → 推进 relay |

### 🔴 坑 23（本轮新踩，已修）：`/health` 的 `agent_alive: true` **仍是陈旧缓存**
第九轮 `/health` 仍返回 `agent_alive: true, agent_pid: 1841`，但 `agent_seen` 已 **4.56 h** 未更新。
⇒ 再次确认坑 22：**判活必须看 `agent_seen` 滞后，绝不信 `agent_alive` 布尔值**。

### 🔴 坑 24：判「实例已停」的**最强单条证据 = 探针无响应**，不是 `/health`
`/health` 的 `agent_alive` 会骗人（坑 23）。真正确证的方法：
下发一条**无害只读**命令 → 等 120~150 s → **无任何 result 落地** ⇒ relay agent 已不执行 ⇒ 容器已退出。
（本轮实测 150 s 无响应，判据成立。）
⚠️ 该探针**必须**用**全新独占 id 段**（本轮用 1020000~1020999，避开并发会话 1005000~1009999、
守护产物段 1010000、历史探查段 1002xxx/1004xxx/1008xxx）⇒ 避免 id 被抢占读回别人的结果（坑 20）。
⚠️ 该探针**不得** `clearall`、不得写文件、不得改码 —— 纯只读。

### 本轮新增/更新的文件
- `tools/tmp/p5probe_r90.py`（新增：只读存活判定探针，独占段 1020000~1020999，150 s 超时判定）
- **未新建**结论文档（`results/第5阶段结果与结论.md` 已于第八轮前交付，09:11 mtime，本轮复核后仍有效）

### 费用
实例 2026-11-23 11:03 ~ 2026-11-24 08:28 ≈ **21.4 h**，RTX 3090 ≈2.2 元/h ⇒ **≈47 元（估算，以 AutoDL 账单为准）**；
**关机后 0 元**，本轮确认无空转烧钱。

### 若后续再被触发（本 automation 已 PAUSED，正常不会再触发）
只需做两件事即可收工：
1. `python tools/relay_status_probe.py` 看 `agent_seen` 滞后 > 120 s ⇒ 实例仍停；
2. `ls results/phase5/` 确认 39 个文件在、`results/第5阶段结果与结论.md` 在。
**绝对不要**在已关机的实例上做任何动作；新的 P0 实验需用户批准后另起任务。
