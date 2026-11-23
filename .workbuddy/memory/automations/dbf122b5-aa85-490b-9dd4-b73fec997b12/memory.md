# 第4阶段自动收尾任务 — 执行记录

## 2026-11-18 15:00 首次执行 → 未收尾（阶段未完成）

### 判定结果（用 /root/HPT/output/.rc_<tag> 判据）
- **阶段 A（通信侧）3/7**：已有 `.rc_c_drop00` `.rc_c_drop25` `.rc_c_drop50`；
  缺 `c_drop75` `c_int8` `c_int4` `c_bit1`。前 3 个配置评估表已 3/3（3 seeds）。
- **阶段 B（DP 效用）0/3**：`.rc_dp_s0001` `.rc_dp_s0003` `.rc_dp_s0012` 全缺；
  `/root/HPT/output/dp_calib/` 目录尚未生成 ⇒ C 标定都没跑。

### 结论
两个阶段都没完成 → **按纪律什么都不做**（未跑 extract、未打包、未拉文件、未填报告、未 commit）。

### 云端实况（15:01 采样）
- `chain_phase4_dp.sh` 存活（已运行 3h16m），正在等通信扫描 7/7；链日志最后进度 3/7 @ 13:57。
  ⇒ 通信扫描结束后会自动接力：calibrate_dp_clip.sh → run_dp_utility.sh → extract_dp_utility.py。
- 3 路并行训练中：c_drop75 r38/50、c_int8 r29/50、c_int4 r16/50；c_bit1 未启动（等空槽位）。
- GPU 99%（12856/24576 MiB），磁盘 69%（余 9.5G）。

### ETA（按 ~3.4 分钟/轮 3 路并行推算）
- 阶段 A 收尾 ≈ 19:00（c_bit1 是关键路径）；阶段 B 收尾 ≈ 22:00–23:30。

### 本地监控仍在跑
- `tools/monitor_comm.py`（11:17 启动，600s 间隔，全部 7 个 tables 齐了会打印 DONE 退出）
- `tools/monitor_phase4.py`（10:41 启动）
- 监控只写 tools/monitor_comm.log，不进对话上下文。

### 下次执行的正确做法
1. 先跑同一条状态命令拿齐 `.rc_*` 列表（不要按配置逐个调）。
2. 7 个 rc 全齐 → 才跑 extract_comm_results.py（读基线自检：c_drop00 必须 0.667±0.20）、
   打包 comm_raw.json + fig7_comm_*.png → pull_files.py 拉回 results/（本地路径用 C:/ 形式）→
   填 results/第4阶段结果与结论.md 第三节表格 + 第14周判据结论。
3. 3 个 dp rc 全齐 → 才跑 extract_dp_utility.py → 打包拉回 → 填 4.6 节表格，
   结论必须按 4.6 节已写好的「判定标准表」四种情形分别写，并同步第4阶段隐私侧结果与结论.md 局限第 1 条。
4. 填完再 `git add -A && git commit`（不 push）。
5. 本次未 commit（工作区本来就是干净的，git status 无改动）。
