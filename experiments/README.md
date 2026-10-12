# experiments/ — 实验矩阵与结果（待建）

第 17–20 周落地。

- `configs/`：12 个主配置 × seeds 的超参 yaml（对齐 `docs/planning/最低难度方案与路径.md` 的实验矩阵）。
- `eval.py`：统一评测入口，输出成功率均值 ± 95% CI（3–5 seeds × 20 rollouts）。
- `results/`：每次运行的 CSV/JSON（**每周五打包下载到本地**）。
- `figures/`：13 图 6 表的源数据（丢包扫描、压缩比、流量预算、EDA 权重、消融等）。

统计底线：**3 seeds × 20 rollouts**，报均值 ± 95% CI。算力充裕提到 5 seeds。
