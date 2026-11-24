# 第 5 阶段主实验矩阵（成功率，均值 ± 95% CI，t df=n−1=2 ⇒ t_crit=4.303）

统计口径：3 seeds × 20 rollouts/seed；CI 与 p 均由本脚本独立复算（不采信上游 stats）。

## 参考上下界（第 2 阶段门2实验，不重跑）

| 配置 | 说明 | reach-v2 (task1) | push-v2 (task2) | button-press (task3) |
|---|---|---|---|---|
| 集中式等预算 | 性能上界（无联邦） | 0.933 ± 0.190 | — | — |
| 异构联邦基线（无 MoE） | 联邦下界 | 0.017 ± 0.072 | 0.100 ± 0.248 | 0.450 ± 0.215 |

## 主表（第 5 阶段主实验矩阵）

| 配置 | 划分 | 来源 | reach-v2 (task1) | push-v2 (task2) | button-press (task3) | 三任务均值 | MoE 参数量(M) |
|---|---|---|---|---|---|---|---|
| 均衡划分 + FedAvg | balanced | 第3阶段复用 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.283 ± 0.587 | 0.094 | 22.38 |
| 均衡划分 + EDA 中心性 | balanced | 第3阶段复用 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.250 ± 0.569 | 0.083 | 22.38 |
| 均衡划分 + 簇感知（本文） | balanced | 第3阶段复用 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.667 ± 0.517 | 0.222 | 22.38 |
| 均衡划分 + FedProx | balanced | 第5阶段新跑 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.350 ± 0.497 | 0.117 | 22.38 |
| 均衡划分 + SCAFFOLD | balanced | 第5阶段新跑 | 0.133 ± 0.190 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.044 | 22.38 |
| 孤立客户端 + FedAvg | isolated | 第5阶段新跑 | 0.600 ± 0.430 | 0.000 ± 0.000 | 0.783 ± 0.287 | 0.461 | 22.38 |
| 孤立客户端 + FedProx | isolated | 第5阶段新跑 | 0.567 ± 0.190 | 0.000 ± 0.000 | 0.683 ± 0.190 | 0.417 | 22.38 |
| 孤立客户端 + SCAFFOLD | isolated | 第5阶段新跑 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 | 22.38 |
| 孤立客户端 + EDA 中心性 | isolated | 第5阶段新跑 | 0.617 ± 0.313 | 0.000 ± 0.000 | 0.750 ± 0.248 | 0.456 | 22.38 |
| 孤立客户端 + 簇感知（本文） | isolated | 第5阶段新跑 | 0.550 ± 0.215 | 0.000 ± 0.000 | 0.300 ± 0.329 | 0.283 | 22.38 |

## 配对 t 检验（同 seed 配对，df = n−1 = 2）

| 比较 | domain | Δ均值 | t | p(双尾) | 结论 |
|---|---|---|---|---|---|
| H1 均衡臂 EDA(中心性) vs FedAvg | reach-v2 (task1) | +0.000 | 0.000 | 1.000 | 无差异 |
| H1 均衡臂 EDA(中心性) vs FedAvg | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| H1 均衡臂 EDA(中心性) vs FedAvg | button-press (task3) | -0.033 | -2.000 | 0.184 | 不显著 |
| H2 均衡臂 簇感知 vs FedAvg | reach-v2 (task1) | +0.000 | 0.000 | 1.000 | 无差异 |
| H2 均衡臂 簇感知 vs FedAvg | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| H2 均衡臂 簇感知 vs FedAvg | button-press (task3) | +0.383 | 23.000 | 0.002 | **显著** |
| H2 均衡臂 簇感知 vs 中心性 EDA | reach-v2 (task1) | +0.000 | 0.000 | 1.000 | 无差异 |
| H2 均衡臂 簇感知 vs 中心性 EDA | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| H2 均衡臂 簇感知 vs 中心性 EDA | button-press (task3) | +0.417 | 25.000 | 0.002 | **显著** |
| H2 均衡臂 簇感知 vs 异构基线(无MoE) | reach-v2 (task1) | -0.017 | -1.000 | 0.423 | 不显著 |
| H2 均衡臂 簇感知 vs 异构基线(无MoE) | push-v2 (task2) | -0.100 | -1.732 | 0.225 | 不显著 |
| H2 均衡臂 簇感知 vs 异构基线(无MoE) | button-press (task3) | +0.217 | 1.857 | 0.204 | 不显著 |
| 均衡臂 FedProx vs FedAvg | reach-v2 (task1) | +0.000 | 0.000 | 1.000 | 无差异 |
| 均衡臂 FedProx vs FedAvg | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| 均衡臂 FedProx vs FedAvg | button-press (task3) | +0.067 | 1.512 | 0.270 | 不显著 |
| 均衡臂 SCAFFOLD vs FedAvg | reach-v2 (task1) | +0.133 | 3.024 | 0.094 | 不显著 |
| 均衡臂 SCAFFOLD vs FedAvg | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| 均衡臂 SCAFFOLD vs FedAvg | button-press (task3) | -0.283 | -2.077 | 0.173 | 不显著 |
| 均衡臂 SCAFFOLD vs FedProx | reach-v2 (task1) | +0.133 | 3.024 | 0.094 | 不显著 |
| 均衡臂 SCAFFOLD vs FedProx | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| 均衡臂 SCAFFOLD vs FedProx | button-press (task3) | -0.350 | -3.031 | 0.094 | 不显著 |
| 孤立臂 EDA(中心性) vs FedAvg | reach-v2 (task1) | +0.017 | 0.378 | 0.742 | 不显著 |
| 孤立臂 EDA(中心性) vs FedAvg | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| 孤立臂 EDA(中心性) vs FedAvg | button-press (task3) | -0.033 | -1.000 | 0.423 | 不显著 |
| H2 孤立臂 簇感知 vs FedAvg | reach-v2 (task1) | -0.050 | -1.000 | 0.423 | 不显著 |
| H2 孤立臂 簇感知 vs FedAvg | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| H2 孤立臂 簇感知 vs FedAvg | button-press (task3) | -0.483 | -8.043 | 0.015 | **显著** |
| H2 孤立臂 簇感知 vs 中心性 EDA | reach-v2 (task1) | -0.067 | -2.000 | 0.184 | 不显著 |
| H2 孤立臂 簇感知 vs 中心性 EDA | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| H2 孤立臂 簇感知 vs 中心性 EDA | button-press (task3) | -0.450 | -15.588 | 0.004 | **显著** |
| 孤立臂 FedProx vs FedAvg | reach-v2 (task1) | -0.033 | -0.555 | 0.635 | 不显著 |
| 孤立臂 FedProx vs FedAvg | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| 孤立臂 FedProx vs FedAvg | button-press (task3) | -0.100 | -3.464 | 0.074 | 不显著 |
| 孤立臂 SCAFFOLD vs FedAvg | reach-v2 (task1) | -0.600 | -6.000 | 0.027 | **显著** |
| 孤立臂 SCAFFOLD vs FedAvg | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| 孤立臂 SCAFFOLD vs FedAvg | button-press (task3) | -0.783 | -11.750 | 0.007 | **显著** |
| 孤立臂 FedProx vs SCAFFOLD | reach-v2 (task1) | +0.567 | 12.851 | 0.006 | **显著** |
| 孤立臂 FedProx vs SCAFFOLD | push-v2 (task2) | +0.000 | 0.000 | 1.000 | 无差异 |
| 孤立臂 FedProx vs SCAFFOLD | button-press (task3) | +0.683 | 15.497 | 0.004 | **显著** |

> p 用 df=2 的闭式解 p = 1 − |t|/√(t²+2)（双尾，已用 F(1,2) 关系独立复核）；显著性门槛 p < 0.05。
