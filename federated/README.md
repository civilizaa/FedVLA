# federated/ — 联邦训练主模块（待建）

第 5–12 周落地。计划内容：

- `server.py`：中央聚合循环（FedAvg / FedProx / FedAvgM / SCAFFOLD / EDA）。
- `client.py`：4 个串行客户端，本地训练 trunk（stem+head 个性化）。
- `moewrapper.py`：把 `fedvla_scaffold/fl_core.py` 的 DGMoE/EDA 接到 HPT 真实模型上。
- `run_round.py`：一轮通信的完整流程，统计上传/聚合耗时。

对应假设：H1 均衡划分下 EDA 退化为 FedAvg；H2 按簇聚合更优。
