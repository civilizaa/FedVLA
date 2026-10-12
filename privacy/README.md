# privacy/ — 隐私模块（待建）

第 13–16 周与 comms 并行。对应 H3（模态感知 DP 预算优于统一预算）：

- `dp_sgd.py`：按模态（视觉/语言/动作）分配不同 DP 噪声预算。
- `feature_inversion.py`：从 Split Learning 的切分激活反演输入图像，量化泄露。
- `membership.py`：成员推断攻击基线（可选）。

注意：DP 与 EDA 聚合可能相互作用，需在实验中单独消融。
