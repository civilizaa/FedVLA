"""FedVLA：top-k MoE 前馈块（并联残差式），配合 HPT 的 BlockWithMasking 使用。

═══════════════════════════════════════════════════════════════════════════════
为什么不用 FedVLA 原文的双门控
═══════════════════════════════════════════════════════════════════════════════
原文的双门控有两个参数 `W_e`（专家自感知）与 `lam`（缩放因子），
初始化尺度不对时会出现「要么全激活、要么几乎不激活」的二值化崩塌，
是整个方案里最容易训崩的环节（已在 fedvla_scaffold 的 toy 实验里观察到）。
所以本实现只用**最标准的 top-k 路由**：

    logits  = W_g · x                      (N, E)
    topk    = TopK(logits, k)              (N, k)
    weights = softmax(topk_values)         (N, k)   —— 在被选中的 k 个里归一化
    y       = Σ_e weights_e · Expert_e(x)

═══════════════════════════════════════════════════════════════════════════════
关键设计：稠密 FFN + 并联 MoE 残差（而不是用 MoE 替换稠密 FFN）
═══════════════════════════════════════════════════════════════════════════════
    输出(x) = Mlp_dense(x) + MoE(x)

理由（三条，都很重要）：

1) **训练起点严格等于预训练模型**
   MoE 分支的输出投影（专家的 fc2）零初始化 → MoE(x) ≡ 0 →
   前向函数与原始 HPT 的 Mlp **逐位相同**。
   这样 MoE 版本与稠密基线的比较完全公平，不会因为"随机初始化的新模块"
   而在起点上吃亏。（本地单测已数值验证：max err < 1e-6）

2) **预训练权重可以原样载入，不需要任何形状变换**
   稠密部分的键名仍然是 `fc1/fc2`，与 HPT 完全一致。
   （早期曾试过"用 MoE 替换 FFN + 把稠密权重复制到各专家"的方案，
   但专家的 hidden dim = ratio × 稠密 hidden dim，形状对不上，权重无法复制；
   若强行把 ratio 设为 1.0，8 个专家会让 trunk 参数量涨 8 倍，
   直接毁掉第 5 章的通信论证。并联残差避开了这两个坑。）

3) **参数与算力开销可控**
   每个专家的 hidden dim = expert_ratio × 稠密 FFN 的 hidden dim。
   默认 E=8, k=2, ratio=0.25：
     · 参数量 = 8 × 0.25 = 2× 稠密 FFN（trunk 整体约 +17%）
     · 每 token 额外算力 = 2/8 × 0.25 = 0.5× 稠密 FFN（+50% FLOPs）

═══════════════════════════════════════════════════════════════════════════════
专家使用计数
═══════════════════════════════════════════════════════════════════════════════
每个 MoE 层维护一个 `expert_counts` 缓冲区（非 persistent，不进 state_dict），
统计每个专家被多少 token 路由到。训练结束后由 run.py 导出到 `expert_counts.pt`，
供 EDA（专家驱动聚合）计算客户端相似度与聚合权重。
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


class _ExpertMlp(nn.Module):
    """单个专家：Linear -> act -> dropout -> Linear -> dropout（与 HPT 的 Mlp 同构）"""

    def __init__(self, in_features: int, hidden_features: int,
                 out_features: Optional[int] = None,
                 act_layer: Callable = nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TopKMoE(nn.Module):
    """标准 top-k MoE 前馈分支。"""

    def __init__(self, in_features: int, hidden_features: int,
                 out_features: Optional[int] = None,
                 num_experts: int = 8, top_k: int = 2,
                 act_layer: Callable = nn.GELU, drop: float = 0.0,
                 gate_init_std: float = 0.02, zero_output_init: bool = False):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k={top_k} 必须落在 [1, {num_experts}]")
        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(in_features, num_experts, bias=False)
        # 小随机初始化：让路由一开始就对称性破缺，但不能太大
        nn.init.normal_(self.gate.weight, std=gate_init_std)
        self.experts = nn.ModuleList([
            _ExpertMlp(in_features, hidden_features, out_features, act_layer, drop)
            for _ in range(num_experts)
        ])

        # 非 persistent 缓冲区：统计专家使用次数，不进 state_dict
        self.register_buffer("expert_counts",
                             torch.zeros(num_experts, dtype=torch.float32),
                             persistent=False)
        self.counting = True

        if zero_output_init:
            zero_moe_output_projections(self)

    def reset_counts(self):
        self.expert_counts.zero_()

    def get_counts(self) -> torch.Tensor:
        return self.expert_counts.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xf = x.reshape(-1, shape[-1])

        logits = self.gate(xf)
        topk_val, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_val = torch.softmax(topk_val, dim=-1)

        out = torch.zeros_like(xf)
        for e in range(self.num_experts):
            hit_mask = (topk_idx == e)
            hit = hit_mask.any(dim=-1)
            if not bool(hit.any()):
                continue
            rows = hit.nonzero(as_tuple=True)[0]
            y = self.experts[e](xf[rows])
            w = (topk_val * hit_mask).sum(dim=-1)[rows].unsqueeze(-1)
            out = out.index_add(0, rows, y * w)

        if self.counting and self.training:
            with torch.no_grad():
                cnt = torch.bincount(topk_idx.reshape(-1),
                                     minlength=self.num_experts).float()
                self.expert_counts += cnt

        return out.reshape(shape)


class MoEBlock(nn.Module):
    """稠密 FFN + 并联 top-k MoE 残差分支。

    稠密部分的子模块名与 HPT 的 Mlp 完全一致（fc1 / act / fc2 / drop），
    因此预训练权重可以**原样载入**，不需要任何键名或形状变换。
    """

    def __init__(self, in_features: int, hidden_features: int,
                 out_features: Optional[int] = None,
                 num_experts: int = 8, top_k: int = 2,
                 act_layer: Callable = nn.GELU, drop: float = 0.0,
                 expert_ratio: float = 0.25, gate_init_std: float = 0.02):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.moe = TopKMoE(
            in_features=in_features,
            hidden_features=max(1, int(hidden_features * expert_ratio)),
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            act_layer=act_layer,
            drop=drop,
            gate_init_std=gate_init_std,
            zero_output_init=True,     # 训练起点严格等于稠密 FFN
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return y + self.moe(x)


# ── 工具函数 ──────────────────────────────────────────────────────────────
def has_moe(module: nn.Module) -> bool:
    """模块树里是否存在 TopKMoE"""
    return any(isinstance(m, TopKMoE) for m in module.modules())


def zero_moe_output_projections(module: nn.Module) -> int:
    """把所有 MoE 分支的输出投影（专家的 fc2）置零。

    必须在模型权重初始化**之后**调用一次：
    HPT 的 SimpleTransformer 在构造末尾会 `self.apply(self._init_weights)`，
    那一步会把构造时的零初始化覆盖掉。
    """
    n = 0
    for m in module.modules():
        if isinstance(m, TopKMoE):
            for e in m.experts:
                nn.init.zeros_(e.fc2.weight)
                if e.fc2.bias is not None:
                    nn.init.zeros_(e.fc2.bias)
            n += 1
    return n


def moe_param_keys(module: nn.Module) -> set:
    """模块里所有 MoE 相关的 state_dict 键（这些键在载入预训练权重时是"允许缺失"的）。"""
    keys = set()
    for name, mod in module.named_modules():
        if isinstance(mod, TopKMoE):
            prefix = f"{name}." if name else ""
            for k in mod.state_dict().keys():
                keys.add(prefix + k)
    return keys


def collect_expert_counts(module: nn.Module) -> dict:
    """遍历模块，收集所有 MoE 层的专家使用计数。"""
    out = {}
    for name, mod in module.named_modules():
        if isinstance(mod, TopKMoE):
            out[name] = mod.get_counts()
    return out


def stack_expert_counts(counts: dict) -> torch.Tensor:
    """把 {层路径: tensor(E)} 叠成 (L, E) 矩阵（按层路径排序）。"""
    if not counts:
        return torch.zeros(0, 0)
    keys = sorted(counts.keys())
    return torch.stack([counts[k].detach().cpu().float() for k in keys], dim=0)
