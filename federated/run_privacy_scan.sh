#!/usr/bin/env bash
# FedVLA 第 4 阶段隐私侧：DP-SGD 上行加噪 + 梯度反演攻击（路线图 5.2）。
#
# 与通信侧（run_comm_scan.sh）的关系：
#   通信侧回答"链路损伤下成功率掉多少"，隐私侧回答"上行更新泄露多少、DP 能挡多少"。
#   两者共用第 3 阶段的最终方法模型（异构 + MoE + 簇感知 = P3-C），
#   因为论文第 4 章要回答的是"**我们的方法**在链路受损与隐私威胁下还站不站得住"。
#
# 关键口径（2026-11-08 全部实测确认，不是推测）：
#   · **反演目标是 ResNet 特征图，不是像素**：zarr 的 image 是 (N,1,49,512) 的
#     512 维 ResNet 特征（取值 [0,15.77]）⇒ 潜变量不用 sigmoid，指标用逐 patch 相关。
#   · **梯度匹配取交集**：MoE 的 top-k 路由随输入变化 ⇒ 候选梯度键 ⊂ 服务器收到的
#     梯度键（实测 15.82M vs 17.13M）⇒ 只能交集匹配；默认只匹配稠密 blocks 0-7。
#   · **语言通路不存在**：实测 policy.modalities=['image','state']，
#     loss(有语言)−loss(无语言)=0 ⇒ 模态敏感度覆盖 视觉/状态/动作 三通路。
#   · **DP 作用在"上行"上**：上行 = trunk + normalizer + modalities_tokens；
#     stems/heads 是个性化参数，永不上行。
#   · MoE 环境变量必须逐字复刻训练侧（否则是残缺模型，参数量自检会拦下）。
#
# 幂等：privacy_experiment.py 按 (样本, 模式, ε, 目标) 增量续跑；
#       结果文件存在且 PRIV_FORCE != 1 时跳过整个扫描。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
HPT_PY=/root/miniconda3/envs/hpt/bin/python

PRIV_DIR=${PRIV_DIR:-$OUT/privacy}
# 被攻击的客户端：c2 = button-press-topdown-v2（第 3 阶段唯一有信号的域）
PRETRAINED_DIR=${PRETRAINED_DIR:-$OUT/p3_moe_eda_cluster/global_r49}
# ⚠️ 2026-11-08 修正：原值 /root/HPT/data/metaworld_1 **不存在**。
#    实际的数据是 zarr（ResNet 特征）目录，privacy_experiment.py 直接读它，
#    不经过 LocalTrajDataset（后者在无 env 渲染时会退化成每次重渲染 16 分钟）。
DATASET_ROOT=${DATASET_ROOT:-/root/HPT/data/zarr_mujoco_metaworld_task3_resnet_traj100}
DOMAIN=${DOMAIN:-mujoco_metaworld_task3}

EPSILONS=${EPSILONS:-}
# ⚠️ 自变量是**噪声倍率 σ**而不是 ε（实测理由见 privacy_experiment.py 4.4 节）：
#    trunk 有 17.1M 参数 ⇒ 客户端级 DP 的噪声总范数 = σ·C·√17.1M = σ·4139。
#    σ=1（ε≈1）时噪声范数 4139 ≫ 信号范数 1 ⇒ ε∈[1,32] 全落在噪声地板上，
#    "ε 越大越容易反演"的规律根本看不出来。真正的转折在 σ≈1e-4（ε≈1e5~1e7）。
SIGMAS=${SIGMAS:-1e-4,1e-3,1e-2,1e-1,1}
MODES=${MODES:-uniform,modality}
TARGETS=${TARGETS:-image,action,state}
SAMPLES=${SAMPLES:-0,1,2}
DELTA=${DELTA:-1e-5}
# RDP 复合轮数 = 主实验的联邦轮数（第3阶段 P3-C = 50 轮）
DP_STEPS=${DP_STEPS:-50}
CLIENTS=${CLIENTS:-4}
CLIP_NORM=${CLIP_NORM:-1.0}
INV_STEPS=${INV_STEPS:-200}
INV_LR=${INV_LR:-0.05}
MATCH_MODE=${MATCH_MODE:-dense}
SEED=${SEED:-1234}

log () { echo "[$(date '+%F %T')] $*"; }

# ===== MoE 环境变量（训练与评估必须逐字一致）=====
# ⚠️ 铁律：patch_hpt_moe.py 只在 FEDVLA_MOE=1 时构造 MoE 结构，
#    Policy.load_model 用 strict=False → 漏设会**静默丢弃 264 个专家权重**。
#    实测：完整 MoE = 22.380M，残缺 = 13.94M。privacy_experiment.py 内有参数量自检。
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half
export MUJOCO_GL=egl

mkdir -p "$PRIV_DIR"

if [ -f "$PRIV_DIR/privacy_results.json" ] && [ "${PRIV_FORCE:-0}" != "1" ]; then
  log "隐私侧结果已存在（$PRIV_DIR/privacy_results.json），跳过。设 PRIV_FORCE=1 可强制重跑"
  exit 0
fi

log "===== FedVLA 第4阶段：隐私侧（DP-SGD 上行加噪 + 梯度反演）====="
log "模型=$PRETRAINED_DIR"
log "数据=$DATASET_ROOT"
log "σ 序列=$SIGMAS  模式=$MODES  目标=$TARGETS  样本=$SAMPLES"
log "δ=$DELTA  联邦轮数=$DP_STEPS  客户端=$CLIENTS  C=$CLIP_NORM"
log "反演: steps=$INV_STEPS lr=$INV_LR match=$MATCH_MODE"

if [ ! -d "$PRETRAINED_DIR" ]; then
  log "[!!] 模型目录不存在：$PRETRAINED_DIR"
  log "    第3阶段的 global_r49 若已被清理，请改用 PRETRAINED_DIR 指定其他轮次"
  exit 1
fi
if [ ! -d "$DATASET_ROOT" ]; then
  log "[!!] 数据目录不存在：$DATASET_ROOT"
  ls -d /root/HPT/data/* 2>/dev/null | sed 's/^/    候选: /'
  exit 1
fi

"$HPT_PY" "$SCRIPT_DIR/privacy_experiment.py" \
  --model-dir "$PRETRAINED_DIR" \
  --zarr "$DATASET_ROOT" \
  --domain "$DOMAIN" \
  --out "$PRIV_DIR" \
  --epsilons "$EPSILONS" \
  --sigmas "$SIGMAS" \
  --modes "$MODES" \
  --targets "$TARGETS" \
  --samples "$SAMPLES" \
  --delta "$DELTA" \
  --dp-steps "$DP_STEPS" \
  --clients "$CLIENTS" \
  --clip-norm "$CLIP_NORM" \
  --inv-steps "$INV_STEPS" \
  --inv-lr "$INV_LR" \
  --match-mode "$MATCH_MODE" \
  --seed "$SEED" \
  2>&1 | tee "$PRIV_DIR/privacy_run.log"

rc=${PIPESTATUS[0]}
if [ "$rc" = "0" ]; then
  touch "$PRIV_DIR/PRIVACY_DONE"
  log "隐私侧完成，PRIVACY_DONE"
else
  log "[!!] 隐私侧失败 rc=$rc —— 查看 $PRIV_DIR/privacy_run.log"
fi
exit "$rc"
