echo "=== global model.pth 里所有 normalizer 键 ==="
/root/miniconda3/envs/hpt/bin/python - <<'PYEOF'
import torch, collections
p="/root/fedcore/keep/c_int4_global_r49/model.pth"
o=torch.load(p,map_location="cpu",weights_only=False)
if isinstance(o,dict) and "model" in o and isinstance(o["model"],dict): o=o["model"]
ks=[k for k in o if isinstance(k,str) and "normalizer" in k]
doms=collections.Counter(k.split(".")[1] for k in ks if k.count(".")>1)
print("normalizer 张量数:",len(ks))
print("按域:",dict(doms))
for k in ks[:6]: print("   ",k, tuple(o[k].shape) if torch.is_tensor(o[k]) else type(o[k]))
print("顶层键样本:", [k for k in list(o)[:5]])
PYEOF
echo "=== HPT 里 normalizer 的来源 ==="
grep -rn "normalizer" /root/HPT/hpt/policies/transformer_policy.py 2>/dev/null | head -25
echo "=== run_eval 怎么建 policy ==="
sed -n '1,80p' /root/HPT/hpt/run_eval.py 2>/dev/null
