# 第 4 阶段隐私侧结果表（DP-SGD 上行加噪 + 梯度反演）

## A. 逐通路敏感度（消融法：把该通路换成另一条轨迹的输入）

| 样本 | 视觉 D_m | 状态 D_m | 动作 D_m | 视觉 1−cos | 状态 1−cos | 动作 1−cos |
|---|---|---|---|---|---|---|
| 0 | 3.8164 | 1.7595 | 2.2393 | 0.7599 | 0.1356 | 1.3581 |
| 1 | 7.8518 | 3.5542 | 1.7943 | 0.7801 | 0.8143 | 1.0143 |
| 2 | 5.4342 | 5.3721 | 1.2947 | 0.8822 | 0.9340 | 0.6759 |

## B. 重建质量 vs 噪声倍率 σ（均值，样本数见 n）

| 目标 | 方案 | σ_ref | 总 ε | 指标 | 均值 | std | 相对L2 | 学习信号保真度(shared) | n |
|---|---|---|---|---|---|---|---|---|---|
| Action (label) | modality | 0.0001 | 5e+09 | cos | 0.9722 | 0.0010 | 0.2428 | 0.8630 | 3 |
| Action (label) | modality_rev | 0.0001 | 5e+09 | cos | 0.9739 | 0.0029 | 0.2376 | 0.7339 | 3 |
| Action (label) | uniform | 0.0001 | 5e+09 | cos | 0.9709 | 0.0018 | 0.2501 | 0.9240 | 3 |
| Action (label) | modality | 0.001 | 5e+07 | cos | 0.9750 | 0.0039 | 0.2284 | 0.1686 | 3 |
| Action (label) | modality_rev | 0.001 | 5e+07 | cos | 0.9725 | 0.0047 | 0.2378 | 0.1076 | 3 |
| Action (label) | uniform | 0.001 | 5e+07 | cos | 0.9742 | 0.0036 | 0.2349 | 0.2350 | 3 |
| Action (label) | modality | 0.01 | 4.999e+05 | cos | 0.9455 | 0.0130 | 0.3301 | 0.0173 | 3 |
| Action (label) | modality_rev | 0.01 | 4.999e+05 | cos | 0.9425 | 0.0119 | 0.3397 | 0.0110 | 3 |
| Action (label) | uniform | 0.01 | 4.999e+05 | cos | 0.9562 | 0.0176 | 0.2943 | 0.0244 | 3 |
| Action (label) | modality | 0.1 | 4873 | cos | 0.9368 | 0.0107 | 0.3547 | 0.0019 | 3 |
| Action (label) | modality_rev | 0.1 | 4873 | cos | 0.9359 | 0.0107 | 0.3570 | 0.0013 | 3 |
| Action (label) | uniform | 0.1 | 4873 | cos | 0.9381 | 0.0116 | 0.3514 | 0.0026 | 3 |
| Action (label) | modality | 1 | 16.55 | cos | 0.9342 | 0.0108 | 0.3610 | 0.0004 | 3 |
| Action (label) | modality_rev | 1 | 16.55 | cos | 0.9341 | 0.0111 | 0.3610 | 0.0003 | 3 |
| Action (label) | uniform | 1 | 16.55 | cos | 0.9341 | 0.0112 | 0.3610 | 0.0005 | 3 |
| Action (label) | clean | clean | — | cos | 0.9722 | 0.0019 | 0.2436 | — | 3 |
| Vision (input) | modality | 0.0001 | 5e+09 | corr | 0.0250 | 0.0028 | 1.3169 | 0.8630 | 3 |
| Vision (input) | modality_rev | 0.0001 | 5e+09 | corr | 0.0260 | 0.0012 | 1.3204 | 0.7339 | 3 |
| Vision (input) | uniform | 0.0001 | 5e+09 | corr | 0.0244 | 0.0029 | 1.3172 | 0.9240 | 3 |
| Vision (input) | modality | 0.001 | 5e+07 | corr | 0.0247 | 0.0022 | 1.3168 | 0.1686 | 3 |
| Vision (input) | modality_rev | 0.001 | 5e+07 | corr | 0.0260 | 0.0026 | 1.3171 | 0.1076 | 3 |
| Vision (input) | uniform | 0.001 | 5e+07 | corr | 0.0257 | 0.0009 | 1.3199 | 0.2350 | 3 |
| Vision (input) | modality | 0.01 | 4.999e+05 | corr | 0.0242 | 0.0030 | 1.3202 | 0.0173 | 3 |
| Vision (input) | modality_rev | 0.01 | 4.999e+05 | corr | 0.0227 | 0.0020 | 1.3234 | 0.0110 | 3 |
| Vision (input) | uniform | 0.01 | 4.999e+05 | corr | 0.0236 | 0.0031 | 1.3199 | 0.0244 | 3 |
| Vision (input) | modality | 0.1 | 4873 | corr | 0.0093 | 0.0012 | 1.3596 | 0.0019 | 3 |
| Vision (input) | modality_rev | 0.1 | 4873 | corr | 0.0060 | 0.0022 | 1.3778 | 0.0013 | 3 |
| Vision (input) | uniform | 0.1 | 4873 | corr | 0.0086 | 0.0031 | 1.3555 | 0.0026 | 3 |
| Vision (input) | modality | 1 | 16.55 | corr | -0.0004 | 0.0017 | 1.3738 | 0.0004 | 3 |
| Vision (input) | modality_rev | 1 | 16.55 | corr | -0.0014 | 0.0026 | 1.3773 | 0.0003 | 3 |
| Vision (input) | uniform | 1 | 16.55 | corr | 0.0019 | 0.0067 | 1.3776 | 0.0005 | 3 |
| Vision (input) | clean | clean | — | corr | 0.0258 | 0.0008 | 1.3204 | — | 3 |
| Proprioception (input) | modality | 0.0001 | 5e+09 | cos | 0.3170 | 0.0318 | 2.5429 | 0.8630 | 3 |
| Proprioception (input) | modality_rev | 0.0001 | 5e+09 | cos | 0.2811 | 0.1193 | 2.6222 | 0.7339 | 3 |
| Proprioception (input) | uniform | 0.0001 | 5e+09 | cos | 0.3312 | 0.0551 | 2.6278 | 0.9240 | 3 |
| Proprioception (input) | modality | 0.001 | 5e+07 | cos | 0.1927 | 0.1061 | 3.1249 | 0.1686 | 3 |
| Proprioception (input) | modality_rev | 0.001 | 5e+07 | cos | 0.2756 | 0.0528 | 2.7675 | 0.1076 | 3 |
| Proprioception (input) | uniform | 0.001 | 5e+07 | cos | 0.2544 | 0.0542 | 2.7151 | 0.2350 | 3 |
| Proprioception (input) | modality | 0.01 | 4.999e+05 | cos | 0.3481 | 0.1069 | 1.7842 | 0.0173 | 3 |
| Proprioception (input) | modality_rev | 0.01 | 4.999e+05 | cos | 0.2710 | 0.0236 | 2.0406 | 0.0110 | 3 |
| Proprioception (input) | uniform | 0.01 | 4.999e+05 | cos | 0.4443 | 0.0997 | 1.5704 | 0.0244 | 3 |
| Proprioception (input) | modality | 0.1 | 4873 | cos | -0.0621 | 0.0396 | 2.8069 | 0.0019 | 3 |
| Proprioception (input) | modality_rev | 0.1 | 4873 | cos | 0.0018 | 0.0607 | 2.5678 | 0.0013 | 3 |
| Proprioception (input) | uniform | 0.1 | 4873 | cos | 0.0787 | 0.0562 | 2.4306 | 0.0026 | 3 |
| Proprioception (input) | modality | 1 | 16.55 | cos | -0.0009 | 0.0697 | 2.8952 | 0.0004 | 3 |
| Proprioception (input) | modality_rev | 1 | 16.55 | cos | -0.0470 | 0.0216 | 2.9418 | 0.0003 | 3 |
| Proprioception (input) | uniform | 1 | 16.55 | cos | -0.0138 | 0.0490 | 2.7153 | 0.0005 | 3 |
| Proprioception (input) | clean | clean | — | cos | 0.3014 | 0.0821 | 2.6778 | — | 3 |
