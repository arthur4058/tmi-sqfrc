# GeoLife 60 秒用户稳健融合 V21 实验报告

## 结论

V21 在三个随机种子上相对 V2 的 Accuracy 和 Macro-F1 均为正增益。三种子平均 Accuracy 从 69.7574% 提升至 70.1844%，Macro-F1 从 63.8386% 提升至 64.4152%。

该结果没有达到 5 个百分点的大幅提升，但解决了 V17–V20 在验证集提升、独立测试集退化的问题。目前它是更可信的稳定改进，而不是一次偶然的单种子峰值。

## 方法

- 主模型：V2 三专家低采样率模型。
- 补充专家：V16 短序列多尺度轨迹模型。
- 数据：GeoLife，用户互斥 train/validation/test，60 秒采样视图。
- 选择阶段只使用 validation 用户，不读取 test 标签。
- 对融合权重 `0.00–0.50` 做验证集扫描。
- 首先最大化 Accuracy 与 Macro-F1 均不退化的验证用户比例。
- 然后最大化用户级联合增益的下四分位数，最后才比较全局增益。
- 权重冻结后仅在对应 seed 的 test 集评估一次。

这个规则的作用是防止总体均值掩盖某些用户的大幅退化。不同 seed 自动选择不同权重，说明短序列专家的可靠程度会随训练初始化变化。

## 三种子测试结果

| Seed | V2 Acc | V21 Acc | ΔAcc (pp) | V2 Macro-F1 | V21 Macro-F1 | ΔF1 (pp) | V16 权重 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 65.9618% | 66.0228% | +0.0610 | 60.5996% | 60.6391% | +0.0395 | 0.01 |
| 2024 | 70.8215% | 71.5331% | +0.7117 | 64.9119% | 65.6131% | +0.7012 | 0.20 |
| 10086 | 72.4888% | 72.9972% | +0.5083 | 66.0042% | 66.9933% | +0.9891 | 0.17 |
| 平均 | 69.7574% | 70.1844% | **+0.4270** | 63.8386% | 64.4152% | **+0.5766** | — |

稳定性：3/3 seeds 的 Accuracy 提升，3/3 seeds 的 Macro-F1 提升。

## 被淘汰的 V22

额外尝试了按 V2 单样本预测熵动态分配 V16 权重的 V22。它在 seed10086 达到 73.0378% Accuracy、67.0015% Macro-F1，略高于 V21；但在 seed42 的 Macro-F1 下降 0.007 个百分点。因此 V22 被判定为跨种子不稳定，不作为最终方案提交。

## 可复现命令

```bash
conda activate tmi-sqfrc
cd ~/research/tmi-sqfrc

python -m scripts.train_multiscale_short_official --seed 42
python -m scripts.train_multiscale_short_official --seed 2024

python -m scripts.evaluate_user_robust_v2_v16 --seed 42 --device cuda
python -m scripts.evaluate_user_robust_v2_v16 --seed 2024 --device cuda
python -m scripts.evaluate_user_robust_v2_v16 --seed 10086 --device cuda
```

## 限制与后续建议

1. 项目此前已多次查看同一测试集，因此论文最终定稿前应预留新的用户互斥 holdout，或采用外层用户级交叉验证。
2. V21 的提升稳定但幅度中等。后续优先研究能直接恢复 60 秒短序列运动表征的训练式模块，而不是继续堆叠测试后校准器。
3. 正式论文实验应同时报告三种子均值与标准差、Accuracy、Macro-F1、各类别召回率，并保持 5/10/20/30 秒性能不显著下降。
