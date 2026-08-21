# GeoLife V13 保守 Bus-Car 纠错 Dev-A 实验

## 结论

**V13 FAIL**。

本轮只使用原训练用户内部的 Train-A/Dev-A，正式 validation/test 未加载。
B0 在 Train-A 重新训练，因此 Dev-A 用户对 B0、V12 和 V13 都是未见用户。

| Method | Accuracy | Macro-F1 | Walk F1 | Bike F1 | Bus F1 | Car F1 | Train F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6575 | 0.5847 | 0.8102 | 0.6693 | 0.6244 | 0.4955 | 0.3243 |
| V12（不改 logits） | 0.6575 | 0.5847 | 0.8102 | 0.6693 | 0.6244 | 0.4955 | 0.3243 |
| V12+V13 | 0.6575 | 0.5850 | 0.8102 | 0.6693 | 0.6237 | 0.4973 | 0.3243 |

## 配对增量

- Accuracy：+0.00 pp
- Macro-F1：+0.02 pp
- Bus F1：-0.07 pp
- Car F1：+0.18 pp

## Bus-Car 混淆

- B0 Bus→Car：52；V13：53
- B0 Car→Bus：125；V13：124

## Gate

- Dev-A 平均 sparsity gate：0.8561
- Dev-A 平均 uncertainty gate：0.1849
- 最大 |delta|：0.0157

高采样率理论保护值（按平均点数代入同一 gate）：

- 5s：0.0000
- 10s：0.0000
- 20s：0.0350
- 30s：0.4900
- 60s：0.9570

## 验收门槛

- ΔAccuracy ≥ +1.0 pp；
- ΔMacro-F1 ≥ +2.0 pp；
- Car F1 Δ ≥ -0.5 pp；
- Bus F1 明显改善。

正式测试是否访问：False。
