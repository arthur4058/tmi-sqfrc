# GeoLife V4 时间可靠性融合模块实验结果

## 最终结论

V4 在固定 60 秒采样间隔的匹配训练/测试协议上，**Accuracy 和 Macro-F1 均超过 B0**，通过最低筛选标准；但 Accuracy 只提高 0.05 个百分点，当前仍属于单随机种子的初步结果，不能直接写成稳定提升。

| 方法 | Accuracy | Macro-F1 | 相对 B0 Accuracy | 相对 B0 Macro-F1 |
|---|---:|---:|---:|---:|
| B0 原基线 | 71.61% | 63.55% | — | — |
| M1 逐点质量门控 | 71.43% | 63.82% | -0.18 pp | +0.27 pp |
| **V4 时间可靠性融合** | **71.66%** | **64.14%** | **+0.05 pp** | **+0.59 pp** |

V4 相对 M1 的 Accuracy 提高 0.23 pp、Macro-F1 提高 0.32 pp。

## 解决的问题

固定采样间隔增大后，每个 300 秒窗口中的 GPS 点显著减少，速度、加速度、急动度和航向变化率等运动特征的可靠性也会改变。B0 把所有运动特征表示以相同方式参与双分支融合，无法根据采样质量调整其贡献。

早期 M1 在输入端逐点缩放运动特征，可能破坏原始时序模式。V4 保留 B0 的轨迹和运动特征编码器，只在运动特征分支完成编码后、与轨迹分支融合前进行轻量校准。

## V4 方法

每个轨迹段构造四个质量描述量：

1. 标准化 `delta_t` 的平均绝对值；
2. 标准化 `delta_t` 的段内离散程度；
3. 有效点数占最大序列长度的比例；
4. 相对 5 秒基准的名义采样间隔对数。

一个隐藏维度为 8 的 MLP 输出运动特征分支 128 个潜在通道的可靠性修正：

```text
reliability = 1 + 0.25 * tanh(MLP(quality_descriptor))
feature_output = feature_output * reliability
```

- 门控严格限制在 `[0.75, 1.25]`；
- 最后一层零初始化，初始状态严格等价于 B0；
- 门控在所有 B0 层初始化完成后才创建；
- 已用自动测试验证同一 seed 下 V4 与 B0 的共享主干初始参数逐项完全一致；
- 仅增加 1,192 个参数，占 B0 参数量的 0.2204%。

## 公平实验协议

- 基线提交：`4ffb01d`
- 数据协议：`geolife-five-rate-matched-v1`
- 数据：`geolife_five_rate_fixed_60s`
- 用户划分：训练、验证、测试用户严格互斥
- 物理窗口：300 秒，步长 150 秒
- 训练/测试采样间隔：均为 60 秒
- 训练样本：62,786
- 验证样本：7,129
- 测试样本：4,918
- seed：10086
- 最大训练轮数：200
- Early Stopping patience：40
- Batch size：64
- 优化器：RAdam，学习率 0.001
- 模型其余结构和超参数与 B0 一致

## 训练结果

- 第 42 轮触发早停
- 最佳 checkpoint：第 2 轮
- 最佳验证 Accuracy：73.2221%
- 最佳验证 Macro-F1：62.0402%
- 正式训练耗时：493.54 秒
- checkpoint 有限值检查：通过
- checkpoint SHA-256：`7fc88416494b8c555ccbb5089804cadab982f551d8933f33897fc21aeed58489`

最佳轮次较早，说明 60 秒档输入信息较少，模型在后续训练中容易过拟合。独立测试只使用验证 Accuracy 选出的第 2 轮 checkpoint。

## 独立测试结果

| 类别 | Precision | Recall | F1 | 相对 B0 F1 | 样本数 |
|---|---:|---:|---:|---:|---:|
| Walk | 79.91% | 90.34% | 84.80% | +0.12 pp | 1,356 |
| Bike | 63.25% | 78.98% | 70.25% | +2.47 pp | 571 |
| Bus | 44.12% | 69.44% | 53.96% | +1.49 pp | 638 |
| Car | 85.42% | 65.41% | 74.09% | -0.69 pp | 2,015 |
| Train | 69.60% | 25.74% | 37.58% | -0.44 pp | 338 |
| **Macro / Overall** | **68.46%** | **65.98%** | **64.14%** | **+0.59 pp** | **4,918** |

整体 Accuracy 为 **71.66%**。Macro-F1 的提升主要来自 Bike 和 Bus；Car 与 Train 略有下降，因此 V4 还不是各类别都稳定获益的最终模块。

## 无效运行的处理

第一次正式运行中，V4 门控在主干网络之前创建，改变了同 seed 下主干参数的随机初始化。该结果存在初始化混杂，已明确排除，没有进入本报告。其本地产物保存在：

```text
experiments/geolife_v4_temporal_reliability_fixed_60s_seed10086_invalid_rng_initialization
```

最终结果来自修复后重新从头运行的唯一公平实验。

## 判断与后续建议

当前判断为：**初步通过，但证据强度不足**。

- 最低标准通过：Accuracy 和 Macro-F1 同时高于 B0；
- Macro-F1 提升达到 0.59 pp；
- Accuracy 仅提升 0.05 pp，可能处于随机波动范围；
- 目前只有 seed 10086，不能声称稳定有效。

下一步不建议立即修改模块或直接跑完整五档。应保持代码与超参数不变，只在 60 秒档补跑两个随机种子（建议 42、2026）：

1. 若三种子平均 Accuracy 和 Macro-F1 均高于 B0，再扩展到 20、30 秒及完整五档；
2. 若 Macro-F1 稳定提升但 Accuracy 波动，可把 Macro-F1 明确设为论文主指标，并报告 Accuracy 的真实结果；
3. 若多种子提升消失，则停止 V4，不再继续微调门控权重，转向类别不平衡或稀疏序列正则化方案。

## 可复现文件

- 训练配置：`configs/five_rate/geolife_temporal_reliability_v4_60s_train.json`
- 测试配置：`configs/five_rate/geolife_temporal_reliability_v4_60s_test.json`
- 结构化结果：`reports/experiments/geolife_temporal_reliability_fusion_v4_60s.json`
- 表格结果：`reports/experiments/geolife_temporal_reliability_fusion_v4_60s.csv`
- checkpoint、日志和生成数据位于本地 `experiments/` 与 `data/`，不提交 Git。
