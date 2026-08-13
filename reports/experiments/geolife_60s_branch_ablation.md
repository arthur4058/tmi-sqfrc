# GeoLife 60 秒分支消融诊断报告

## 1. 目的

本实验用于回答一个具体问题：现有 B0 双分支基线在 60 秒稀疏采样条件下，轨迹坐标分支与运动特征分支分别承担什么作用？

这不是新的改进版本，也不用于宣称性能提升。实验只移除 B0 的一个输入分支，定位后续正式模块应该作用的位置。

## 2. 固定实验协议

- 基线提交：`4ffb01d`
- 诊断分支：`diag/60s-branch-ablation`
- 数据协议：`geolife-five-rate-matched-v1`
- 数据：`geolife_five_rate_fixed_60s`
- 划分：用户互斥的固定 train / val / test
- 训练样本：62,786
- 验证样本：7,129
- 测试样本：4,918
- 随机种子：10,086
- 最大训练轮数：200
- 验证间隔：1 epoch
- 早停 patience：40
- 优化器：RAdam
- 学习率：0.001
- batch size：64
- 输入：`50%noise`
- GPU：NVIDIA GeForce RTX 5070
- 模型选择：验证集 Accuracy 最优 checkpoint
- 最终结果：独立测试集一次性评估

三种模型使用同一数据划分、随机种子、优化器和训练选择规则：

- **B0 双分支**：轨迹坐标分支 + 运动特征分支。
- **B1 仅轨迹**：只输入经纬度序列，结构与 B0 的轨迹分支一致。
- **B2 仅运动特征**：只输入速度、加速度、jerk、航向变化率，结构与 B0 的运动特征分支一致。

## 3. 模型规模与 checkpoint

| 模型 | 参数量 | 最佳 epoch | checkpoint SHA-256 |
|---|---:|---:|---|
| B0 双分支 | 540,937 | 2 | `e02cf4ecfeadb214ae48ea73c617edc423044ae2dd1f70d93705ada269234e13` |
| B1 仅轨迹 | 300,810 | 3 | `48e818c9cebe504788cbc3a8f9bb60caf7e7d1864dae6e63659157efd808af3f` |
| B2 仅运动特征 | 400,268 | 19 | `279110d6c376935df5f27d488f3bd35cdee51c3903e2fcff41fb56632a5148ad` |

训练没有要求跑满 200 轮。三种模型都按相同的 `patience=40` 规则早停，并保留验证 Accuracy 最优的 checkpoint，因此训练轮数差异不构成协议差异。

## 4. 独立测试结果

| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 | 相对 B0 Accuracy | 相对 B0 Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0 双分支 | **71.61%** | **64.55%** | 65.26% | **63.55%** | — | — |
| B1 仅轨迹 | 24.77% | 28.81% | 23.88% | 15.37% | -46.84 pp | -48.18 pp |
| B2 仅运动特征 | 67.47% | 60.68% | **66.87%** | 61.97% | -4.14 pp | -1.58 pp |

### 4.1 逐类 F1

| 模型 | Walk | Bike | Bus | Car | Train |
|---|---:|---:|---:|---:|---:|
| B0 双分支 | 84.68% | **67.78%** | 52.47% | **74.78%** | **38.02%** |
| B1 仅轨迹 | 32.70% | 0.00% | 29.67% | 14.46% | 0.00% |
| B2 仅运动特征 | **85.22%** | 67.60% | **53.89%** | 65.62% | 37.54% |

### 4.2 混淆矩阵

B0 双分支：

```text
             Pred: Walk  Bike  Bus   Car  Train
True Walk          1222   103   29     1      1
True Bike            79   468   24     0      0
True Bus             51    68  361   153      5
True Car             78   166  318  1371     82
True Train          100     5    6   127    100
```

B1 仅轨迹：

```text
             Pred: Walk  Bike  Bus  Car  Train
True Walk           659     0  694    0      3
True Bike           163     0  408    0      0
True Bus            236     0  402    0      0
True Car           1505     0  341  157     12
True Train          111     0  227    0      0
```

B2 仅运动特征：

```text
             Pred: Walk  Bike  Bus   Car  Train
True Walk          1240    84   23     4      5
True Bike            86   459   24     1      1
True Bus             54    78  388   111      7
True Car             74   160  352  1066    363
True Train          100     6   15    52    165
```

## 5. 诊断结论

### 5.1 基线没有选错

B0 在 Accuracy 和 Macro F1 上都高于 B1、B2。双分支融合确实有效，不能因为 V5/V6 没有通过提升门槛，就认为原作者基线无效或必须重新复现另一套基线。

### 5.2 60 秒下运动特征是主要信息来源

B2 保留了 B0 的大部分性能：Accuracy 只下降 4.14 个百分点，Macro F1 只下降 1.58 个百分点。说明速度、加速度、jerk 和航向变化率是当前 60 秒识别的主要有效输入。

### 5.3 原始轨迹分支单独不可用，但融合时仍有补充价值

60 秒窗口中每段最多只有 6 个坐标点。B1 无法识别 Bike 和 Train，测试 Accuracy 只有 24.77%。因此不能把 B1 当作新的主基线或正式模块载体。

不过，B0 相比 B2 的 Accuracy 提高 4.14 个百分点，尤其 Car F1 从 65.62% 提升至 74.78%。这说明轨迹分支虽然单独很弱，但与运动特征联合时仍提供了互补信息。

### 5.4 后续问题定位

当前最值得解决的不是“重新选择基线”，也不是继续增强已经很强的运动特征分支，而是：

> 如何让极短的稀疏轨迹坐标序列产生更可靠的表示，并只在它确实有用时与运动特征融合。

## 6. 下一步建议

停止继续堆叠 V7/V8 编号。下一次只建立一个正式候选分支，并做一个受控改动：

1. 保留 B0 的运动特征分支与分类头。
2. 将轨迹分支从“6 个点直接展平”改为适合极短序列的位移/时间间隔编码与掩码池化。
3. 增加一个轻量可靠性门控，根据有效点数、时间跨度和间隔稳定性控制轨迹分支对融合结果的贡献。
4. 先只跑 60 秒、seed 10086。
5. 预先设定通过条件：Accuracy 和 Macro F1 都超过 B0；未通过则停止，不扩展五档和多 seed。

这里的目标不是强行获得 5 个百分点提升。对于已经达到 71.61% 的 B0，在完全相同数据与协议下，若单个轻量模块能够稳定同时提升 Accuracy 和 Macro F1，且在多 seed 下成立，才是可信的有效结果。

## 7. 可复现配置

训练配置：

- `configs/diagnostics/geolife_60s_b1_trajectory_train.json`
- `configs/diagnostics/geolife_60s_b2_feature_train.json`

独立测试配置：

- `configs/diagnostics/geolife_60s_b1_trajectory_test.json`
- `configs/diagnostics/geolife_60s_b2_feature_test.json`

冒烟配置：

- `configs/diagnostics/geolife_60s_b1_trajectory_smoke.json`
- `configs/diagnostics/geolife_60s_b2_feature_smoke.json`

原始数据、生成的 NPY/PKL、实验 checkpoint 和工作簿均未提交 Git。
