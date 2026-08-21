# V12/V13 稠密行为恢复与保守纠错实验报告

> 日期：2026-08-21
> 仓库：`arthur4058/tmi-sqfrc`
> 分支：`feat/dense-privileged-behavior-recovery-v12`
> 起点：`36b34d2`（V11）
> 数据：GeoLife 用户互斥、300 秒物理窗口、60 秒稀疏视图

## 1. 最终结论

本任务书已经执行到其条件规则允许的终点：

- **V12 行为恢复：PASS**；
- **V13 保守 Bus-Car 纠错：FAIL**；
- 因 V13 没有达到 Dev-A 门槛，严格按照任务书停止；
- 没有访问原 validation、正式 test；
- 没有运行 30 秒扩展和多随机种子正式实验；
- 不能把本轮 Dev-A 结果包装成正式测试提升。

V12 能从 60 秒稀疏区间恢复部分由 5 秒轨迹监督的物理行为统计量，但这些信息没有通过当前只修改 Bus/Car logits 的 V13 机制转化为分类收益。

## 2. 数据与防泄漏协议

从原五档数据的 44 个 training users 内重新建立用户互斥开发折：

| 项目 | Train-A | Dev-A |
|---|---:|---:|
| 用户数 | 35 | 9 |
| 原始物理窗口 | 47,307 | 1,786 |
| 60 秒稀疏区间 | 209,641 | 7,943 |
| 有效 5 秒行为目标 | 209,641 | 7,943 |
| 无效区间 | 0 | 0 |

检查结果：

- Train-A 与 Dev-A 用户无交集；
- 两个开发折都覆盖 Walk、Bike、Bus、Car、Train；
- 5 秒和 60 秒视图的标签、用户、来源和 `pair_id` 完全一致；
- 使用真实时间戳选择落入每个 60 秒区间的 5 秒真实观测；
- 没有插值；
- target normalization 只使用 Train-A；
- 原 validation 和正式 test 没有被加载。

## 3. V12：稠密先验监督的行为恢复

### 3.1 模型

V12 输入仅来自测试时可见的 60 秒稀疏区间：

1. `delta_t`
2. `distance`
3. `average_speed`
4. `heading`
5. `heading_change`
6. `heading_change_rate`
7. `valid_interval`

轻量网络结构：

```text
7 → Linear(32) → GELU → Dropout → Linear(6)
```

新增参数量：454。训练损失为 Train-A 统计归一化后的 SmoothL1。

恢复目标：平均速度、速度标准差、最大速度、停留比例、加速度能量和转向能量。

### 3.2 Dev-A 恢复结果

| Target | 常数 MAE | 稀疏直接观测 MAE | V12 MAE | V12 相对常数改善 | R² | Pearson |
|---|---:|---:|---:|---:|---:|---:|
| mean_speed | 6.480941 | **0.377591** | 0.493200 | **92.39%** | 0.9393 | 0.9727 |
| std_speed | 1.404871 | 1.561185 | **0.926527** | **34.05%** | 0.0558 | 0.2476 |
| max_speed | 7.632505 | 2.925929 | **1.893194** | **75.20%** | 0.7231 | 0.8593 |
| stop_ratio | 0.093230 | **0.061560** | 0.079865 | **14.34%** | 0.0434 | 0.4508 |
| acceleration_energy | 0.178192 | 0.221227 | **0.124857** | **29.93%** | 0.1074 | 0.3898 |
| turning_energy | 41.105431 | 73.844978 | **26.686531** | **35.08%** | 0.4195 | 0.6689 |

训练状态：

- 最佳 epoch：49；
- 最佳 Dev normalized MAE：0.434738；
- Train normalized MAE：0.380591；
- Train/Dev gap ratio：1.1423；
- 训练时间：42.89 秒；
- GPU：RTX 5070。

### 3.3 V12 判定

预设 PASS 规则：

- `mean_speed`、`max_speed`、`stop_ratio` 至少两个相对常数预测器改善不低于 10%；
- 六个目标至少四个优于常数预测器；
- Train/Dev normalized MAE 比值不超过 2。

实际结果：关键目标 3/3 通过、全部目标 6/6 优于常数、无严重 Train/Dev 分离。因此：

> **V12 PASS。**

但需注意：`mean_speed` 和 `stop_ratio` 的简单稀疏直接观测基线仍优于 V12，说明这两个量的大部分可预测性来自 60 秒端点本身。V12 真正额外体现价值的是最大速度、速度波动、加速度能量和转向能量。

## 4. V13 开发基线

旧 B0 checkpoint 已经见过 Dev-A 用户，不能用于新的开发协议。因此重新完成：

1. 只用 Train-A 用户进行 S3 增强；
2. 对 Train-A/Dev-A 分别运行与五档基线相同参数的 S4；
3. 只用 Train-A 训练 B0；
4. 只用 Dev-A 选最佳 checkpoint。

分段样本：

- Train-A：60,221；
- Dev-A：1,740；
- 五类全部覆盖。

B0 最大 200 epochs、patience 40，epoch 51 早停，最佳 checkpoint 为 epoch 11：

- Dev-A Accuracy：65.7471%；
- Dev-A Macro-F1：58.4738%。

## 5. V13：稀疏度与不确定性感知的保守纠错

V13 冻结 B0 和 V12，只训练一个小型纠错器。输入包括：

- V12 六种恢复行为的窗口均值与最大值；
- B0 Bus/Car logits；
- Bus-Car margin；
- B0 entropy 和 confidence；
- 有效 GPS 点数与有效区间比例。

V13 只做反对称更新：

```text
Bus logit += delta
Car logit -= delta
```

Walk、Bike、Train logits 完全不变。`delta` 同时受到 observation-density gate、Bus-Car uncertainty gate 和 `alpha` 上界限制。

只在 Dev-A 搜索：

- `alpha`：0.25、0.50、0.75、1.00；
- uncertainty `tau`：0.50、1.00；
- temperature：0.50；
- anchor weight：0.50。

八个冻结候选均未通过门槛。最佳候选为 `alpha=0.25, tau=0.5`。

## 6. V13 Dev-A 结果

| Method | Accuracy | Macro-F1 | Walk F1 | Bike F1 | Bus F1 | Car F1 | Train F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 65.7471% | 58.4738% | 81.0170% | 66.9307% | 62.4409% | 49.5479% | 32.4324% |
| V12（只恢复行为，不改 logits） | 65.7471% | 58.4738% | 81.0170% | 66.9307% | 62.4409% | 49.5479% | 32.4324% |
| V12+V13 | 65.7471% | 58.4959% | 81.0170% | 66.9307% | 62.3697% | 49.7297% | 32.4324% |

配对变化：

- Accuracy：**+0.00 pp**；
- Macro-F1：**+0.022 pp**；
- Bus F1：**-0.071 pp**；
- Car F1：**+0.182 pp**。

Bus-Car 混淆：

- Bus→Car：B0 为 52，V13 为 53；
- Car→Bus：B0 为 125，V13 为 124。

V13 只把一个 Car→Bus 修正为 Car，同时又把一个 Bus 改错为 Car，总体正确数没有变化。

Gate 统计：

- Dev-A 平均 sparsity gate：0.8561；
- Dev-A 平均 uncertainty gate：0.1849；
- 最大 `|delta|`：0.0157；
- 所有 correction 均满足预设上界。

按五档平均点数代入同一个 sparsity gate：

| 采样间隔 | 理论平均 gate |
|---:|---:|
| 5 s | 0.0000 |
| 10 s | 0.0000 |
| 20 s | 0.0350 |
| 30 s | 0.4900 |
| 60 s | 0.9570 |

高采样率保护机制在公式和单元测试层面成立，但因 60 秒 Dev-A 未通过，没有继续运行五档分类实验。

## 7. V13 判定

预设 Dev-A 门槛：

- Accuracy 至少 `+1.0 pp`；
- Macro-F1 至少 `+2.0 pp`；
- Car F1 不低于 `-0.5 pp`；
- Bus F1 明显改善。

实际仅为 `+0.00 pp Accuracy / +0.022 pp Macro-F1`，Bus F1 还略有下降。因此：

> **V13 FAIL。**

按照任务书规则，本轮没有访问正式 test，也没有补跑 30 秒和多随机种子。

## 8. 失败原因分析

1. **纠错范围过窄。** Dev-A 不仅存在 Bus↔Car 混淆，还存在 Bus→Walk/Bike、Car→Train、Train→Car/Bus 等错误；只交换 Bus/Car logits 的理论收益上限较低。
2. **V12 恢复不等于分类判别。** V12 对物理统计回归有效，但可恢复的连续行为量未必正好对应 Bus/Car 的稳定跨用户边界。
3. **不确定性门过于保守。** 平均 uncertainty gate 只有 0.1849，最佳候选最大修正仅 0.0157 logit，几乎不改变预测。
4. **增大上界没有解决问题。** `alpha=0.5–1.0` 的候选 Accuracy 均下降约 0.115 pp，Bus F1 下降而 Car F1 上升，说明不是简单把 correction 放大就能获得双向收益。
5. **跨用户类别边界仍不稳定。** Dev-A 的 Train F1 仅 32.43%，Car F1 仅 49.55%；主要瓶颈不是单一 Bus-Car 边界。

## 9. 与当前最好 V2 的关系

本轮不能替代 V2：

- V2 在正式 60 秒协议三个种子上平均提升 `+1.21 pp Accuracy / +2.03 pp Macro-F1`；
- V12 只证明部分行为量可恢复；
- V13 未把恢复量转化为明显分类提升；
- 本轮未访问正式 test，因此不存在可与 V2 主结果直接对比的新测试数字。

## 10. 后续建议

建议保留 V12 作为“稠密训练先验可以恢复部分区间行为”的有效诊断模块，但停止当前 V13 Bus-Car 二分类边界纠正路线。

下一步若继续利用 V12，应先在全新开发折上验证以下更直接的问题：

1. V12 恢复表示对五类标签是否提供超出稀疏直接观测的互信息；
2. 使用线性 probe 比较 `B0 representation` 与 `B0 + frozen V12 representation`，而不是先手工限定 Bus-Car；
3. 若线性 probe 仍无稳定增益，则停止行为恢复分类路线；
4. 论文主线暂时保持 V2，并补充代表方法对比、计算成本和显著性检验。

禁止根据正式 test 继续调整当前 V13 gate 或 alpha。

## 11. 工程验证范围

- V12/V13 专项测试通过；
- 时间戳对齐、无泄漏、gate 端点、uncertainty 单调性、correction bound、高采样率不变性均有自动测试；
- checkpoint 和大型数据只保存在本地 `data/`、`experiments/`；
- Git 只提交代码、配置、清单和报告。
