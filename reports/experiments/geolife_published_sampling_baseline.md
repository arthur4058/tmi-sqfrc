# GeoLife 公开时间分箱采样协议与 Original TMI 基线报告

## 1. 结论

本实验按照公开文献中的时间分箱采样算子重建了 GeoLife 多采样率数据：将物理时间划分为等长区间，每个非空区间仅保留第一个真实 GPS 观测点，不进行位置插值。数据完整性验证通过。

在用户互斥划分上训练、不包含任何采样感知模块的 Original TMI 后，使用同一个最佳 checkpoint 测试四种采样视图，得到以下 Macro-F1：

- 固定 5 秒：0.7042；
- 固定 30 秒：0.4974，下降 0.2068；
- 固定 60 秒：0.3833，下降 0.3209；
- 动态 5–60 秒：0.6112，下降 0.0930。

结果表明，采样变稀和轨迹内采样率变化都会降低原基线性能。这组结果冻结为后续采样质量感知与运动特征可靠性模块需要超过的 B0。

## 2. 实验标识

- Git 分支：`feat/variable-sampling-benchmark`
- 协议标识：`geolife-published-episode-sampling-v2`
- 数据随机种子：42
- 模型随机种子：10086
- 物理窗口：300 秒
- 滑动步长：150 秒
- 最少 GPS 点数：5
- 训练模型：Original TMI 双分支模型
- 最佳 checkpoint：epoch 11
- checkpoint SHA-256：`9b97b2c1864c81cd193a3c86a1beafac42954efbf52bb92cc36fa6dd3a96b984`
- GeoLife 1.3 压缩包 SHA-256：`1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6`
- 完成日期：2026-08-04

原始数据、生成的 NPY/PKL、模型 checkpoint 和日志均保存在本机，未提交到 Git。

## 3. 公开方法依据

固定采样视图采用 Burkhard 等人在采样率影响研究中使用的算子：把时间划分成等长 episodes，在每个非空 episode 中保留第一个真实观测点。

- Burkhard, O., Becker, H., Weibel, R., Axhausen, K. W. (2020). *On the requirements on spatial accuracy and sampling rate for transport mode detection in view of a shift to passive signalling data*. Transportation Research Part C, 114, 99–117. DOI: <https://doi.org/10.1016/j.trc.2020.01.021>
- 公开全文：<https://www.research-collection.ethz.ch/server/api/core/bitstreams/896120d2-e7e0-4882-8f1c-f4cccad18630/content>

Takahashi 和 Fujii 也在公开 GeoLife 上通过保留满足目标时间间隔的下一个真实点构造 1、3、5 分钟稀疏数据，说明使用稠密公开轨迹构造可控采样视图是已有研究采用的可复现路径：<https://doi.org/10.2197/ipsjjip.34.309>。

动态 `variable_5_60s` 不是缺失数据或随机删点：它在连续 60 秒物理时间块中切换 5、10、15、30、60 秒 episode 长度，每个窗口使用由固定种子产生的排列。每个 episode 仍执行相同的公开采样算子。该视图在报告中明确称为“公开固定采样算子的分段变率扩展”。

## 4. 数据构造

### 4.1 数据流程

```text
GeoLife 真实带标签轨迹
    → 清洗空轨迹、重复时间戳和非有限值
    → 按用户划分 train / validation / test
    → 300 秒物理时间窗口，150 秒滑动
    → 同一窗口构造 fixed_5s / fixed_30s / fixed_60s / variable_5_60s
    → 每种视图重新计算轨迹与运动特征
```

全部视图只选择原始 GeoLife 点；没有插值、均值坐标或合成坐标。

### 4.2 用户和样本规模

| 划分 | 有效用户数 | 配对窗口数 | Walk | Bike | Bus | Car | Train |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 44 | 49,093 | 11,050 | 7,096 | 13,111 | 9,943 | 7,893 |
| Validation | 6 | 7,505 | 2,742 | 901 | 2,122 | 1,159 | 581 |
| Test | 12 | 4,941 | 1,370 | 573 | 640 | 2,016 | 342 |

轨迹级用户划分原为 45/6/13 名用户；一名训练用户和一名测试用户没有满足 300 秒窗口与最少点数要求的片段，因此窗口数据中为 44/6/12。三个集合仍严格用户互斥。

### 4.3 测试视图统计

| 视图 | 平均点数 | 中位点数 | 平均实际间隔 | 中位实际间隔 |
|---|---:|---:|---:|---:|
| fixed_5s | 51.49 | 60 | 5.78 秒 | 5 秒 |
| fixed_30s | 9.94 | 10 | 31.56 秒 | 30 秒 |
| fixed_60s | 5.42 | 5 | 60.98 秒 | 60 秒 |
| variable_5_60s | 22.41 | 25 | 13.28 秒 | 10 秒 |

实际均值不必严格等于目标间隔，因为公开分箱算子不会在空 episode 中插值；信号缺失时会自然产生更长的真实间隔。

## 5. 数据正确性验证

全量验证结果：`PASSED`。

已验证：

1. Train、Validation、Test 用户集合两两无交集；
2. 四种视图的标签、用户 ID、源轨迹 ID 和 `pair_id` 完全对齐；
3. 每个稀疏视图都能由对应固定 5 秒真实点视图重新执行采样算子精确生成；
4. 所有坐标均来自原始点，没有插值；
5. 固定视图平均点数严格满足 `5s > 30s > 60s`；
6. 动态视图的时间间隔标准差大于 0，轨迹内部确实存在多个采样率；
7. 相同种子和 `pair_id` 会生成完全一致的动态视图；
8. 每个生成的轨迹文件均保存 SHA-256。

机器可读验证文件：

- `reports/manifests/geolife_published_sampling_seed42.json`
- `reports/manifests/geolife_published_sampling_validation.json`

S4 特征计算会按照原作者的运动有效性规则过滤极少量片段，因此最终测试样本数为 4,918–4,940，视图间差异小于 0.5%。采样前的 4,941 个窗口严格配对。

## 6. Original TMI 训练

主要配置：

- 训练视图：`fixed_5s`；
- 轨迹分支：4 层 Transformer，`d_model=64`，8 个注意力头；
- 特征分支：1 层 Transformer，`d_model=128`，16 个注意力头；
- 参数量：540,937；
- Optimizer：RAdam；
- Learning rate：0.001；
- Batch size：64；
- 最大 epochs：200；
- Early stopping patience：40；
- 输入：50% noisy / 50% clean；
- 独立用户验证集。

训练在 epoch 51 早停，最佳 checkpoint 为 epoch 11：

| 验证指标 | 数值 |
|---|---:|
| Loss | 0.543637 |
| Accuracy | 0.846798 |
| Macro Precision | 0.833868 |
| Macro Recall | 0.791247 |
| Macro-F1 | 0.809214 |

训练和验证损失保持有限值，未出现 NaN。

## 7. 正式基线结果

四种测试条件均加载同一个 epoch 11 checkpoint，没有重新训练或微调。

| 测试视图 | 样本数 | Accuracy | Macro-F1 | ΔAccuracy | ΔMacro-F1 |
|---|---:|---:|---:|---:|---:|
| fixed_5s | 4,932 | 0.7419 | 0.7042 | 0.0000 | 0.0000 |
| fixed_30s | 4,940 | 0.6366 | 0.4974 | -0.1053 | -0.2068 |
| fixed_60s | 4,918 | 0.4929 | 0.3833 | -0.2490 | -0.3209 |
| variable_5_60s | 4,940 | 0.6836 | 0.6112 | -0.0583 | -0.0930 |

### 7.1 各类别 F1

| 测试视图 | Walk | Bike | Bus | Car | Train |
|---|---:|---:|---:|---:|---:|
| fixed_5s | 0.8333 | 0.6860 | 0.5926 | 0.7800 | 0.6291 |
| fixed_30s | 0.8043 | 0.5790 | 0.2303 | 0.6955 | 0.1777 |
| fixed_60s | 0.7279 | 0.4882 | 0.1054 | 0.4612 | 0.1339 |
| variable_5_60s | 0.8022 | 0.6061 | 0.5389 | 0.7339 | 0.3750 |

Bus 和 Train 对采样变稀最敏感，因此后续模块应以 Macro-F1 和类别 F1 为主要判断依据，不能只看 Accuracy。

## 8. 后续模块的冻结对照

后续实验必须保持以下内容不变：

- 用户划分和窗口 `pair_id`；
- 四种采样视图及随机种子；
- 数据清洗和 S4 特征参数；
- Original TMI 主干与分类头；
- 训练轮次上限、早停规则、优化器和学习率；
- 测试指标计算方法。

加入采样质量感知与运动特征可靠性模块后，至少应满足：

1. `fixed_30s`、`fixed_60s` 和 `variable_5_60s` 的 Macro-F1 高于本报告 B0；
2. 四视图平均 Macro-F1 提高；
3. `fixed_5s` 性能没有明显下降。

## 9. 可复现入口

- 数据构造与验证：`bash scripts/generate_published_sampling_dataset.sh`
- 特征生成：`bash scripts/generate_variable_sampling_features.sh`
- 基线训练：`bash scripts/run_variable_sampling_training.sh`
- 四视图评测：`python scripts/evaluate_variable_sampling.py`
- 训练配置：`configs/variable_sampling/geolife_published_fixed5_seed10086.json`
- 机器可读结果：
  - `reports/experiments/geolife_published_sampling_results.json`
  - `reports/experiments/geolife_published_sampling_results.csv`
