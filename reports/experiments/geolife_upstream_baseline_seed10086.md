# GeoLife 上游基线复现实验报告

## 1. 实验结论

本次实验已经完成 GeoLife 上游五分类基线的端到端复现：数据预处理、全量训练启动、最佳 checkpoint 保存以及独立测试均已验证。

有效结果来自 **epoch 25 的最佳 checkpoint**。该模型在 5,321 个独立测试样本上取得：

- Accuracy：**0.8395**
- Macro precision：**0.8280**
- Macro recall：**0.8285**
- Macro F1：**0.8273**

训练在 epoch 40 出现非有限损失并发生类别坍缩。现有 EarlyStopping 将 NaN 错误识别为性能改善，因此没有正常停止。本报告保留该现象作为上游实现兼容性问题；测试结果使用发散前保存且已经过有限值检查的 epoch 25 checkpoint。

## 2. 复现范围

- 数据集：Microsoft GeoLife GPS Trajectories 1.3
- 数据协议：`geolife-upstream-random-v1`
- 类别：Walk、Bike、Bus、Car、Train
- 训练目标：`dual_branch_classification_from_scratch`
- 数据表示：`trajectory_with_feature`
- 输入模式：`50%noise`
- 数据划分：上游轨迹级随机划分
- 最终论文的用户互斥变采样协议：尚未实现

本次结果用于确认上游基线能够在当前环境中运行，不能当作最终用户互斥论文协议的结果。

## 3. 代码、环境与硬件

- 仓库：`arthur4058/tmi-sqfrc`
- 实验分支：`exp/geolife-baseline-reproduction`
- 测试配置提交：`dcbb0a5`
- 环境基线：`v0.1.0-env`
- 数据基线：`v0.2.0-data`
- Python：3.10.20
- NumPy：1.24.4
- PyTorch：2.7.1+cu128
- PyTorch CUDA runtime：12.8
- GPU：NVIDIA GeForce RTX 5070
- GPU compute capability：12.0

## 4. 数据产物

上游 S1–S4 预处理已经完成，主要统计如下：

| 阶段 | 产物 | 数量 |
| --- | --- | ---: |
| 原始数据 | 用户目录 | 182 |
| 原始数据 | 有标签用户 | 69 |
| 原始数据 | PLT 文件 | 18,670 |
| S1 | 提取轨迹 | 14,686 |
| S2 | 训练轨迹 | 11,748 |
| S2 | 测试轨迹 | 2,938 |
| S3 | 增强后训练轨迹 | 25,840 |
| S4 | 训练分段 | 43,546 |
| S4 | 测试分段 | 5,321 |

数据协议、参数、统计及 SHA-256 详见：

- `docs/dataset_protocol.md`
- `reports/manifests/v0.2.0_geolife_stats.json`
- `reports/manifests/v0.2.0_geolife_sha256.txt`

## 5. 正式训练配置

配置文件：`configs/baseline/geolife_upstream_seed10086.json`

关键参数：

| 参数 | 值 |
| --- | --- |
| epochs | 200 |
| batch size | 64 |
| optimizer | RAdam |
| learning rate | 0.001 |
| validation interval | 1 |
| validation ratio | 0.1 |
| patience | 40 |
| seed | 10086 |
| GPU | 0 |
| num workers | 0 |

全量数据规模：

- 训练：39,191
- 验证：4,355
- 测试：5,321

## 6. 验证集最佳结果

最佳 checkpoint 保存于 epoch 25：

| 指标 | 值 |
| --- | ---: |
| Validation loss | 0.464723 |
| Validation accuracy | 0.836510 |
| Validation precision | 0.835078 |
| Validation recall | 0.838720 |
| Validation F1 | 0.834900 |

有效 checkpoint：

`experiments/geolife_upstream_baseline_seed10086/checkpoints/model_best_epoch25_valid.pth`

SHA-256：

`9b61373f24c726087b20680fb3ab1b2ff47950aae5a7eb23cfb9cd04c2d7c8d9`

安全有限值检查结果：

- checkpoint epoch：25
- 模型张量数：117
- 含 NaN/Inf 的张量数：0

## 7. 数值发散记录

训练在 epoch 39 仍然正常：

- Training loss：0.372982
- Validation loss：0.498434
- Validation accuracy：0.830540
- Validation F1：0.830024

epoch 40 首次出现：

- Training loss：NaN
- Validation loss：NaN
- Validation accuracy：0.161424
- Validation precision：0.161424
- Validation recall：0.200000
- Validation F1：0.055595

这组指标表示模型坍缩为只预测一个类别。由于 EarlyStopping 对 NaN 的比较逻辑不完整，它随后持续输出“发现更好的性能，重置计数器”，训练没有自动终止。实验最终在 epoch 138 手动停止。

无效 checkpoint：

`experiments/geolife_upstream_baseline_seed10086/checkpoints/model_last.pth`

有限值检查结果：

- checkpoint epoch：138
- 总张量数：275
- 含 NaN/Inf 的张量数：259
- 状态：**禁止用于恢复、推理或测试**

该问题不是数据下载、用户操作或 RTX 5070 硬件故障，而是训练过程的数值稳定性与非有限指标处理缺陷。

## 8. 独立测试

独立测试配置：

`configs/baseline/geolife_upstream_epoch25_test.json`

测试模型：epoch 25 有效 checkpoint  
测试样本数：5,321  
测试运行时间：45.62 秒（shell 总时间 49.64 秒）

总体结果：

| 指标 | 值 |
| --- | ---: |
| Accuracy | 0.8395 |
| Macro precision | 0.8280 |
| Macro recall | 0.8285 |
| Macro F1 | 0.8273 |

控制台显示的 `Overall accuracy: 0.840` 是对精确值 0.8395 的四舍五入。

## 9. 各类别测试结果

| 类别 | Precision | Recall | F1 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| Walk | 0.8401 | 0.8718 | 0.8556 | 1,856 |
| Bike | 0.8052 | 0.8601 | 0.8317 | 908 |
| Bus | 0.8859 | 0.7945 | 0.8377 | 1,241 |
| Car | 0.8943 | 0.8600 | 0.8768 | 836 |
| Train | 0.7146 | 0.7562 | 0.7348 | 480 |

Train 是表现最弱的类别，14.58% 的 Train 样本被误判为 Walk。其他主要混淆包括：

- Bike → Walk：10.35%
- Bus → Walk：8.86%
- Walk → Bike：6.73%
- Car → Bus：6.82%

## 10. 测试混淆矩阵（计数）

| True \ Pred | Walk | Bike | Bus | Car | Train |
| --- | ---: | ---: | ---: | ---: | ---: |
| Walk | 1,618 | 125 | 43 | 10 | 60 |
| Bike | 94 | 781 | 16 | 4 | 13 |
| Bus | 110 | 41 | 986 | 48 | 56 |
| Car | 34 | 10 | 57 | 719 | 16 |
| Train | 70 | 13 | 11 | 23 | 363 |

本次测试的完整本地产物：

- `experiments/geolife_upstream_epoch25_test/classification_results.xlsx`
- `experiments/geolife_upstream_epoch25_test/configuration.json`
- `experiments/geolife_upstream_epoch25_test/data_indices.json`
- `experiments/geolife_upstream_epoch25_test/output.log`

`experiments/` 已被 Git 忽略，模型、日志和 Excel 文件不会提交到仓库。

## 11. 复现状态判定

| 项目 | 状态 |
| --- | --- |
| 环境部署 | 通过 |
| CUDA/GPU 验证 | 通过 |
| S1–S4 上游预处理 | 通过 |
| 全量训练启动 | 通过 |
| 最佳 checkpoint 保存 | 通过 |
| 最佳 checkpoint 有限值检查 | 通过 |
| 独立测试 | 通过 |
| 200 epoch 完整稳定训练 | 未通过：epoch 40 数值发散 |
| EarlyStopping 非有限指标处理 | 未通过 |
| 最终用户互斥变采样论文协议 | 尚未实施 |

因此，本阶段可以认定为：**上游数据与模型基线复现成功，并发现一个需要单独修复的训练稳定性缺陷。**

## 12. 后续工作

1. 从 `develop` 新建独立修复分支。
2. EarlyStopping 遇到 NaN/Inf 时立即停止或明确判定为无改进。
3. 在训练循环中增加非有限 loss 检测，防止继续更新并覆盖有效状态。
4. 保留原始 RAdam、学习率和 seed 配置作为忠实上游基线。
5. 将数值稳定性修改作为单独的兼容性实验，不与原始基线结果混合。
6. 完成实验报告提交及 PR 后，再进入用户互斥、变采样率论文协议实现阶段。
