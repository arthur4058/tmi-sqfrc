# V6 配对多采样率一致性实验报告

## 1. 结论

V6 已完成代码实现、单元测试、GPU 冒烟、60 秒正式训练和独立测试。

结果呈现明显的指标分化：Accuracy 相对 B0 提升 0.74 个百分点，但 Macro-F1 下降 1.49 个百分点。由于预先规定 Accuracy 与 Macro-F1 必须同时至少提升 0.5 个百分点，V6 判定为**未通过**，不扩展到其余采样档。

V6 证明“同窗口稠密/稀疏联合监督”能够提高总体正确数，但当前损失会偏向样本较多的类别，尚不能作为论文最终模块。

## 2. 与 V3/V3.1 的区别

- V3/V3.1：冻结 5 秒教师，单向指导 60 秒学生。
- V6：只有一个共享模型；5 秒和 60 秒视图都接受真实标签监督，并使用对称 Jensen–Shannon 一致性约束。
- V6 不加载教师 checkpoint，不增加推理参数；测试阶段仍只输入 60 秒视图。

## 3. 固定协议

- 起点提交：`4ffb01d`
- 分支：`feat/paired-multirate-consistency-v6`
- 稀疏训练/验证/测试数据：`geolife_v3_fixed_60s`
- 配对稠密训练视图：`geolife_v3_fixed_5s`
- 配对训练样本：45,672
- 独立测试样本：4,918
- 用户互斥划分：保持不变
- 随机种子：10086
- 最大轮数：200
- 早停 patience：40
- 优化器：RAdam
- 学习率：0.001
- Batch size：64
- 稀疏监督权重：0.7
- 稠密监督权重：0.3
- 一致性权重：0.2
- 一致性 ramp：前 10 轮从 0 线性增至 1
- 模型参数量：540,937，与 B0 完全相同

## 4. 训练目标

同一物理窗口的 5 秒和 60 秒视图共享模型参数：

```text
L = 0.7 * CE(sparse, y)
  + 0.3 * CE(dense, y)
  + 0.2 * ramp(epoch) * JS(p_sparse, p_dense)
```

两种视图使用同一个噪声选择结果，避免把一致性任务变成额外的去噪任务。两种序列保持各自自然长度，避免把 60 秒序列补齐至 5 秒长度后污染卷积池化；前向顺序按 batch 交替，降低 BatchNorm 顺序偏差。

## 5. 预设门槛

同配对数据 B0：

- Accuracy：0.6846
- Macro-F1：0.6293

V6 必须同时达到：

- Accuracy ≥ 0.6896
- Macro-F1 ≥ 0.6343

只有两项同时通过才扩展其它采样档。

## 6. 独立测试结果

| 模型 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| B0 | 0.6846 | - | - | 0.6293 |
| V6 | 0.6919 | 0.6271 | 0.6439 | 0.6144 |
| V6 - B0 | **+0.0074** | - | - | **-0.0149** |

判定：Accuracy 通过门槛，Macro-F1 未通过，整体未通过。

### 各类别结果

| 类别 | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|
| Walk | 0.7815 | 0.9417 | 0.8542 | 1,356 |
| Bike | 0.5090 | 0.7881 | 0.6186 | 571 |
| Bus | 0.4625 | 0.6097 | 0.5260 | 638 |
| Car | 0.8692 | 0.5901 | 0.7029 | 2,015 |
| Train | 0.5131 | 0.2899 | 0.3705 | 338 |

总体准确率主要受 Walk、Bike 和 Car 改善推动，但 Train recall 只有 0.2899，导致 Macro-F1 明显下降。该结果说明当前普通交叉熵联合监督仍受类别不平衡影响。

## 7. 训练与验证

- 正式训练在 epoch 41 后因 patience=40 早停。
- 最佳 checkpoint：epoch 1。
- 最佳验证 Accuracy：0.7188。
- 最佳验证 Macro-F1：0.6231。
- 正式训练耗时：533.79 秒。
- checkpoint SHA-256：`f343c278dea509b0f7427f9e84a70653cfcf5c7020a05ebf7125722dd9ebb8d7`
- 正式测试使用 `model_best.pth`，不是最后一轮模型。

## 8. 验证记录

- V6 专项单元测试：4/4 通过。
- 全套回归测试：15/15 通过。
- Python 编译检查：通过。
- `pip check`：通过。
- `git diff --check`：通过。
- 配对窗口、标签一致性检查：通过。
- GPU 正反向冒烟：通过，无 NaN/Inf。

## 9. 下一步建议

V6 比 V3/V3.1 更接近有效方向，因为 Accuracy 已显著转正，但下一步不能只补 seed，也不应直接加大一致性权重。更合理的 V6.1 是只解决当前暴露出的类别失衡：

1. 保持 V6 配对结构和一致性权重不变。
2. 将两项监督交叉熵替换为按训练类别频率计算的 class-balanced loss，或使用 balanced sampler。
3. 仍然先跑 60 秒单 seed，要求 Accuracy 不低于 V6，同时 Macro-F1 超过 B0。
4. 只有通过后再扩展五档和多 seed。

## 10. 可复现命令

```bash
python main.py --config configs/five_rate/geolife_v6_paired_consistency_60s_train.json
python main.py --config configs/five_rate/geolife_v6_paired_consistency_60s_test.json
```

数据、checkpoint 和 `experiments/` 输出保留本地，不提交 Git。
