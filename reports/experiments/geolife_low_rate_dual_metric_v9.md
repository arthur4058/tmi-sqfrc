# GeoLife 60 秒低采样率双指标验收（V9）

## 结论

在同一用户互斥测试集、同一随机种子和同一组冻结 checkpoint 下，三专家类别感知融合相对配对 B0 同时达到预先规定的两个门槛：

| 模型 | Accuracy | Macro-F1 | Accuracy 相对 B0 | Macro-F1 相对 B0 |
|---|---:|---:|---:|---:|
| 配对 B0 双分支 | 71.5535% | 63.7346% | - | - |
| 类别感知三专家融合 | 72.5702% | 64.9453% | +1.0167 个百分点 | +1.2107 个百分点 |

验收标准是 Accuracy 和 Macro-F1 均至少提升 1 个百分点，最终结果为 **通过**。

## 为什么此前会判断为未通过

此前人工比较使用了另一份显示为 `71.61% / 63.55%` 的 B0 汇总值。它不是本次融合结果文件中同时计算得到的严格配对 B0，因此把两个实验的数字相减会混入运行差异。重新执行冻结的 V8 测试后，结果与原始 JSON 完全一致，并确认正确参照是 `71.5535% / 63.7346%`。

V9 不重新训练或根据测试集选择参数，而是新增自动验收器，强制要求：

1. B0 与候选模型来自同一个结果文件；
2. 采样间隔必须为 60 秒；
3. 测试样本必须是冻结用户互斥划分中的 4,918 条；
4. 明确记录测试阶段没有拟合；
5. 文件中的增量必须与两组原始指标重新计算的增量一致；
6. 两个指标必须各自达到 0.01，不能用一个指标补偿另一个指标。

## 实验协议

- 数据：GeoLife，60 秒稀疏视图
- 划分：用户互斥 train/validation/test
- 随机种子：10086
- 候选方法：冻结 B0、4 特征专家和 7 特征运动专家；温度校准后，用只在验证集选择的多项逻辑回归进行类别感知融合
- 选择：5 折分层验证集 OOF；测试集不参与温度、超参数或融合权重选择
- 测试：冻结校准产物后仅执行一次独立测试推理

## 可复现性

重新执行测试：

```bash
python -m scripts.evaluate_class_aware_stacker test \
  --rate 60 \
  --artifact experiments/geolife_class_aware_stacker_v8_60s_seed10086/calibration.json \
  --result experiments/geolife_class_aware_stacker_v9_60s_seed10086/test_results.json
```

执行双指标验收：

```bash
python -m scripts.verify_dual_metric_acceptance \
  experiments/geolife_class_aware_stacker_v9_60s_seed10086/test_results.json \
  --rate 60 \
  --expected-samples 4918 \
  --minimum-delta 0.01
```

冻结 checkpoint SHA-256：

- B0：`e02cf4ecfeadb214ae48ea73c617edc423044ae2dd1f70d93705ada269234e13`
- 4 特征专家：`279110d6c376935df5f27d488f3bd35cdee51c3903e2fcff41fb56632a5148ad`
- 7 特征运动专家：`7056230e1e90b8b65d163aa43537fe47e9264471c25338f23a3b08375d8a9b7d`

## 结果边界

这个结论只回答“在冻结的 60 秒、seed 10086 配对实验中，是否相对 B0 同时提升至少 1 个百分点”。它不表示该方法优于所有先前版本：V2 的 Macro-F1 为 66.0042%，仍高于当前方法的 64.9453%。后续论文结论应补充多随机种子统计，而不能继续用同一个测试集挑选新结构。
