# SORF-TMI model complexity benchmark

> Device: NVIDIA GeForce RTX 5070
> PyTorch: 2.7.1+cu128
> CUDA runtime: 12.8
> Batch size: 64
> Warm-up: 40 iterations; measurement: 5 repeats × 120 iterations

FLOPs use PyTorch `FlopCounterMode` for one forward sample. Latency and peak allocated CUDA memory use actual test tensors already resident on the GPU; data loading and host-to-device transfer are excluded.

| Rate | Model | Parameters | GFLOPs/sample | Batch latency (ms) | Latency/sample (ms) | Throughput (sample/s) | Peak memory (MiB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 s | B0 | 540,937 | 0.0110 | 1.727 | 0.0270 | 37051.2 | 13.9 |
| 30 s | Relation expert | 331,497 | 0.0017 | 1.261 | 0.0197 | 50756.2 | 29.6 |
| 30 s | SORF-TMI | 872,434 | 0.0127 | 2.802 | 0.0438 | 22842.8 | 31.8 |
| 60 s | B0 | 540,937 | 0.0060 | 1.863 | 0.0291 | 34348.0 | 12.7 |
| 60 s | Relation expert | 331,497 | 0.0007 | 1.024 | 0.0160 | 62512.2 | 15.1 |
| 60 s | SORF-TMI | 872,434 | 0.0066 | 2.853 | 0.0446 | 22435.4 | 17.2 |

## Interpretation

SORF-TMI executes B0 and the relation expert and then fuses their probabilities, so its parameter count and compute are necessarily above B0. The relation expert is cheaper at 60 seconds because six real anchors generate fewer pairwise tokens than the eleven anchors at 30 seconds. The benchmark supports an accuracy-efficiency trade-off claim; it does not support a claim that SORF-TMI is faster than B0.
