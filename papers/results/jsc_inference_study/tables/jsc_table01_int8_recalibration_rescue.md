**Table J1. Post-audit INT8 quantization rescue across 5 CARLA architectures via input-clipping saturation, INT8 embedding-bank recalibration, and $M=32$ prototype compression (F1 and FP32 decision discordance %).**

| Architecture | INT8 Runtime | Naive FP32 Bank F1 (Disc.) | Recal. `k-NN (k=5)` Zero-`p95` F1 (Disc.) | Recal. `Proto-32` Val-Opt F1 | Recal. `Proto-32` Zero-`p95` F1 | Detector Size [MB] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Conv1D` | `TFLite INT8` | 0.7421 (51.0%) | **0.8425** (6.3%) | **0.8513** | 0.8408 | 5.04 $\to$ **0.37** |
| `Conv1D` | `ONNX INT8` | 0.7421 (51.0%) | **0.8518** (9.3%) | **0.8104** | 0.8003 | 5.07 $\to$ **0.41** |
| `MLP` | `TFLite INT8` | 0.7421 (49.8%) | **0.8287** (6.7%) | **0.8107** | 0.7978 | 4.90 $\to$ **0.23** |
| `MLP` | `ONNX INT8` | 0.7421 (49.8%) | **0.8248** (8.6%) | **0.8367** | 0.8138 | 4.94 $\to$ **0.28** |
| `Transformer` | `TFLite INT8` | 0.7421 (50.5%) | **0.8087** (9.8%) | **0.8410** | 0.8367 | 1.56 $\to$ **0.39** |
| `Transformer` | `ONNX INT8` | 0.7421 (50.5%) | **0.7808** (13.3%) | **0.8018** | 0.8003 | 1.66 $\to$ **0.50** |
| `LSTM` | `TFLite INT8` | 0.7421 (52.4%) | **0.2806** (39.4%) | **0.5312** | 0.0957 | 5.30 $\to$ **0.64** |
| `LSTM` | `ONNX INT8` | 0.7421 (52.4%) | **0.7991** (10.7%) | **0.7961** | 0.7788 | 6.78 $\to$ **2.11** |
| `GRU` | `TFLite INT8` | 0.7421 (54.6%) | **0.7708** (10.6%) | **0.7846** | 0.7684 | 4.06 $\to$ **1.73** |
| `GRU` | `ONNX INT8` | 0.7421 (54.6%) | **0.7786** (6.2%) | **0.7708** | 0.7482 | 7.70 $\to$ **5.37** |
