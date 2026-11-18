**Table J3. Fastest validated edge runtime per architecture on NVIDIA Jetson Nano (Maxwell 128-core GPU) vs. baseline `Keras FP32 (CPU)` latency.**

| Architecture | Fastest Validated Backend | Graph Latency [ms] | Speedup vs Keras CPU | Test F1 | $\mathrm{ROC\text{-}AUC}$ | Detector Size [MB] |
| :--- | :--- | ---: | ---: | :---: | :---: | ---: |
| `Conv1D` | `TFLite FP16` | **0.492** | 47.4$\times$ | 0.8905 | 0.9150 | 5.33 |
| `MLP` | `TFLite FP32` | **0.168** | 156.2$\times$ | 0.8815 | 0.9161 | 5.38 |
| `Transformer` | `TensorRT FP32 (GPU)` | **2.993** | 31.1$\times$ | 0.8808 | 0.9100 | 2.74 |
| `LSTM` | `ONNX FP32 (CPU)` | **14.772** | 70.0$\times$ | 0.8436 | 0.8835 | 6.96 |
| `GRU` | `TensorRT FP16 (GPU)` | **29.196** | 43.0$\times$ | 0.8218 | 0.8678 | 12.79 |
