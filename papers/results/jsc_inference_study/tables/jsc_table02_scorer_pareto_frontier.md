**Table J2. Embedding anomaly-scorer Pareto trade-off between reference-bank memory, median CPU scoring latency ($p_{50}$), and zero-label `p95` F1.**

| Backbone | Scorer Variant | Vectors / Rank | Bank [MB] | Total [MB] | Scorer $p_{50}$ [ms] | $\mathrm{ROC\text{-}AUC}$ | Zero-`p95` F1 |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: |
| `Conv1D` | `Legacy Sklearn k-NN (N=2419)` | 2419 | 4.7246 | 5.923 | 262.215 | 0.9160 | 0.8823 |
| `Conv1D` | `Vectorized BLAS k-NN (N=2419)` | 2419 | 4.7246 | 5.923 | 11.965 | 0.9160 | 0.8823 |
| `Conv1D` | `Ridge Mahalanobis (d=513)` | 513 | 1.0020 | 2.200 | 0.051 | 0.9157 | 0.8867 |
| `Conv1D` | `Ledoit-Wolf Shrinkage (d=513)` | 513 | 1.0020 | 2.200 | 0.037 | 0.8849 | 0.8172 |
| `Conv1D` | **`Prototype k-NN (M=128)`** | 128 | 0.2500 | 1.448 | 0.069 | 0.9063 | 0.8744 |
| `Conv1D` | **`Prototype k-NN (M=64)`** | 64 | 0.1250 | 1.323 | 0.063 | 0.9002 | 0.8622 |
| `Conv1D` | **`Prototype k-NN (M=32)`** | 32 | 0.0625 | 1.261 | 0.060 | 0.8911 | 0.8375 |
| `Conv1D` | `Prototype k-NN (M=16)` | 16 | 0.0312 | 1.229 | 0.059 | 0.8891 | 0.8216 |
| `Conv1D` | `Single Centroid (M=1)` | 1 | 0.0020 | 1.200 | 0.015 | 0.8688 | 0.7991 |
| `MLP` | `Legacy Sklearn k-NN (N=2419)` | 2419 | 4.7246 | 5.384 | 292.342 | 0.9049 | 0.8646 |
| `MLP` | `Vectorized BLAS k-NN (N=2419)` | 2419 | 4.7246 | 5.384 | 13.026 | 0.9049 | 0.8646 |
| `MLP` | `Ridge Mahalanobis (d=513)` | 513 | 1.0020 | 1.661 | 0.037 | 0.9161 | 0.8815 |
| `MLP` | `Ledoit-Wolf Shrinkage (d=513)` | 513 | 1.0020 | 1.661 | 0.038 | 0.8508 | 0.7666 |
| `MLP` | **`Prototype k-NN (M=128)`** | 128 | 0.2500 | 0.909 | 0.070 | 0.8834 | 0.8232 |
| `MLP` | **`Prototype k-NN (M=64)`** | 64 | 0.1250 | 0.784 | 0.041 | 0.8727 | 0.7975 |
| `MLP` | **`Prototype k-NN (M=32)`** | 32 | 0.0625 | 0.722 | 0.060 | 0.8681 | 0.7810 |
| `MLP` | `Prototype k-NN (M=16)` | 16 | 0.0312 | 0.691 | 0.058 | 0.8561 | 0.7764 |
| `MLP` | `Single Centroid (M=1)` | 1 | 0.0020 | 0.661 | 0.014 | 0.8325 | 0.7732 |
