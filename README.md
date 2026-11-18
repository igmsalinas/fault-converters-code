# Fault Converters: Self-Supervised Anomaly Detection & Edge Deployment for Power Electronics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open Source Contribution](https://img.shields.io/badge/Open%20Source-fault__converters-brightgreen.svg)](https://github.com/igmsalinas/fault_converters)

An open-source research and engineering framework for physics-informed, self-supervised anomaly detection on power electronic converters (PECs) using frequency-domain transfer function (Bode) signatures, accompanied by an edge compilation and hardware-in-the-loop benchmarking suite for embedded accelerators (NVIDIA Jetson Nano).

Project repository: [https://github.com/igmsalinas/fault_converters](https://github.com/igmsalinas/fault_converters)

---

## 📄 Research Publications

This repository houses the source code, simulation digital twins, training pipelines, and reproducible benchmarking suites for two companion research papers:

### 1. Paper A: Representation Learning & Physics-Informed Anomaly Detection
> **"Self-Supervised Contrastive Anomaly Detection in Power-Converter Frequency-Response Signatures"**  
> *Ignacio Martin-Salinas, Jose A. Belloch, Adrián Amor-Martín, Pablo Zumel, Cristina Fernandez* (Universidad Carlos III de Madrid)  
> Manuscript directory: [`papers/mmas_anomaly_detection/`](papers/mmas_anomaly_detection/)

- **Physics-Informed Negative Sampler**: Replaces domain-agnostic heuristics with analytic small-signal transfer ratios ($\tilde{H}(s) / H^{(0)}(s)$), synthesizing physically consistent hard negatives where magnitude and phase evolve coherently under circuit laws.
- **Formal Set-Membership Formulation**: Grounds normal and anomalous states in non-dimensional component multiplier bands derived from manufacturing tolerances and accelerated aging literature.
- **Encoder Family Benchmark**: Evaluates Conv1D, LSTM, GRU, MLP, and Transformer encoders under a unified hyperparameter search and non-parametric $k$-NN reference bank scoring protocol ($n=1{,}268$ held-out test states).
- **Key Results**: Conv1D-AE achieves the highest F1-score (**0.8897**) and PR-AUC (**0.9570**). In paired ablations with identical initial weight states, physics-informed negative synthesis improves mean F1 by **$+0.0181$** and recall by **$+0.0339$** over heuristic synthesis.

### 2. Paper B: Edge Optimization, Operator Lowering & Energy Characterization
> **"Validated TensorRT Optimization of Self-Supervised Power-Converter Anomaly Detectors on Jetson Nano"**  
> *Ignacio Martin-Salinas, Jose A. Belloch, Adrián Amor-Martín, Pablo Zumel, Cristina Fernandez* (Universidad Carlos III de Madrid)  
> Manuscript directory: [`papers/jsc_edge_deployment/`](papers/jsc_edge_deployment/)

- **Cross-Runtime Optimization Matrix**: 45 distinct model–runtime–precision pathways spanning Keras (CPU/GPU), TensorFlow Lite, ONNX Runtime, and NVIDIA TensorRT in FP32, FP16, and INT8 precisions.
- **Single-Sample ($n=1$) Streaming Protocol**: 15 warm-up iterations, 150 timed executions using monotonic host clocks for CPU and CUDA execution stream events for GPU.
- **Downstream Decision Acceptance Gate**: Enforces a $\le 1.0\%$ threshold decision discordance rule on the full test set, identifying 37 validated and 8 fidelity-rejected lowering paths.
- **Rail-Level Power & Energy Telemetry**: Direct 10-Hz INA3221 sensor profiling via PMLib across module input, GPU, and CPU power rails.
- **Key Results**: MLP in TFLite FP32 delivers the lowest latency (**0.168 ms**) and energy consumption (**6.33 mJ/inference**). TensorRT FP16 accelerates recurrent models by **$43\times$** (GRU, 29.20 ms) and **$52\times$** (LSTM, 19.81 ms) with $16\times$--$21\times$ energy savings. INT8 is shown to fail due to hardware lack of DP4A on Maxwell Tegra, parser QDQ import errors, and CPU decision distortion.

---

## 📁 Repository Structure

```
fault_converters/
├── data/                                # PSIM schematics, component ranges, simulation data
│   └── buck/
│       ├── buck.psimsch                 # Reference PSIM circuit schematic
│       ├── component_ranges.json        # Component tolerance & fault multiplier intervals
│       ├── parameters.txt               # Nominal physical baseline parameters
│       └── buck_data/                   # Raw simulation manifests and frequency responses
├── docs/                                # Technical documentation, parameter specs, references
│   ├── CONVERTER_PARAMETERS.md          # Circuit parameters and physical modeling
│   ├── FUTURE_RESEARCH.md               # Research roadmap (Mamba, KAN, PINN loss)
│   ├── REFERENCES.md                    # Degradation literature and component standards
│   ├── TECHNICAL_DOCUMENTATION.md       # Comprehensive architectural & algorithmic guide
│   └── plots/                           # Generated figures and architecture diagrams
├── experiments/                         # Saved checkpoints, weights, preprocessors, metrics
├── notebooks/                           # Interactive exploration and diagnostic dashboards
│   └── explore_results.ipynb            # Result analysis, error distributions, ROC/PR curves
├── papers/                              # LaTeX paper manuscripts and benchmarking results
│   ├── mmas_anomaly_detection/          # Paper A LaTeX source, figures, and bib
│   ├── jsc_edge_deployment/             # Paper B LaTeX source, figures, and bib
│   └── results/                         # Raw CSV result files for tables and plots
├── scripts/                             # Automation scripts for training, evaluation, edge runs
│   ├── run_corrected_jetson_pipeline.sh # Full target-side Jetson Nano compilation & benchmarking
│   ├── run_negative_sampler_ablation.py # Paired physics-informed vs heuristic ablation
│   ├── train_all.sh                     # Full multi-architecture training pipeline
│   ├── evaluate_all.sh                  # Evaluation of all trained experiments
│   └── generate_synthetic_buck.py       # Standalone synthetic dataset generator (no PSIM)
├── src/                                 # Modular Python source library
│   ├── data/                            # Loaders, preprocessors, LHS/grid, negative synthesis
│   ├── models/                          # BaseAutoencoder, Conv1D, LSTM, GRU, MLP, Transformer, VAE, CARLA
│   ├── losses/                          # NT-Xent contrastive, triplet, composite CARLA losses
│   ├── training/                        # BaseTrainer, CarlaTrainer, Random-search HPO
│   ├── evaluation/                      # Metrics, threshold calibration, Bode & PCA visualizers
│   ├── inference/                       # AnomalyPredictor, k-NN reference bank scoring
│   └── deployment/                      # TFLite, ONNX, TensorRT, Vitis AI export & benchmarking
└── tests/                               # Comprehensive pytest unit and integration test suite
```

---

## 🚀 Getting Started

### Installation & Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency resolution.

```bash
# 1. Clone the repository
git clone https://github.com/igmsalinas/fault_converters.git
cd fault_converters

# 2. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment and sync dependencies
uv sync

# 4. Activate the virtual environment
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows
```

---

## 🔬 Workflow & Usage

### 1. Dataset Generation

The dataset represents small-signal control-to-output transfer functions $G_{vd}(s) = \hat{v}_{\mathrm{out}}(s) / \hat{d}(s)$ across 101 log-spaced frequencies ($100\text{ Hz}$ to $50\text{ kHz}$).

- **Option A: Synthetic Digital Twin (No PSIM Required)**  
  Generate a physically consistent buck converter dataset directly from the analytic small-signal equations:
  ```bash
  python scripts/generate_synthetic_buck.py \
      --out data/buck/buck_data \
      --n-normal 1000 \
      --n-fault 500
  ```

- **Option B: High-Fidelity PSIM Simulation**  
  Execute automated PSIM batch simulations using Latin Hypercube Sampling (LHS):
  ```bash
  python data/generate_data.py --converter buck --normal-mode lhs --fault-mode lhs --n-normal 10000 --n-fault 2000
  ```

### 2. Model Training & Hyperparameter Search

Train standard autoencoders or self-supervised CARLA contrastive models. All models execute an automated hyperparameter search over layer topologies, latent dimensions, and regularization, followed by a final training run.

```bash
# Train a standard Conv1D Autoencoder
python -m src.train --model conv1d_ae --n-trials 20 --final-epochs 100 --data-dir data/buck/buck_data

# Train Physics-Informed CARLA Contrastive Autoencoder
python -m src.train --model carla --n-trials 20 --final-epochs 150 --data-dir data/buck/buck_data

# Fast debug run
python -m src.train --model carla --debug
```

### 3. Model Evaluation

Evaluate trained models on held-out test partitions and generate standardized artifacts (`metrics.json`, `threshold.json`, `test_results.csv`):

```bash
# Evaluate single model
python -m src.evaluate --model-dir experiments/dataset_00_run_01/carla

# Evaluate all experiments in directory
bash scripts/evaluate_all.sh
```

### 4. Edge Compilation & Hardware Benchmarking

Export trained models to edge runtimes (TFLite, ONNX, TensorRT, Vitis AI) and execute latency, memory, and classification fidelity benchmarking:

```bash
# 1. Export and optimize model across formats
python -m src.deployment_optimization \
    --model-dir experiments/dataset_00_run_01/carla \
    --data-dir data/buck/buck_data \
    --run-all

# 2. Benchmark single-sample (n=1) latency, memory, and decision concordance
python -m src.deployment_evaluation \
    --model-dir experiments/dataset_00_run_01/carla \
    --data-dir data/buck/buck_data
```

To run the complete Jetson Nano automated pipeline (model conversion, engine compilation, fidelity gates, and PMLib INA3221 rail power profiling):
```bash
bash scripts/run_corrected_jetson_pipeline.sh
```

---

## 🧪 Testing & Code Quality

The codebase enforces strict Test-Driven Development (TDD), high unit-test coverage, and PEP 8 linting.

```bash
# Run pytest test suite
uv run pytest

# Run with test coverage report
uv run pytest --cov=src tests/

# Formatting and linting
uv run ruff check .
uv run ruff format .
uv run isort .
```

---

## 📊 Summary of Main Benchmark Findings

| Architecture | Model Family | Test F1-Score | PR-AUC | Optimal Edge Runtime | Batch-1 Latency (ms) | Energy (mJ/inf.) |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **Conv1D-AE** | Convolutional | **0.8897** | **0.9570** | TFLite FP16 | 0.492 | 8.41 |
| **MLP-AE** | Feedforward | 0.8815 | 0.9563 | TFLite FP32 | **0.168** | **6.33** |
| **Transformer-AE**| Self-Attention| 0.8808 | 0.9538 | TensorRT FP32 | 2.993 | 28.12 |
| **LSTM-AE** | Recurrent | 0.8442 | 0.9385 | TensorRT FP16 | 19.814 | 148.07 |
| **GRU-AE** | Recurrent | 0.8218 | 0.9290 | TensorRT FP16 | 29.196 | 239.82 |

---

## 📚 Citation

If you use this codebase, models, datasets, or benchmarking methodology in your research, please cite our open-source project and publications:

```bibtex
@misc{martinsalinas2026faultconverters,
  author = {Martin-Salinas, Ignacio and Belloch, Jose A. and Amor-Martin, Adrian and Zumel, Pablo and Fernandez, Cristina},
  title = {Fault Converters: Self-Supervised Anomaly Detection and Edge Deployment for Power Electronic Converters},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/igmsalinas/fault_converters}}
}

@article{martinsalinas2026carla_pec,
  author = {Martin-Salinas, Ignacio and Belloch, Jose A. and Amor-Martin, Adrian and Zumel, Pablo and Fernandez, Cristina},
  title = {Self-Supervised Contrastive Anomaly Detection in Power-Converter Frequency-Response Signatures},
  journal = {Under review},
  year = {2026}
}

@article{martinsalinas2026jetson_tensorrt,
  author = {Martin-Salinas, Ignacio and Belloch, Jose A. and Amor-Martin, Adrian and Zumel, Pablo and Fernandez, Cristina},
  title = {Validated TensorRT Optimization of Self-Supervised Power-Converter Anomaly Detectors on Jetson Nano},
  journal = {Under review},
  year = {2026}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
