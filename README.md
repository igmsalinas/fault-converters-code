# Fault diagnosis in power converters

Research code for detecting component degradation from complex frequency responses of synchronous buck converters, with a companion Jetson Nano deployment study. The code implements PSIM sweep generation, complex spectral preprocessing, self-supervised training and post-training anomaly scoring.

## Companion repositories

- [Variable-load paper](https://github.com/igmsalinas/variable-load-fra-paper) is the maintained IEEE manuscript, with figures, tables and the reproducibility review. The copy under [papers/variable_load_fra_ad](papers/variable_load_fra_ad) is an archived draft.
- [Data and archived experiments](https://github.com/igmsalinas/fault-converters-data) contains the variable-load sweeps and manifest, frozen splits, checkpoints, per-run records and deployment artifacts. GitHub permissions are required. See its README for selective Git LFS retrieval.

| Study | Source and evidence |
| --- | --- |
| Fixed-load detection and negative-sampler ablation | [MMAS draft](papers/mmas_anomaly_detection/main.tex). The corrected Dataset 01 campaign records are in the data repository under `experiments/dataset_01_run_06_corrected`. |
| Embedded deployment and inference fidelity | [Jetson Nano draft](papers/jsc_edge_deployment/main.tex). The same corrected campaign contains model-specific scorers, precision exports and target reports. |
| Detection under variable load | [IEEE draft](https://github.com/igmsalinas/variable-load-fra-paper). The [variable-load result tables and per-seed records](papers/results/mlwa_study_dataset02) are in this repository. The primary result uses the final ten-seed healthy-only calibration table. |

## Installation and source checks

Use Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

~~~sh
git clone https://github.com/igmsalinas/fault-converters-code.git
cd fault-converters-code
uv sync --locked
uv run --locked python scripts/audit_publication_artifacts.py --papers-only
uv run --locked python scripts/audit_variable_load_manuscript.py
uv run --locked pytest -q
~~~

The first paper audit checks the two fixed-load drafts. The second checks the archived variable-load draft and tracked result summaries. TensorRT and CUDA dependencies target Linux x86-64. Existing Jetson Nano binaries and benchmarks depend on their recorded JetPack and runtime versions. The Nano setup uses [its separate requirements](requirements-jetpack46.txt).

## Verify the archived results

The baseline publication snapshots are [code at 9e13ccd](https://github.com/igmsalinas/fault-converters-code/tree/9e13ccdd263346e2eadd15102fe096393a64d38a) and [data at 4950ecf](https://github.com/igmsalinas/fault-converters-data/tree/4950ecf604fab8b63395201b2d4a28af88fae8e8). Pair the splits, manifests and weights from the same snapshot.

1. Follow the data README to retrieve the split and selected experiment files. The primary variable-load sweeps are tracked directly in Git. The in-tree Dataset 02 copy matches the data repository's 800 sweeps, manifest and metadata after normalizing line endings.
2. For the fixed-load archive audit, make the hydrated `dataset_01_run_06_corrected` directory available under `experiments/` in the code checkout and run `python scripts/audit_publication_artifacts.py` in the project environment. The archive provides stored test scores and evaluation provenance. Exact raw Dataset 01 sweeps are pending archival.
3. Run `python scripts/generate_latest_results_figures.py` in a scratch checkout with the corrected campaign to regenerate the companion figure source from archived outputs.
4. For a full variable-load run, inspect `python scripts/run_dataset02_mlwa_benchmark.py --help`. The runner accepts data, frozen-configuration, study, cache and output directories. The [extended study runner](scripts/run_dataset02_extended_study.py) and [final analysis runner](scripts/finalize_simplified_mlwa_study.py) use the standard archive directory layout, can train on a cache miss, and write analysis outputs. Use a scratch checkout for those runs. Retrieve all checkpoint components used in a run, including portable NumPy weights.

The variable-load protocol generates one single-component negative per anchor from a nominal-load conduction-loss transfer ratio. Its final score uses distance to five healthy training embeddings and a threshold set on healthy validation sweeps. The fixed-load publications use model-specific archived scorers and select thresholds using fit-validation labels. Keep these evaluation protocols distinct when comparing metrics.

## Source layout

- [data/](data) contains simulator inputs and generation tools.
- [src/](src) contains loaders, models, scoring and deployment.
- [scripts/](scripts) contains research runners, figure generation and audits.
- [papers/results/](papers/results) contains derived result records.
- [tests/](tests) covers the Python workflows.
