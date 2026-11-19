# Fault diagnosis in power converters

Research code accompanying the [variable-load frequency-response manuscript](https://github.com/igmsalinas/variable-load-fra-paper). It implements PSIM sweep generation, complex spectral preprocessing, self-supervised training and anomaly scoring for synchronous buck converters.

- The [manuscript repository](https://github.com/igmsalinas/variable-load-fra-paper) contains the maintained IEEE draft, figures, tables and reproducibility review. The [in-repository copy](papers/variable_load_fra_ad) is an archived draft.
- The [data repository](https://github.com/igmsalinas/fault-converters-data) contains the variable-load sweeps and manifest, frozen split and checkpoints. See its README for selective Git LFS retrieval.
- The [publication results](papers/results/mlwa_study_dataset02) include the final ten-seed table calibrated on healthy validation data and the supporting per-seed records.

## Installation and source checks

Use Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

~~~sh
git clone https://github.com/igmsalinas/fault-converters-code.git
cd fault-converters-code
uv sync --locked
uv run --locked python scripts/audit_variable_load_manuscript.py
uv run --locked pytest -q
~~~

The audit checks the archived variable-load draft and the tracked result summaries. TensorRT and CUDA dependencies target Linux x86-64.

## Verify the archived results

The baseline publication snapshots are [code at 9e13ccd](https://github.com/igmsalinas/fault-converters-code/tree/9e13ccdd263346e2eadd15102fe096393a64d38a) and [data at 4950ecf](https://github.com/igmsalinas/fault-converters-data/tree/4950ecf604fab8b63395201b2d4a28af88fae8e8). Pair the splits, manifests and weights from the same snapshot.

1. Follow the data README to retrieve the split and selected experiment files. The primary variable-load sweeps are tracked directly in Git. The in-tree Dataset 02 copy matches the data repository's 800 sweeps, manifest and metadata after normalizing line endings.
2. The paper’s fixed-load comparison uses the saved test scores and evaluation provenance in the data repository’s corrected Dataset 01 campaign. The exact raw fixed-load cohort remains to be archived.
3. For a full variable-load run, inspect `python scripts/run_dataset02_mlwa_benchmark.py --help`. The runner accepts data, frozen-configuration, study, cache and output directories. The [extended study runner](scripts/run_dataset02_extended_study.py) and [final analysis runner](scripts/finalize_simplified_mlwa_study.py) use the standard archive directory layout, can train on a cache miss, and write analysis outputs. Use a scratch checkout for those runs. Retrieve all checkpoint components used in a run, including portable NumPy weights.

The variable-load protocol generates one single-component negative per anchor from a nominal-load conduction-loss transfer ratio. Its final score uses distance to five healthy training embeddings and a threshold set on healthy validation sweeps.

## Source layout

- [data/](data) contains simulator inputs and generation tools.
- [src/](src) contains loaders, models, scoring and deployment.
- [scripts/](scripts) contains research runners, figure generation and audits.
- [papers/results/](papers/results) contains derived result records.
- [tests/](tests) covers the Python workflows.
