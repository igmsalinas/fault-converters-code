# Corrected Training and Jetson Nano Campaign: Manual Runbook

This runbook reproduces the complete `dataset_01_run_06_corrected` study without
an automated assistant. It covers safe training resume, model verification,
workstation ablations, Jetson conversion and evaluation, PMLib measurements,
result retrieval, paper regeneration, audits, and the final Git/LFS commit.

The hardware split is strict:

- The workstation RTX 4070 is used only for model search and training.
- The Jetson Nano is used for detector/threshold selection, held-out test
  inference, format conversion, TensorRT engine construction, latency, power,
  RAM, CUDA-memory, utilization, and thermal measurements.
- Never report workstation inference as Jetson inference. Never build a
  TensorRT engine on the workstation for use on the Nano.

The commands assume:

```text
Workstation: /home/nacho/projects/fault_converters
Jetson:      /home/ignaciouc3m/projects/fault_converters
PMLib:       /home/ignaciouc3m/projects/pmlib
Experiment:  experiments/dataset_01_run_06_corrected
Dataset:     data/buck/buck_data/dataset_01
```

Credentials are intentionally not stored here. Enter SSH and `sudo` passwords
interactively. Do not use `--overwrite` in a recovery workflow.

## 1. Safely stop training before shutdown

On the workstation, terminate the whole training tmux session:

```bash
tmux kill-session -t fault_training
pgrep -af 'src.train.*dataset_01_run_06_corrected'
```

No `pgrep` output means it is safe to shut down normally. Do not send a single
`Ctrl+C` to the training wrapper: it may treat the active model as failed and
continue to the next one. An interrupted active trial restarts at epoch 1 with
its saved configuration and deterministic trial seed; completed trials remain
recorded in `search_progress.json`.

## 2. Resume the five-model search and training

On the workstation:

```bash
cd /home/nacho/projects/fault_converters
tmux new-session -d -s fault_training \
  "bash scripts/train_all.sh --carla-only \
  --experiment-name dataset_01_run_06_corrected \
  --data-dir data/buck/buck_data/dataset_01 \
  --output-dir experiments --n-trials 50 --batch-size 128 \
  --epochs-per-trial 30 --final-epochs 150 --seed 42 \
  >> experiments/dataset_01_run_06_corrected/training.log 2>&1"
```

Monitor it without attaching to the training process:

```bash
tail -f experiments/dataset_01_run_06_corrected/training.log
pgrep -af 'src.train.*dataset_01_run_06_corrected'
nvidia-smi
```

`Ctrl+C` while viewing `tail -f` stops only the log viewer. The wrapper skips an
architecture whose top-level `model_config.json` already exists and resumes a
partial search from its recorded completed trials.

## 3. Confirm that main training finished

The log must end with `All models trained successfully!`:

```bash
tail -50 experiments/dataset_01_run_06_corrected/training.log
find experiments/dataset_01_run_06_corrected \
  -mindepth 2 -maxdepth 2 -name model_config.json | sort
```

Exactly five model directories must be represented: `carla_conv1d`,
`carla_gru`, `carla_lstm`, `carla_mlp`, and `carla_transformer`.

Check the selected artifacts for every model:

```bash
for model in conv1d gru lstm mlp transformer; do
  model_dir="experiments/dataset_01_run_06_corrected/carla_${model}"
  test -s "$model_dir/search_results.json"
  test -s "$model_dir/checkpoints/best/autoencoder.weights.h5"
  test -s "$model_dir/checkpoints/best/reference_embeddings.npy"
  test -s "$model_dir/split_indices.npz"
  test -s "$model_dir/training_provenance.json"
  printf '%-12s trial directories: ' "$model"
  find "$model_dir/hp_tuning" -maxdepth 2 -type d -name 'trial_*' | wc -l
done
```

Each count must be 50. After training no longer occupies the GPU, run:

```bash
uv run pytest -q
```

Do not run final held-out inference on the workstation.

## 4. Train the paired negative-sampler ablation

These are six additional Conv1D training runs: physics and heuristic samplers
for three paired seeds. Training happens on the RTX workstation; scorer
selection and test inference happen later on the Nano.

```bash
tmux new-session -d -s fault_ablation \
  "cd /home/nacho/projects/fault_converters && \
  uv run python scripts/run_negative_sampler_ablation.py \
  --experiment experiments/dataset_01_run_06_corrected \
  --data-dir data/buck/buck_data/dataset_01 \
  --seeds 200042 200043 200044 --epochs 150 --batch-size 128 \
  --train-only \
  >> experiments/dataset_01_run_06_corrected/ablation_training.log 2>&1"

tail -f experiments/dataset_01_run_06_corrected/ablation_training.log
```

The command skips a replicate with `ablation_training_provenance.json`. If it
was interrupted inside a replicate, remove only that incomplete replicate
directory before restarting. There must eventually be six records:

```bash
find experiments/dataset_01_run_06_corrected/ablations/negative_sampler \
  -name ablation_training_provenance.json | sort
```

## 5. Align source and transfer trained artifacts

Record the workstation revision and ensure there are no unexpected tracked
modifications. Generated experiment directories may still be untracked.

```bash
cd /home/nacho/projects/fault_converters
git rev-parse HEAD
git status --short
ssh ignaciouc3m@192.168.55.1 \
  'cd /home/ignaciouc3m/projects/fault_converters && git rev-parse HEAD && git status --short'
```

The hashes must match. Preserve the Jetson's `.venv-jp46`, dataset, PMLib, and
generated outputs. Never use `git reset --hard` or `rsync --delete`.

If a committed source revision must be transferred without pushing:

```bash
git bundle create /tmp/fault_converters_source.bundle HEAD
scp /tmp/fault_converters_source.bundle \
  ignaciouc3m@192.168.55.1:/tmp/fault_converters_source.bundle
ssh ignaciouc3m@192.168.55.1
```

On the Jetson, only if tracked files are clean:

```bash
cd /home/ignaciouc3m/projects/fault_converters
git fetch /tmp/fault_converters_source.bundle HEAD
git merge --ff-only FETCH_HEAD
rm /tmp/fault_converters_source.bundle
```

Back on the workstation, transfer the stopped, completed experiment:

```bash
cd /home/nacho/projects/fault_converters
rsync -a --partial --info=progress2 \
  experiments/dataset_01_run_06_corrected/ \
  ignaciouc3m@192.168.55.1:/home/ignaciouc3m/projects/fault_converters/experiments/dataset_01_run_06_corrected/
```

Verify the target dataset:

```bash
ssh ignaciouc3m@192.168.55.1 \
  'test -d /home/ignaciouc3m/projects/fault_converters/data/buck/buck_data/dataset_01 && echo dataset-present'
```

If absent, transfer it separately, again without `--delete`:

```bash
rsync -a --partial --info=progress2 \
  data/buck/buck_data/dataset_01/ \
  ignaciouc3m@192.168.55.1:/home/ignaciouc3m/projects/fault_converters/data/buck/buck_data/dataset_01/
```

## 6. Jetson software and PMLib preflight

Connect to the Jetson:

```bash
ssh ignaciouc3m@192.168.55.1
cd /home/ignaciouc3m/projects/fault_converters
```

The target is JetPack 4.6.x/L4T R32.7.x, CUDA 10.2, TensorRT 8.2, and Python
3.6. Reuse `.venv-jp46`; do not copy the workstation environment.

```bash
uname -m
head -1 /etc/nv_tegra_release
.venv-jp46/bin/python --version
.venv-jp46/bin/python -c \
  'import tensorflow, tensorrt, onnx, onnxruntime, tf2onnx, pycuda.driver; print(tensorflow.__version__, tensorrt.__version__, onnx.__version__, onnxruntime.__version__, tf2onnx.__version__)'
```

`uname -m` must be `aarch64`. First-time environment requirements, system
packages, and NVIDIA aarch64 wheel notes are in `requirements-jetpack46.txt`.
Ordinary PyPI cannot provide the required NVIDIA TensorFlow and Jetson ONNX
Runtime wheels.

PMLib installation/configuration is documented in
`/home/ignaciouc3m/projects/pmlib/README.md`. It must expose `POM_5V_IN`,
`POM_5V_GPU`, and `POM_5V_CPU`. `POM_5V_IN` is total module power; never sum the
three rails.

In a separate Jetson terminal, start the daemon and leave it running:

```bash
cd /home/ignaciouc3m/projects/pmlib/server
sudo ./build/pmlib_server --configfile settings-nano.json
```

In the repository terminal, verify it:

```bash
pgrep -a pmlib_server
.venv-jp46/bin/python scripts/check_pmlib_counter.py \
  --client-dir /home/ignaciouc3m/projects/pmlib/client --windows 2
```

Do not continue if this check fails or the rails are absent.

## 7. Lock and record the Nano operating state

On the Jetson:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
echo 80 | sudo tee /sys/devices/pwm-fan/target_pwm
nvpmodel -q
sudo jetson_clocks --show
mkdir -p experiments/dataset_01_run_06_corrected
{ date -Is; sudo jetson_clocks --show; } \
  > experiments/dataset_01_run_06_corrected/jetson_clocks_show.txt
```

The clock record must be created less than ten minutes before the campaign.
The preflight requires MAXN, all four CPU clocks fixed, the GPU clock fixed,
and the EMC override fixed at maximum.

## 8. Run the complete target campaign

The preferred entry point performs the required stages in order:

1. Jetson detector/scorer/threshold selection and held-out FP32 evaluation.
2. Target evaluation of the six ablation models.
3. Detector assets and FP32 ONNX serialization.
4. TFLite and ONNX FP32/FP16/INT8 exports with shared calibration data.
5. TensorRT FP32/FP16/INT8 build attempts on the Nano.
6. Loaded-weight and reference-bank provenance audit.
7. Converted-model fidelity and decision-discordance evaluation.
8. Native TensorFlow GPU FP32 and mixed-FP16 evaluation.
9. Three isolated repetitions of timing, memory, utilization, thermal, and
   PMLib power measurements.

Start it in tmux on the Jetson:

```bash
tmux new-session -d -s jetson_campaign
tmux send-keys -t jetson_campaign \
  "cd /home/ignaciouc3m/projects/fault_converters && \
  set -o pipefail && \
  PYTHON_BIN=.venv-jp46/bin/python \
  EXPERIMENT_NAME=dataset_01_run_06_corrected \
  DATA_DIR=data/buck/buck_data/dataset_01 \
  PMLIB_CLIENT_DIR=/home/ignaciouc3m/projects/pmlib/client \
  REPETITIONS=3 \
  bash scripts/run_corrected_jetson_pipeline.sh \
  2>&1 | tee experiments/dataset_01_run_06_corrected/jetson_campaign.log" C-m
```

Monitor it from another terminal:

```bash
tail -f /home/ignaciouc3m/projects/fault_converters/experiments/dataset_01_run_06_corrected/jetson_campaign.log
```

The final line must be:

```text
Corrected Jetson Nano campaign completed.
```

Unsupported, build-failed, executed-but-fidelity-rejected, and validated are
different scientific outcomes. Preserve every recorded outcome. An executable
path whose accuracy is degraded still receives runtime/resource measurements
and is labelled rejected rather than removed. GRU/LSTM TensorRT support is
decided empirically on this JetPack/TensorRT stack.

## 9. Resume or diagnose a failed target campaign

Preserve the log and identify the first failing stage. Fix only its underlying
cause. Before every retry, refresh the target state and clock record:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
echo 80 | sudo tee /sys/devices/pwm-fan/target_pwm
{ date -Is; sudo jetson_clocks --show; } \
  > experiments/dataset_01_run_06_corrected/jetson_clocks_show.txt
```

Rerunning the full command in Section 8 is the safest resume method. Completed
TensorRT engines are retained only when their ONNX source hash matches;
completed resource JSON repetitions are retained; detector and fidelity reports
are deliberately refreshed to prevent stale-checkpoint results.

For expert diagnosis, these are the component stages in required order:

```bash
FAULT_CONVERTERS_PYTHON=.venv-jp46/bin/python \
  bash scripts/evaluate_all.sh \
  --experiment-name dataset_01_run_06_corrected \
  --data-dir data/buck/buck_data/dataset_01

for model in conv1d gru lstm mlp transformer; do
  .venv-jp46/bin/python -m src.deployment_optimization \
    --model-dir "experiments/dataset_01_run_06_corrected/carla_${model}" \
    --data-dir data/buck/buck_data/dataset_01 --portable-export-only
done

.venv-jp46/bin/python scripts/export_precision_variants.py \
  --experiment experiments/dataset_01_run_06_corrected \
  --data-dir data/buck/buck_data/dataset_01

bash scripts/build_jetson_engines.sh \
  --experiment-name dataset_01_run_06_corrected

.venv-jp46/bin/python scripts/audit_model_provenance.py \
  experiments/dataset_01_run_06_corrected/carla_{conv1d,gru,lstm,mlp,transformer}

FAULT_CONVERTERS_PYTHON=.venv-jp46/bin/python \
  bash scripts/evaluate_inference.sh \
  --experiment-name dataset_01_run_06_corrected \
  --data-dir data/buck/buck_data/dataset_01

for model in conv1d gru lstm mlp transformer; do
  for precision in fp32 fp16; do
    .venv-jp46/bin/python scripts/benchmark_native_gpu_precisions.py \
      --model-dir "experiments/dataset_01_run_06_corrected/carla_${model}" \
      --data-dir data/buck/buck_data/dataset_01 \
      --precision "$precision" \
      --output "experiments/dataset_01_run_06_corrected/carla_${model}/deployment/native_gpu_${precision}.json"
  done
done

PYTHON_BIN=.venv-jp46/bin/python \
EXPERIMENT=experiments/dataset_01_run_06_corrected \
PMLIB_CLIENT_DIR=/home/ignaciouc3m/projects/pmlib/client \
REPETITIONS=3 bash scripts/run_jetson_resource_suite.sh all
```

The full campaign also evaluates and summarizes ablations between the first and
second deployment stages. Prefer it unless diagnosing one specific failure.

## 10. Measurement interpretation rules

Use generated metadata rather than informal timings:

- CPU, TFLite, ONNX, scorer, and end-to-end detector latency use the monotonic
  `perf_counter` clock.
- Native TensorFlow GPU and TensorRT device execution use synchronized CUDA
  events. GPU reports also retain detector end-to-end wall time.
- The standardized protocol is batch size 1, 15 warm-ups, and 150 timed runs.
- Resource studies use a fresh process per backend and three repetitions.
- PMLib brackets separate idle and sustained-active windows.
- RAM comes from `tegrastats`; process RSS/HWM comes from `/proc`.
- GPU-path CUDA allocation is sampled with `pycuda.driver.mem_get_info`.
- TFLite FP16 weight PTQ is not FP16 CPU arithmetic.
- INT8 claims require a calibrated artifact and recorded calibration hash.

## 11. Retrieve and cross-check Nano artifacts

After the campaign has stopped, run on the workstation:

```bash
cd /home/nacho/projects/fault_converters
rsync -a --partial --info=progress2 \
  ignaciouc3m@192.168.55.1:/home/ignaciouc3m/projects/fault_converters/experiments/dataset_01_run_06_corrected/ \
  experiments/dataset_01_run_06_corrected/
```

Do not use `--delete`. This retrieves target-built engines, build logs,
conversion statuses, evaluations, raw power/resource records, and target
capability/provenance evidence.

Run the provenance audit again on the workstation:

```bash
uv run python scripts/audit_model_provenance.py \
  experiments/dataset_01_run_06_corrected/carla_{conv1d,gru,lstm,mlp,transformer}
```

Any cross-host mismatch in loaded weights, reference bank, threshold, tensor
shapes, or scorer is a publication blocker. Weight content is compared through
the order-insensitive `detector_loaded_weights_multiset_sha256` invariant:
workstation Keras enumerates the final dense tensors in a different order than
JetPack Keras 2.7, so the aggregate order-sensitive digest differs across hosts
even when every tensor is byte-identical. For a genuine content check,
`scripts/dump_detector_weight_hashes.py MODEL_DIRS...` prints per-tensor hashes
that can be compared as multisets, and the audit must be run on both hosts with
the same script revision so both `model_provenance_*.json` records carry the
multiset field.

The TensorRT INT8 control experiment (Section "TensorRT INT8 barriers" of the
JSC manuscript) is reproduced on the target with:

```bash
.venv-jp46/bin/python scripts/run_int8_qdq_control.py
/usr/src/tensorrt/bin/trtexec --onnx=/tmp/model_int8_qdq_control.onnx \
  --saveEngine=/tmp/model_int8_qdq_control.engine --workspace=512 --buildOnly --int8
```

It rebuilds MLP's calibrated INT8 graph restricted to Conv/MatMul/Gemm with
FP32 biases (identical calibration SHA-256) and records the importer rejection
and exit code next to the campaign build logs.

## 12. Generate final tables and figures

On the workstation:

```bash
uv run python scripts/generate_latest_results_figures.py
uv run python scripts/generate_arch_diagrams.py
```

The result generator writes canonical CSV files under `papers/results/` and
copies figures into both paper trees. Verify required tables:

```bash
for file in \
  papers/results/latest_carla_metrics.csv \
  papers/results/jetson_deployment_results.csv \
  papers/results/jetson_resource_measurements.csv \
  papers/results/jetson_resource_summary.csv \
  papers/results/native_gpu_precision_results.csv \
  papers/results/tensorrt_layer_precision_summary.csv; do
  test -s "$file" || echo "MISSING: $file"
done
```

## 13. Update and compile the papers

Canonical manuscripts and their independent bibliography files are:

```text
papers/mmas_anomaly_detection/main.tex
papers/mmas_anomaly_detection/references.bib
papers/jsc_edge_deployment/main.tex
papers/jsc_edge_deployment/references.bib
```

Rewrite stale numerical prose, tables, abstract statements, discussion, and
conclusions from final CSV/JSON evidence. These must be scientific papers that
present final methodology and results, not a chronological correction log.

Review all of the following:

- RTX 4070 is mentioned only as the training/search host.
- All final evaluation, conversion, timing, power, and memory data are Jetson
  Nano results.
- Dataset and split counts agree everywhere.
- Fit-validation detector selection is not confused with held-out testing.
- Unsupported, failed, rejected, and validated paths remain distinct.
- Precision labels match the artifact's actual computation.
- GRU/LSTM TensorRT statements match empirical build records.
- Total module power is `POM_5V_IN`; CPU/GPU rails are not added to it.
- Every number traces to a generated table or raw report.
- Citation keys resolve in that paper's own `references.bib`.
- Every `\includegraphics` path resolves inside that paper tree.

Compile with BibTeX when LaTeX is installed:

```bash
cd /home/nacho/projects/fault_converters/papers/mmas_anomaly_detection
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

cd /home/nacho/projects/fault_converters/papers/jsc_edge_deployment
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Run the repository publication audit regardless of LaTeX availability:

```bash
cd /home/nacho/projects/fault_converters
uv run python scripts/audit_publication_artifacts.py
```

It must report `Publication manuscripts and final artifacts passed all audits.`

## 14. Final tests, Git LFS, and commit

On the workstation:

```bash
uv run pytest -q
git diff --check
git lfs version
git lfs install
```

Stage documentation, code, papers, and the entire corrected experiment. `-f`
is required because target ONNX, TensorRT, and build logs are normally ignored:

```bash
git add README.md MANUAL_CAMPAIGN_README.md .gitattributes
git add scripts src tests requirements-jetpack46.txt
git add papers docs
git add -f experiments/dataset_01_run_06_corrected
```

Do not add these superseded or smoke directories:

```text
experiments/dataset_01_run_06_superseded_nuisance_search/
experiments/dataset_01_run_06_superseded_pre_keras_seed_reset/
tmp/training_smoke/
tmp/training_smoke2/
```

Verify LFS and the entire staged set before committing:

```bash
git lfs ls-files
git status --short
git diff --cached --check
git diff --cached --name-status
```

Then create the local commit:

```bash
git commit -m "Complete corrected Jetson Nano inference study"
```

Do not push unless explicitly intended. The eventual push will be large because
it contains every retained trial/final checkpoint, converted model, target
engine, measurement record, result table, figure, and manuscript artifact.

## 15. Completion checklist

- [ ] Five architectures each completed 50 trials and final training.
- [ ] Best checkpoints load their intended weights and reference embeddings.
- [ ] Six paired ablation models were trained on the workstation.
- [ ] Source revisions match across workstation and Nano.
- [ ] Detector/threshold selection and held-out evaluation ran on the Nano.
- [ ] TFLite and ONNX FP32/FP16/INT8 attempts are recorded for every model.
- [ ] TensorRT FP32/FP16/INT8 attempts were made on the Nano.
- [ ] Native Jetson GPU FP32 and mixed-FP16 paths were evaluated.
- [ ] Discordance/fidelity reports exist for executable paths.
- [ ] Three isolated resource repetitions exist per executable path.
- [ ] Power, RAM, RSS/HWM, CUDA memory, utilization, and thermal data exist.
- [ ] Workstation/Jetson loaded-model provenance matches.
- [ ] Final tables and figures were regenerated.
- [ ] Both manuscripts contain consistent final methods and results.
- [ ] Tests and publication audits pass.
- [ ] Required models and artifacts are staged through Git LFS.
- [ ] The final local commit exists; no push occurred unintentionally.
