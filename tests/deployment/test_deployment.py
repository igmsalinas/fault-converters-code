import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
import keras
import shutil
from types import SimpleNamespace

from src.models.conv1d_ae import Conv1DAutoencoder
from src.models.gru_ae import GRUAutoencoder
from src.models.lstm_ae import LSTMAutoencoder
from src.models.contrastive_ae import ContrastiveAutoencoder
from src.deployment.keras_quantization import quantize_keras_native, convert_to_tflite
from src.deployment.onnx_trt import (
    TensorRTCompatibleMultiHeadAttention,
    compile_onnx_to_tensorrt,
    convert_keras_to_onnx,
    make_tensorrt_compatible_model,
)
from src.deployment.vitis_ai import quantize_for_vitis_ai
from src.deployment.utils import build_carla_embedding_model, get_file_size_mb
from src.deployment.runners import (
    create_keras_runner,
    create_tflite_runner,
    create_onnx_runner,
    create_tensorrt_runner,
    run_inference_loop,
    measure_inference_latency,
    compare_runner_outputs,
)
from src.deployment.timing_memory_benchmark import (
    run_full_performance_suite,
    save_performance_report,
)
from src.deployment.timing_evaluation import (
    run_deployment_evaluations,
    save_evaluation_report,
)
from scripts.export_precision_variants import (
    export_onnx_variants,
    onnx_qdq_compute_coverage,
)
from scripts.audit_model_provenance import sanitize_hostname, weight_multiset_sha256


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------


def test_provenance_hostname_is_filename_safe():
    assert sanitize_hostname("jetson nano/lab:01") == "jetson_nano_lab_01"


def test_weight_multiset_digest_is_order_insensitive():
    rng = np.random.default_rng(7)
    weights = [rng.normal(size=(3, 4)).astype(np.float32), rng.normal(size=5).astype(np.float64)]
    permuted = [weights[1], weights[0].copy()]
    assert weight_multiset_sha256(weights)[0] == weight_multiset_sha256(permuted)[0]
    altered = [weights[0], weights[1] + 1]
    assert weight_multiset_sha256(weights)[0] != weight_multiset_sha256(altered)[0]

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_model(sequence_length, n_features):
    ae = Conv1DAutoencoder(
        input_shape=(sequence_length, n_features),
        latent_dim=4,
        filters=[8],
        kernel_size=3,
    ).build()
    return ae


@pytest.fixture
def model_dir_with_config(sample_model, temp_dir):
    """Save a minimal model directory with config and weights (shared fixture).

    This eliminates the copy-paste of model_config.json + save_weights
    across multiple test functions.
    """
    with open(temp_dir / "model_config.json", "w") as f:
        json.dump({
            "name": "conv1d_ae",
            "type": "standard",
            "input_shape": [24, 5],
            "latent_dim": 4,
            "filters_idx": 0,
            "kernel_size": 3,
        }, f)
    sample_model.autoencoder.save_weights(str(temp_dir / "best_model.weights.h5"))
    return temp_dir


@pytest.fixture
def deployment_dir_with_tflite(sample_model, model_dir_with_config):
    """Save a TFLite dynamic model alongside the model dir for benchmark tests."""
    path_dynamic = str(model_dir_with_config / "model_dynamic.tflite")
    convert_to_tflite(
        sample_model.autoencoder,
        optimization_mode="dynamic",
        output_path=path_dynamic,
    )
    return model_dir_with_config


# ---------------------------------------------------------------------------
# Component 1: Keras & TFLite Quantization
# ---------------------------------------------------------------------------

def test_keras_and_tflite_quantization(sample_model, mock_data, temp_dir):
    # Test Keras-native quantization
    quant_keras = quantize_keras_native(
        sample_model.autoencoder,
        mode="int8",
        output_dir=str(temp_dir),
    )
    assert isinstance(quant_keras, keras.Model)
    assert (temp_dir / "keras_quantized_int8.weights.h5").exists()

    # Test TFLite conversions
    path_dynamic = str(temp_dir / "model_dynamic.tflite")
    convert_to_tflite(
        sample_model.autoencoder,
        optimization_mode="dynamic",
        output_path=path_dynamic,
    )
    assert Path(path_dynamic).exists()
    assert Path(path_dynamic).stat().st_size > 0

    path_fp16 = str(temp_dir / "model_fp16.tflite")
    convert_to_tflite(
        sample_model.autoencoder,
        optimization_mode="float16",
        output_path=path_fp16,
    )
    assert Path(path_fp16).exists()

    # Full INT8 calibration conversion
    path_int8 = str(temp_dir / "model_int8.tflite")
    convert_to_tflite(
        sample_model.autoencoder,
        optimization_mode="int8",
        calibration_data=mock_data,
        output_path=path_int8,
    )
    assert Path(path_int8).exists()


# ---------------------------------------------------------------------------
# Component 2: ONNX Conversion
# ---------------------------------------------------------------------------

def test_onnx_conversion(sample_model, temp_dir):
    path_onnx = str(temp_dir / "model.onnx")
    success = convert_keras_to_onnx(sample_model.autoencoder, path_onnx)

    # Try importing tf2onnx; if import fails, success is False but execution is safe
    try:
        import tf2onnx
        assert success is True
        assert Path(path_onnx).exists()
    except ImportError:
        assert success is False


def test_tensorrt_compatible_attention_preserves_outputs():
    inputs = keras.Input(shape=(7, 8))
    outputs = keras.layers.MultiHeadAttention(num_heads=2, key_dim=4)(
        inputs, inputs
    )
    model = keras.Model(inputs, outputs)
    compatible = make_tensorrt_compatible_model(model)
    compatible_attention = next(
        layer
        for layer in compatible.layers
        if isinstance(layer, TensorRTCompatibleMultiHeadAttention)
    )
    assert compatible_attention.get_config()["output_shape"] is None
    assert compatible_attention._output_shape_override is None
    sample = np.random.default_rng(42).normal(size=(2, 7, 8)).astype(np.float32)

    np.testing.assert_allclose(
        model(sample, training=False).numpy(),
        compatible(sample, training=False).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_tensorrt_compatible_carla_transformer_preserves_loaded_weights():
    """Exercise the nested encoder/projection graph used for final deployment."""
    carla = ContrastiveAutoencoder(
        input_shape=(7, 2),
        latent_dim=4,
        projection_dim=5,
        encoder_type="transformer",
        encoder_filters=[8, 8],
        num_heads=2,
        projection_hidden_dim=6,
        projection_layers=2,
        dropout_rate=0.1,
    ).build()
    detector = build_carla_embedding_model(carla)
    compatible = make_tensorrt_compatible_model(detector)
    sample = np.random.default_rng(43).normal(size=(3, 7, 2)).astype(np.float32)

    assert len(detector.weights) == len(compatible.weights)
    for source, target in zip(detector.get_weights(), compatible.get_weights()):
        np.testing.assert_array_equal(source, target)
    np.testing.assert_allclose(
        detector(sample, training=False).numpy(),
        compatible(sample, training=False).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_transformer_onnx_avoids_legacy_tensorrt_softmax_fusion(temp_dir):
    pytest.importorskip("tf2onnx")
    onnx = pytest.importorskip("onnx")
    inputs = keras.Input(shape=(7, 8))
    outputs = keras.layers.MultiHeadAttention(num_heads=2, key_dim=4)(
        inputs, inputs
    )
    model = keras.Model(inputs, outputs)
    path = temp_dir / "attention.onnx"

    assert convert_keras_to_onnx(model, str(path))
    graph = onnx.load(str(path)).graph
    op_types = [node.op_type for node in graph.node]
    assert "Softmax" not in op_types
    assert "Exp" in op_types
    assert "ReduceSum" in op_types


def test_onnx_qdq_coverage_identifies_floating_recurrent_operator():
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    graph = helper.make_graph(
        [
            helper.make_node(
                "DequantizeLinear", ["xq", "scale", "zero"], ["xdq"]
            ),
            helper.make_node("MatMul", ["xdq", "weight"], ["dense"]),
            helper.make_node("GRU", ["sequence", "w", "r"], ["recurrent"]),
        ],
        "coverage_probe",
        [],
        [],
    )
    coverage = onnx_qdq_compute_coverage(helper.make_model(graph))

    assert coverage["compute_operators_with_dequantized_inputs"] == {"MatMul": 1}
    assert coverage["compute_operators_without_dequantized_inputs"] == {"GRU": 1}


def test_onnx_calibration_paths_remain_absolute_inside_tempdir(
    temp_dir, monkeypatch
):
    onnx = pytest.importorskip("onnx")
    quantization = pytest.importorskip("onnxruntime.quantization")
    helper = onnx.helper
    tensor = onnx.TensorProto
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "path_resolution",
        [helper.make_tensor_value_info("input", tensor.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", tensor.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    model_path = temp_dir / "model.onnx"
    onnx.save(model, str(model_path))
    captured = {}

    def inspect_paths(model_input, model_output, **_kwargs):
        captured.update({"input": model_input, "output": model_output})
        raise RuntimeError("stop after checking resolved calibration paths")

    monkeypatch.setattr(quantization, "quantize_static", inspect_paths)
    records = []
    monkeypatch.chdir(temp_dir.parent)
    export_onnx_variants(
        Path(temp_dir.name) / model_path.name,
        np.zeros((1, 2), dtype=np.float32),
        records,
    )

    assert Path(captured["input"]).is_absolute()
    assert Path(captured["output"]).is_absolute()
    assert Path(captured["input"]) == model_path.resolve()


@pytest.mark.parametrize(
    ("model_class", "cell_name", "unit_argument"),
    [
        (GRUAutoencoder, "GRU", {"gru_units": [4, 3]}),
        (LSTMAutoencoder, "LSTM", {"lstm_units": [4, 3]}),
    ],
)
def test_recurrent_onnx_uses_native_operator_and_preserves_outputs(
    temp_dir, model_class, cell_name, unit_argument
):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    autoencoder = model_class(
        input_shape=(7, 2),
        latent_dim=3,
        dropout_rate=0.1,
        **unit_argument,
    ).build().autoencoder
    path = temp_dir / f"{cell_name.lower()}_native.onnx"

    assert convert_keras_to_onnx(autoencoder, str(path))
    graph = onnx.load(str(path)).graph
    operator_types = [node.op_type for node in graph.node]
    assert operator_types.count(cell_name) == 4
    assert "Loop" not in operator_types

    sample = np.random.default_rng(42).normal(size=(1, 7, 2)).astype(np.float32)
    expected = autoencoder.predict(sample, verbose=0)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    # ONNX Runtime and TensorFlow use different fused recurrent kernels; their
    # float32 accumulation order can differ by a few micro-units.
    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("encoder_type", "cell_name"), [("gru", "GRU"), ("lstm", "LSTM")])
def test_recurrent_carla_detector_exports_native_embedding_graph(
    temp_dir, encoder_type, cell_name
):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    carla = ContrastiveAutoencoder(
        input_shape=(7, 2),
        latent_dim=3,
        projection_dim=4,
        projection_hidden_dim=5,
        encoder_type=encoder_type,
        encoder_units=[4, 3],
        dropout_rate=0.0,
    ).build()
    detector = build_carla_embedding_model(carla)
    path = temp_dir / f"carla_{encoder_type}_detector.onnx"

    assert convert_keras_to_onnx(detector, str(path))
    graph = onnx.load(str(path)).graph
    operator_types = [node.op_type for node in graph.node]
    assert operator_types.count(cell_name) == 2
    assert "Loop" not in operator_types

    sample = np.random.default_rng(7).normal(size=(1, 7, 2)).astype(np.float32)
    expected = detector.predict(sample, verbose=0)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Component 3: ONNX to TensorRT
# ---------------------------------------------------------------------------

def test_onnx_to_tensorrt_compilation(sample_model, temp_dir):
    # Convert Keras model to ONNX
    path_onnx = str(temp_dir / "model.onnx")
    convert_keras_to_onnx(sample_model.autoencoder, path_onnx)

    # Attempt to compile ONNX to TensorRT
    path_engine = str(temp_dir / "model.engine")
    success = compile_onnx_to_tensorrt(
        onnx_path=path_onnx,
        engine_path=path_engine,
        precision_mode="FP16",
    )
    # The return value should be a boolean (either True on successful build or False if TRT not installed/supported)
    assert isinstance(success, bool)


# ---------------------------------------------------------------------------
# Component 4: Vitis AI Targets
# ---------------------------------------------------------------------------

def test_vitis_ai_targets(sample_model, mock_data, temp_dir):
    # Verify Vitis AI quantizer fallback handles environment issues gracefully
    # and generates matching configs for different target boards.
    for target in ["kv260", "pynq_z2"]:
        target_dir = temp_dir / target
        success = quantize_for_vitis_ai(
            model=sample_model.autoencoder,
            calibration_data=mock_data,
            output_dir=str(target_dir),
            target_hardware=target,
        )
        assert isinstance(success, bool)

        # Verify configuration files were generated
        assert (target_dir / f"{target}_arch.json").exists()
        assert (target_dir / "deploy_pynq.py").exists()

        # Read arch.json to verify target parameters
        with open(target_dir / f"{target}_arch.json", "r") as f:
            arch_data = json.load(f)
            assert "target" in arch_data
            assert "cpu_arch" in arch_data
            if target == "pynq_z2":
                assert arch_data["cpu_arch"] == "arm32"
            else:
                assert arch_data["cpu_arch"] == "arm64"


# ---------------------------------------------------------------------------
# Component 5: Shared Utilities
# ---------------------------------------------------------------------------

def test_get_file_size_mb(temp_dir):
    # Test file size
    test_file = temp_dir / "test.bin"
    test_file.write_bytes(b"\x00" * 1024)  # 1KB
    size = get_file_size_mb(str(test_file))
    assert abs(size - 1.0 / 1024) < 0.001

    # Test nonexistent path
    size = get_file_size_mb(str(temp_dir / "nonexistent"))
    assert size == 0.0

    # Test directory size
    size = get_file_size_mb(str(temp_dir))
    assert size > 0.0


# ---------------------------------------------------------------------------
# Component 6: Inference Runners
# ---------------------------------------------------------------------------

def test_keras_runner(sample_model, mock_data):
    runner = create_keras_runner(sample_model.autoencoder)
    output = runner(mock_data)
    assert output.shape == mock_data.shape


def test_tflite_runner(sample_model, mock_data, temp_dir):
    path = str(temp_dir / "model_dynamic.tflite")
    convert_to_tflite(sample_model.autoencoder, "dynamic", output_path=path)

    runner = create_tflite_runner(path)
    # Models are exported with a fixed batch dimension of 1.
    single = np.expand_dims(mock_data[0], axis=0)
    output = runner(single)
    assert output.shape == single.shape


def test_onnx_runner(sample_model, mock_data, temp_dir):
    path = str(temp_dir / "model.onnx")
    success = convert_keras_to_onnx(sample_model.autoencoder, path)
    if not success:
        pytest.skip("tf2onnx not installed")

    runner = create_onnx_runner(path)
    assert runner is not None
    # Models are exported with a fixed batch dimension of 1.
    single = np.expand_dims(mock_data[0], axis=0)
    output = runner(single)
    assert output.shape == single.shape


def test_tensorrt_runner(sample_model, temp_dir, mock_data):
    """Test TRT runner creation — handles multiple unavailability scenarios.

    TensorRT engine compilation (needs `tensorrt`) and runner creation
    (additionally needs `cuda-python` or `pycuda`) can fail independently.
    """
    path_onnx = str(temp_dir / "model.onnx")
    success = convert_keras_to_onnx(sample_model.autoencoder, path_onnx)
    if not success:
        pytest.skip("tf2onnx not installed")

    path_engine = str(temp_dir / "model.engine")
    trt_success = compile_onnx_to_tensorrt(path_onnx, path_engine, "FP16")

    if not trt_success:
        # TensorRT package not available — runner should also fail gracefully
        runner = create_tensorrt_runner(path_engine)
        assert runner is None
        return

    # Engine compiled successfully — runner may or may not work
    # depending on whether cuda-python/pycuda is installed
    runner = create_tensorrt_runner(path_engine)
    if runner is None:
        # TRT compiled but CUDA memory packages not available — acceptable
        pytest.skip("TensorRT engine compiled but cuda-python/pycuda not installed for runtime")
    else:
        # Full TRT stack available — verify runner produces correct output.
        # Engines are built with a fixed batch dimension of 1.
        single = np.expand_dims(mock_data[0], axis=0)
        output = runner(single)
        assert output.shape == single.shape


def test_run_inference_loop(sample_model, mock_data):
    runner = create_keras_runner(sample_model.autoencoder)
    output = run_inference_loop(runner, mock_data, batch_size=2)
    assert output.shape == mock_data.shape


def test_run_inference_loop_uses_requested_batch_size(mock_data):
    observed_batch_sizes = []

    def runner(batch):
        observed_batch_sizes.append(len(batch))
        return batch

    output = run_inference_loop(runner, mock_data, batch_size=1)
    assert output.shape == mock_data.shape
    assert observed_batch_sizes == [1] * len(mock_data)


def test_runner_loop_accepts_projection_embeddings(mock_data):
    def embedding_runner(batch):
        return np.mean(batch, axis=1)

    embedding_runner.output_shape = (mock_data.shape[-1],)
    output = run_inference_loop(embedding_runner, mock_data, batch_size=2)

    assert output.shape == (len(mock_data), mock_data.shape[-1])


def test_cpu_timing_uses_perf_counter(mock_data):
    runner = lambda batch: batch
    result = measure_inference_latency(
        runner, mock_data[:1], num_warmup=2, num_runs=3
    )
    assert result["timing_source"] == "perf_counter"
    assert result["timing_batch_size"] == 1
    assert result["timing_warmup_runs"] == 2
    assert result["timing_runs"] == 3
    assert result["latency_mean_ms"] == result["end_to_end_latency_mean_ms"]


def test_gpu_timing_uses_cuda_event_hook(mock_data):
    def runner(batch):
        return batch

    def cuda_timed_call(batch):
        return batch, 0.25

    runner.cuda_timed_call = cuda_timed_call
    result = measure_inference_latency(
        runner, mock_data[:1], num_warmup=2, num_runs=3
    )
    assert result["timing_source"] == "cuda_event"
    assert result["latency_mean_ms"] == pytest.approx(0.25)
    assert result["end_to_end_latency_mean_ms"] >= 0.0


def test_runner_consistency_reports_numerical_and_decision_drift(mock_data):
    reference_runner = lambda batch: batch
    candidate_runner = lambda batch: np.zeros_like(batch)
    result = compare_runner_outputs(
        reference_runner,
        candidate_runner,
        mock_data,
        threshold=0.0,
        batch_size=1,
        max_samples=3,
    )
    assert result["consistency_samples"] == min(3, len(mock_data))
    assert result["reconstruction_mse_vs_baseline"] > 0.0
    assert result["reconstruction_within_tolerance"] is False
    assert result["anomaly_disagreement_rate_vs_baseline"] == 1.0


# ---------------------------------------------------------------------------
# Component 7: Performance Suite (Timing & Memory)
# ---------------------------------------------------------------------------

def test_timing_memory_suite(mock_data, deployment_dir_with_tflite):
    res = run_full_performance_suite(
        model_dir=str(deployment_dir_with_tflite),
        deployment_dir=str(deployment_dir_with_tflite),
        test_data=mock_data,
    )
    assert len(res) >= 1
    # Check that baseline has results
    assert res[0]["model_name"] == "Keras FP32 (Baseline)"
    assert "latency_mean_ms" in res[0]

    # Save results
    report_path = str(deployment_dir_with_tflite / "performance_benchmark_report")
    save_performance_report(res, report_path)

    # Assert reports exist
    assert Path(report_path + ".json").exists()


# ---------------------------------------------------------------------------
# Component 9: Classification Evaluation
# ---------------------------------------------------------------------------

def test_timing_evaluation_suite(mock_data, deployment_dir_with_tflite):
    mock_labels = np.random.choice([0, 1], size=(mock_data.shape[0],))
    (deployment_dir_with_tflite / "threshold.json").write_text(
        json.dumps({"threshold": 0.1})
    )

    res = run_deployment_evaluations(
        model_dir=str(deployment_dir_with_tflite),
        deployment_dir=str(deployment_dir_with_tflite),
        test_data=mock_data,
        test_labels=mock_labels,
    )
    assert len(res) >= 1
    # Check that baseline has results
    assert res[0]["model_name"] == "Keras FP32 (Baseline)"
    assert "latency_per_sample_mean_ms" in res[0]
    assert "f1" in res[0]
    assert res[0]["full_test_consistency_samples"] == len(mock_data)
    assert res[0]["full_test_anomaly_disagreement_rate_vs_baseline"] == 0.0

    # Save results
    report_path = str(deployment_dir_with_tflite / "evaluation_benchmark_report")
    save_evaluation_report(res, report_path)

    # Assert reports exist
    assert Path(report_path + ".json").exists()


def test_deployment_evaluation_rejects_missing_calibrated_threshold(
    mock_data, deployment_dir_with_tflite
):
    predictor = SimpleNamespace(threshold=None)

    with pytest.raises(RuntimeError, match="fit-validation threshold"):
        run_deployment_evaluations(
            model_dir=str(deployment_dir_with_tflite),
            deployment_dir=str(deployment_dir_with_tflite),
            test_data=mock_data,
            test_labels=np.zeros(len(mock_data), dtype=np.int64),
            predictor=predictor,
        )


def test_tflite_int8_runner_saturates_extreme_inputs_without_wraparound(
    sample_model, mock_data, temp_dir
):
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, mock_data.shape[1], mock_data.shape[2]], dtype=tf.float32)])
    def serve_fn(x):
        return sample_model.autoencoder(x, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serve_fn.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        for i in range(min(10, len(mock_data))):
            yield [mock_data[i : i + 1].astype(np.float32)]

    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    int8_path = temp_dir / "sat_int8.tflite"
    int8_path.write_bytes(converter.convert())

    runner = create_tflite_runner(str(int8_path))
    extreme = np.full((1, mock_data.shape[1], mock_data.shape[2]), 50.0, dtype=np.float32)
    bigger = np.full((1, mock_data.shape[1], mock_data.shape[2]), 80.0, dtype=np.float32)
    # Once saturated at +127 int8 input, increasing from +50 to +80 must produce
    # identical quantized input (no two's-complement modulo-256 wraparound!).
    out_50 = runner(extreme)
    out_80 = runner(bigger)
    np.testing.assert_allclose(out_50, out_80, rtol=1e-6, atol=1e-6)
