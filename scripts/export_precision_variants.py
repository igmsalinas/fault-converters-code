#!/usr/bin/env python3
"""Export the supported TFLite and ONNX precision variants for CARLA models."""

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
from pathlib import Path
import socket
import sys
import tempfile

import numpy as np

# Export portable CPU graphs; CUDA-backed Keras tracing otherwise captures
# CudnnRNNV3, which TensorFlow Lite cannot lower on JetPack 4.6.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.keras_quantization import convert_to_tflite
from src.deployment.onnx_graph import topologically_sort_graph
from src.deployment.utils import build_carla_embedding_model, get_calibration_dataset
from src.inference.predictor import AnomalyPredictor


MODELS = ("conv1d", "gru", "lstm", "mlp", "transformer")
ONNX_COMPUTE_OPERATORS = {
    "Attention", "Conv", "ConvTranspose", "Gemm", "GRU", "LSTM", "MatMul",
}


class ArrayCalibrationReader:
    """ONNX Runtime calibration reader with deterministic batch-1 samples."""

    def __init__(self, data: np.ndarray, input_name: str):
        self.data = np.asarray(data, dtype=np.float32)
        self.input_name = input_name
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        sample = self.data[self.index : self.index + 1]
        self.index += 1
        return {self.input_name: sample}


def onnx_qdq_compute_coverage(model):
    """Describe Q/DQ adjacency without overstating effective hardware precision."""
    producer = {
        output: node
        for node in model.graph.node
        for output in node.output
    }
    with_dequantized_inputs = Counter()
    without_dequantized_inputs = Counter()
    for node in model.graph.node:
        if node.op_type not in ONNX_COMPUTE_OPERATORS:
            continue
        input_producers = [producer.get(name) for name in node.input]
        if any(
            parent is not None and parent.op_type == "DequantizeLinear"
            for parent in input_producers
        ):
            with_dequantized_inputs[node.op_type] += 1
        else:
            without_dequantized_inputs[node.op_type] += 1
    return {
        "interpretation": (
            "Structural Q/DQ adjacency only; effective TensorRT layer precision is "
            "determined separately by the target builder/runtime."
        ),
        "compute_operators_with_dequantized_inputs": dict(
            sorted(with_dequantized_inputs.items())
        ),
        "compute_operators_without_dequantized_inputs": dict(
            sorted(without_dequantized_inputs.items())
        ),
    }


def export_onnx_variants(base_path: Path, calibration_data: np.ndarray, records: list):
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    model = onnx.load(str(base_path))
    input_name = model.graph.input[0].name

    fp16_path = base_path.with_name("model_fp16.onnx")
    try:
        fp16_model = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=False,
        )
        topologically_sort_graph(fp16_model.graph)
        onnx.checker.check_model(fp16_model)
        onnx.save(fp16_model, str(fp16_path))
        records.append({"backend": "onnx", "precision": "fp16", "status": "exported"})
    except Exception as error:
        records.append({"backend": "onnx", "precision": "fp16", "status": "unsupported", "reason": str(error)})

    int8_path = base_path.with_name("model_int8.onnx")
    try:
        # ONNX Runtime 1.11 writes ``augmented_model.onnx`` in the process
        # working directory during calibration. Isolate that intermediate so
        # consecutive architectures cannot overwrite or reuse one another.
        # Resolve both paths before changing directory; resolving a relative
        # path afterward incorrectly anchors it below the temporary directory.
        model_input_path = base_path.resolve()
        model_output_path = int8_path.resolve()
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="fault_ort_calibration_") as work_dir:
            try:
                os.chdir(work_dir)
                quantize_static(
                    model_input=str(model_input_path),
                    model_output=str(model_output_path),
                    calibration_data_reader=ArrayCalibrationReader(calibration_data, input_name),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
                )
            finally:
                os.chdir(original_cwd)
        quantized = onnx.load(str(int8_path))
        onnx.checker.check_model(quantized)
        counts = Counter(node.op_type for node in quantized.graph.node)
        if counts["QuantizeLinear"] == 0 or counts["DequantizeLinear"] == 0:
            raise RuntimeError("quantizer produced no Q/DQ operators")
        records.append({
            "backend": "onnx",
            "precision": "int8",
            "status": "exported",
            "method": "static QDQ PTQ",
            "calibration_samples": int(len(calibration_data)),
            "quantize_nodes": counts["QuantizeLinear"],
            "dequantize_nodes": counts["DequantizeLinear"],
            "qdq_compute_coverage": onnx_qdq_compute_coverage(quantized),
        })
    except Exception as error:
        if int8_path.exists():
            int8_path.unlink()
        records.append({"backend": "onnx", "precision": "int8", "status": "unsupported", "reason": str(error)})

    # Record CPU Execution Provider compatibility independently of export.
    for precision, path in (("fp32", base_path), ("fp16", fp16_path), ("int8", int8_path)):
        if not path.exists():
            continue
        try:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            shape = tuple(int(dim) for dim in session.get_inputs()[0].shape)
            session.run(None, {session.get_inputs()[0].name: np.zeros(shape, np.float32)})
            status = "runnable"
            reason = None
        except Exception as error:
            status = "unsupported"
            reason = str(error)
        records.append({
            "backend": "onnx-cpu",
            "precision": precision,
            "status": status,
            **({"reason": reason} if reason else {}),
        })


def export_model(model_dir: Path, data_dir: Path, max_files: int, num_calib: int,
                 tflite_only=False):
    predictor = AnomalyPredictor(model_dir=str(model_dir))
    embedding_model = build_carla_embedding_model(predictor.model)
    # Reset for every architecture so all conversions see the same persisted
    # healthy training-state indices rather than successive RNG slices.
    np.random.seed(42)
    calibration_data = get_calibration_dataset(
        data_dir=str(data_dir),
        max_files=max_files,
        num_samples=num_calib,
        preprocessor=predictor.preprocessor,
        split_indices_path=str(model_dir / "split_indices.npz"),
    )
    calibration_sha256 = hashlib.sha256(
        np.ascontiguousarray(calibration_data, dtype=np.float32).tobytes()
    ).hexdigest()
    deployment = model_dir / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    status_path = deployment / "precision_export_status.json"
    records = []
    if tflite_only and status_path.exists():
        records = [
            record for record in json.loads(status_path.read_text())
            if not record.get("backend", "").startswith("tflite")
        ]

    import tensorflow as tf
    with tempfile.TemporaryDirectory(prefix="fault_tflite_export_") as saved_dir:
        fixed_input = tf.keras.Input(
            batch_shape=(1,) + tuple(embedding_model.input_shape[1:])
        )
        fixed_model = tf.keras.Model(
            fixed_input, embedding_model(fixed_input)
        )
        tf.saved_model.save(fixed_model, saved_dir)

        for mode, precision in (("none", "fp32"), ("float16", "fp16"), ("int8", "int8")):
            output = deployment / ("model_float16.tflite" if mode == "float16" else f"model_{precision}.tflite")
            try:
                kwargs = {"calibration_data": calibration_data} if mode == "int8" else {}
                convert_to_tflite(
                    embedding_model,
                    mode,
                    output_path=str(output),
                    saved_model_dir=saved_dir,
                    **kwargs,
                )
                record = {
                    "backend": "tflite-cpu",
                    "precision": precision,
                    "status": "exported",
                }
                record["converter_tensorflow_version"] = tf.__version__
                if mode == "float16":
                    record["method"] = "FP16 weight PTQ; FP32 CPU compute and I/O"
                elif mode == "int8":
                    interpreter = tf.lite.Interpreter(model_path=str(output))
                    interpreter.allocate_tensors()
                    int8_tensors = sum(
                        detail["dtype"] == np.int8
                        for detail in interpreter.get_tensor_details()
                    )
                    if int8_tensors == 0:
                        raise RuntimeError("strict INT8 conversion contains no INT8 tensors")
                    record.update({
                        "method": "TensorFlow Lite full-integer static PTQ",
                        "calibration_samples": int(len(calibration_data)),
                        "int8_tensor_count": int(int8_tensors),
                        "io_dtype": "float32",
                        "calibration_sha256": calibration_sha256,
                    })
                records.append(record)
            except Exception as error:
                if output.exists():
                    output.unlink()
                records.append({"backend": "tflite-cpu", "precision": precision, "status": "unsupported", "reason": str(error)})

    if not tflite_only:
        export_onnx_variants(deployment / "model.onnx", calibration_data, records)
        for record in records:
            if record.get("backend") == "onnx" and record.get("precision") == "int8":
                record["calibration_sha256"] = calibration_sha256
    for record in records:
        record.setdefault("conversion_hostname", socket.gethostname())
        record.setdefault("conversion_machine", platform.machine())
    status_path.write_text(json.dumps(records, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", default="experiments/dataset_01_run_06_corrected"
    )
    parser.add_argument("--data-dir", default="data/buck/buck_data/dataset_01")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--num-calib", type=int, default=100)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--tflite-only", action="store_true",
        help="Export only TFLite variants (useful with the target runtime version).",
    )
    args = parser.parse_args()

    np.random.seed(42)
    summary = {}
    for model in args.models:
        model_dir = Path(args.experiment) / f"carla_{model}"
        print(f"Exporting {model_dir}...", flush=True)
        summary[model] = export_model(
            model_dir, Path(args.data_dir), args.max_files, args.num_calib,
            tflite_only=args.tflite_only,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
