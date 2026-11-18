#!/usr/bin/env python3
"""Format-restricted QDQ control for the TensorRT INT8 study.

Rebuilds one model's calibrated INT8 ONNX graph with quantization restricted
to the compute operators TensorRT accepts (Conv/MatMul/Gemm), leaving every
other operator floating point. The restricted graph exercises the TensorRT
8.2 ONNX importer with a fully supported quantized subset, so a subsequent
build attempt separates the graph-format barrier from the hardware barrier
(no TensorRT INT8 below compute capability 6.1). The calibration set is
identical to the campaign's: same persisted split indices, same seed, same
100 representatives, and the SHA-256 is reported for comparison.

Runs on the target (JetPack 4.6 stack) but only writes to /tmp; it never
touches campaign artifacts.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.utils import get_calibration_dataset
from src.inference.predictor import AnomalyPredictor


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


def strip_bias_quantization(model):
    """Restore FP32 biases around Gemm/Conv constants.

    TensorRT 8.2's importer accepts DequantizeLinear only on activations and
    rejects it on constants, but ONNX Runtime also quantizes Gemm biases (as
    bare INT8 initializers feeding DequantizeLinear, with QuantizeLinear
    folded away). Dequantize the bias in numpy, rewire each bias input to the
    restored FP32 initializer, and drop the orphaned DQ nodes so the remaining
    quantized constants are only the compute-operator weights that TensorRT's
    Q/DQ flow is documented for.
    """
    from onnx import numpy_helper

    initializers = {init.name: init for init in model.graph.initializer}
    producers = {}
    for node in model.graph.node:
        for output in node.output:
            producers[output] = node
    removable = set()
    rewired = 0
    for node in model.graph.node:
        if node.op_type not in ("Gemm", "MatMul", "Conv") or len(node.input) < 3:
            continue
        dq = producers.get(node.input[2])
        if dq is None or dq.op_type != "DequantizeLinear":
            continue
        source = dq.input[0]
        if source not in initializers:
            continue
        int8_bias = numpy_helper.to_array(initializers[source])
        if int8_bias.dtype not in (np.int8, np.uint8, np.int32, np.uint32):
            continue
        scale = numpy_helper.to_array(initializers[dq.input[1]]).astype(np.float32)
        zero_point = (
            numpy_helper.to_array(initializers[dq.input[2]]).astype(np.float32)
            if len(dq.input) > 2
            else np.float32(0.0)
        )
        fp32_bias = (int8_bias.astype(np.float32) - zero_point) * scale
        restored = numpy_helper.from_array(
            np.ascontiguousarray(fp32_bias), name=source + "_fp32"
        )
        model.graph.initializer.append(restored)
        initializers[restored.name] = restored
        node.input[2] = restored.name
        removable.add(dq.name)
        rewired += 1
    if rewired:
        kept = [node for node in model.graph.node if node.name not in removable]
        del model.graph.node[:]
        model.graph.node.extend(kept)
    return rewired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="experiments/dataset_01_run_06_corrected/carla_mlp",
    )
    parser.add_argument("--data-dir", default="data/buck/buck_data/dataset_01")
    parser.add_argument("--num-calib", type=int, default=100)
    parser.add_argument(
        "--output", default="/tmp/model_int8_qdq_control.onnx"
    )
    parser.add_argument("--summary", default="/tmp/int8_qdq_control.json")
    args = parser.parse_args()

    import onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    model_dir = Path(args.model_dir)
    predictor = AnomalyPredictor(model_dir=str(model_dir))
    # Identical to the campaign export: reset for the same persisted healthy
    # training-state indices rather than successive RNG slices.
    np.random.seed(42)
    calibration_data = get_calibration_dataset(
        data_dir=str(args.data_dir),
        max_files=None,
        num_samples=args.num_calib,
        preprocessor=predictor.preprocessor,
        split_indices_path=str(model_dir / "split_indices.npz"),
    )
    calibration_sha256 = hashlib.sha256(
        np.ascontiguousarray(calibration_data, dtype=np.float32).tobytes()
    ).hexdigest()

    base_path = model_dir / "deployment" / "model.onnx"
    model = onnx.load(str(base_path))
    input_name = model.graph.input[0].name

    # Resolve both paths before changing directory; resolving a relative path
    # afterward incorrectly anchors it below the temporary directory (the same
    # isolation bug fixed in export_precision_variants.py).
    model_input_path = base_path.resolve()
    output_path = Path(args.output).resolve()
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="fault_qdq_control_") as work_dir:
        try:
            os.chdir(work_dir)
            quantize_static(
                model_input=str(model_input_path),
                model_output=str(output_path),
                calibration_data_reader=ArrayCalibrationReader(
                    calibration_data, input_name
                ),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                extra_options={
                    "ActivationSymmetric": True,
                    "WeightSymmetric": True,
                },
            )
        finally:
            os.chdir(original_cwd)

    quantized = onnx.load(str(output_path))
    rewired = strip_bias_quantization(quantized)
    if rewired:
        onnx.save(quantized, str(output_path))
        quantized = onnx.load(str(output_path))
    onnx.checker.check_model(quantized)
    counts = Counter(node.op_type for node in quantized.graph.node)
    summary = {
        "control": "format-restricted QDQ (Conv/MatMul/Gemm only, FP32 biases)",
        "model_dir": str(model_dir),
        "output": str(output_path),
        "calibration_samples": int(len(calibration_data)),
        "calibration_sha256": calibration_sha256,
        "bias_qdq_pairs_removed": rewired,
        "quantize_nodes": counts["QuantizeLinear"],
        "dequantize_nodes": counts["DequantizeLinear"],
        "quantized_compute_nodes": {
            op: counts[op] for op in ("Conv", "MatMul", "Gemm") if counts[op]
        },
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
