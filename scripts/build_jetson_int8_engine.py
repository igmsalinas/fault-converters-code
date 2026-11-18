#!/usr/bin/env python3
"""Build one calibrated TensorRT INT8 engine on JetPack 4.6."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.onnx_trt import compile_onnx_to_tensorrt
from src.deployment.utils import get_calibration_dataset
from src.inference.predictor import AnomalyPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", default="data/buck/buck_data/dataset_01")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--num-calib", type=int, default=100)
    args = parser.parse_args()

    # Calibration/model loading is CPU-only; leave Nano's GPU memory available
    # for TensorRT and PyCUDA in the same process.
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    np.random.seed(42)
    model_dir = Path(args.model_dir)
    predictor = AnomalyPredictor(model_dir=str(model_dir))
    calibration_data = get_calibration_dataset(
        data_dir=args.data_dir,
        max_files=args.max_files,
        num_samples=args.num_calib,
        preprocessor=predictor.preprocessor,
        split_indices_path=str(model_dir / "split_indices.npz"),
    )
    deployment = model_dir / "deployment"
    output = deployment / "model_int8.engine"
    success = compile_onnx_to_tensorrt(
        str(deployment / "model.onnx"),
        str(output),
        precision_mode="INT8",
        calibration_data=calibration_data,
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
