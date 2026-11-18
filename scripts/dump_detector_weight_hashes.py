#!/usr/bin/env python3
"""Dump per-tensor SHA-256 hashes of the detector embedding model weights.

Unlike the aggregate order-sensitive digest in ``audit_model_provenance.py``,
this report hashes every weight tensor individually so that cross-host
comparisons can distinguish genuine weight differences from framework
enumeration-order differences (e.g. Keras 2.7 vs newer Keras returning the
final dense tensors in a different order).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import AnomalyPredictor
from src.deployment.utils import build_carla_embedding_model


def tensor_sha256(array):
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dirs", nargs="+")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    report = {}
    for model_dir in args.model_dirs:
        model_dir = Path(model_dir)
        predictor = AnomalyPredictor(model_dir=str(model_dir))
        detector = build_carla_embedding_model(predictor.model)
        tensors = []
        for index, value in enumerate(detector.get_weights()):
            array = np.ascontiguousarray(np.asarray(value))
            tensors.append({
                "index": index,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": tensor_sha256(array),
            })
        report[model_dir.name] = {
            "tensor_count": len(tensors),
            "tensors": tensors,
        }

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
