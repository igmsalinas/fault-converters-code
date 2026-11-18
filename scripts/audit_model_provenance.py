#!/usr/bin/env python3
"""Record checkpoint, loaded-weight, threshold, and probe-output provenance."""

import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path
import socket
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import AnomalyPredictor
from src.deployment.utils import build_carla_embedding_model


def sanitize_hostname(value):
    """Return a hostname that is safe to embed in an artifact filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays_sha256(arrays):
    digest = hashlib.sha256()
    shapes = []
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        shapes.append(list(array.shape))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest(), shapes


def weight_multiset_sha256(arrays):
    """Order-insensitive digest over per-tensor weight hashes.

    Keras versions enumerate a model's final weights in different orders, so
    the aggregate order-sensitive digest differs across hosts even when every
    tensor is byte-identical. Comparing sorted per-tensor digests keeps the
    cross-host invariant about content, not enumeration order.
    """
    digests = []
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        digests.append(digest.hexdigest())
    combined = hashlib.sha256()
    for digest in sorted(digests):
        combined.update(digest.encode("ascii"))
    return combined.hexdigest(), digests


def audit(model_dir):
    model_dir = Path(model_dir)
    predictor = AnomalyPredictor(model_dir=str(model_dir))
    if predictor.reference_embeddings is None:
        raise RuntimeError(f"Missing CARLA reference bank under {model_dir}")
    if predictor.threshold is None:
        raise RuntimeError(f"Missing fit-validation threshold under {model_dir}")
    detector = build_carla_embedding_model(predictor.model)
    weights = detector.get_weights()
    weight_digest, weight_shapes = arrays_sha256(weights)
    multiset_digest, tensor_digests = weight_multiset_sha256(weights)
    probe = np.zeros((1,) + tuple(detector.input_shape[1:]), np.float32)
    output = detector(probe, training=False).numpy()
    output_digest, _ = arrays_sha256([output])

    tracked_files = []
    for relative in (
        "checkpoints/best/autoencoder.weights.h5",
        "checkpoints/best/autoencoder.weights.npz",
        "checkpoints/best/encoder.weights.npz",
        "checkpoints/best/decoder.weights.npz",
        "checkpoints/best/projection_head.weights.npz",
        "checkpoints/best/reference_embeddings.npy",
        "model_config.json",
        "threshold.json",
        "preprocessor/config.json",
        "preprocessor/scaler.json",
    ):
        path = model_dir / relative
        if path.exists():
            tracked_files.append({
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })

    result = {
        "device": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "model_dir": str(model_dir),
        "checkpoint_selection": "checkpoints/best",
        "threshold": float(predictor.threshold),
        "detector_parameter_count": int(sum(np.asarray(w).size for w in weights)),
        "detector_weight_tensor_count": len(weights),
        "detector_loaded_weights_sha256": weight_digest,
        "detector_weight_shapes": weight_shapes,
        "detector_loaded_weights_multiset_sha256": multiset_digest,
        "detector_weight_tensor_sha256": tensor_digests,
        "zero_probe_embedding_sha256": output_digest,
        "reference_embedding_shape": list(predictor.reference_embeddings.shape),
        "reference_embeddings_sha256": arrays_sha256(
            [predictor.reference_embeddings]
        )[0],
        "scoring_method": predictor.scoring_method,
        "k_neighbors": predictor.k_neighbors,
        "tracked_files": tracked_files,
    }
    deployment_dir = model_dir / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    safe_hostname = sanitize_hostname(socket.gethostname())
    output_path = deployment_dir / ("model_provenance_%s.json" % safe_hostname)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    # A workstation report is transferred with the selected model. On the
    # Jetson, compare the values actually loaded by its older TensorFlow/Keras
    # stack against that report. Output tensors may differ by harmless floating
    # point rounding, so the invariant covers weights, reference bank,
    # threshold, and scorer rather than the probe-output hash. Weight content
    # is compared through the order-insensitive multiset digest because Keras
    # versions enumerate the same byte-identical tensors in different orders.
    invariant_fields = (
        "detector_loaded_weights_multiset_sha256",
        "reference_embedding_shape",
        "reference_embeddings_sha256",
        "threshold",
        "scoring_method",
        "k_neighbors",
    )
    comparisons = []
    for reference_path in sorted(deployment_dir.glob("model_provenance_*.json")):
        if reference_path == output_path:
            continue
        reference = json.loads(reference_path.read_text())
        mismatches = [
            field for field in invariant_fields
            if reference.get(field) != result.get(field)
        ]
        comparisons.append({
            "report": reference_path.name,
            "matched": not mismatches,
            "mismatched_fields": mismatches,
        })
    result["cross_host_comparisons"] = comparisons
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    (deployment_dir / "model_provenance.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    failed = [item for item in comparisons if not item["matched"]]
    if failed:
        raise RuntimeError(
            "Loaded detector provenance differs across hosts: %s" % failed
        )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dirs", nargs="+")
    args = parser.parse_args()
    for model_dir in args.model_dirs:
        audit(model_dir)


if __name__ == "__main__":
    main()
