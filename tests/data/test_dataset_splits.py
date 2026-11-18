"""Regression tests for leakage-safe dataset preparation."""

import numpy as np
import pytest

from src.data.dataset import PowerConverterDataset
from src.data.loader import SimulationMetadata
from src.data.preprocessor import DataPreprocessor


class _IndexLoader:
    def __init__(self, normal, anomaly):
        self._normal = np.asarray(normal, dtype=np.int64)
        self._anomaly = np.asarray(anomaly, dtype=np.int64)

    def get_normal_indices(self):
        return self._normal.copy()

    def get_anomaly_indices(self):
        return self._anomaly.copy()


def _dataset_with_repeated_components():
    # Component A is deliberately repeated among normals, and F among faults.
    components = ["A", "A", "B", "C", "C", "D", "E", "F", "F", "G", "H"]
    component_values = {name: float(ord(name) - ord("A")) for name in set(components)}
    raw = np.zeros((len(components), 3, 2), dtype=np.float64)
    raw[:, :, 0] = np.arange(len(components), dtype=np.float64)[:, None]

    dataset = PowerConverterDataset.__new__(PowerConverterDataset)
    dataset.loader = _IndexLoader(normal=range(7), anomaly=range(7, 11))
    dataset.preprocessor = DataPreprocessor(use_real_imag=False)
    dataset._raw_data = raw
    dataset._processed_data = None
    dataset._metadata = [
        SimulationMetadata(f"{index}.txt", {"R": component_values[name]})
        for index, name in enumerate(components)
    ]
    dataset._splits = None
    dataset._is_loaded = True
    dataset._is_processed = False
    return dataset


def _component_keys(dataset, indices):
    return {
        tuple(sorted(dataset._metadata[int(index)].variations.items()))
        for index in indices
    }


def test_grouped_split_prevents_duplicate_component_leakage():
    dataset = _dataset_with_repeated_components()
    splits = dataset.prepare_splits(
        train_ratio=0.4, val_ratio=0.2, test_ratio=0.4, seed=7
    )

    train_keys = _component_keys(dataset, splits["train_idx"])
    val_keys = _component_keys(dataset, splits["val_idx"])
    test_keys = _component_keys(dataset, splits["test_idx"])
    fit_anomaly_idx = splits["fit_val_idx"][splits["fit_val_labels"] == 1]
    fit_anomaly_keys = _component_keys(dataset, fit_anomaly_idx)

    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)
    assert fit_anomaly_keys.isdisjoint(test_keys)
    selected_indices = np.concatenate(
        [splits["train_idx"], splits["fit_val_idx"], splits["test_idx"]]
    )
    assert len(selected_indices) == len(_component_keys(dataset, selected_indices))


def test_preprocessor_fits_training_normals_only():
    dataset = _dataset_with_repeated_components()
    splits = dataset.prepare_splits(
        train_ratio=0.4, val_ratio=0.2, test_ratio=0.4, seed=7
    )
    expected_mean = dataset._raw_data[splits["train_idx"], :, 0].mean()

    dataset.preprocess(fit=True)

    assert dataset.preprocessor.scaler.mean_.item() == pytest.approx(expected_mean)
    assert splits["train"].shape[0] == len(splits["train_idx"])
    assert splits["fit_val_data"].shape[0] == len(splits["fit_val_idx"])


def test_preprocessing_requires_prepared_training_split():
    dataset = _dataset_with_repeated_components()
    with pytest.raises(RuntimeError, match="Prepare splits"):
        dataset.preprocess(fit=True)


def test_saved_split_assignment_roundtrip(tmp_path):
    original = _dataset_with_repeated_components()
    splits = original.prepare_splits(0.4, 0.2, 0.4, seed=7)
    path = tmp_path / "split_indices.npz"
    np.savez(path, **{
        key: splits[key] for key in (
            "train_idx", "val_idx", "fit_val_idx", "fit_val_labels",
            "test_idx", "test_labels",
        )
    })

    restored = _dataset_with_repeated_components()
    loaded = restored.load_split_indices(str(path))
    for key in (
        "train_idx", "val_idx", "fit_val_idx", "fit_val_labels",
        "test_idx", "test_labels",
    ):
        np.testing.assert_array_equal(loaded[key], splits[key])


def test_saved_split_assignment_rejects_overlap(tmp_path):
    path = tmp_path / "overlap.npz"
    np.savez(
        path,
        train_idx=np.array([0]),
        val_idx=np.array([1]),
        fit_val_idx=np.array([0, 7]),
        fit_val_labels=np.array([0, 1]),
        test_idx=np.array([2, 8]),
        test_labels=np.array([0, 1]),
    )
    dataset = _dataset_with_repeated_components()
    with pytest.raises(ValueError, match="overlap"):
        dataset.load_split_indices(str(path))


@pytest.mark.parametrize("ratios", [(0.7, 0.2, 0.2), (-0.1, 0.5, 0.6)])
def test_invalid_split_ratios_are_rejected(ratios):
    dataset = _dataset_with_repeated_components()
    with pytest.raises(ValueError):
        dataset.prepare_splits(*ratios)
