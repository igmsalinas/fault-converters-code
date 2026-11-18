"""
Dataset Classes
===============

PyTorch/Keras-compatible dataset classes for power converter data.
"""

import hashlib
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from pathlib import Path

from .loader import DataLoader, SimulationMetadata
from .preprocessor import DataPreprocessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TransferFunctionData:
    """
    Container for a single transfer function sample.

    Attributes:
        frequency: Frequency values (Hz)
        amplitude: Amplitude values (dB)
        phase: Phase values (degrees or radians)
        metadata: Optional simulation metadata
    """

    frequency: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    metadata: Optional[SimulationMetadata] = None

    @property
    def num_points(self) -> int:
        """Number of frequency points."""
        return len(self.frequency)

    def to_array(self, include_frequency: bool = False) -> np.ndarray:
        """
        Convert to numpy array.

        Args:
            include_frequency: Whether to include frequency as feature

        Returns:
            Array of shape (num_points, n_features)
        """
        if include_frequency:
            return np.stack([self.frequency, self.amplitude, self.phase], axis=-1)
        return np.stack([self.amplitude, self.phase], axis=-1)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        frequency: Optional[np.ndarray] = None,
        metadata: Optional[SimulationMetadata] = None,
    ) -> "TransferFunctionData":
        """
        Create from numpy array.

        Args:
            data: Array of shape (num_points, 2 or 3)
            frequency: Optional frequency values
            metadata: Optional metadata

        Returns:
            TransferFunctionData instance
        """
        if data.shape[-1] == 3:
            return cls(
                frequency=data[:, 0],
                amplitude=data[:, 1],
                phase=data[:, 2],
                metadata=metadata,
            )
        elif data.shape[-1] == 2:
            if frequency is None:
                frequency = np.arange(data.shape[0])
            return cls(
                frequency=frequency,
                amplitude=data[:, 0],
                phase=data[:, 1],
                metadata=metadata,
            )
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")


class PowerConverterDataset:
    """
    Dataset class for power converter transfer function data.

    Provides data loading, preprocessing, and batching functionality
    compatible with Keras training loops.
    """

    def __init__(
        self,
        data_dir,
        cache_dir: Optional[str] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        normal_threshold: float = 5.0,
        use_component_ranges: bool = True,
        component_ranges=None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory, or list of directories, with simulation files
            cache_dir: Directory for caching
            preprocessor: Optional preprocessor instance
            normal_threshold: Fallback flat threshold (%) for components without
                a declared range (and the rule when component ranges are off)
            use_component_ranges: Label samples with the per-component
                tolerance/degradation bands (default) vs. the legacy flat rule.
            component_ranges: Optional ranges spec; if None it is auto-discovered
                from ``data/<converter>/component_ranges.json``.
        """
        self.data_dir = data_dir
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.normal_threshold = normal_threshold

        # Data loader
        self.loader = DataLoader(
            data_dir=data_dir,
            cache_dir=cache_dir,
            normal_threshold=normal_threshold,
            use_component_ranges=use_component_ranges,
            component_ranges=component_ranges,
        )

        # Preprocessor
        self.preprocessor = preprocessor or DataPreprocessor()

        # Data storage
        self._raw_data = None
        self._processed_data = None
        self._metadata = None
        self._splits = None

        self._is_loaded = False
        self._is_processed = False

    def load(
        self,
        max_files: Optional[int] = None,
        use_cache: bool = True,
    ) -> "PowerConverterDataset":
        """
        Load raw data from files.

        Args:
            max_files: Maximum number of files to load
            use_cache: Whether to use cache

        Returns:
            Self
        """
        self.loader.load(max_files=max_files, use_cache=use_cache)
        self._raw_data = self.loader.data
        self._metadata = self.loader.metadata
        self._is_loaded = True

        logger.info(f"Loaded {len(self._raw_data)} samples")
        return self

    def preprocess(self, fit: bool = True) -> "PowerConverterDataset":
        """
        Preprocess loaded data.

        Args:
            fit: Whether to fit preprocessor (True for training data)

        Returns:
            Self
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        if fit:
            if self._splits is None or "train_idx" not in self._splits:
                raise RuntimeError(
                    "Prepare splits before fitting preprocessing so the scaler is "
                    "fit on training normals only. Call prepare_splits() first."
                )
            train_idx = self._splits["train_idx"]
            if len(train_idx) == 0:
                raise RuntimeError("Cannot fit preprocessing: training split is empty.")
            self.preprocessor.fit(self._raw_data[train_idx])

        self._processed_data = self.preprocessor.transform(self._raw_data)
        self._is_processed = True
        if self._splits is not None:
            self._materialize_splits()

        logger.info(f"Preprocessed data shape: {self._processed_data.shape}")
        return self

    def get_train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        include_anomalies_in_test: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Split raw data into train/val/test sets.

        Delegates to loader's method.
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self.loader.get_train_val_test_split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            include_anomalies_in_test=include_anomalies_in_test,
        )

    def prepare_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        deduplicate_groups: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Prepare train/val/test splits.

        Training and validation use only normal data.
        Test includes both normal and anomalous data.

        Args:
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            seed: Random seed
            deduplicate_groups: Use one representative for every repeated
                physical component vector. Raw source files are never modified.

        Returns:
            Dictionary with split arrays and labels
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if min(train_ratio, val_ratio, test_ratio) < 0:
            raise ValueError("Split ratios must be non-negative")

        rng = np.random.default_rng(seed)

        # Get indices
        normal_idx = self.loader.get_normal_indices()
        anomaly_idx = self.loader.get_anomaly_indices()

        logger.info(
            f"Normal samples: {len(normal_idx)}, Anomaly samples: {len(anomaly_idx)}"
        )

        def group_key(index: int) -> str:
            metadata = self._metadata[int(index)]
            if metadata.variations:
                values = "|".join(
                    f"{name}={float(value):.8g}"
                    for name, value in sorted(metadata.variations.items())
                )
                return "components:" + values
            return "signal:" + hashlib.sha256(
                np.ascontiguousarray(self._raw_data[int(index)]).tobytes()
            ).hexdigest()

        def shuffled_groups(indices: np.ndarray):
            groups = {}
            for index in indices:
                groups.setdefault(group_key(int(index)), []).append(int(index))
            keys = list(groups)
            rng.shuffle(keys)
            if deduplicate_groups:
                return [[groups[key][0]] for key in keys]
            return [groups[key] for key in keys]

        def take_groups(groups, target_rows):
            selected, count = [], 0
            while groups and count < target_rows:
                group = groups.pop()
                selected.extend(group)
                count += len(group)
            return np.asarray(selected, dtype=np.int64)

        # Keep every repeated physical component vector in exactly one split.
        normal_groups = shuffled_groups(normal_idx)
        n_normal = sum(len(group) for group in normal_groups)
        train_idx = take_groups(normal_groups, int(n_normal * train_ratio))
        val_idx = take_groups(normal_groups, int(n_normal * val_ratio))
        test_normal_idx = np.asarray(
            [index for group in normal_groups for index in group], dtype=np.int64
        )

        # Validation receives approximately one anomaly per validation normal;
        # all repeated fault vectors remain together.
        anomaly_groups = shuffled_groups(anomaly_idx)
        n_anomaly = sum(len(group) for group in anomaly_groups)
        fit_val_anom_idx = take_groups(
            anomaly_groups, min(len(val_idx), n_anomaly)
        )
        test_anomaly_idx = np.asarray(
            [index for group in anomaly_groups for index in group], dtype=np.int64
        )

        fit_val_idx = np.concatenate([val_idx, fit_val_anom_idx])
        fit_val_labels = np.concatenate(
            [np.zeros(len(val_idx)), np.ones(len(fit_val_anom_idx))]
        )

        fit_val_perm = rng.permutation(len(fit_val_idx))
        fit_val_idx = fit_val_idx[fit_val_perm]
        fit_val_labels = fit_val_labels[fit_val_perm]

        # Create test set with remaining anomalies
        test_idx = np.concatenate([test_normal_idx, test_anomaly_idx])
        test_labels = np.concatenate(
            [
                np.zeros(len(test_normal_idx)),
                np.ones(len(test_anomaly_idx)),
            ]
        )

        # Shuffle test set
        test_perm = rng.permutation(len(test_idx))
        test_idx = test_idx[test_perm]
        test_labels = test_labels[test_perm]

        self._splits = {
            "test_labels": test_labels,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "fit_val_labels": fit_val_labels,
            "fit_val_idx": fit_val_idx,
        }
        if self._is_processed:
            self._materialize_splits()

        logger.info(
            f"Splits - Train: {len(train_idx)}, Val: {len(val_idx)}, "
            f"Fit Val: {len(fit_val_idx)} (Normal: {len(val_idx)}, Anomaly: {len(fit_val_anom_idx)}), "
            f"Test: {len(test_idx)} (Normal: {len(test_normal_idx)}, Anomaly: {len(test_anomaly_idx)})"
        )
        if deduplicate_groups:
            selected = len(train_idx) + len(fit_val_idx) + len(test_idx)
            logger.info(
                "Deduplicated %d repeated component-vector rows for splitting; "
                "%d unique samples retained.",
                len(self._raw_data) - selected,
                selected,
            )

        return self._splits

    def _materialize_splits(self) -> None:
        """Populate split arrays after indices and preprocessing are available."""
        if self._splits is None or self._processed_data is None:
            return
        self._splits["train"] = self._processed_data[self._splits["train_idx"]]
        self._splits["val"] = self._processed_data[self._splits["val_idx"]]
        self._splits["test"] = self._processed_data[self._splits["test_idx"]]
        self._splits["fit_val_data"] = self._processed_data[
            self._splits["fit_val_idx"]
        ]

    def load_split_indices(self, path: str) -> Dict[str, np.ndarray]:
        """Load and validate the exact split assignment saved during training."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        required = {
            "train_idx", "val_idx", "fit_val_idx", "fit_val_labels",
            "test_idx", "test_labels",
        }
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Saved split file is missing keys: {sorted(missing)}")
            splits = {key: np.asarray(archive[key]) for key in required}

        for key in ("train_idx", "val_idx", "fit_val_idx", "test_idx"):
            indices = splits[key].astype(np.int64, copy=False)
            if (
                indices.ndim != 1
                or np.any(indices < 0)
                or np.any(indices >= self.num_samples)
            ):
                raise ValueError(f"Invalid indices in saved split field {key}")
            splits[key] = indices
        if len(splits["fit_val_idx"]) != len(splits["fit_val_labels"]):
            raise ValueError("fit-validation indices and labels have different lengths")
        if len(splits["test_idx"]) != len(splits["test_labels"]):
            raise ValueError("test indices and labels have different lengths")

        groups = [
            set(splits["train_idx"]),
            set(splits["fit_val_idx"]),
            set(splits["test_idx"]),
        ]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Saved training, fit-validation, and test splits overlap")

        self._splits = splits
        if self._is_processed:
            self._materialize_splits()
        logger.info("Loaded exact training split assignment from %s", path)
        return self._splits

    @property
    def train_data(self) -> np.ndarray:
        """Get training data."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits["train"]

    @property
    def val_data(self) -> np.ndarray:
        """Get validation data."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits["val"]

    @property
    def fit_val_data(self) -> np.ndarray:
        """Get fit validation data."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits.get("fit_val_data", self._splits["val"])

    @property
    def fit_val_labels(self) -> np.ndarray:
        """Get fit validation labels."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits.get("fit_val_labels", np.zeros(len(self._splits["val"])))

    @property
    def test_data(self) -> np.ndarray:
        """Get test data."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits["test"]

    @property
    def test_labels(self) -> np.ndarray:
        """Get test labels."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits["test_labels"]

    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        return len(self._raw_data) if self._is_loaded else 0

    @property
    def sequence_length(self) -> int:
        """Sequence length (number of frequency points)."""
        if self._processed_data is not None:
            return self._processed_data.shape[1]
        return 0

    @property
    def num_features(self) -> int:
        """Number of features per time step."""
        if self._processed_data is not None:
            return self._processed_data.shape[2]
        return 0

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Input shape for model (seq_len, n_features)."""
        return (self.sequence_length, self.num_features)

    @property
    def frequencies(self) -> Optional[np.ndarray]:
        """Frequency grid (Hz) shared by every sample, or ``None`` if unavailable."""
        return self.loader.frequencies

    def get_metadata_for_split(self, split: str) -> List[SimulationMetadata]:
        """Get metadata for a specific split."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")

        idx_key = f"{split}_idx"
        if idx_key not in self._splits:
            raise ValueError(f"Unknown split: {split}")

        indices = self._splits[idx_key]
        return [self._metadata[i] for i in indices]

    def save(self, path: str) -> None:
        """Save dataset state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save preprocessor
        self.preprocessor.save(str(path / "preprocessor"))

        # Save processed data and splits
        if self._processed_data is not None:
            np.save(path / "processed_data.npy", self._processed_data)

        if self._splits is not None:
            for key, value in self._splits.items():
                np.save(path / f"split_{key}.npy", value)

    @classmethod
    def load_processed(cls, path: str) -> "PowerConverterDataset":
        """Load preprocessed dataset."""
        path = Path(path)

        # Create instance with minimal setup
        dataset = cls.__new__(cls)
        dataset.preprocessor = DataPreprocessor.load(str(path / "preprocessor"))

        # Load processed data
        dataset._processed_data = np.load(path / "processed_data.npy")
        dataset._is_processed = True

        # Load splits
        dataset._splits = {}
        for split_file in path.glob("split_*.npy"):
            key = split_file.stem.replace("split_", "")
            dataset._splits[key] = np.load(split_file)

        return dataset


# For backward compatibility
BuckConverterDataset = PowerConverterDataset
