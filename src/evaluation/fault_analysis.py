"""Per-fault recall diagnostics, FRA measurement-noise injection, prevalence
adjustment, and exact paired statistical tests for multi-seed ablations.
"""

from itertools import product
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

# Default anomalous percent-deviation bands: (multiplier - 1.0) * 100
DEFAULT_ANOMALY_PCT_BANDS: Dict[str, Tuple[float, float]] = {
    "Cout": (-70.0, -30.0),
    "Esr_C": (200.0, 700.0),
    "Lout": (-60.0, -30.0),
    "Esr_L": (100.0, 500.0),
    "Rds_1": (200.0, 1400.0),
    "Rds_2": (200.0, 1400.0),
}


def compute_per_fault_recall(
    predictions: np.ndarray,
    variations_list: Sequence[Mapping[str, float]],
    bands: Optional[Mapping[str, Tuple[float, float]]] = None,
    tol: float = 1e-3,
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Compute recall disaggregated by faulted component (any-fault and single-fault).

    Args:
        predictions: Binary anomaly predictions ``(N,)``.
        variations_list: Per-sample dictionary of component percent deviations
            ``(m - 1) * 100`` matching ``SimulationMetadata.variations``.
        bands: Component anomalous percent-deviation intervals ``(low_pct, high_pct)``.
        tol: Numerical boundary tolerance on interval endpoints.
    """
    preds = np.asarray(predictions, dtype=int)
    if preds.shape[0] != len(variations_list):
        raise ValueError("predictions and variations_list must have identical length")

    spec = bands or DEFAULT_ANOMALY_PCT_BANDS
    active_per_sample = []
    for v in variations_list:
        active = [
            comp
            for comp, (lo, hi) in spec.items()
            if (lo - tol) <= float(v.get(comp, 0.0)) <= (hi + tol)
        ]
        active_per_sample.append(active)

    results: Dict[str, Dict[str, Union[int, float]]] = {}
    for comp in spec:
        any_mask = np.array([comp in act for act in active_per_sample], dtype=bool)
        single_mask = np.array([act == [comp] for act in active_per_sample], dtype=bool)
        results[comp] = {
            "n_any": int(any_mask.sum()),
            "recall_any": float(preds[any_mask].mean()) if any_mask.any() else float("nan"),
            "n_single": int(single_mask.sum()),
            "recall_single": float(preds[single_mask].mean()) if single_mask.any() else float("nan"),
        }

    multi_mask = np.array([len(act) >= 2 for act in active_per_sample], dtype=bool)
    results["multi_fault"] = {
        "n_any": int(multi_mask.sum()),
        "recall_any": float(preds[multi_mask].mean()) if multi_mask.any() else float("nan"),
        "n_single": 0,
        "recall_single": float("nan"),
    }
    return results


def inject_fra_measurement_noise(
    raw_bode: np.ndarray,
    sigma_mag_db: float,
    sigma_phase_deg: float,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Inject realistic swept-sine FRA magnitude (dB) and phase (deg) noise.

    Operates on un-standardized physical Bode responses of shape ``(..., 2)``
    where channel 0 is ``|G|_dB`` and channel 1 is ``angle(G)_deg``.
    """
    arr = np.asarray(raw_bode, dtype=np.float64).copy()
    if sigma_mag_db <= 0.0 and sigma_phase_deg <= 0.0:
        return arr

    gen = rng if rng is not None else np.random.default_rng(seed)
    if sigma_mag_db > 0.0:
        arr[..., 0] += gen.normal(0.0, float(sigma_mag_db), size=arr[..., 0].shape)
    if sigma_phase_deg > 0.0 and arr.shape[-1] >= 2:
        arr[..., 1] += gen.normal(0.0, float(sigma_phase_deg), size=arr[..., 1].shape)
    return arr


def compute_prevalence_adjusted_prauc(
    labels: np.ndarray,
    scores: np.ndarray,
    target_prevalence: float,
) -> float:
    """Compute Precision-Recall AUC reweighted to a target anomaly prevalence ``pi``."""
    pi = float(target_prevalence)
    if not (0.0 < pi < 1.0):
        raise ValueError("target_prevalence must lie strictly in (0, 1)")

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    if not np.any(pos) or not np.any(neg):
        raise ValueError("Both classes must be present to compute PR-AUC")

    weights = np.empty_like(s, dtype=np.float64)
    weights[pos] = pi / float(pos.sum())
    weights[neg] = (1.0 - pi) / float(neg.sum())
    return float(average_precision_score(y, s, sample_weight=weights))


def paired_sign_flip_test(
    a: Sequence[float],
    b: Sequence[float],
) -> Dict[str, float]:
    """Exact paired sign-flip permutation test and Wilcoxon signed-rank summary."""
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    if xa.shape != xb.shape or xa.ndim != 1 or xa.size == 0:
        raise ValueError("Inputs must be non-empty 1-D arrays of identical length")

    diffs = xa - xb
    n = diffs.size
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

    # Exact two-sided sign-flip test for n <= 20
    if n <= 20:
        signs = np.asarray(list(product([-1.0, 1.0], repeat=n)), dtype=np.float64)
        perm_means = (signs * diffs[None, :]).mean(axis=1)
        p_sign_flip = float(np.mean(np.abs(perm_means) >= abs(mean_diff) - 1e-15))
    else:
        rng = np.random.default_rng(42)
        signs = rng.choice([-1.0, 1.0], size=(20000, n))
        perm_means = (signs * diffs[None, :]).mean(axis=1)
        p_sign_flip = float(np.mean(np.abs(perm_means) >= abs(mean_diff) - 1e-15))

    p_wilcoxon = float("nan")
    if n >= 5 and np.any(diffs != 0.0):
        try:
            _, p_w = wilcoxon(diffs, alternative="two-sided")
            p_wilcoxon = float(p_w)
        except ValueError:
            p_wilcoxon = float("nan")

    return {
        "n": float(n),
        "mean_diff": mean_diff,
        "sample_std_diff": std_diff,
        "p_value_two_sided": p_sign_flip,
        "wilcoxon_p_two_sided": p_wilcoxon,
    }
