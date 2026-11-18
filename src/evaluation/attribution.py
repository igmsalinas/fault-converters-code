"""Transformer Self-Attention Extraction and Jacobian Frequency Attribution.

Provides exact, weight-preserving introspection utilities to test whether
self-attention and anomaly-score gradients concentrate on physically meaningful
converter landmarks (DC resistive drop, LC resonance peak, and ESR high-frequency zero).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import Model, layers

from ..models.transformer_ae import TransformerBlock

DEFAULT_FREQUENCY_BANDS_HZ: Dict[str, Tuple[float, float]] = {
    "dc_low": (100.0, 400.0),
    "resonance": (700.0, 2000.0),
    "mid_inductive": (2000.0, 8000.0),
    "hf_esr": (8000.0, 50000.0),
}


def _discover_encoder_attention_probes(encoder: Model) -> List[Tuple[Model, layers.MultiHeadAttention]]:
    """Return ``(pre_input_submodel, mha_layer)`` pairs in forward order."""
    probes: List[Tuple[Model, layers.MultiHeadAttention]] = []
    for lyr in encoder.layers:
        if isinstance(lyr, layers.MultiHeadAttention):
            in_tensor = lyr.input[0] if isinstance(lyr.input, (list, tuple)) else lyr.input
            probes.append((Model(encoder.inputs, in_tensor), lyr))
        elif isinstance(lyr, TransformerBlock) and hasattr(lyr, "att"):
            in_tensor = lyr.input[0] if isinstance(lyr.input, (list, tuple)) else lyr.input
            probes.append((Model(encoder.inputs, in_tensor), lyr.att))
    return probes


def extract_transformer_attention_maps(
    model: Any,
    data: np.ndarray,
    batch_size: int = 128,
) -> List[np.ndarray]:
    """Extract exact ``(N, num_heads, seq_len, seq_len)`` self-attention weights per encoder layer.

    Raises ``ValueError`` if ``model.encoder`` contains no ``MultiHeadAttention`` layers.
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("Model does not expose an 'encoder' sub-model with attention layers.")

    probes = _discover_encoder_attention_probes(encoder)
    if not probes:
        raise ValueError(
            f"Encoder '{encoder.name}' contains no MultiHeadAttention layers to inspect."
        )

    arr = np.asarray(data, dtype=np.float32)
    per_layer_chunks: List[List[np.ndarray]] = [[] for _ in probes]

    for start in range(0, len(arr), batch_size):
        batch = tf.convert_to_tensor(arr[start : start + batch_size], dtype=tf.float32)
        for idx, (pre_model, mha_layer) in enumerate(probes):
            h = pre_model(batch, training=False)
            _, attn_scores = mha_layer(h, h, return_attention_scores=True, training=False)
            per_layer_chunks[idx].append(np.asarray(attn_scores, dtype=np.float32))

    return [np.concatenate(chunks, axis=0) for chunks in per_layer_chunks]


def summarize_frequency_attention(
    attention_maps: Sequence[np.ndarray],
    frequencies: Sequence[float],
    bands: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Quantify key-attention concentration, entropy, and cross-band routing across frequencies."""
    if not attention_maps:
        raise ValueError("attention_maps must be non-empty")

    freqs = np.asarray(frequencies, dtype=np.float64)
    seq_len = int(freqs.shape[0])
    band_defs = dict(DEFAULT_FREQUENCY_BANDS_HZ if bands is None else bands)

    layer_profiles: List[np.ndarray] = []
    layer_entropies: List[float] = []
    layer_matrices: List[np.ndarray] = []

    for attn in attention_maps:
        a = np.asarray(attn, dtype=np.float64)
        if a.ndim != 4 or a.shape[-1] != seq_len or a.shape[-2] != seq_len:
            raise ValueError(f"Expected (N, H, {seq_len}, {seq_len}) attention tensor, got {a.shape}")
        # Row-normalized entropy per (sample, head, query)
        clipped = np.clip(a, 1e-12, 1.0)
        ent = -np.sum(clipped * np.log(clipped), axis=-1) / max(np.log(seq_len), 1e-12)
        layer_entropies.append(float(np.mean(ent)))

        # Mean (S_query, S_key) matrix and incoming key marginal
        mean_mat = a.mean(axis=(0, 1))
        key_prof = mean_mat.mean(axis=0)
        key_prof = key_prof / max(float(key_prof.sum()), 1e-12)
        layer_matrices.append(mean_mat)
        layer_profiles.append(key_prof)

    mean_matrix = np.mean(np.stack(layer_matrices, axis=0), axis=0)
    mean_key = np.mean(np.stack(layer_profiles, axis=0), axis=0)
    mean_key = mean_key / max(float(mean_key.sum()), 1e-12)

    peak_idx = int(np.argmax(mean_key))
    band_stats: Dict[str, Dict[str, float]] = {}
    band_masks: Dict[str, np.ndarray] = {}
    for bname, (f_lo, f_hi) in band_defs.items():
        mask = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
        band_masks[bname] = mask
        frac = float(np.mean(mask))
        mass = float(np.sum(mean_key[mask])) if np.any(mask) else 0.0
        band_stats[bname] = {
            "freq_min_hz": float(f_lo),
            "freq_max_hz": float(f_hi),
            "bin_fraction": frac,
            "attention_mass": mass,
            "enrichment_ratio": float(mass / frac) if frac > 0.0 else 0.0,
        }

    # Cross-band query -> key routing: average attention mass sent from query band b_q to key band b_k
    cross_band: Dict[str, Dict[str, float]] = {}
    for q_name, q_mask in band_masks.items():
        cross_band[q_name] = {}
        if not np.any(q_mask):
            for k_name in band_masks:
                cross_band[q_name][k_name] = 0.0
            continue
        q_rows = mean_matrix[q_mask]  # (n_q, S)
        for k_name, k_mask in band_masks.items():
            if not np.any(k_mask):
                cross_band[q_name][k_name] = 0.0
            else:
                mass_qk = float(np.mean(np.sum(q_rows[:, k_mask], axis=-1)))
                k_frac = float(np.mean(k_mask))
                cross_band[q_name][k_name] = float(mass_qk / k_frac) if k_frac > 0.0 else 0.0

    return {
        "num_layers": len(attention_maps),
        "peak_attended_index": peak_idx,
        "peak_attended_freq_hz": float(freqs[peak_idx]),
        "peak_attention_mass": float(mean_key[peak_idx]),
        "normalized_entropy_mean": float(np.mean(layer_entropies)),
        "normalized_entropy_by_layer": layer_entropies,
        "mean_key_attention": mean_key.tolist(),
        "layer_key_attention": [p.tolist() for p in layer_profiles],
        "bands": band_stats,
        "cross_band_enrichment": cross_band,
    }


def compute_knn_score_frequency_saliency(
    model: Any,
    query_data: np.ndarray,
    reference_data: np.ndarray,
    space: str = "projection",
    k: int = 5,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute exact Jacobian norm ``||d s_kNN(x) / d x_f||_2`` per frequency bin ``(N, seq_len)``."""
    ref_arr = np.asarray(reference_data, dtype=np.float32)
    q_arr = np.asarray(query_data, dtype=np.float32)

    def _forward_emb(x_tensor: tf.Tensor) -> tf.Tensor:
        z = model.encoder(x_tensor, training=False)
        if space == "projection" and hasattr(model, "projection_head"):
            p = model.projection_head(z, training=False)
            norms = tf.maximum(tf.norm(p, axis=-1, keepdims=True), 1e-8)
            return p / norms
        return z

    ref_chunks = []
    for i in range(0, len(ref_arr), batch_size):
        ref_chunks.append(_forward_emb(tf.convert_to_tensor(ref_arr[i : i + batch_size])))
    ref_emb = tf.concat(ref_chunks, axis=0)  # (N_ref, D)
    k_eff = min(int(k), int(ref_emb.shape[0]))

    sal_chunks: List[np.ndarray] = []
    for i in range(0, len(q_arr), batch_size):
        xb = tf.Variable(q_arr[i : i + batch_size], dtype=tf.float32)
        with tf.GradientTape() as tape:
            qb = _forward_emb(xb)  # (B, D)
            # Exact pairwise Euclidean distance to reference bank
            diff_sq = (
                tf.reduce_sum(tf.square(qb), axis=-1, keepdims=True)
                + tf.reshape(tf.reduce_sum(tf.square(ref_emb), axis=-1), (1, -1))
                - 2.0 * tf.matmul(qb, ref_emb, transpose_b=True)
            )
            dists = tf.sqrt(tf.maximum(diff_sq, 1e-12))
            neg_topk, _ = tf.math.top_k(-dists, k=k_eff)
            knn_scores = tf.reduce_mean(-neg_topk, axis=-1)
            total_score = tf.reduce_sum(knn_scores)
        grads = tape.gradient(total_score, xb)  # (B, S, C)
        token_norm = tf.linalg.norm(grads, axis=-1)  # (B, S)
        sal_chunks.append(np.asarray(token_norm, dtype=np.float32))

    return np.concatenate(sal_chunks, axis=0)
