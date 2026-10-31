"""Accuracy metrics for estimated network sequences.

Every function takes one ``(T, d_m, d_m)`` array per mode.  Edge recovery is
scored on signed edges: the positive and negative edges are separate
channels, so an edge estimated with the wrong sign is never counted as
recovered.  Change-point detection reads the temporal deviation of ``A_t``.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def temporal_deviations(networks: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Return the relative Frobenius change at every transition.

    The temporal deviation at time ``t`` is the Frobenius change from
    ``t - 1`` divided by the Frobenius norm at ``t``, as defined for KTVGL.
    A machine-precision floor keeps an all-zero estimated network finite; its
    zero-to-zero deviation remains zero.
    """
    deviations = []
    for network in networks:
        sequence = np.asarray(network, dtype=float)
        changes = np.linalg.norm(np.diff(sequence, axis=0), axis=(1, 2))
        current_norms = np.linalg.norm(sequence[1:], axis=(1, 2))
        deviations.append(
            changes / np.maximum(current_norms, np.finfo(float).eps)
        )
    return deviations


def signed_edge_score_labels(
    estimated: Sequence[np.ndarray],
    truth: Sequence[np.ndarray],
    *,
    truth_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand every off-diagonal edge into positive and negative channels.

    The positive channel scores ``+A_hat`` against ``A_true > 0`` and the
    negative channel scores ``-A_hat`` against ``A_true < 0``.  A sign flip
    therefore never counts as a recovered edge, unlike an absolute-value
    ranking.
    """
    scores = []
    labels = []
    for estimated_mode, true_mode in zip(estimated, truth):
        estimated_mode = np.asarray(estimated_mode, dtype=float)
        true_mode = np.asarray(true_mode, dtype=float)
        dimension = true_mode.shape[1]
        upper = np.triu_indices(dimension, k=1)
        for t in range(true_mode.shape[0]):
            estimated_edges = estimated_mode[t][upper]
            true_edges = true_mode[t][upper]
            scores.extend(
                (
                    estimated_edges,
                    -estimated_edges,
                )
            )
            labels.extend(
                (
                    (true_edges > truth_tolerance).astype(int),
                    (true_edges < -truth_tolerance).astype(int),
                )
            )
    return np.concatenate(scores), np.concatenate(labels)


def edge_recovery_metrics(
    estimated: Sequence[np.ndarray],
    truth: Sequence[np.ndarray],
    *,
    truth_tolerance: float = 1e-12,
) -> dict[str, float]:
    """Compute signed micro AUCs by separating positive/negative edges."""
    scores, labels = signed_edge_score_labels(
        estimated,
        truth,
        truth_tolerance=truth_tolerance,
    )
    if labels.min() == labels.max():
        roc_auc = np.nan
        average_precision = np.nan
    else:
        roc_auc = float(roc_auc_score(labels, scores))
        average_precision = float(average_precision_score(labels, scores))
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "AUC-ROC": roc_auc,
        "AUC-PR": average_precision,
        "Best-F1": float(np.nanmax(f1)),
    }


def change_point_metrics(
    estimated: Sequence[np.ndarray],
    change_points: Sequence[Sequence[int]],
    *,
    tolerance: int = 1,
) -> dict[str, float]:
    """Compute Temporal Deviation Ratio and top-k change-point recall."""
    deviations = temporal_deviations(estimated)
    tdr_values = []
    recalled = []
    for mode_deviation, mode_points in zip(deviations, change_points):
        if not mode_points:
            continue
        if float(np.max(mode_deviation)) <= np.finfo(float).tiny:
            tdr_values.append(0.0)
            recalled.extend(False for _ in mode_points)
            continue
        denominator = max(float(np.mean(mode_deviation)), np.finfo(float).tiny)
        tdr_values.append(
            float(np.mean([mode_deviation[point - 1] for point in mode_points]))
            / denominator
        )
        top_indexes = np.argsort(mode_deviation)[-len(mode_points) :] + 1
        recalled.extend(
            any(
                abs(int(candidate) - point) <= tolerance
                for candidate in top_indexes
            )
            for point in mode_points
        )
    return {
        "TDR": float(np.mean(tdr_values)) if tdr_values else np.nan,
        "CP-recall@K": float(np.mean(recalled)) if recalled else np.nan,
    }


def project_flat_network_to_modes(
    flat_network: np.ndarray,
    dims: Sequence[int],
) -> list[np.ndarray]:
    """Recover mode scores from flat edges whose other coordinates agree."""
    flat_network = np.asarray(flat_network, dtype=float)
    if flat_network.ndim != 3:
        raise ValueError("flat_network must have shape (T, D, D).")
    T, D, width = flat_network.shape
    if D != width or D != int(np.prod(dims)):
        raise ValueError("flat_network shape does not match dims.")

    projected = []
    for mode, dimension in enumerate(dims):
        values = np.zeros((T, dimension, dimension), dtype=float)
        other_modes = [index for index in range(len(dims)) if index != mode]
        other_coordinates = itertools.product(
            *[range(dims[index]) for index in other_modes]
        )
        other_coordinates = list(other_coordinates)
        for i in range(dimension):
            for j in range(i + 1, dimension):
                pairs = []
                for fixed in other_coordinates:
                    left = [0] * len(dims)
                    right = [0] * len(dims)
                    left[mode] = i
                    right[mode] = j
                    for index, coordinate in zip(other_modes, fixed):
                        left[index] = coordinate
                        right[index] = coordinate
                    left_flat = np.ravel_multi_index(tuple(left), dims)
                    right_flat = np.ravel_multi_index(tuple(right), dims)
                    pairs.append(flat_network[:, left_flat, right_flat])
                edge_values = np.mean(np.stack(pairs, axis=0), axis=0)
                values[:, i, j] = edge_values
                values[:, j, i] = edge_values
        projected.append(values)
    return projected
