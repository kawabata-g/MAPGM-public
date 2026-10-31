"""Diagnostic plots for MAPGM fits.

Each function draws one figure per mode (per time, for the edge heatmaps)
and saves it as a PNG under a subdirectory of ``save_dir``, or shows it when
``save_dir`` is None.  ``A[m]`` has shape ``(T, d_m, d_m)`` and ``b[m]``
shape ``(T, d_m)``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def _finish(path):
    plt.xlabel("Time")
    plt.grid()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def _mark_change_points(change_points):
    for cp in change_points or ():
        plt.axvline(cp, color="red", linestyle="--", linewidth=0.8)


def viz_temporal_deviation(A, change_points=None, save_dir=None):
    """||A_t - A_{t-1}||_F over time; A[m] has shape (T, d_m, d_m)."""
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "temporal_deviation")
        os.makedirs(save_dir, exist_ok=True)
    for m, A_m in enumerate(A):
        deviation = np.linalg.norm(np.diff(A_m, axis=0), axis=(1, 2))
        plt.figure()
        plt.plot(range(1, len(A_m)), deviation)
        _mark_change_points(None if change_points is None else change_points[m])
        plt.ylabel(f"||A_t - A_(t-1)||_F (mode {m + 1})")
        _finish(None if save_dir is None else
                os.path.join(save_dir, f"mode_{m + 1}.png"))


def viz_error(A_hat, A_true, save_dir=None):
    """Error ||A_hat_t - A_t||_F over time.

    Absolute rather than relative: a segment's true graph can have no edges,
    which leaves ||A_t||_F = 0.
    """
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "error")
        os.makedirs(save_dir, exist_ok=True)
    for m, (A_hat_m, A_true_m) in enumerate(zip(A_hat, A_true)):
        error = np.linalg.norm(A_hat_m - A_true_m, axis=(1, 2))
        plt.figure()
        plt.plot(range(len(A_hat_m)), error)
        plt.ylabel(f"||A_hat_t - A_t||_F (mode {m + 1})")
        _finish(None if save_dir is None else
                os.path.join(save_dir, f"mode_{m + 1}.png"))


def _edge_heatmap(ax, A_t, limit, labels, title):
    image = ax.imshow(A_t, cmap="RdBu_r", vmin=-limit, vmax=limit)
    if labels is not None:
        ticks = range(len(labels))
        ax.set_xticks(ticks, labels, rotation=90)
        ax.set_yticks(ticks, labels)
    ax.set_title(title)
    return image


def viz_edges(A, times, A_true=None, labels=None, save_dir=None):
    """Heatmaps of A_t at ``times``, with the truth beside it when given.

    The colour range is symmetric and shared by every time (and the truth)
    of a mode, so the same colour means the same value across snapshots.
    ``labels[m]`` names the d_m nodes of mode m.
    """
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "edges")
        os.makedirs(save_dir, exist_ok=True)
    for m, A_m in enumerate(A):
        panels = [] if A_true is None else [("Truth", A_true[m])]
        panels.append(("Estimate", A_m))
        limit = max(np.abs(values).max() for _, values in panels)
        limit = max(limit, np.finfo(float).tiny)
        mode_labels = None if labels is None else labels[m]
        for t in times:
            fig, axes = plt.subplots(1, len(panels), squeeze=False,
                                     figsize=(5 * len(panels), 4.5))
            for ax, (name, values) in zip(axes[0], panels):
                image = _edge_heatmap(ax, values[t], limit, mode_labels,
                                      f"{name} (mode {m + 1}, t={t})")
            fig.colorbar(image, ax=axes[0].tolist())
            if save_dir is not None:
                fig.savefig(os.path.join(save_dir, f"mode_{m + 1}_t{t:03d}.png"),
                            bbox_inches="tight")
            else:
                plt.show()
            plt.close(fig)


def viz_base(b, save_dir=None):
    """Every element of b_t over time; b[m] has shape (T, d_m)."""
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "base")
        os.makedirs(save_dir, exist_ok=True)
    for m, b_m in enumerate(b):
        plt.figure()
        plt.plot(range(len(b_m)), b_m, linewidth=0.8)
        plt.ylabel(f"b_t (mode {m + 1})")
        _finish(None if save_dir is None else
                os.path.join(save_dir, f"mode_{m + 1}.png"))
