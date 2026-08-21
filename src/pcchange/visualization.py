"""Red/blue change colouring and 2D/3D summary figures."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

STABLE_GRAY = np.array([0.70, 0.70, 0.70], dtype=np.float64)
_AXIS_LABELS = {0: "X (m)", 1: "Y (m)", 2: "Z (m)"}


def change_vmax(
    signed: np.ndarray,
    *,
    threshold: float,
    distances: np.ndarray | None = None,
) -> float:
    """Colour-bar half-range from changed points, not the whole-cloud percentile.

    Using a global 98th percentile collapses the scale to millimetres of noise.
    Changed-point robust peak keeps the bar on the actual C2C magnitude.
    """
    signed = np.asarray(signed, dtype=np.float64)
    if distances is None:
        changed = np.abs(signed) > float(threshold)
    else:
        changed = np.asarray(distances, dtype=np.float64) > float(threshold)
    if signed.size == 0:
        return max(float(threshold), 1e-6)
    if np.any(changed):
        mag = np.abs(signed[changed])
        peak = float(np.percentile(mag, 90))
    else:
        peak = float(np.max(np.abs(signed)))
    return max(peak, float(threshold), 1e-6)


def signed_to_rgb(
    signed: np.ndarray, *, vmax: float | None = None
) -> tuple[np.ndarray, float]:
    signed = np.asarray(signed, dtype=np.float64)
    if vmax is None:
        vmax = change_vmax(signed, threshold=0.0)
    vmax = float(vmax) if vmax > 1e-9 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    # RdBu: red = negative (loss / excavation), blue = positive (gain)
    rgb = cm.RdBu(norm(signed))[:, :3]
    return rgb, vmax


def color_point_cloud_signed(
    pcd: o3d.geometry.PointCloud,
    signed: np.ndarray,
    *,
    threshold: float = 0.20,
    distances: np.ndarray | None = None,
    vmax: float | None = None,
) -> tuple[o3d.geometry.PointCloud, float]:
    signed = np.asarray(signed, dtype=np.float64)
    if distances is None:
        distances = np.abs(signed)
    else:
        distances = np.asarray(distances, dtype=np.float64)
    changed = distances > float(threshold)
    vmax_used = vmax if vmax is not None else change_vmax(
        signed, threshold=threshold, distances=distances
    )
    colors = np.broadcast_to(STABLE_GRAY, (signed.shape[0], 3)).copy()
    if np.any(changed):
        rgb, vmax_used = signed_to_rgb(signed[changed], vmax=vmax_used)
        colors[changed] = rgb
    out = copy.deepcopy(pcd)
    out.colors = o3d.utility.Vector3dVector(colors)
    return out, vmax_used


def save_distance_histogram(
    distances: np.ndarray,
    path: str | Path,
    *,
    threshold: float | None = None,
    title: str = "C2C distance",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(distances, bins=60, color="#4c72b0", edgecolor="none")
    if threshold is not None:
        ax.axvline(threshold, color="#c44e52", linestyle="--", label=f"threshold={threshold:.2f} m")
        ax.legend()
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _changed_mask(
    signed: np.ndarray,
    *,
    threshold: float,
    distances: np.ndarray | None,
    changed_mask: np.ndarray | None,
) -> np.ndarray:
    if changed_mask is not None:
        return np.asarray(changed_mask, dtype=bool)
    if distances is not None:
        return np.asarray(distances, dtype=np.float64) > float(threshold)
    return np.abs(np.asarray(signed, dtype=np.float64)) > float(threshold)


def save_red_blue_map(
    points: np.ndarray,
    signed: np.ndarray,
    path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    threshold: float = 0.20,
    distances: np.ndarray | None = None,
    changed_mask: np.ndarray | None = None,
    vmax: float | None = None,
    title: str = "Signed C2C (gray = stable; red/blue = |C2C| > threshold)",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points)
    signed = np.asarray(signed, dtype=np.float64)
    changed = _changed_mask(
        signed, threshold=threshold, distances=distances, changed_mask=changed_mask
    )
    stable = ~changed
    vmax_used = (
        vmax
        if vmax is not None
        else change_vmax(signed, threshold=threshold, distances=distances)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    if np.any(stable):
        ax.scatter(
            points[stable, axes[0]],
            points[stable, axes[1]],
            c=[STABLE_GRAY],
            s=2,
            linewidths=0,
            zorder=1,
            label="stable",
        )
    sc = None
    if np.any(changed):
        sc = ax.scatter(
            points[changed, axes[0]],
            points[changed, axes[1]],
            c=signed[changed],
            s=8,
            cmap="RdBu",
            vmin=-vmax_used,
            vmax=vmax_used,
            linewidths=0,
            zorder=2,
            label="|C2C| > threshold",
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("signed distance (m)")
    ax.set_xlabel(_AXIS_LABELS[axes[0]])
    ax.set_ylabel(_AXIS_LABELS[axes[1]])
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_red_blue_map_3d(
    points: np.ndarray,
    signed: np.ndarray,
    path: str | Path,
    *,
    threshold: float = 0.20,
    distances: np.ndarray | None = None,
    changed_mask: np.ndarray | None = None,
    vmax: float | None = None,
    max_stable: int = 20000,
    title: str = "3D signed C2C (gray = stable)",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points)
    signed = np.asarray(signed, dtype=np.float64)
    changed = _changed_mask(
        signed, threshold=threshold, distances=distances, changed_mask=changed_mask
    )
    stable = np.flatnonzero(~changed)
    if stable.size > max_stable:
        rng = np.random.default_rng(0)
        stable = rng.choice(stable, size=max_stable, replace=False)
    vmax_used = (
        vmax
        if vmax is not None
        else change_vmax(signed, threshold=threshold, distances=distances)
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if stable.size:
        ax.scatter(
            points[stable, 0],
            points[stable, 1],
            points[stable, 2],
            c=[STABLE_GRAY],
            s=1,
            linewidths=0,
            depthshade=False,
        )
    if np.any(changed):
        sc = ax.scatter(
            points[changed, 0],
            points[changed, 1],
            points[changed, 2],
            c=signed[changed],
            s=6,
            cmap="RdBu",
            vmin=-vmax_used,
            vmax=vmax_used,
            linewidths=0,
            depthshade=False,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
        cb.set_label("signed distance (m)")
    ax.set_xlabel(_AXIS_LABELS[0])
    ax.set_ylabel(_AXIS_LABELS[1])
    ax.set_zlabel(_AXIS_LABELS[2])
    ax.set_title(title)
    ax.view_init(elev=18, azim=-70)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
