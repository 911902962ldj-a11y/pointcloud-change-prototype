"""Cloud-to-cloud distances and change statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class ChangeResult:
    distances: np.ndarray
    signed_distances: np.ndarray
    threshold: float
    changed_mask: np.ndarray
    mean: float
    std: float
    median: float
    p95: float
    p99: float
    n_changed: int
    fraction_changed: float


def cloud_to_cloud_distances(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> np.ndarray:
    if source.is_empty() or target.is_empty():
        raise ValueError("source and target must be non-empty")
    return np.asarray(source.compute_point_cloud_distance(target), dtype=np.float64)


def _ensure_normals(pcd: o3d.geometry.PointCloud, radius: float) -> None:
    if pcd.has_normals():
        return
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    pcd.orient_normals_towards_camera_location(pcd.get_center())


def signed_cloud_to_cloud_distances(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    normal_radius: float = 0.25,
) -> np.ndarray:
    """Unsigned C2C magnitude with sign from the target normal at the nearest neighbour.

    Positive: source lies on the normal side of the target surface.
    """
    _ensure_normals(target, normal_radius)
    src_pts = np.asarray(source.points)
    tgt_pts = np.asarray(target.points)
    tgt_n = np.asarray(target.normals)
    tree = o3d.geometry.KDTreeFlann(target)
    signed = np.empty(src_pts.shape[0], dtype=np.float64)
    for i, p in enumerate(src_pts):
        _k, idx, _d = tree.search_knn_vector_3d(p, 1)
        j = idx[0]
        signed[i] = float(np.dot(p - tgt_pts[j], tgt_n[j]))
    return signed


def change_statistics(
    distances: np.ndarray,
    *,
    threshold: float,
    signed_distances: np.ndarray | None = None,
) -> ChangeResult:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.size == 0:
        raise ValueError("distances must be non-empty")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    changed = distances > threshold
    signed = (
        np.asarray(signed_distances, dtype=np.float64)
        if signed_distances is not None
        else distances.copy()
    )
    return ChangeResult(
        distances=distances,
        signed_distances=signed,
        threshold=float(threshold),
        changed_mask=changed,
        mean=float(np.mean(distances)),
        std=float(np.std(distances)),
        median=float(np.median(distances)),
        p95=float(np.percentile(distances, 95)),
        p99=float(np.percentile(distances, 99)),
        n_changed=int(np.count_nonzero(changed)),
        fraction_changed=float(np.mean(changed)),
    )


def detect_change(
    source_aligned: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    threshold: float = 0.20,
    normal_radius: float = 0.25,
) -> ChangeResult:
    distances = cloud_to_cloud_distances(source_aligned, target)
    signed = signed_cloud_to_cloud_distances(
        source_aligned, target, normal_radius=normal_radius
    )
    return change_statistics(distances, threshold=threshold, signed_distances=signed)
