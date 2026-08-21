"""Generalized ICP with residual-based outlier rejection after alignment."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class RegistrationConfig:
    voxel_size: float = 0.05
    n_scales: int = 4
    max_corr_voxel_multiplier: float = 5.0
    max_iteration: int = 80
    residual_k_mad: float = 3.0
    residual_min_threshold: float = 0.05
    refine: bool = True


@dataclass
class RegistrationResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    voxel_sizes: list[float]
    residual_inlier_mask: np.ndarray
    residuals: np.ndarray


def centroid_translation_init(
    source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud
) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(target.get_center()) - np.asarray(source.get_center())
    return T


def voxel_schedule(finest: float, n_scales: int = 4) -> list[float]:
    if finest <= 0:
        raise ValueError("finest voxel size must be positive")
    if n_scales < 1:
        raise ValueError("n_scales must be >= 1")
    return [finest * (2 ** k) for k in range(n_scales - 1, -1, -1)]


def _estimate_normals(pcd: o3d.geometry.PointCloud, radius: float) -> None:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    pcd.orient_normals_towards_camera_location(pcd.get_center())


def generalized_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float,
    init: np.ndarray | None = None,
    *,
    max_iteration: int = 80,
) -> o3d.pipelines.registration.RegistrationResult:
    if init is None:
        init = np.eye(4)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iteration,
    )
    estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    return o3d.pipelines.registration.registration_generalized_icp(
        source,
        target,
        float(max_correspondence_distance),
        np.asarray(init, dtype=np.float64),
        estimation,
        criteria,
    )


def residual_inlier_mask(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    *,
    k_mad: float = 3.0,
    min_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = copy.deepcopy(source)
    aligned.transform(np.asarray(transformation, dtype=np.float64))
    dist = np.asarray(aligned.compute_point_cloud_distance(target), dtype=np.float64)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    sigma = 1.4826 * mad
    threshold = max(min_threshold, med + k_mad * sigma)
    return dist <= threshold, dist


def multi_scale_generalized_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    voxel_sizes: list[float],
    init: np.ndarray | None = None,
    max_corr_voxel_multiplier: float = 5.0,
    max_iteration: int = 80,
) -> o3d.pipelines.registration.RegistrationResult:
    T = np.eye(4, dtype=np.float64) if init is None else np.asarray(init, dtype=np.float64)
    last = None
    for vs in voxel_sizes:
        src = source.voxel_down_sample(vs)
        tgt = target.voxel_down_sample(vs)
        _estimate_normals(src, radius=vs * 2.5)
        _estimate_normals(tgt, radius=vs * 2.5)
        max_corr = vs * max_corr_voxel_multiplier
        last = generalized_icp(
            src, tgt, max_corr, T, max_iteration=max_iteration
        )
        T = np.asarray(last.transformation, dtype=np.float64)
    if last is None:
        raise RuntimeError("empty voxel schedule")
    return last


def register_point_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    cfg = config or RegistrationConfig()
    sizes = voxel_schedule(cfg.voxel_size, cfg.n_scales)
    init = centroid_translation_init(source, target)
    icp = multi_scale_generalized_icp(
        source,
        target,
        voxel_sizes=sizes,
        init=init,
        max_corr_voxel_multiplier=cfg.max_corr_voxel_multiplier,
        max_iteration=cfg.max_iteration,
    )
    T = np.asarray(icp.transformation, dtype=np.float64)

    mask, residuals = residual_inlier_mask(
        source,
        target,
        T,
        k_mad=cfg.residual_k_mad,
        min_threshold=cfg.residual_min_threshold,
    )
    if cfg.refine and int(np.count_nonzero(mask)) > 50:
        indices = np.flatnonzero(mask).tolist()
        src_in = source.select_by_index(indices)
        finest = sizes[-1]
        _estimate_normals(src_in, radius=finest * 2.5)
        tgt_n = copy.deepcopy(target)
        _estimate_normals(tgt_n, radius=finest * 2.5)
        refined = generalized_icp(
            src_in,
            tgt_n,
            max_correspondence_distance=finest * 3.0,
            init=T,
            max_iteration=cfg.max_iteration,
        )
        T = np.asarray(refined.transformation, dtype=np.float64)
        icp = refined
        mask, residuals = residual_inlier_mask(
            source,
            target,
            T,
            k_mad=cfg.residual_k_mad,
            min_threshold=cfg.residual_min_threshold,
        )

    return RegistrationResult(
        transformation=T,
        fitness=float(icp.fitness),
        inlier_rmse=float(icp.inlier_rmse),
        voxel_sizes=sizes,
        residual_inlier_mask=mask,
        residuals=residuals,
    )
