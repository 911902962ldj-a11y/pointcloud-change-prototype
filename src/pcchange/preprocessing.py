"""Outlier filtering and voxel downsampling. Plane fitting is not used."""

from __future__ import annotations

from dataclasses import dataclass

import open3d as o3d


@dataclass(frozen=True)
class PreprocessConfig:
    nb_neighbors: int = 20
    std_ratio: float = 2.0
    voxel_size: float = 0.05


def statistical_outlier_filter(
    pcd: o3d.geometry.PointCloud,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    cleaned, _ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return cleaned


def voxel_downsample(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> o3d.geometry.PointCloud:
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    return pcd.voxel_down_sample(voxel_size)


def preprocess(
    pcd: o3d.geometry.PointCloud,
    config: PreprocessConfig | None = None,
) -> o3d.geometry.PointCloud:
    """Statistical outlier filter, then voxel downsample."""
    cfg = config or PreprocessConfig()
    filtered = statistical_outlier_filter(
        pcd, nb_neighbors=cfg.nb_neighbors, std_ratio=cfg.std_ratio
    )
    return voxel_downsample(filtered, cfg.voxel_size)
