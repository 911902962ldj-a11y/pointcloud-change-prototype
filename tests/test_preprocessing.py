from __future__ import annotations

import inspect

import numpy as np
import open3d as o3d

from pcchange import preprocessing
from pcchange.preprocessing import (
    PreprocessConfig,
    preprocess,
    statistical_outlier_filter,
    voxel_downsample,
)


def _cloud_with_outliers(n: int = 400, n_out: int = 25, seed: int = 0) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=0.05, size=(n, 3))
    outliers = rng.uniform(-4.0, 4.0, size=(n_out, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([pts, outliers]))
    return pcd


def test_statistical_filter_removes_far_points() -> None:
    pcd = _cloud_with_outliers()
    cleaned = statistical_outlier_filter(pcd, nb_neighbors=20, std_ratio=1.5)
    assert 50 < len(cleaned.points) < len(pcd.points)


def test_voxel_downsample_reduces_count() -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.0, 1.0, size=(5000, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    down = voxel_downsample(pcd, 0.2)
    assert len(down.points) < len(pcd.points)
    assert len(down.points) > 10


def test_preprocess_order_filter_then_voxel() -> None:
    pcd = _cloud_with_outliers(n=2000, n_out=80)
    out = preprocess(pcd, PreprocessConfig(nb_neighbors=20, std_ratio=1.5, voxel_size=0.1))
    assert 10 < len(out.points) < len(pcd.points)


def test_preprocessing_source_has_no_plane_fitting() -> None:
    src = inspect.getsource(preprocessing)
    assert "segment_plane" not in src
    assert "registration_ransac" not in src
    assert "RANSACConvergenceCriteria" not in src
