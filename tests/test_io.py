from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from pcchange.io import load_point_cloud, save_point_cloud


def _xyz_cloud(n: int = 32, seed: int = 0) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_ply_roundtrip(tmp_path: Path) -> None:
    src = _xyz_cloud()
    path = tmp_path / "cloud.ply"
    save_point_cloud(src, path)
    loaded = load_point_cloud(path)
    np.testing.assert_allclose(
        np.asarray(src.points), np.asarray(loaded.points), atol=1e-6
    )


def test_rejects_non_ply(tmp_path: Path) -> None:
    path = tmp_path / "cloud.las"
    path.write_text("nope")
    with pytest.raises(ValueError, match="PLY"):
        load_point_cloud(path)
    with pytest.raises(ValueError, match="PLY"):
        save_point_cloud(_xyz_cloud(), path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_point_cloud(tmp_path / "missing.ply")


def test_empty_cloud_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\n"
        "property float y\nproperty float z\nend_header\n"
    )
    with pytest.raises(ValueError, match="No points"):
        load_point_cloud(path)
