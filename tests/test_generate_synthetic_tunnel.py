from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from generate_synthetic_tunnel import (
    DEFAULT_TRANSLATION,
    DEFAULT_YAW_DEG,
    generate_synthetic_tunnel,
    rotation_matrix_z,
    save_synthetic_tunnel,
)


def test_rotation_is_vertical_yaw() -> None:
    R = rotation_matrix_z(DEFAULT_YAW_DEG)
    expected = np.array(
        [
            [np.cos(np.deg2rad(5.0)), -np.sin(np.deg2rad(5.0)), 0.0],
            [np.sin(np.deg2rad(5.0)), np.cos(np.deg2rad(5.0)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(R, expected, atol=1e-12)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_generate_default_pose_and_excavation() -> None:
    data = generate_synthetic_tunnel(point_spacing=0.15, length=16.0, n_outliers=20, seed=1)
    np.testing.assert_allclose(data.t_gt, np.asarray(DEFAULT_TRANSLATION))
    assert data.yaw_deg == DEFAULT_YAW_DEG
    np.testing.assert_allclose(data.R_gt, rotation_matrix_z(DEFAULT_YAW_DEG))
    assert data.t0_points.shape[0] > 1000
    assert data.t1_points.shape[0] == data.t1_excavation_mask.shape[0]
    assert int(np.count_nonzero(data.t1_excavation_mask)) > 50
    assert data.excavation_depth > 0.3


def test_save_ply_and_ground_truth(tmp_path: Path) -> None:
    data = generate_synthetic_tunnel(point_spacing=0.2, length=12.0, n_outliers=10, seed=0)
    paths = save_synthetic_tunnel(data, tmp_path)
    t0 = o3d.io.read_point_cloud(str(paths["T0"]))
    t1 = o3d.io.read_point_cloud(str(paths["T1"]))
    assert not t0.is_empty()
    assert not t1.is_empty()
    assert paths["ground_truth"].is_file()
    mask = np.load(paths["mask"])
    assert mask.dtype == bool
    assert mask.shape[0] == np.asarray(t1.points).shape[0]
