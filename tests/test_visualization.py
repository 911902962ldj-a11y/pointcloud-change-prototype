from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from pcchange.visualization import (
    STABLE_GRAY,
    change_vmax,
    color_point_cloud_signed,
    save_distance_histogram,
    save_red_blue_map,
    save_red_blue_map_3d,
    signed_to_rgb,
)


def test_signed_to_rgb_diverging() -> None:
    rgb, vmax = signed_to_rgb(np.array([-1.0, 0.0, 1.0]), vmax=1.0)
    assert rgb.shape == (3, 3)
    assert vmax == 1.0
    # RdBu: more red on the negative side, more blue on the positive side
    assert rgb[0, 0] > rgb[0, 2]
    assert rgb[2, 2] > rgb[2, 0]


def test_change_vmax_uses_changed_magnitude_not_noise_percentile() -> None:
    signed = np.full(2000, 0.01)
    signed[-80:] = 0.70
    distances = np.abs(signed)
    vmax = change_vmax(signed, threshold=0.20, distances=distances)
    assert vmax >= 0.60
    assert vmax < 1.5


def test_stable_points_are_gray() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3))
    signed = np.full(200, 0.01)
    signed[-10:] = -0.70
    distances = np.abs(signed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    colored, vmax = color_point_cloud_signed(
        pcd, signed, threshold=0.20, distances=distances
    )
    colors = np.asarray(colored.colors)
    np.testing.assert_allclose(colors[:-10], np.broadcast_to(STABLE_GRAY, (190, 3)))
    assert not np.allclose(colors[-10:], STABLE_GRAY)
    assert vmax >= 0.60


def test_saves_figures_and_colored_cloud(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3))
    signed = pts[:, 1]
    distances = np.abs(signed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    colored, _ = color_point_cloud_signed(
        pcd, signed, threshold=0.20, distances=distances
    )
    assert colored.has_colors()
    hist = save_distance_histogram(distances, tmp_path / "h.png", threshold=0.20)
    cmap = save_red_blue_map(
        pts,
        signed,
        tmp_path / "m.png",
        threshold=0.20,
        distances=distances,
    )
    yz = save_red_blue_map(
        pts,
        signed,
        tmp_path / "yz.png",
        axes=(1, 2),
        threshold=0.20,
        distances=distances,
    )
    p3d = save_red_blue_map_3d(
        pts, signed, tmp_path / "3d.png", threshold=0.20, distances=distances
    )
    assert hist.is_file() and hist.stat().st_size > 0
    assert cmap.is_file() and cmap.stat().st_size > 0
    assert yz.is_file() and yz.stat().st_size > 0
    assert p3d.is_file() and p3d.stat().st_size > 0
