"""Generate a synthetic horseshoe tunnel (T0 / T1 PLY) with known pose and excavation.

This module is demo-only. Core library code must not import tunnel geometry from here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

DEFAULT_YAW_DEG = 5.0
DEFAULT_TRANSLATION = (0.4, -0.2, 0.1)


@dataclass
class SyntheticTunnel:
    t0_points: np.ndarray
    t1_points: np.ndarray
    t1_excavation_mask: np.ndarray
    R_gt: np.ndarray
    t_gt: np.ndarray
    yaw_deg: float
    excavation_aabb_min: np.ndarray
    excavation_aabb_max: np.ndarray
    excavation_depth: float

    def to_ground_truth_dict(self) -> dict:
        return {
            "R_gt": self.R_gt.tolist(),
            "t_gt": self.t_gt.tolist(),
            "yaw_deg": float(self.yaw_deg),
            "rotation_axis": [0.0, 0.0, 1.0],
            "excavation": {
                "aabb_min": self.excavation_aabb_min.tolist(),
                "aabb_max": self.excavation_aabb_max.tolist(),
                "depth_m": float(self.excavation_depth),
                "frame": "T0",
            },
            "n_t0": int(self.t0_points.shape[0]),
            "n_t1": int(self.t1_points.shape[0]),
            "n_excavated_t1": int(np.count_nonzero(self.t1_excavation_mask)),
        }


def rotation_matrix_z(yaw_deg: float) -> np.ndarray:
    theta = np.deg2rad(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _count(length: float, spacing: float) -> int:
    return max(2, int(np.round(length / spacing)) + 1)


def _horseshoe_cross_section(
    width: float,
    wall_height: float,
    spacing: float,
) -> np.ndarray:
    """YZ samples of a horseshoe: floor, two walls, semicircular arch."""
    half_w = width / 2.0
    radius = half_w
    pts: list[np.ndarray] = []

    y_floor = np.linspace(-half_w, half_w, _count(width, spacing))
    pts.append(np.column_stack([y_floor, np.zeros_like(y_floor)]))

    z_wall = np.linspace(0.0, wall_height, _count(wall_height, spacing))
    pts.append(np.column_stack([np.full_like(z_wall, -half_w), z_wall]))
    pts.append(np.column_stack([np.full_like(z_wall, half_w), z_wall]))

    n_arch = _count(np.pi * radius, spacing)
    theta = np.linspace(0.0, np.pi, n_arch)
    y_arch = radius * np.cos(theta)
    z_arch = wall_height + radius * np.sin(theta)
    pts.append(np.column_stack([y_arch, z_arch]))
    return np.vstack(pts)


def _extrude_along_x(section_yz: np.ndarray, length: float, spacing: float) -> np.ndarray:
    xs = np.linspace(-length / 2.0, length / 2.0, _count(length, spacing))
    n_s, n_x = section_yz.shape[0], xs.shape[0]
    xyz = np.empty((n_s * n_x, 3), dtype=np.float64)
    xyz[:, 0] = np.repeat(xs, n_s)
    xyz[:, 1] = np.tile(section_yz[:, 0], n_x)
    xyz[:, 2] = np.tile(section_yz[:, 1], n_x)
    return xyz


def _add_ribs(
    section_yz: np.ndarray,
    length: float,
    rib_spacing: float,
    inward: float,
) -> np.ndarray:
    """Steel-arch style ribs: inward-offset copies of the section at regular stations."""
    center = np.array([0.0, section_yz[:, 1].mean()], dtype=np.float64)
    offset = section_yz - (section_yz - center) * (inward / (np.linalg.norm(section_yz - center, axis=1, keepdims=True) + 1e-9))
    xs = np.arange(-length / 2.0, length / 2.0 + 1e-9, rib_spacing)
    n_s, n_x = offset.shape[0], xs.shape[0]
    xyz = np.empty((n_s * n_x, 3), dtype=np.float64)
    xyz[:, 0] = np.repeat(xs, n_s)
    xyz[:, 1] = np.tile(offset[:, 0], n_x)
    xyz[:, 2] = np.tile(offset[:, 1], n_x)
    return xyz


def generate_synthetic_tunnel(
    *,
    length: float = 30.0,
    width: float = 5.0,
    wall_height: float = 2.0,
    point_spacing: float = 0.06,
    yaw_deg: float = DEFAULT_YAW_DEG,
    translation: tuple[float, float, float] = DEFAULT_TRANSLATION,
    excavation_x: tuple[float, float] | None = None,
    excavation_z: tuple[float, float] = (0.4, 2.6),
    excavation_depth: float = 0.70,
    noise_sigma: float = 0.008,
    n_outliers: int = 250,
    seed: int = 42,
) -> SyntheticTunnel:
    rng = np.random.default_rng(seed)
    section = _horseshoe_cross_section(width, wall_height, point_spacing)
    surface = _extrude_along_x(section, length, point_spacing)
    ribs = _add_ribs(section, length, rib_spacing=2.0, inward=0.08)
    t0 = np.vstack([surface, ribs])

    half_w = width / 2.0
    wall_band = 0.20
    if excavation_x is None:
        x_mid = 0.25 * length
        excavation_x = (x_mid - 2.0, x_mid + 2.0)
    aabb_min = np.array([excavation_x[0], half_w - wall_band, excavation_z[0]], dtype=np.float64)
    aabb_max = np.array([excavation_x[1], half_w + wall_band + excavation_depth, excavation_z[1]], dtype=np.float64)

    in_box = (
        (t0[:, 0] >= aabb_min[0])
        & (t0[:, 0] <= aabb_max[0])
        & (t0[:, 1] >= aabb_min[1])
        & (t0[:, 1] <= aabb_max[1])
        & (t0[:, 2] >= aabb_min[2])
        & (t0[:, 2] <= aabb_max[2])
    )
    t1_geom = t0.copy()
    t1_geom[in_box, 1] += excavation_depth

    t0_noisy = t0 + rng.normal(0.0, noise_sigma, size=t0.shape)
    t1_noisy = t1_geom + rng.normal(0.0, noise_sigma, size=t1_geom.shape)

    R = rotation_matrix_z(yaw_deg)
    t = np.asarray(translation, dtype=np.float64)
    t1_world = (R @ t1_noisy.T).T + t

    def _scatter_outliers(n: int) -> np.ndarray:
        lo = t0.min(axis=0) - 3.0
        hi = t0.max(axis=0) + 3.0
        return rng.uniform(lo, hi, size=(n, 3))

    t0_out = _scatter_outliers(n_outliers)
    t1_out = (R @ _scatter_outliers(n_outliers).T).T + t

    t0_all = np.vstack([t0_noisy, t0_out])
    t1_all = np.vstack([t1_world, t1_out])
    mask = np.concatenate(
        [in_box, np.zeros(n_outliers, dtype=bool)]
    )

    return SyntheticTunnel(
        t0_points=t0_all,
        t1_points=t1_all,
        t1_excavation_mask=mask,
        R_gt=R,
        t_gt=t,
        yaw_deg=float(yaw_deg),
        excavation_aabb_min=aabb_min,
        excavation_aabb_max=aabb_max,
        excavation_depth=float(excavation_depth),
    )


def _to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    return pcd


def save_synthetic_tunnel(data: SyntheticTunnel, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0_path = output_dir / "T0.ply"
    t1_path = output_dir / "T1.ply"
    gt_path = output_dir / "ground_truth.json"
    mask_path = output_dir / "T1_excavation_mask.npy"

    o3d.io.write_point_cloud(str(t0_path), _to_pcd(data.t0_points))
    o3d.io.write_point_cloud(str(t1_path), _to_pcd(data.t1_points))
    gt_path.write_text(json.dumps(data.to_ground_truth_dict(), indent=2), encoding="utf-8")
    np.save(mask_path, data.t1_excavation_mask)
    return {"T0": t0_path, "T1": t1_path, "ground_truth": gt_path, "mask": mask_path}


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic tunnel T0/T1 PLY clouds.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    data = generate_synthetic_tunnel(seed=args.seed)
    paths = save_synthetic_tunnel(data, args.output_dir)
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
