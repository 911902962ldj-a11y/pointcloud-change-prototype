"""PLY-only point cloud I/O."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d

SUPPORTED_SUFFIXES = {".ply"}


def _as_path(path: str | Path) -> Path:
    return Path(path)


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    path = _as_path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Only PLY is supported, got suffix {path.suffix!r} for {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"No points in {path}")
    return pcd


def save_point_cloud(pcd: o3d.geometry.PointCloud, path: str | Path) -> Path:
    path = _as_path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Only PLY is supported, got suffix {path.suffix!r} for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    if not ok:
        raise IOError(f"Failed to write {path}")
    return path
