from __future__ import annotations

import sys
from pathlib import Path

import open3d as o3d
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO_TUNNEL = ROOT / "demos" / "tunnel"
if str(DEMO_TUNNEL) not in sys.path:
    sys.path.insert(0, str(DEMO_TUNNEL))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def synthetic_tunnel():
    from generate_synthetic_tunnel import generate_synthetic_tunnel

    return generate_synthetic_tunnel(
        length=18.0,
        point_spacing=0.10,
        n_outliers=80,
        seed=7,
    )


@pytest.fixture(scope="session")
def synthetic_tunnel_pcds(synthetic_tunnel):
    t0 = o3d.geometry.PointCloud()
    t0.points = o3d.utility.Vector3dVector(synthetic_tunnel.t0_points)
    t1 = o3d.geometry.PointCloud()
    t1.points = o3d.utility.Vector3dVector(synthetic_tunnel.t1_points)
    return t0, t1

