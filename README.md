# pointcloud-change-prototype

Phase 1: **Synthetic Tunnel** — generate two epoch PLY clouds, register them with Generalized ICP, and measure change with cloud-to-cloud (C2C) distances.

Core library (`src/pcchange`) is scene-agnostic. The horseshoe tunnel, ground-truth pose, and excavation patch live only in `demos/tunnel`.

## Environment

Python 3.10, packages in `.venv`:

```text
open3d==0.19.0
numpy
matplotlib
pytest
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Pipeline

Read PLY → statistical outlier filter → voxel downsample → Open3D Generalized ICP → residual outlier rejection → C2C → change stats + red/blue map → Markdown report.

Preprocessing does **not** use RANSAC. Large residuals are rejected after registration so the rigid pose is not pulled by the excavated patch; C2C still uses the aligned full clouds.

## Generate synthetic data

From the repo root:

```powershell
.\.venv\Scripts\python.exe demos\tunnel\generate_synthetic_tunnel.py
```

Writes `demos/tunnel/data/`:

| File | Contents |
| --- | --- |
| `T0.ply` | Epoch 0 |
| `T1.ply` | Epoch 1 (yaw 5° about Z, translation `[0.4, -0.2, 0.1]` m, local excavation) |
| `ground_truth.json` | `R_gt`, `t_gt`, excavation AABB (T0 frame) |
| `T1_excavation_mask.npy` | Per-point excavation label on T1 |

## Run the pipeline (Python, not a CLI)

```python
from pathlib import Path
from pcchange.pipeline import PipelineConfig, run_pipeline

root = Path("demos/tunnel/data")
run_pipeline(
    source_path=root / "T1.ply",
    target_path=root / "T0.ply",
    output_dir=Path("output/tunnel"),
    ground_truth_path=root / "ground_truth.json",
    excavation_mask_path=root / "T1_excavation_mask.npy",
    config=PipelineConfig(),
)
```

Outputs under `output/tunnel/`: aligned cloud, red/blue coloured PLY, histogram PNG, map PNG, `report.md`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Ground-truth checks:

- rotation error `< 0.5°`
- translation error `< 0.05 m`
- excavation recall `≥ 0.8`
