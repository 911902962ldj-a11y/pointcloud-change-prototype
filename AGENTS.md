# Agent notes — pointcloud-change-prototype

This repository is a **point-cloud change detection** prototype. Follow these constraints unless a later phase explicitly lifts them.

## Scope (phase 1)

- In: synthetic tunnel demo + PLY I/O + classical pipeline.
- Out: MCP, LAS/LAZ, slope/open-pit scenes, real survey data, a project CLI, and deep learning.

## Layout

- `src/pcchange/` is **scene-agnostic**. Do not hard-code tunnel dimensions, horseshoe sections, excavation boxes, or “巷道” logic there.
- Tunnel geometry and its ground-truth pose/excavation live only in `demos/tunnel/`.
- Tests import the demo generator; they must not re-implement a second tunnel model inside `src/`.

## Pipeline (fixed order)

1. Read PLY
2. Statistical outlier filter
3. Voxel downsample
4. Open3D Generalized ICP
5. Residual-based outlier rejection (after registration, not as a preprocess)
6. Cloud-to-cloud (C2C) distances
7. Change statistics + red/blue map
8. Markdown report

Do **not** use RANSAC (or `segment_plane`) in preprocessing. Registration may refine with residual/MAD inliers, then C2C runs on the aligned full (preprocessed) clouds so real change is not discarded.

## I/O

- Only `.ply` in the core IO module.
- Ground truth for the demo is JSON (`R_gt`, `t_gt`) plus a boolean mask for excavated T1 points.

## Tests

Registration and change tests **must** use the synthetic ground truth:

- rotation error `< 0.5°`
- translation error `< 0.05 m`
- excavation-region recall `≥ 0.8`

Generate demo data with `demos/tunnel/generate_synthetic_tunnel.py`. Run tests with the project venv: `.venv/Scripts/python.exe -m pytest`.
