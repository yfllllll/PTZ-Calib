# Python Port Implementation Status

## ✅ CORE PIPELINE COMPLETE

Successfully ported the PTZ-Calib C++ project to Python using `pyceres` (Python bindings for Ceres Solver).

## Strategy

To achieve **numerically equivalent** behavior to the original C++ implementation:
- **pyceres 2.6**: Official Python bindings to Ceres Solver (same solver as C++)
- **opencv-python**: Same OpenCV functions (Rodrigues, projectPoints, etc.)
- **numpy**: For matrix operations matching cv::Mat

## Completed Modules

| Python File | C++ Source | Status | Lines |
|-------------|-----------|--------|-------|
| `types.py` | `types.{h,cc}` | ✅ Done | ~200 |
| `union_find.py` | `union_find.h` | ✅ Done | ~80 |
| `tracks.py` | `tracks.{h,cc}` | ✅ Done | ~194 |
| `data_io.py` | `data_io.{h,cc}` | ✅ Done | ~300 |
| `krt_optimizer.py` | `krt_optimizer.{h,cc}` | ✅ Done | ~500 |
| `ptzray_optimizer.py` | `ptzray_optimizer.{h,cc}` | ✅ Done | ~430 |
| `ptz_incremental_optimizer.py` | `ptz_incremental_optimizer.{h,cc}` | ✅ Done | ~270 |
| `run_ptz_ba.py` | `run_ptz_ba.{h,cc}` | ✅ Done | ~90 |
| `run_ptz_reloc.py` | `run_ptz_reloc.{h,cc}` | 🚧 Optional | - |

**Total ported**: ~2064 lines of Python code across 8 modules.

## Cost Functions (all pyceres-based)

### krt_optimizer.py (6 factors)
- Factor2d2d, Factor2d2dFxfy
- Factor2d2dDist, Factor2d2dFxfyDist
- Factor2d3dDist, Factor2d3dFxfyDist

### ptzray_optimizer.py (6 factors)
- PTZRayFactor (base ray projection)
- PTZRayDistFactor (with distortion)
- PTZRayFxfyDistFactor (independent fx, fy)
- PTZRayDistDispFactor (with displacement model)
- Reproj2d3dFactor (2d-3d reprojection)
- Reproj2d3dDispFactor (2d-3d with displacement)

All cost functions faithfully port the C++ math:
- Same distortion model (k1, k2, k3, p1, p2)
- Same projection equations
- Same displacement model for zoom lenses
- Same Rodrigues rotation representation

## Quick Start

```bash
# Install
cd python
pip install -r requirements.txt

# Verify pyceres
python test_pyceres.py

# Run BA
python run_ptz_ba.py \
  --features_dir data/features \
  --matches_file data/matches.json \
  --cameras_in data/cameras_init.json \
  --cameras_out data/cameras_optimized.json \
  --max_iter 100
```

## Known Simplifications

- **Initial pair estimation**: Uses heuristics based on feature count and disparity
  (C++ uses full essential matrix decomposition). Cameras are expected to be
  pre-initialized with reasonable values.
- **Parameter constraints**: The `SubsetParameterization` (for fixing specific
  parameter indices) is simplified — full support requires additional pyceres
  manifold setup.
- **Global BA frequency**: Simplified triggering compared to C++'s adaptive strategy.

## Optional Future Enhancements

- [ ] `run_ptz_reloc.py` — Relocalization application (~250 lines)
- [ ] Full SubsetManifold integration for exact C++ parity
- [ ] Comprehensive unit tests
- [ ] Benchmark against C++ on WorldCup14 and synthetic datasets

## Git History

All changes on branch `python-port`:
```
e15ca62 Add run_ptz_ba.py application entry point
3ce6c50 Add PtzIncrementalOptimizer for incremental BA
ef81af9 Complete PTZRayOptimizer with full solve() pipeline
5faf7f4 Add ptzray_optimizer with all 6 cost functions (pyceres)
... (earlier commits for basic modules)
```
