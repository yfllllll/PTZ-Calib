# Python Port Implementation Status

## Strategy: Faithful C++ Port using pyceres

To achieve **100% numerically identical** behavior to the original C++ implementation, we use:
- **pyceres 2.6**: Official Python bindings to Ceres Solver (same solver as C++)
- **opencv-python**: Same OpenCV functions (undistortPoints, projectPoints, etc.)
- **numpy**: For matrix operations matching cv::Mat

## Completed Modules ✅

### Core Data Structures (100% faithful)
- ✅ `types.py`: Camera, ImageFeatures, MatchesInfo, Ray
- ✅ `union_find.py`: DisjointSet matching `src/core/union_find.h`
- ✅ `tracks.py`: TracksBuilder matching `src/core/tracks.{h,cc}`
- ✅ `data_io.py`: All I/O functions matching `src/core/data_io.{h,cc}`

### Optimization (scipy version - 99% faithful)
- ✅ `krt_optimizer.py`: All 6 cost functions with identical formulas
  - Factor2d2d, Factor2d2dFxfy, Factor2d2dDist, Factor2d2dFxfyDist
  - Factor2d3dDist, Factor2d3dFxfyDist
  - Uses scipy.optimize.least_squares (LM/TRF algorithm)
  - **Note**: Numerical results may differ ~1% from Ceres due to different solvers

### Infrastructure ✅
- ✅ pyceres installed and tested (see `test_pyceres.py`)
- ✅ Package setup with pip installable

## Remaining Work 🚧

### 1. krt_optimizer_pyceres.py (Priority: High)
Rewrite krt_optimizer using pyceres for 100% numerical equivalence:
- Port all 6 cost functions as `pyceres.CostFunction` subclasses
- Use `problem.set_manifold()` with SubsetManifold for fixed parameters
- Match `ceres::SubsetParameterization` semantics exactly
- ~500 lines of code

### 2. ptzray_optimizer.py (Priority: High)
Port `src/core/ptzray_optimizer.{h,cc}` (~1074 lines):
- 6 cost functions (PTZRayFactor, PTZRayDistFactor, etc.)
- Ray parametrization
- Shared intrinsics via SubsetManifold
- Track-based residuals
- Global BA logic

### 3. ptz_incremental_optimizer.py (Priority: Medium)
Port `src/core/ptz_incremental_optimizer.{h,cc}` (~441 lines):
- Seed pair selection heuristics
- Incremental registration loop
- Progressive global BA triggers
- Track management integration

### 4. Application Scripts (Priority: Low)
- `scripts/run_ptz_ba.py`: Main pipeline
- `scripts/run_ptz_reloc.py`: Relocalization pipeline

## Technical Design Notes

### pyceres CostFunction Pattern
```python
class MyFactor(pyceres.CostFunction):
    def __init__(self, uv, cam1):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([15])  # camera params
        self.uv = uv
        self.cam1 = cam1
    
    def Evaluate(self, parameters, residuals, jacobians):
        # Same formula as C++ Factor2d2d
        # ... compute residuals
        # jacobians = None means use numeric differentiation
        return True
```

### Fixed Parameter Handling
C++ uses `SubsetParameterization`. In pyceres 2.6, use:
```python
manifold = pyceres.SubsetManifold(15, fixed_indices)
problem.set_manifold(param_block, manifold)
```

### Coordinate Transformations
Follow C++ pattern strictly:
- `T_curr_local = T_curr_world * T_local_world^{-1}`
- Then optimize in local frame
- Transform back: `T_curr_world = T_curr_local * T_local_world`

## Usage

### Install
```bash
cd python
pip install -r requirements.txt
```

### Verify pyceres Works
```bash
python test_pyceres.py
```

### Current Working Features
```python
from ptzcalib import Camera
from ptzcalib.data_io import (
    load_imgs_and_features, load_matches_info, 
    read_from_json, save_to_json
)
from ptzcalib.tracks import TracksBuilder
from ptzcalib.krt_optimizer import KRTOptimizer, FactorType

# Load data
ok, fnames, features, sizes = load_imgs_and_features(img_dir, feature_dir)
ok, matches_info = load_matches_info(matches_path, fnames, features)

# Build tracks
builder = TracksBuilder()
builder.build(matches_info)
builder.filter(min_track_length=2)
tracks = builder.export_to_stl()

# Optimize a camera (with scipy backend for now)
opt = KRTOptimizer(max_iter=100, max_reproj_error=50, factor_type=FactorType.FDist)
opt.set_init_params(K, R, t, dist)
opt.add_2d2d_constraints(cam_ref, kpts_ref, kpts_curr, matches)
opt.add_2d3d_constraints(pts2d, pts3d)
ok, K_opt, R_opt, t_opt, dist_opt = opt.solve()
```

## Files Summary

| Python File | C++ Source | Status | Lines |
|-------------|-----------|--------|-------|
| `types.py` | `types.{h,cc}` | ✅ Done | 200 |
| `union_find.py` | `union_find.h` | ✅ Done | 80 |
| `tracks.py` | `tracks.{h,cc}` | ✅ Done | 150 |
| `data_io.py` | `data_io.{h,cc}` | ✅ Done | 300 |
| `krt_optimizer.py` | `krt_optimizer.{h,cc}` | ⚠️ scipy | 500 |
| `krt_optimizer_pyceres.py` | `krt_optimizer.{h,cc}` | 🚧 TODO | ~500 |
| `ptzray_optimizer.py` | `ptzray_optimizer.{h,cc}` | 🚧 TODO | ~1200 |
| `ptz_incremental_optimizer.py` | `ptz_incremental_optimizer.{h,cc}` | 🚧 TODO | ~500 |
| `run_ptz_ba.py` | `run_ptz_ba.{h,cc}` | 🚧 TODO | ~200 |
| `run_ptz_reloc.py` | `run_ptz_reloc.{h,cc}` | 🚧 TODO | ~250 |

**Total estimated remaining**: ~2650 lines of Python code

## Next Steps

Continue in a new session to complete:
1. Rewrite krt_optimizer with pyceres backend
2. Implement ptzray_optimizer (largest module)
3. Implement ptz_incremental_optimizer
4. Create application entry scripts
5. Test end-to-end with provided sample data
