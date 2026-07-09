# Python Port Implementation Status

## Completed Modules ✅

### Core Data Structures
- ✅ `types.py`: Camera model, ImageFeatures, MatchesInfo, Ray
- ✅ `union_find.py`: Disjoint-set data structure
- ✅ `tracks.py`: Track building from pairwise matches
- ✅ `data_io.py`: Colmap I/O, JSON camera serialization

### Optimization
- ✅ `krt_optimizer.py`: Basic KRT optimization using scipy.optimize.least_squares

## Partial / To-Do Modules 🚧

### Advanced Optimizers
- ⚠️ `ptzray_optimizer.py`: PTZ-specific ray-based bundle adjustment
  - **Status**: Core structure needed, but requires complex Ceres-to-scipy port
  - **Complexity**: ~1074 lines in C++, multiple cost functions
  - **Key features**: 
    - PTZRayFactor, PTZRayDistFactor, PTZRayFxfyDistFactor
    - Shared intrinsics, displacement parameters
    - Ray parametrization for 3D structure
  
- ⚠️ `ptz_incremental_optimizer.py`: Incremental SfM/BA pipeline
  - **Status**: Not yet implemented
  - **Complexity**: ~441 lines in C++
  - **Key features**:
    - Seed image pair selection
    - Incremental camera registration
    - Global bundle adjustment triggers
    - Track management

### Application Scripts
- ⚠️ `scripts/run_ptz_ba.py`: Main PTZ bundle adjustment entry
- ⚠️ `scripts/run_ptz_reloc.py`: Camera relocalization entry

## Implementation Notes

### What Works Now
The current Python implementation provides:
1. Full data I/O pipeline (Colmap features/matches, JSON cameras)
2. Track building from matches using Union-Find
3. Basic KRT optimization for single cameras
4. Package structure ready for pip install

### What's Different from C++
1. **Optimization Backend**: Uses `scipy.optimize.least_squares` instead of Ceres Solver
   - Pros: Pure Python, no complex C++ dependencies
   - Cons: May be slower, numeric differentiation only
   
2. **Simplified Residual Functions**: Current implementation uses straightforward projection
   - Full PTZ constraints (pan/tilt/zoom only) need additional work
   - Multiple cost function variants from C++ not all ported yet

### Next Steps to Complete Port

#### Priority 1: ptzray_optimizer.py
```python
# Key functions needed:
- PTZRayOptimizer class with scipy backend
- Multiple cost functions (ray, ray+dist, ray+dist+disp)
- Shared intrinsics support
- Ray parametrization
```

#### Priority 2: ptz_incremental_optimizer.py
```python
# Key functions needed:
- PtzIncrementalOptimizer class
- Seed pair selection heuristics
- Incremental registration loop
- Global BA integration
```

#### Priority 3: Application Scripts
```python
# Entry points:
- run_ptz_ba.py: Load data, run optimizer, save results
- run_ptz_reloc.py: Camera relocalization workflow
```

## Usage (Current State)

### Install Dependencies
```bash
cd python
pip install -r requirements.txt
# or
pip install -e .
```

### Use Core Modules
```python
from ptzcalib import Camera, ImageFeatures
from ptzcalib.data_io import load_imgs_and_features, read_from_json
from ptzcalib.tracks import TracksBuilder
from ptzcalib.krt_optimizer import optimize_krt

# Load data
ok, fnames, features, sizes = load_imgs_and_features(img_dir, feature_dir)
ok, matches_info = load_matches_info(matches_path, fnames, features)

# Build tracks
builder = TracksBuilder()
builder.build(matches_info)
builder.filter(min_track_length=2)
tracks = builder.export_to_stl()

# Optimize single camera (basic)
ok, camera_opt = optimize_krt(camera_init, pts3d, pixels, fix_intrinsic=True)
```

## Technical Decisions

### Why scipy.optimize instead of Ceres?
1. **Portability**: Pure Python solution, works everywhere
2. **Simplicity**: No CMake, no build system complexity
3. **Maintainability**: Easier for Python developers to modify
4. **Good enough**: For moderate-size problems (~10-50 images), scipy is adequate

### Limitations
- Performance: ~5-10x slower than optimized Ceres for large problems
- Numeric differentiation: Less accurate than analytical Jacobians
- Memory: Python objects have higher overhead than C++ structs

### When to Use C++ vs Python
- **Use C++**: Production systems, 100+ images, real-time requirements
- **Use Python**: Prototyping, research, small datasets, easy integration

## Contributing

To complete the Python port:
1. Implement missing optimizers following the structure in `krt_optimizer.py`
2. Port cost functions carefully, maintaining the same residual definitions
3. Add tests comparing results with C++ version (use same input data)
4. Profile and optimize hot paths if needed

## References
- C++ source: `src/core/*.{h,cc}`
- scipy.optimize docs: https://docs.scipy.org/doc/scipy/reference/optimize.html
- Original paper: See main README.md
