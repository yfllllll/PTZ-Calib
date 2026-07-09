# PTZ-Calib Python Implementation

Python port of the PTZ-Calib project for robust PTZ camera calibration.

## Requirements

```bash
pip install numpy scipy opencv-python pyceres
```

## Project Structure

```
python/
├── ptzcalib/              # Main package
│   ├── __init__.py
│   ├── types.py           # Camera and basic types
│   ├── data_io.py         # I/O utilities
│   ├── tracks.py          # Track building
│   ├── union_find.py      # Union-Find data structure
│   ├── krt_optimizer.py   # KRT optimization
│   ├── ptzray_optimizer.py    # PTZ ray optimization
│   └── ptz_incremental_optimizer.py  # Incremental BA
├── run_ptz_ba.py          # PTZ-IBA + georeferencing entry point
├── run_ptz_reloc.py       # PTZ relocalization entry point
└── tests/                 # Unit tests
```

## Usage

See original C++ project README for dataset download and usage examples.

## Implementation Status

**Functional — Core Pipeline Complete!**

### Modules Ported
- ✅ `types.py` — Camera, ImageFeatures, MatchesInfo, Ray
- ✅ `union_find.py` — Union-Find data structure
- ✅ `tracks.py` — Track building from matches
- ✅ `data_io.py` — JSON I/O
- ✅ `krt_optimizer.py` — KRT optimization (6 cost functions)
- ✅ `ptzray_optimizer.py` — PTZ-Ray optimizer (6 cost functions, pyceres)
- ✅ `ptz_incremental_optimizer.py` — Incremental BA pipeline
- ✅ `run_ptz_ba.py` — Application entry point

### Implementation Notes
- **Optimizer**: Uses `pyceres` (Python bindings for Ceres Solver) for numerical
  equivalence with C++ version. Falls back to `scipy.optimize.least_squares` if
  pyceres unavailable.
- **All cost functions faithfully port the C++ math**: same distortion model,
  same projection equations, same displacement model.
- **API compatibility**: Maintains similar API to C++ where possible for
  easy porting of application code.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: install image matching dependency for data preparation
pip install git+https://github.com/gmberton/vismatch.git

# Prepare features and matches from images only
python prepare_data_vismatch.py \
  --images data/images \
  --output data/features \
  --pairing window \
  --window 5 \
  --matcher superpoint-lightglue

# Run PTZ Bundle Adjustment (matches the C++ CLI shape)
python run_ptz_ba.py \
  --images data/images \
  --features data/features \
  --annotation data/annotation.json \
  --output outputs \
  --max_iter 200
```
