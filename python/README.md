# PTZ-Calib Python Implementation

Python port of the PTZ-Calib project for robust PTZ camera calibration.

## Requirements

```bash
pip install numpy scipy opencv-python
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
├── scripts/               # Executable scripts
│   ├── run_ptz_ba.py
│   └── run_ptz_reloc.py
└── tests/                 # Unit tests
```

## Usage

See original C++ project README for dataset download and usage examples.

## Implementation Notes

- Uses `scipy.optimize.least_squares` for nonlinear optimization (replaces Ceres)
- Numeric differentiation for all cost functions
- Maintains API compatibility with C++ version where possible
