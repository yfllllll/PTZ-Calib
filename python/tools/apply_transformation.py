#!/usr/bin/env python3
"""Apply similarity transformation to points/cameras.

Utility to batch-apply the transformation solved by control_point_picker.py
to convert coordinates from local to world coordinate system.

Usage:
    # Transform single point
    python apply_transformation.py --transform transformation.json \
        --point 100.5 200.3 50.2
    
    # Transform camera poses file
    python apply_transformation.py --transform transformation.json \
        --cameras cameras_local.json --output cameras_world.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_transformation(transform_file: str):
    """Load transformation parameters from JSON."""
    with open(transform_file) as f:
        t = json.load(f)
    
    R = np.array(t['rotation_matrix'])
    trans = np.array(t['translation'])
    s = t['scale']
    
    return R, trans, s


def transform_point(point_local: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """Transform a single point: world = s * R * local + t."""
    return s * (R @ point_local) + t


def transform_camera(camera_local: dict, R: np.ndarray, t: np.ndarray, s: float) -> dict:
    """Transform camera pose from local to world coordinates.
    
    Camera pose transformation:
    - Position: world_pos = s * R * local_pos + t
    - Rotation: world_R = R * local_R (compose rotations)
    - Intrinsics (K, dist): unchanged
    """
    camera_world = camera_local.copy()
    
    # Transform camera position
    if 't' in camera_local:
        t_local = np.array(camera_local['t'])
        t_world = transform_point(t_local, R, t, s)
        camera_world['t'] = t_world.tolist()
    
    # Transform camera rotation
    if 'R' in camera_local:
        R_local = np.array(camera_local['R'])
        R_world = R @ R_local
        camera_world['R'] = R_world.tolist()
    
    # Scale focal lengths if needed (for scale != 1)
    if 'K' in camera_local and abs(s - 1.0) > 1e-6:
        K = np.array(camera_local['K'])
        K[0, 0] *= s  # fx
        K[1, 1] *= s  # fy
        K[0, 2] *= s  # cx
        K[1, 2] *= s  # cy
        camera_world['K'] = K.tolist()
    
    return camera_world


def main():
    parser = argparse.ArgumentParser(description="Apply similarity transformation")
    parser.add_argument('--transform', type=str, required=True,
                       help='Path to transformation.json')
    parser.add_argument('--point', nargs=3, type=float,
                       help='Transform single point (x y z)')
    parser.add_argument('--cameras', type=str,
                       help='Path to cameras JSON file (local coordinates)')
    parser.add_argument('--output', type=str,
                       help='Output path for transformed cameras')
    
    args = parser.parse_args()
    
    # Load transformation
    R, trans, s = load_transformation(args.transform)
    
    print(f"Loaded transformation:")
    print(f"  Scale: {s:.6f}")
    print(f"  Rotation:\n{R}")
    print(f"  Translation: {trans}\n")
    
    # Transform single point
    if args.point:
        point_local = np.array(args.point)
        point_world = transform_point(point_local, R, trans, s)
        
        print(f"Point transformation:")
        print(f"  Local:  {point_local}")
        print(f"  World:  {point_world}")
    
    # Transform cameras
    if args.cameras:
        if not args.output:
            parser.error("--output is required when using --cameras")
        
        with open(args.cameras) as f:
            cameras_local = json.load(f)
        
        # Transform each camera
        cameras_world = {}
        for cam_id, cam_data in cameras_local.items():
            cameras_world[cam_id] = transform_camera(cam_data, R, trans, s)
        
        # Save
        with open(args.output, 'w') as f:
            json.dump(cameras_world, f, indent=2)
        
        print(f"Transformed {len(cameras_world)} cameras")
        print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
