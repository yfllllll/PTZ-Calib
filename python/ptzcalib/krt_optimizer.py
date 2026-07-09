"""KRT optimizer for initial camera pose estimation.

Python port of src/core/krt_optimizer.h/cc using scipy.optimize.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

from .types import Camera

logger = logging.getLogger(__name__)


def _projection_residual(
    cam_vec: np.ndarray,
    pts3d: List[np.ndarray],
    pixels: List[np.ndarray],
) -> np.ndarray:
    """Compute reprojection residuals for optimization."""
    cam = Camera.from_vector(cam_vec)
    residuals = []
    
    for i, pt3d in enumerate(pts3d):
        pt_cam = cam.R() @ pt3d.reshape(3, 1) + cam.t()
        if pt_cam[2, 0] < 0.1:
            residuals.extend([1000.0, 1000.0])
            continue
        
        # Project to normalized plane
        x_n = pt_cam[0, 0] / pt_cam[2, 0]
        y_n = pt_cam[1, 0] / pt_cam[2, 0]
        
        # Apply distortion
        r2 = x_n * x_n + y_n * y_n
        dist = cam.dist().reshape(-1)
        k1, k2, p1, p2, k3 = dist[:5]
        
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_d = x_n * radial + 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n * x_n)
        y_d = y_n * radial + p1 * (r2 + 2 * y_n * y_n) + 2 * p2 * x_n * y_n
        
        # Project to pixel
        K = cam.K()
        u_pred = K[0, 0] * x_d + K[0, 2]
        v_pred = K[1, 1] * y_d + K[1, 2]
        
        # Residual
        obs = pixels[i].reshape(-1)
        residuals.append(u_pred - obs[0])
        residuals.append(v_pred - obs[1])
    
    return np.array(residuals)


def optimize_krt(
    camera_init: Camera,
    pts3d: List[np.ndarray],
    pixels: List[np.ndarray],
    fix_intrinsic: bool = True,
) -> Tuple[bool, Camera]:
    """Optimize camera extrinsics (and optionally intrinsics) via BA.
    
    Returns (success, optimized_camera).
    """
    if len(pts3d) < 4 or len(pixels) < 4:
        return False, camera_init
    
    x0 = camera_init.to_vector()
    
    try:
        if fix_intrinsic:
            # Only optimize R, t (indices 4-9 in the 15-vector)
            def residual_fn(x_rt):
                x_full = x0.copy()
                x_full[4:10] = x_rt
                return _projection_residual(x_full, pts3d, pixels)
            
            res = least_squares(
                residual_fn,
                x0[4:10],
                method='lm',
                max_nfev=100,
            )
            x_opt = x0.copy()
            x_opt[4:10] = res.x
        else:
            # Optimize all parameters
            res = least_squares(
                lambda x: _projection_residual(x, pts3d, pixels),
                x0,
                method='lm',
                max_nfev=200,
            )
            x_opt = res.x
        
        cam_opt = Camera.from_vector(x_opt)
        return True, cam_opt
    
    except Exception as e:
        logger.warning("KRT optimization failed: %s", e)
        return False, camera_init
