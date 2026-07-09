"""KRT optimizer for camera pose estimation.

Faithful Python port of src/core/krt_optimizer.{h,cc}.
Uses scipy.optimize.least_squares to replace Ceres Solver.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from .types import Camera

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Factor types matching C++ FACTOR_TYPE enum."""
    F = 0           # f, R
    FDist = 1       # f, R, dist
    Fxfy = 2        # fx, fy, R
    FxfyDist = 3    # fx, fy, R, dist


# ====================================== Factor2d2d ======================================

def factor_2d2d_residual(param: np.ndarray, cam1: Camera, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """2D-2D homography residual: f, R (fx=fy)."""
    param_copy = param.copy()
    param_copy[1] = param_copy[0]  # fy = fx
    
    cam2 = Camera.from_vector(param_copy)
    
    pt1 = np.array([[uv1[0]], [uv1[1]], [1.0]])
    ray1 = np.linalg.inv(cam1.R()) @ np.linalg.inv(cam1.K()) @ pt1
    ray1 /= np.linalg.norm(ray1)
    
    # x2 = H21*x1 = K2*R2*R1^(-1)*K1^(-1)*x1
    uv2_predict = cam2.K() @ cam2.R() @ ray1
    uv2_predict /= uv2_predict[2, 0]
    
    residual = np.array([
        uv2[0] - uv2_predict[0, 0],
        uv2[1] - uv2_predict[1, 0]
    ])
    
    return residual


# ====================================== Factor2d2dFxfy ======================================

def factor_2d2d_fxfy_residual(param: np.ndarray, cam1: Camera, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """2D-2D homography residual: fx, fy, R."""
    cam2 = Camera.from_vector(param)
    
    pt1 = np.array([[uv1[0]], [uv1[1]], [1.0]])
    ray1 = np.linalg.inv(cam1.R()) @ np.linalg.inv(cam1.K()) @ pt1
    
    # x2 = H21*x1 = K2*R2*R1^(-1)*K1^(-1)*x1
    uv2_predict = cam2.K() @ cam2.R() @ ray1
    uv2_predict /= uv2_predict[2, 0]
    
    residual = np.array([
        uv2[0] - uv2_predict[0, 0],
        uv2[1] - uv2_predict[1, 0]
    ])
    
    return residual


# ====================================== Factor2d2dDist ======================================

def factor_2d2d_dist_residual(param: np.ndarray, cam1: Camera, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """2D-2D homography residual with distortion: f, R, dist (fx=fy)."""
    param_copy = param.copy()
    param_copy[1] = param_copy[0]  # fy = fx
    
    cam2 = Camera.from_vector(param_copy)
    
    # Undistort uv1
    uv1_origin = np.array([[uv1[0], uv1[1]]], dtype=np.float32)
    uv1_undistort = cv2.undistortPoints(uv1_origin, cam1.K(), cam1.dist(), None, cam1.K())
    pt1 = np.array([[uv1_undistort[0, 0, 0]], [uv1_undistort[0, 0, 1]], [1.0]])
    
    # Check if near border
    width1 = cam1.K()[0, 2] * 2
    height1 = cam1.K()[1, 2] * 2
    if (uv1_undistort[0, 0, 0] < 0 or uv1_undistort[0, 0, 0] >= width1 or
        uv1_undistort[0, 0, 1] < 0 or uv1_undistort[0, 0, 1] >= height1):
        return np.array([0.0, 0.0])
    
    ray1 = np.linalg.inv(cam1.R()) @ np.linalg.inv(cam1.K()) @ pt1
    ray1 /= np.linalg.norm(ray1)
    
    # x2 = K2*R2*X1
    pt3d_2 = cam2.R() @ ray1
    pt3d_2 /= pt3d_2[2, 0]
    x = pt3d_2[0, 0]
    y = pt3d_2[1, 0]
    
    # Apply distortion
    k1, k2, k3 = param[10], param[11], param[12]
    p1, p2 = param[13], param[14]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r2 * r2 * r2
    xy = x * y
    x2 = x * x
    y2 = y * y
    radial_dist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    
    x_distorted = x * radial_dist + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
    y_distorted = y * radial_dist + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
    
    fx, fy = param[0], param[1]
    cx, cy = param[2], param[3]
    x_proj = fx * x_distorted + cx
    y_proj = fy * y_distorted + cy
    
    residual = np.array([
        uv2[0] - x_proj,
        uv2[1] - y_proj
    ])
    
    return residual


# ====================================== Factor2d2dFxfyDist ======================================

def factor_2d2d_fxfy_dist_residual(param: np.ndarray, cam1: Camera, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """2D-2D homography residual with distortion: fx, fy, R, dist."""
    cam2 = Camera.from_vector(param)
    
    # Undistort uv1
    uv1_origin = np.array([[uv1[0], uv1[1]]], dtype=np.float32)
    uv1_undistort = cv2.undistortPoints(uv1_origin, cam1.K(), cam1.dist(), None, cam1.K())
    pt1 = np.array([[uv1_undistort[0, 0, 0]], [uv1_undistort[0, 0, 1]], [1.0]])
    
    # Check if near border
    width1 = cam1.K()[0, 2] * 2
    height1 = cam1.K()[1, 2] * 2
    if (uv1_undistort[0, 0, 0] < 0 or uv1_undistort[0, 0, 0] >= width1 or
        uv1_undistort[0, 0, 1] < 0 or uv1_undistort[0, 0, 1] >= height1):
        return np.array([0.0, 0.0])
    
    ray1 = np.linalg.inv(cam1.R()) @ np.linalg.inv(cam1.K()) @ pt1
    ray1 /= np.linalg.norm(ray1)
    
    # x2 = K2*R2*X1
    pt3d_2 = cam2.R() @ ray1
    pt3d_2 /= pt3d_2[2, 0]
    x = pt3d_2[0, 0]
    y = pt3d_2[1, 0]
    
    # Apply distortion
    k1, k2, k3 = param[10], param[11], param[12]
    p1, p2 = param[13], param[14]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r2 * r2 * r2
    xy = x * y
    x2 = x * x
    y2 = y * y
    radial_dist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    
    x_distorted = x * radial_dist + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
    y_distorted = y * radial_dist + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
    
    fx, fy = param[0], param[1]
    cx, cy = param[2], param[3]
    x_proj = fx * x_distorted + cx
    y_proj = fy * y_distorted + cy
    
    residual = np.array([
        uv2[0] - x_proj,
        uv2[1] - y_proj
    ])
    
    return residual


# ====================================== Factor2d3dDist ======================================

def factor_2d3d_dist_residual(param: np.ndarray, pt2d: np.ndarray, pt3d: np.ndarray) -> np.ndarray:
    """2D-3D reprojection residual: f, R, dist (fx=fy)."""
    param_copy = param.copy()
    param_copy[1] = param_copy[0]  # fy = fx
    
    cam = Camera.from_vector(param_copy)
    
    pts3d = np.array([[pt3d[0], pt3d[1], pt3d[2]]], dtype=np.float64)
    pts2d_proj, _ = cv2.projectPoints(pts3d, cam.rvec(), cam.t(), cam.K(), cam.dist())
    
    residual = np.array([
        pt2d[0] - pts2d_proj[0, 0, 0],
        pt2d[1] - pts2d_proj[0, 0, 1]
    ])
    
    return residual


# ====================================== Factor2d3dFxfyDist ======================================

def factor_2d3d_fxfy_dist_residual(param: np.ndarray, pt2d: np.ndarray, pt3d: np.ndarray) -> np.ndarray:
    """2D-3D reprojection residual: fx, fy, R, dist."""
    cam = Camera.from_vector(param)
    
    pts3d = np.array([[pt3d[0], pt3d[1], pt3d[2]]], dtype=np.float64)
    pts2d_proj, _ = cv2.projectPoints(pts3d, cam.rvec(), cam.t(), cam.K(), cam.dist())
    
    residual = np.array([
        pt2d[0] - pts2d_proj[0, 0, 0],
        pt2d[1] - pts2d_proj[0, 0, 1]
    ])
    
    return residual


# ====================================== KRTOptimizer ======================================

class KRTOptimizer:
    """KRT optimizer matching C++ implementation."""
    
    def __init__(self, max_iter: int, max_reproj_error: float, factor_type: FactorType):
        self.max_iter = max_iter
        self.max_reproj_error = max_reproj_error
        self.factor_type = factor_type
        self.set_fixed_focal = False
        
        self.cam_curr_world = Camera()
        self.cam_curr_local = Camera()
        self.cam_curr_local_param = np.zeros(15)
        
        self.R_local_world = np.eye(3)
        self.t_local_world = np.zeros((3, 1))
        
        self.constraints_2d2d = []  # List of (cam_ref_local, uv1, uv2)
        self.constraints_2d3d = []  # List of (pt2d, pt3d_local)
        
        self.num_iter = 0
    
    def set_init_params(self, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: np.ndarray):
        """Set initial camera parameters."""
        self.cam_curr_world.set_K(K.copy())
        self.cam_curr_world.set_dist(dist.copy())
        self.cam_curr_world.set_R(R.copy())
        self.cam_curr_world.set_t(t.copy())

    SetInitParams = set_init_params
    
    def add_2d2d_constraints(self, cam_ref: Camera, kpts_ref: List, kpts_curr: List, matches: List):
        """Add 2D-2D constraints from keypoint matches."""
        # Set reference frame as local coordinate
        self.R_local_world = cam_ref.R().copy()
        self.t_local_world = cam_ref.t().copy()
        
        cam_ref_local = Camera()
        cam_ref_local.set_K(cam_ref.K().copy())
        cam_ref_local.set_dist(cam_ref.dist().copy())
        cam_ref_local.set_R(np.eye(3))
        cam_ref_local.set_t(np.zeros((3, 1)))
        
        # T_curr_local = T_curr_world * T_local_world^{-1}
        R_local_world_inv = np.linalg.inv(self.R_local_world)
        self.cam_curr_local.set_K(self.cam_curr_world.K().copy())
        self.cam_curr_local.set_dist(self.cam_curr_world.dist().copy())
        self.cam_curr_local.set_R(self.cam_curr_world.R() @ R_local_world_inv)
        self.cam_curr_local.set_t(
            -self.cam_curr_world.R() @ R_local_world_inv @ self.t_local_world + self.cam_curr_world.t()
        )
        
        self.cam_curr_local_param = self.cam_curr_local.to_vector()
        
        for match in matches:
            uv1 = np.array([kpts_ref[match.queryIdx].pt[0], kpts_ref[match.queryIdx].pt[1]])
            uv2 = np.array([kpts_curr[match.trainIdx].pt[0], kpts_curr[match.trainIdx].pt[1]])
            self.constraints_2d2d.append((cam_ref_local, uv1, uv2))

    Add2d2dConstraints = add_2d2d_constraints
    
    def add_2d3d_constraints(self, pts2d: List[np.ndarray], pts3d: List[np.ndarray]):
        """Add 2D-3D constraints."""
        if len(pts2d) != len(pts3d) or len(pts2d) == 0:
            return
        
        # Convert pts3d to local coordinate
        for i in range(len(pts3d)):
            pt3d_world = np.array([[pts3d[i][0]], [pts3d[i][1]], [pts3d[i][2]]])
            pt3d_local = self.R_local_world @ pt3d_world + self.t_local_world
            pt3d_local_vec = np.array([pt3d_local[0, 0], pt3d_local[1, 0], pt3d_local[2, 0]])
            self.constraints_2d3d.append((pts2d[i], pt3d_local_vec))

    Add2d3dConstraints = add_2d3d_constraints
    
    def _build_residual_function(self):
        """Build residual function for all constraints."""
        def residual_fn(x):
            residuals = []
            
            # 2D-2D constraints
            for cam_ref_local, uv1, uv2 in self.constraints_2d2d:
                if self.factor_type == FactorType.F:
                    res = factor_2d2d_residual(x, cam_ref_local, uv1, uv2)
                elif self.factor_type == FactorType.Fxfy:
                    res = factor_2d2d_fxfy_residual(x, cam_ref_local, uv1, uv2)
                elif self.factor_type == FactorType.FDist:
                    res = factor_2d2d_dist_residual(x, cam_ref_local, uv1, uv2)
                elif self.factor_type == FactorType.FxfyDist:
                    res = factor_2d2d_fxfy_dist_residual(x, cam_ref_local, uv1, uv2)
                else:
                    res = np.array([0.0, 0.0])
                residuals.extend(res)
            
            # 2D-3D constraints
            for pt2d, pt3d_local in self.constraints_2d3d:
                if self.factor_type in [FactorType.F, FactorType.FDist]:
                    res = factor_2d3d_dist_residual(x, pt2d, pt3d_local)
                elif self.factor_type in [FactorType.Fxfy, FactorType.FxfyDist]:
                    res = factor_2d3d_fxfy_dist_residual(x, pt2d, pt3d_local)
                else:
                    res = np.array([0.0, 0.0])
                residuals.extend(res)
            
            return np.array(residuals)
        
        return residual_fn
    
    def _get_fixed_indices(self) -> List[int]:
        """Parameter indices fixed by C++ SubsetParameterization."""
        # Fixed parameters based on factor type
        if self.factor_type == FactorType.F:
            # Fix: fy, cx, cy, t[0:2], dist[0:4]
            fixed_indices = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14]
        elif self.factor_type == FactorType.Fxfy:
            # Fix: cx, cy, t[0:2], dist[0:4]
            fixed_indices = [2, 3, 7, 8, 9, 10, 11, 12, 13, 14]
        elif self.factor_type == FactorType.FDist:
            # Fix: fy, cx, cy, t[0:2], dist[1:4]
            fixed_indices = [1, 2, 3, 7, 8, 9, 11, 12, 13, 14]
        elif self.factor_type == FactorType.FxfyDist:
            # Fix: cx, cy, t[0:2], dist[1:4]
            fixed_indices = [2, 3, 7, 8, 9, 11, 12, 13, 14]
        else:
            fixed_indices = []

        return fixed_indices
    
    def solve(self) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve the optimization problem."""
        residual_fn = self._build_residual_function()
        fixed_indices = set(self._get_fixed_indices())
        free_indices = [i for i in range(15) if i not in fixed_indices]
        base_param = self.cam_curr_local_param.copy()

        def pack_residual(free_values):
            x = base_param.copy()
            x[free_indices] = free_values
            return residual_fn(x)
        
        try:
            if not free_indices:
                return False, None, None, None, None

            initial_residuals = residual_fn(self.cam_curr_local.to_vector())
            result = least_squares(
                pack_residual,
                base_param[free_indices],
                method='trf',
                max_nfev=self.max_iter * 100,
                verbose=0
            )
            
            refined = base_param.copy()
            refined[free_indices] = result.x
            self.cam_curr_local_param = refined
            self.num_iter = result.nfev
            
            initial_cost = np.sum(initial_residuals ** 2)
            final_cost = result.cost * 2  # least_squares returns 0.5 * sum(residuals^2)
            num_residuals = len(result.fun)
            
            if not self._check_results(initial_cost, final_cost, num_residuals, result.success):
                return False, None, None, None, None
            
            K, R, t, dist = self._obtain_refined_camera_params()
            return True, K, R, t, dist
        
        except Exception as e:
            logger.warning(f"KRT optimization failed: {e}")
            return False, None, None, None, None

    Solve = solve
    
    def _check_results(self, initial_cost: float, final_cost: float, num_residuals: int, success: bool) -> bool:
        """Check optimization results."""
        if num_residuals <= 0:
            return False
        init_reproj_error = math.sqrt(2) * math.sqrt((2 * initial_cost) / num_residuals)
        final_reproj_error = math.sqrt(2) * math.sqrt((2 * final_cost) / num_residuals)
        
        logger.debug(f"Init reprojection error: {init_reproj_error:.4f}, "
                    f"final reprojection error: {final_reproj_error:.4f}")
        
        # Check convergence
        if not success:
            return False
        
        # Check reprojection error
        if final_reproj_error >= self.max_reproj_error:
            return False
        
        # Check focal length and FOV
        cam = Camera.from_vector(self.cam_curr_local_param)
        fx = cam.K()[0, 0]
        fy = cam.K()[1, 1]
        cx = cam.K()[0, 2]
        cy = cam.K()[1, 2]
        fov_x = math.atan(cx / fx) * 2 * 180 / math.pi
        fov_y = math.atan(cy / fy) * 2 * 180 / math.pi
        
        if fov_x < 0 or fov_x > 170 or fov_y < 0 or fov_y > 170:
            logger.debug(f"FOV invalid! fov_x: {fov_x:.2f}, fov_y: {fov_y:.2f}")
            return False
        
        return True
    
    def _obtain_refined_camera_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform camera parameters from local to world coordinate."""
        param = self.cam_curr_local_param.copy()
        
        # Enforce fx = fy for F and FDist modes
        if self.factor_type in [FactorType.F, FactorType.FDist]:
            param[1] = param[0]
        
        self.cam_curr_local = Camera.from_vector(param)
        
        # Transform from local to world: T_curr_world = T_curr_local * T_local_world
        self.cam_curr_world.set_K(self.cam_curr_local.K().copy())
        self.cam_curr_world.set_dist(self.cam_curr_local.dist().copy())
        self.cam_curr_world.set_R(self.cam_curr_local.R() @ self.R_local_world)
        self.cam_curr_world.set_t(self.cam_curr_local.R() @ self.t_local_world + self.cam_curr_local.t())
        
        return (
            self.cam_curr_world.K(),
            self.cam_curr_world.R(),
            self.cam_curr_world.t(),
            self.cam_curr_world.dist()
        )
    
    def cal_2d2d_reproj_error(self, cam_ref: Camera, kpts_ref: List, kpts_curr: List, matches: List) -> float:
        """Calculate 2D-2D reprojection error."""
        cam_ref_local = Camera()
        cam_ref_local.set_K(cam_ref.K().copy())
        cam_ref_local.set_dist(cam_ref.dist().copy())
        cam_ref_local.set_R(np.eye(3))
        cam_ref_local.set_t(np.zeros((3, 1)))
        
        residuals_sum = 0.0
        
        for match in matches:
            uv1 = np.array([kpts_ref[match.queryIdx].pt[0], kpts_ref[match.queryIdx].pt[1]])
            uv2 = np.array([kpts_curr[match.trainIdx].pt[0], kpts_curr[match.trainIdx].pt[1]])
            
            if self.factor_type == FactorType.F:
                res = factor_2d2d_residual(self.cam_curr_local_param, cam_ref_local, uv1, uv2)
            elif self.factor_type == FactorType.Fxfy:
                res = factor_2d2d_fxfy_residual(self.cam_curr_local_param, cam_ref_local, uv1, uv2)
            elif self.factor_type == FactorType.FDist:
                res = factor_2d2d_dist_residual(self.cam_curr_local_param, cam_ref_local, uv1, uv2)
            elif self.factor_type == FactorType.FxfyDist:
                res = factor_2d2d_fxfy_dist_residual(self.cam_curr_local_param, cam_ref_local, uv1, uv2)
            else:
                res = np.array([0.0, 0.0])
            
            residuals_sum += res[0] ** 2 + res[1] ** 2
        
        num_observation = len(matches)
        reproj_error = math.sqrt(residuals_sum / num_observation)
        
        return reproj_error

    Cal2d2dReprojError = cal_2d2d_reproj_error
    
    def cal_2d3d_reproj_error(self, pts2d: List[np.ndarray], pts3d: List[np.ndarray]) -> float:
        """Calculate 2D-3D reprojection error."""
        if len(pts2d) != len(pts3d) or len(pts2d) == 0:
            return -1.0
        
        # Convert pts3d to local coordinate
        pts3d_local = []
        for pt3d in pts3d:
            pt3d_world = np.array([[pt3d[0]], [pt3d[1]], [pt3d[2]]])
            pt3d_local_mat = self.R_local_world @ pt3d_world + self.t_local_world
            pts3d_local.append(np.array([pt3d_local_mat[0, 0], pt3d_local_mat[1, 0], pt3d_local_mat[2, 0]]))
        
        residuals_sum = 0.0
        
        for i in range(len(pts2d)):
            if self.factor_type in [FactorType.F, FactorType.FDist]:
                res = factor_2d3d_dist_residual(self.cam_curr_local_param, pts2d[i], pts3d_local[i])
            elif self.factor_type in [FactorType.Fxfy, FactorType.FxfyDist]:
                res = factor_2d3d_fxfy_dist_residual(self.cam_curr_local_param, pts2d[i], pts3d_local[i])
            else:
                res = np.array([0.0, 0.0])
            
            residuals_sum += res[0] ** 2 + res[1] ** 2
        
        num_observation = len(pts2d)
        reproj_error = math.sqrt(residuals_sum / num_observation)
        
        return reproj_error

    Cal2d3dReprojError = cal_2d3d_reproj_error

    def set_fixed_focal_flag(self):
        self.set_fixed_focal = True

    SetFixedFocal = set_fixed_focal_flag
