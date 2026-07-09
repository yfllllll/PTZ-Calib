"""PTZ-Ray optimizer using pyceres.

Python port of src/core/ptzray_optimizer.{h,cc}.
"""

from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import cv2
import pyceres

from .types import Camera, ImageFeatures, MatchesInfo, Ray
from .tracks import TracksBuilder, Track, Tracks


class FactorType(Enum):
    """Cost function type for PTZ-Ray optimization."""
    PTZRay = 0
    PTZRayDist = 1
    PTZRayFxfyDist = 2
    PTZRayDistDisp = 3


# ---------------- Helpers ----------------

def _build_camera_from_intr_extr(intrinsics, extrinsics, share_fx=True):
    """Reconstruct the 15-vector camera parameters used in the C++ code.

    intrinsics layout (size 9): fx, fy, cx, cy, k1, k2, k3, p1, p2
    extrinsics layout (size 6): r1, r2, r3, t1, t2, t3
    If share_fx: fy is forced = fx (matches C++ PTZRayFactor / PTZRayDistFactor / PTZRayDistDispFactor).
    """
    param = np.zeros(15, dtype=np.float64)
    param[0] = intrinsics[0]
    param[1] = intrinsics[0] if share_fx else intrinsics[1]
    param[2] = intrinsics[2]
    param[3] = intrinsics[3]
    param[4:10] = extrinsics[:6]
    param[10:15] = intrinsics[4:9]
    return param


def _rodrigues_rvec_to_R(rvec):
    """cv2.Rodrigues wrapper returning 3x3 R from a 3-vector."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def _apply_dist(x, y, k1, k2, k3, p1, p2):
    """Apply radial+tangential distortion to normalized image coords."""
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r2 * r4
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xy = x * y
    xd = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y * y)
    return xd, yd


# ---------------- Cost functions ----------------

class PTZRayFactor(pyceres.CostFunction):
    """f, R, ray. x = KRX, no distortion. shared fx=fy."""

    def __init__(self, uv):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 6, 3])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)

    def Evaluate(self, parameters, residuals, jacobians):
        intr = parameters[0]
        extr = parameters[1]
        ray = parameters[2]

        param = _build_camera_from_intr_extr(intr, extr, share_fx=True)
        R = _rodrigues_rvec_to_R(param[4:7])
        K = np.array([[param[0], 0, param[2]],
                      [0, param[1], param[3]],
                      [0, 0, 1.0]], dtype=np.float64)

        cv_ray = np.asarray(ray, dtype=np.float64).reshape(3, 1)
        n = np.linalg.norm(cv_ray)
        if n > 0:
            cv_ray = cv_ray / n

        uv_predict = K @ R @ cv_ray
        uv_predict = uv_predict / uv_predict[2, 0]

        residuals[0] = self.uv[0] - uv_predict[0, 0]
        residuals[1] = self.uv[1] - uv_predict[1, 0]
        return True



class PTZRayDistFactor(pyceres.CostFunction):
    """f, R, k1, k2, k3, p1, p2, ray. shared fx=fy. Includes penalty if pt3d behind camera."""

    def __init__(self, uv):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 6, 3])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)

    def Evaluate(self, parameters, residuals, jacobians):
        intr = parameters[0]
        extr = parameters[1]
        ray = parameters[2]
        param = _build_camera_from_intr_extr(intr, extr, share_fx=True)
        R = _rodrigues_rvec_to_R(param[4:7])

        cv_ray = np.asarray(ray, dtype=np.float64).reshape(3, 1)
        # Note: C++ code has ray/=norm commented out here
        pt3d = R @ cv_ray

        # penalty if behind camera
        if pt3d[2, 0] < 0:
            residuals[0] = 1000000.0
            residuals[1] = 1000000.0
            return True

        pt3d = pt3d / pt3d[2, 0]
        x, y = pt3d[0, 0], pt3d[1, 0]
        fx, fy, cx, cy = param[0], param[1], param[2], param[3]
        k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
        xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
        x_proj = fx * xd + cx
        y_proj = fy * yd + cy
        residuals[0] = self.uv[0] - x_proj
        residuals[1] = self.uv[1] - y_proj
        return True



class PTZRayFxfyDistFactor(pyceres.CostFunction):
    """fx, fy, R, k1, k2, k3, p1, p2, ray. fx != fy."""

    def __init__(self, uv):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 6, 3])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)

    def Evaluate(self, parameters, residuals, jacobians):
        intr, extr, ray = parameters[0], parameters[1], parameters[2]
        param = _build_camera_from_intr_extr(intr, extr, share_fx=False)
        R = _rodrigues_rvec_to_R(param[4:7])
        cv_ray = np.asarray(ray, dtype=np.float64).reshape(3, 1)
        cv_ray = cv_ray / np.linalg.norm(cv_ray)
        pt3d = R @ cv_ray
        pt3d = pt3d / pt3d[2, 0]
        x, y = pt3d[0, 0], pt3d[1, 0]
        fx, fy, cx, cy = param[0], param[1], param[2], param[3]
        k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
        xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
        residuals[0] = self.uv[0] - (fx * xd + cx)
        residuals[1] = self.uv[1] - (fy * yd + cy)
        return True


class PTZRayDistDispFactor(pyceres.CostFunction):
    """f, R, d1, d2, d3, k1, k2, k3, p1, p2, ray. Displacement model."""

    def __init__(self, uv):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 3, 6, 3])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)

    def Evaluate(self, parameters, residuals, jacobians):
        intr, disp, extr, ray = parameters[0], parameters[1], parameters[2], parameters[3]
        param = _build_camera_from_intr_extr(intr, extr, share_fx=True)
        R = _rodrigues_rvec_to_R(param[4:7])
        cv_ray = np.asarray(ray, dtype=np.float64).reshape(3, 1)
        cv_ray = cv_ray / np.linalg.norm(cv_ray)
        pt3d = R @ cv_ray
        displacement = disp[0] + disp[1] * param[0] + disp[2] * param[0] * param[0]
        pt3d[2, 0] += displacement
        pt3d = pt3d / pt3d[2, 0]
        x, y = pt3d[0, 0], pt3d[1, 0]
        fx, fy, cx, cy = param[0], param[1], param[2], param[3]
        k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
        xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
        residuals[0] = self.uv[0] - (fx * xd + cx)
        residuals[1] = self.uv[1] - (fy * yd + cy)
        return True


class Reproj2d3dFactor(pyceres.CostFunction):
    """2d-3d reprojection error. intrinsics(9), extrinsics(6), tlw(6)."""

    def __init__(self, uv, pt3d):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 6, 6])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)
        self.pt3d = np.asarray(pt3d, dtype=np.float64).reshape(3)

    def Evaluate(self, parameters, residuals, jacobians):
        intr, extr, tlw = parameters[0], parameters[1], parameters[2]
        param = _build_camera_from_intr_extr(intr, extr, share_fx=False)
        R = _rodrigues_rvec_to_R(param[4:7])
        R_l_w = _rodrigues_rvec_to_R(tlw[:3])
        t_l_w = np.asarray(tlw[3:6], dtype=np.float64).reshape(3, 1)
        pt3d_w = self.pt3d.reshape(3, 1)
        pt3d_l = R_l_w @ pt3d_w + t_l_w
        pt_cam = R @ pt3d_l
        pt_cam = pt_cam / pt_cam[2, 0]
        x, y = pt_cam[0, 0], pt_cam[1, 0]
        fx, fy, cx, cy = param[0], param[1], param[2], param[3]
        k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
        xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
        residuals[0] = self.uv[0] - (fx * xd + cx)
        residuals[1] = self.uv[1] - (fy * yd + cy)
        return True


class Reproj2d3dDispFactor(pyceres.CostFunction):
    """2d-3d with displacement. intrinsics(9), disp(3), extrinsics(6), tlw(6)."""

    def __init__(self, uv, pt3d):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 3, 6, 6])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)
        self.pt3d = np.asarray(pt3d, dtype=np.float64).reshape(3)

    def Evaluate(self, parameters, residuals, jacobians):
        intr, disp, extr, tlw = parameters[0], parameters[1], parameters[2], parameters[3]
        param = _build_camera_from_intr_extr(intr, extr, share_fx=False)
        R = _rodrigues_rvec_to_R(param[4:7])
        R_l_w = _rodrigues_rvec_to_R(tlw[:3])
        t_l_w = np.asarray(tlw[3:6], dtype=np.float64).reshape(3, 1)
        pt3d_w = self.pt3d.reshape(3, 1)
        pt3d_l = R_l_w @ pt3d_w + t_l_w
        pt_cam = R @ pt3d_l
        displacement = disp[0] + disp[1] * param[0] + disp[2] * param[0] * param[0]
        pt_cam[2, 0] += displacement
        pt_cam = pt_cam / pt_cam[2, 0]
        x, y = pt_cam[0, 0], pt_cam[1, 0]
        fx, fy, cx, cy = param[0], param[1], param[2], param[3]
        k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
        xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
        residuals[0] = self.uv[0] - (fx * xd + cx)
        residuals[1] = self.uv[1] - (fy * yd + cy)
        return True



class PTZRayOptimizer:
    """PTZ-Ray bundle adjustment optimizer."""
    
    def __init__(self, features, matches_info, cameras, cam_ids, max_iter, factor_type,
                 pixels=None, pts3d=None):
        pass
    
    def solve(self, cameras, rays=None):
        pass
