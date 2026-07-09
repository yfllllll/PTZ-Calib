"""PTZ-Ray optimizer using pyceres.

Python port of src/core/ptzray_optimizer.{h,cc}.
"""

from enum import Enum
import logging
import math
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import cv2
import pyceres

from .types import Camera, ImageFeatures, MatchesInfo, Ray
from .tracks import TracksBuilder, Track, Tracks

logger = logging.getLogger(__name__)


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


def _fill_numeric_jacobians(eval_residuals, parameters, jacobians, eps=1e-6):
    """Fill pyceres jacobians using central differences, like C++ NumericDiff."""
    if jacobians is None:
        return
    base_params = [np.asarray(p, dtype=np.float64) for p in parameters]
    for block_id, jac in enumerate(jacobians):
        if jac is None:
            continue
        block = base_params[block_id]
        num_params = block.size
        num_residuals = len(eval_residuals(base_params))
        for col in range(num_params):
            plus = [p.copy() for p in base_params]
            minus = [p.copy() for p in base_params]
            plus[block_id].reshape(-1)[col] += eps
            minus[block_id].reshape(-1)[col] -= eps
            r_plus = eval_residuals(plus)
            r_minus = eval_residuals(minus)
            deriv = (r_plus - r_minus) / (2.0 * eps)
            for row in range(num_residuals):
                jac[row * num_params + col] = deriv[row]


def _as_uv(uv):
    return np.asarray(uv, dtype=np.float64).reshape(2)


# ---------------- Cost functions ----------------

class PTZRayFactor(pyceres.CostFunction):
    """f, R, ray. x = KRX, no distortion. shared fx=fy."""

    def __init__(self, uv, weight: float = 1.0):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 6, 3])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)
        self.weight = math.sqrt(float(weight))

    def Evaluate(self, parameters, residuals, jacobians):
        def compute(params):
            intr, extr, ray = params[0], params[1], params[2]
            param = _build_camera_from_intr_extr(intr, extr, share_fx=True)
            R = _rodrigues_rvec_to_R(param[4:7])
            K = np.array([[param[0], 0, param[2]],
                          [0, param[1], param[3]],
                          [0, 0, 1.0]], dtype=np.float64)
            cv_ray = np.asarray(ray, dtype=np.float64).reshape(3, 1)
            cv_ray /= np.linalg.norm(cv_ray)
            uv_predict = K @ R @ cv_ray
            uv_predict /= uv_predict[2, 0]
            return self.weight * np.array([self.uv[0] - uv_predict[0, 0],
                                           self.uv[1] - uv_predict[1, 0]], dtype=np.float64)

        res = compute(parameters)
        residuals[0], residuals[1] = res[0], res[1]
        _fill_numeric_jacobians(compute, parameters, jacobians)
        return True



class PTZRayDistFactor(pyceres.CostFunction):
    """f, R, k1, k2, k3, p1, p2, ray. shared fx=fy. Includes penalty if pt3d behind camera."""

    def __init__(self, uv, weight: float = 1.0):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 6, 3])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)
        self.weight = math.sqrt(float(weight))

    def Evaluate(self, parameters, residuals, jacobians):
        def compute(params):
            intr, extr, ray = params[0], params[1], params[2]
            param = _build_camera_from_intr_extr(intr, extr, share_fx=True)
            R = _rodrigues_rvec_to_R(param[4:7])
            cv_ray = np.asarray(ray, dtype=np.float64).reshape(3, 1)
            pt3d = R @ cv_ray
            if pt3d[2, 0] < 0:
                return self.weight * np.array([1000000.0, 1000000.0], dtype=np.float64)
            pt3d /= pt3d[2, 0]
            x, y = pt3d[0, 0], pt3d[1, 0]
            fx, fy, cx, cy = param[0], param[1], param[2], param[3]
            k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
            xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
            return self.weight * np.array([self.uv[0] - (fx * xd + cx),
                                           self.uv[1] - (fy * yd + cy)], dtype=np.float64)

        res = compute(parameters)
        residuals[0], residuals[1] = res[0], res[1]
        _fill_numeric_jacobians(compute, parameters, jacobians)
        return True



class PTZRayFxfyDistFactor(pyceres.CostFunction):
    """fx, fy, R, k1, k2, k3, p1, p2, ray. fx != fy."""

    def __init__(self, uv, weight: float = 1.0):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 6, 3])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)
        self.weight = math.sqrt(float(weight))

    def Evaluate(self, parameters, residuals, jacobians):
        def compute(params):
            intr, extr, ray = params[0], params[1], params[2]
            param = _build_camera_from_intr_extr(intr, extr, share_fx=False)
            R = _rodrigues_rvec_to_R(param[4:7])
            cv_ray = np.asarray(ray, dtype=np.float64).reshape(3, 1)
            cv_ray /= np.linalg.norm(cv_ray)
            pt3d = R @ cv_ray
            pt3d /= pt3d[2, 0]
            x, y = pt3d[0, 0], pt3d[1, 0]
            fx, fy, cx, cy = param[0], param[1], param[2], param[3]
            k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
            xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
            return self.weight * np.array([self.uv[0] - (fx * xd + cx),
                                           self.uv[1] - (fy * yd + cy)], dtype=np.float64)

        res = compute(parameters)
        residuals[0], residuals[1] = res[0], res[1]
        _fill_numeric_jacobians(compute, parameters, jacobians)
        return True


class PTZRayDistDispFactor(pyceres.CostFunction):
    """f, R, d1, d2, d3, k1, k2, k3, p1, p2, ray. Displacement model."""

    def __init__(self, uv, weight: float = 1.0):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 3, 6, 3])
        self.uv = np.asarray(uv, dtype=np.float64).reshape(2)
        self.weight = math.sqrt(float(weight))

    def Evaluate(self, parameters, residuals, jacobians):
        def compute(params):
            intr, disp, extr, ray = params[0], params[1], params[2], params[3]
            param = _build_camera_from_intr_extr(intr, extr, share_fx=True)
            R = _rodrigues_rvec_to_R(param[4:7])
            cv_ray = np.asarray(ray, dtype=np.float64).reshape(3, 1)
            cv_ray /= np.linalg.norm(cv_ray)
            pt3d = R @ cv_ray
            displacement = disp[0] + disp[1] * param[0] + disp[2] * param[0] * param[0]
            pt3d[2, 0] += displacement
            pt3d /= pt3d[2, 0]
            x, y = pt3d[0, 0], pt3d[1, 0]
            fx, fy, cx, cy = param[0], param[1], param[2], param[3]
            k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
            xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
            return self.weight * np.array([self.uv[0] - (fx * xd + cx),
                                           self.uv[1] - (fy * yd + cy)], dtype=np.float64)

        res = compute(parameters)
        residuals[0], residuals[1] = res[0], res[1]
        _fill_numeric_jacobians(compute, parameters, jacobians)
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
        def compute(params):
            intr, extr, tlw = params[0], params[1], params[2]
            param = _build_camera_from_intr_extr(intr, extr, share_fx=False)
            R = _rodrigues_rvec_to_R(param[4:7])
            R_l_w = _rodrigues_rvec_to_R(tlw[:3])
            t_l_w = np.asarray(tlw[3:6], dtype=np.float64).reshape(3, 1)
            pt3d_w = self.pt3d.reshape(3, 1)
            pt3d_l = R_l_w @ pt3d_w + t_l_w
            pt_cam = R @ pt3d_l
            pt_cam /= pt_cam[2, 0]
            x, y = pt_cam[0, 0], pt_cam[1, 0]
            fx, fy, cx, cy = param[0], param[1], param[2], param[3]
            k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
            xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
            return np.array([self.uv[0] - (fx * xd + cx),
                             self.uv[1] - (fy * yd + cy)], dtype=np.float64)

        res = compute(parameters)
        residuals[0], residuals[1] = res[0], res[1]
        _fill_numeric_jacobians(compute, parameters, jacobians)
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
        def compute(params):
            intr, disp, extr, tlw = params[0], params[1], params[2], params[3]
            param = _build_camera_from_intr_extr(intr, extr, share_fx=False)
            R = _rodrigues_rvec_to_R(param[4:7])
            R_l_w = _rodrigues_rvec_to_R(tlw[:3])
            t_l_w = np.asarray(tlw[3:6], dtype=np.float64).reshape(3, 1)
            pt3d_w = self.pt3d.reshape(3, 1)
            pt3d_l = R_l_w @ pt3d_w + t_l_w
            pt_cam = R @ pt3d_l
            displacement = disp[0] + disp[1] * param[0] + disp[2] * param[0] * param[0]
            pt_cam[2, 0] += displacement
            pt_cam /= pt_cam[2, 0]
            x, y = pt_cam[0, 0], pt_cam[1, 0]
            fx, fy, cx, cy = param[0], param[1], param[2], param[3]
            k1, k2, k3, p1, p2 = param[10], param[11], param[12], param[13], param[14]
            xd, yd = _apply_dist(x, y, k1, k2, k3, p1, p2)
            return np.array([self.uv[0] - (fx * xd + cx),
                             self.uv[1] - (fy * yd + cy)], dtype=np.float64)

        res = compute(parameters)
        residuals[0], residuals[1] = res[0], res[1]
        _fill_numeric_jacobians(compute, parameters, jacobians)
        return True



class PTZRayOptimizer:
    """PTZ-Ray bundle adjustment optimizer."""
    
    def __init__(self, features, matches_info, cameras, *args):
        self.features_ = features
        self.matches_info_ = matches_info
        self.cameras_ = [cam.clone() for cam in cameras]
        self.num_cams_ = len(cameras)

        if len(args) == 3:
            pixels, pts3d = [], []
            cam_ids, max_iter, factor_type = args
        elif len(args) == 5:
            pixels, pts3d, cam_ids, max_iter, factor_type = args
        else:
            raise TypeError(
                "PTZRayOptimizer expects either (cam_ids, max_iter, factor_type) "
                "or (pixels, pts3d, cam_ids, max_iter, factor_type)"
            )

        self.cam_ids_ = set(cam_ids) if cam_ids else set(range(self.num_cams_))
        self.max_iter_ = max_iter
        self.type_ = factor_type
        self.pixels_ = pixels if pixels else []
        self.pts3d_ = pts3d if pts3d else []
        
        self.shared_ic_ids_ = list(range(self.num_cams_))
        self.intrinsics_param_: Dict[int, np.ndarray] = {}
        self.extrinsics_param_: Dict[int, np.ndarray] = {}
        self.rays_param_: Dict[int, np.ndarray] = {}
        self.disp_param_ = np.zeros(3, dtype=np.float64)
        self.tlw_param_ = np.zeros(6, dtype=np.float64)
        self.tracks_: Tracks = {}
        self.track_len_ = 0
        self.max_track_len_ = 0
        self.min_track_len_ = 0
        self.summary_ = None
        self.final_reproj_error_all_ = 0.0
        self.final_reproj_error_2d2d_ = 0.0
        self.final_reproj_error_2d3d_ = 0.0
    
    def _is_candidate(self, image_id):
        return image_id in self.cam_ids_

    isCandidate = _is_candidate

    def set_shared_intrinsics(self, shared_ic_ids: List[int]) -> None:
        if len(shared_ic_ids) != self.num_cams_:
            logger.warning("Set shared intrinsics failed, length not matched")
            return
        self.shared_ic_ids_ = list(shared_ic_ids)

    SetSharedIntrinsics = set_shared_intrinsics

    def _check_valid(self) -> bool:
        if not self.features_ or len(self.features_) != self.num_cams_:
            return False
        if self.max_iter_ <= 0:
            return False
        if self.pixels_:
            if len(self.pixels_) != self.num_cams_ or len(self.pts3d_) != self.num_cams_:
                return False
            for i in range(self.num_cams_):
                if len(self.pixels_[i]) != len(self.pts3d_[i]):
                    return False
        return True
    
    def _find_tracks(self):
        builder = TracksBuilder()
        builder.build(self.matches_info_)
        builder.filter(4)
        self.tracks_ = builder.export_to_stl()
        lengths = [len(t) for t in self.tracks_.values()]
        self.track_len_ = int(sum(lengths))
        self.max_track_len_ = int(max(lengths)) if lengths else 0
        self.min_track_len_ = int(min(lengths)) if lengths else 0

    FindTracks = _find_tracks

    @staticmethod
    def T_l_w(tlw):
        R_l_w = _rodrigues_rvec_to_R(np.asarray(tlw[:3], dtype=np.float64))
        t_l_w = np.asarray(tlw[3:6], dtype=np.float64).reshape(3, 1)
        return R_l_w, t_l_w

    def _set_init_trans_local_to_world(self) -> bool:
        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            if not self.pixels_ or i >= len(self.pixels_) or len(self.pixels_[i]) == 0:
                continue

            pts3d = np.asarray(self.pts3d_[i], dtype=np.float64).reshape(-1, 3)
            pixels = np.asarray(self.pixels_[i], dtype=np.float64).reshape(-1, 2)
            if len(pts3d) < 4:
                continue

            ok, rvec, tvec = cv2.solvePnP(
                pts3d,
                pixels,
                self.cameras_[i].K(),
                self.cameras_[i].dist(),
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok:
                continue

            R, _ = cv2.Rodrigues(rvec)
            p3d = pts3d[0].reshape(3, 1)
            p3d_cam = R @ p3d + tvec.reshape(3, 1)
            if p3d_cam[2, 0] < 0 or np.linalg.det(R) < 0.0:
                continue

            predict_pixels, _ = cv2.projectPoints(
                pts3d.astype(np.float32),
                rvec,
                tvec,
                self.cameras_[i].K(),
                None,
            )
            diff = predict_pixels.reshape(-1, 2) - pixels
            reproj_error = math.sqrt(float(np.sum(diff * diff)) / len(pixels))
            if reproj_error > 300:
                continue

            T_i_w = np.eye(4, dtype=np.float64)
            T_i_w[:3, :3] = R
            T_i_w[:3, 3:4] = tvec.reshape(3, 1)
            T_i_l = np.eye(4, dtype=np.float64)
            T_i_l[:3, :3] = self.cameras_[i].R()
            T_i_l[:3, 3:4] = self.cameras_[i].t()
            T_l_w = np.linalg.inv(T_i_l) @ T_i_w
            rvec_l_w, _ = cv2.Rodrigues(T_l_w[:3, :3])
            self.tlw_param_ = np.array(
                [
                    rvec_l_w[0, 0],
                    rvec_l_w[1, 0],
                    rvec_l_w[2, 0],
                    T_l_w[0, 3],
                    T_l_w[1, 3],
                    T_l_w[2, 3],
                ],
                dtype=np.float64,
            )
            return True

        self.tlw_param_ = np.zeros(6, dtype=np.float64)
        return False

    SetInitTransLocalToWorld = _set_init_trans_local_to_world
    
    def _set_up_initial_camera_params(self):
        self.intrinsics_param_.clear()
        self.extrinsics_param_.clear()
        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            ic_id = self.shared_ic_ids_[i]
            cam = self.cameras_[i]
            if ic_id not in self.intrinsics_param_:
                param = cam.to_vector()
                self.intrinsics_param_[ic_id] = np.array([
                    param[0], param[1], param[2], param[3],
                    param[10], param[11], param[12], param[13], param[14],
                ], dtype=np.float64)
            param = cam.to_vector()
            self.extrinsics_param_[i] = param[4:10].copy()
        
        self.disp_param_ = np.zeros(3, dtype=np.float64)
        self.rays_param_.clear()
        for track_id, track in self.tracks_.items():
            ray = self._pix2ray(track)
            if ray is not None:
                self.rays_param_[track_id] = ray.reshape(-1).copy()

    SetUpInitialCameraParams = _set_up_initial_camera_params
    
    def _pix2ray(self, track):
        ray_sum = np.zeros((3, 1), dtype=np.float64)
        count = 0
        for img_id, feat_id in track.items():
            if not self._is_candidate(img_id):
                continue
            if img_id >= len(self.features_) or feat_id >= len(self.features_[img_id].keypoints):
                continue
            pt = self.features_[img_id].keypoints[feat_id].pt
            uv = np.array([[pt[0]], [pt[1]], [1.0]], dtype=np.float64)
            cam = self.cameras_[img_id]
            ray_temp = np.linalg.inv(cam.R_) @ np.linalg.inv(cam.K_) @ uv
            ray_temp = ray_temp / np.linalg.norm(ray_temp)
            ray_sum += ray_temp
            count += 1
        if count == 0:
            return None
        ray_sum /= count
        ray_sum /= np.linalg.norm(ray_sum)
        return ray_sum

    Pix2Ray = _pix2ray
    
    def solve(self, cameras_out, rays_out=None):
        if not self._check_valid():
            return False

        self._find_tracks()
        self._set_init_trans_local_to_world()
        self._set_up_initial_camera_params()

        problem = pyceres.Problem()
        self._add_constraints_2d2d(problem)
        if self.pixels_ and any(len(p) > 0 for p in self.pixels_):
            self._add_constraints_2d3d(problem)
        
        # Add parameter manifolds (SubsetParameterization in C++)
        self._set_parameter_manifolds(problem)
        
        options = pyceres.SolverOptions()
        options.max_num_iterations = self.max_iter_
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.minimizer_progress_to_stdout = True
        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)
        self.summary_ = summary

        self._cal_reproj_error()

        if summary.termination_type == pyceres.TerminationType.CONVERGENCE:
            self._obtain_refined_camera_params(cameras_out, rays_out)
            return True
        return False

    Solve = solve
    
    def _set_parameter_manifolds(self, problem):
        """Set SubsetManifold for parameters (matches C++ SubsetParameterization)."""
        # Intrinsics: fix certain parameters based on factor type
        for ic_id in self.intrinsics_param_.keys():
            if self.type_ == FactorType.PTZRay:
                # Fix cx, cy, k1, k2, k3, p1, p2 (indices 2,3,4,5,6,7,8)
                manifold = pyceres.SubsetManifold(9, [2, 3, 4, 5, 6, 7, 8])
                problem.set_manifold(self.intrinsics_param_[ic_id], manifold)
            elif self.type_ in (FactorType.PTZRayDist, FactorType.PTZRayFxfyDist, FactorType.PTZRayDistDisp):
                # Fix cx, cy, k2, k3, p1, p2 (indices 2,3,5,6,7,8)
                manifold = pyceres.SubsetManifold(9, [2, 3, 5, 6, 7, 8])
                problem.set_manifold(self.intrinsics_param_[ic_id], manifold)
        
        # Extrinsics: fix translation t (indices 3,4,5)
        for i in self.extrinsics_param_.keys():
            manifold = pyceres.SubsetManifold(6, [3, 4, 5])
            problem.set_manifold(self.extrinsics_param_[i], manifold)

    
    def _add_constraints_2d2d(self, problem):
        for track_id, track in self.tracks_.items():
            if track_id not in self.rays_param_:
                continue
            for img_id, feat_id in track.items():
                if not self._is_candidate(img_id):
                    continue
                if img_id >= len(self.features_) or feat_id >= len(self.features_[img_id].keypoints):
                    continue
                uv = self.features_[img_id].keypoints[feat_id].pt
                ic_id = self.shared_ic_ids_[img_id]
                weight = float(len(track))
                
                if self.type_ == FactorType.PTZRay:
                    cost = PTZRayFactor(uv, weight)
                    problem.add_residual_block(
                        cost, None,
                        [self.intrinsics_param_[ic_id], self.extrinsics_param_[img_id], self.rays_param_[track_id]],
                    )
                elif self.type_ == FactorType.PTZRayDist:
                    cost = PTZRayDistFactor(uv, weight)
                    problem.add_residual_block(
                        cost, None,
                        [self.intrinsics_param_[ic_id], self.extrinsics_param_[img_id], self.rays_param_[track_id]],
                    )
                elif self.type_ == FactorType.PTZRayFxfyDist:
                    cost = PTZRayFxfyDistFactor(uv, weight)
                    problem.add_residual_block(
                        cost, None,
                        [self.intrinsics_param_[ic_id], self.extrinsics_param_[img_id], self.rays_param_[track_id]],
                    )
                elif self.type_ == FactorType.PTZRayDistDisp:
                    cost = PTZRayDistDispFactor(uv, weight)
                    problem.add_residual_block(
                        cost, None,
                        [self.intrinsics_param_[ic_id], self.disp_param_, self.extrinsics_param_[img_id], self.rays_param_[track_id]],
                    )

    AddConstraints2d2d = _add_constraints_2d2d
    
    def _add_constraints_2d3d(self, problem):
        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            if not self.pixels_ or i >= len(self.pixels_) or not self.pixels_[i]:
                continue
            ic_id = self.shared_ic_ids_[i]
            for j in range(len(self.pixels_[i])):
                if self.type_ == FactorType.PTZRayDistDisp:
                    cost = Reproj2d3dDispFactor(self.pixels_[i][j], self.pts3d_[i][j])
                    problem.add_residual_block(
                        cost, None,
                        [self.intrinsics_param_[ic_id], self.disp_param_, self.extrinsics_param_[i], self.tlw_param_],
                    )
                else:
                    cost = Reproj2d3dFactor(self.pixels_[i][j], self.pts3d_[i][j])
                    problem.add_residual_block(
                        cost, None,
                        [self.intrinsics_param_[ic_id], self.extrinsics_param_[i], self.tlw_param_],
                    )

    AddConstraints2d3d = _add_constraints_2d3d
    
    def _obtain_refined_camera_params(self, cameras_out, rays_out):
        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            ic_id = self.shared_ic_ids_[i]
            intr = self.intrinsics_param_[ic_id]
            extr = self.extrinsics_param_[i]
            param = np.zeros(15, dtype=np.float64)
            if self.type_ == FactorType.PTZRayFxfyDist:
                param[0], param[1] = intr[0], intr[1]
            else:
                param[0], param[1] = intr[0], intr[0]
            param[2:4] = intr[2:4]
            param[4:9] = extr[:5]
            displacement = self.disp_param_[0] + self.disp_param_[1] * param[0] + self.disp_param_[2] * param[0] * param[0]
            param[9] = extr[5] + displacement
            param[10:15] = intr[4:9]
            cameras_out[i] = Camera.from_vector(param)

        R_l_w, t_l_w = self.T_l_w(self.tlw_param_)
        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            cameras_out[i].set_t(cameras_out[i].R() @ t_l_w + cameras_out[i].t())
            cameras_out[i].set_R(cameras_out[i].R() @ R_l_w)
        
        if rays_out is not None:
            rays_out.clear()
            rays_out.extend([[] for _ in range(self.num_cams_)])
            R_w_l = R_l_w.T
            t_w_l = -R_w_l @ t_l_w
            for track_id, track in self.tracks_.items():
                if track_id not in self.rays_param_:
                    continue
                ray_l = self.rays_param_[track_id].reshape(3, 1)
                ray_w = R_w_l @ ray_l + t_w_l
                for img_id, feat_id in track.items():
                    if not self._is_candidate(img_id):
                        continue
                    if img_id >= len(self.features_) or feat_id >= len(self.features_[img_id].keypoints):
                        continue
                    uv = self.features_[img_id].keypoints[feat_id].pt
                    rays_out[img_id].append(Ray(track_id, ray_w, uv))

    ObtainRefinedCameraParams = _obtain_refined_camera_params

    def _eval_2d2d_residual(self, track_id, img_id, feat_id):
        uv = self.features_[img_id].keypoints[feat_id].pt
        ic_id = self.shared_ic_ids_[img_id]
        params = [self.intrinsics_param_[ic_id], self.extrinsics_param_[img_id], self.rays_param_[track_id]]
        if self.type_ == FactorType.PTZRay:
            cost = PTZRayFactor(uv)
        elif self.type_ == FactorType.PTZRayDist:
            cost = PTZRayDistFactor(uv)
        elif self.type_ == FactorType.PTZRayFxfyDist:
            cost = PTZRayFxfyDistFactor(uv)
        else:
            cost = PTZRayDistDispFactor(uv)
            params = [self.intrinsics_param_[ic_id], self.disp_param_, self.extrinsics_param_[img_id], self.rays_param_[track_id]]
        residuals = np.zeros(2, dtype=np.float64)
        cost.Evaluate(params, residuals, None)
        return residuals

    def _eval_2d3d_residual(self, img_id, obs_id):
        ic_id = self.shared_ic_ids_[img_id]
        if self.type_ == FactorType.PTZRayDistDisp:
            cost = Reproj2d3dDispFactor(self.pixels_[img_id][obs_id], self.pts3d_[img_id][obs_id])
            params = [self.intrinsics_param_[ic_id], self.disp_param_, self.extrinsics_param_[img_id], self.tlw_param_]
        else:
            cost = Reproj2d3dFactor(self.pixels_[img_id][obs_id], self.pts3d_[img_id][obs_id])
            params = [self.intrinsics_param_[ic_id], self.extrinsics_param_[img_id], self.tlw_param_]
        residuals = np.zeros(2, dtype=np.float64)
        cost.Evaluate(params, residuals, None)
        return residuals

    def _cal_reproj_error(self):
        if self.summary_ is not None and self.summary_.num_residuals > 0:
            self.final_reproj_error_all_ = math.sqrt(2.0) * math.sqrt((2.0 * self.summary_.final_cost) / self.summary_.num_residuals)
        self._cal_reproj_error_2d2d()
        self._cal_reproj_error_2d3d()

    CalReprojError = _cal_reproj_error

    def _cal_reproj_error_2d2d(self):
        total = 0.0
        count = 0
        for track_id, track in self.tracks_.items():
            if track_id not in self.rays_param_:
                continue
            for img_id, feat_id in track.items():
                if not self._is_candidate(img_id):
                    continue
                res = self._eval_2d2d_residual(track_id, img_id, feat_id)
                total += float(res[0] * res[0] + res[1] * res[1])
                count += 1
        self.final_reproj_error_2d2d_ = math.sqrt(total / count) if count else 0.0

    CalReprojError2d2d = _cal_reproj_error_2d2d

    def _cal_reproj_error_2d3d(self):
        total = 0.0
        count = 0
        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            if not self.pixels_ or i >= len(self.pixels_) or not self.pixels_[i]:
                continue
            for j in range(len(self.pixels_[i])):
                res = self._eval_2d3d_residual(i, j)
                total += float(res[0] * res[0] + res[1] * res[1])
                count += 1
        self.final_reproj_error_2d3d_ = math.sqrt(total / count) if count else 0.0

    CalReprojError2d3d = _cal_reproj_error_2d3d

    def final_reproj_error_all(self):
        return self.final_reproj_error_all_

    def final_reproj_error_2d2d(self):
        return self.final_reproj_error_2d2d_

    def final_reproj_error_2d3d(self):
        return self.final_reproj_error_2d3d_
