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
        self.features_ = features
        self.matches_info_ = matches_info
        self.cameras_ = [cam.clone() for cam in cameras]
        self.num_cams_ = len(cameras)
        self.cam_ids_ = set(cam_ids)
        self.max_iter_ = max_iter
        self.type_ = factor_type
        self.pixels_ = pixels if pixels else []
        self.pts3d_ = pts3d if pts3d else []
        
        self.shared_ic_ids_ = list(range(len(cameras)))
        self.intrinsics_param_ = {}
        self.extrinsics_param_ = {}
        self.rays_param_ = {}
        self.disp_param_ = [0.0, 0.0, 0.0]
        self.tlw_param_ = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.tracks_ = {}
        
        self._find_tracks()
        self._set_up_initial_camera_params()
        self._set_init_trans_local_to_world()
    
    def _is_candidate(self, image_id):
        return image_id in self.cam_ids_
    
    def _find_tracks(self):
        builder = TracksBuilder()
        for m in self.matches_info_:
            if m.src_img_idx >= len(self.features_) or m.dst_img_idx >= len(self.features_):
                continue
            src_feat = self.features_[m.src_img_idx]
            dst_feat = self.features_[m.dst_img_idx]
            for dm in m.matches:
                if dm.queryIdx < len(src_feat.keypoints) and dm.trainIdx < len(dst_feat.keypoints):
                    if m.inliers_mask and len(m.inliers_mask) > len(builder._nodes):
                        if m.inliers_mask[dm.queryIdx]:
                            builder.insert(m.src_img_idx, dm.queryIdx, m.dst_img_idx, dm.trainIdx)
                    else:
                        builder.insert(m.src_img_idx, dm.queryIdx, m.dst_img_idx, dm.trainIdx)
        builder.filter(2)
        self.tracks_ = builder.export_to_STL()
    
    def _set_up_initial_camera_params(self):
        for i in range(self.num_cams_):
            ic_id = self.shared_ic_ids_[i]
            cam = self.cameras_[i]
            if ic_id not in self.intrinsics_param_:
                self.intrinsics_param_[ic_id] = np.array([
                    cam.K_[0,0], cam.K_[1,1], cam.K_[0,2], cam.K_[1,2],
                    cam.dist_[0,0], cam.dist_[1,0], cam.dist_[2,0], cam.dist_[3,0], cam.dist_[4,0]
                ], dtype=np.float64)
            rvec, _ = cv2.Rodrigues(cam.R_)
            self.extrinsics_param_[i] = np.array([rvec[0,0], rvec[1,0], rvec[2,0], 0, 0, 0], dtype=np.float64)
        
        for track_id, track in self.tracks_.items():
            ray = self._pix2ray(track)
            if ray is not None and ray.shape[0] == 3:
                self.rays_param_[track_id] = ray.reshape(-1).copy()
    
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
        if count > 0:
            ray_sum = ray_sum / count
            ray_sum = ray_sum / np.linalg.norm(ray_sum)
            return ray_sum
        return None
    
    def _set_init_trans_local_to_world(self):
        self.tlw_param_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    def solve(self, cameras_out, rays_out=None):
        problem = pyceres.Problem()
        self._add_constraints_2d2d(problem)
        if self.pixels_ and any(len(p) > 0 for p in self.pixels_):
            self._add_constraints_2d3d(problem)
        
        options = pyceres.SolverOptions()
        options.max_num_iterations = self.max_iter_
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.minimizer_progress_to_stdout = False
        summary = pyceres.Summary()
        pyceres.Solve(options, problem, summary)
        
        self._obtain_refined_camera_params(cameras_out, rays_out)
        return True
    
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
                
                if self.type_ == FactorType.PTZRay:
                    cost = PTZRayFactor(uv)
                    problem.AddResidualBlock(cost, None, 
                        self.intrinsics_param_[ic_id], self.extrinsics_param_[img_id], self.rays_param_[track_id])
                elif self.type_ == FactorType.PTZRayDist:
                    cost = PTZRayDistFactor(uv)
                    problem.AddResidualBlock(cost, None,
                        self.intrinsics_param_[ic_id], self.extrinsics_param_[img_id], self.rays_param_[track_id])
                elif self.type_ == FactorType.PTZRayFxfyDist:
                    cost = PTZRayFxfyDistFactor(uv)
                    problem.AddResidualBlock(cost, None,
                        self.intrinsics_param_[ic_id], self.extrinsics_param_[img_id], self.rays_param_[track_id])
                elif self.type_ == FactorType.PTZRayDistDisp:
                    cost = PTZRayDistDispFactor(uv)
                    problem.AddResidualBlock(cost, None,
                        self.intrinsics_param_[ic_id], self.disp_param_, self.extrinsics_param_[img_id], self.rays_param_[track_id])
    
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
                    problem.AddResidualBlock(cost, None,
                        self.intrinsics_param_[ic_id], self.disp_param_, self.extrinsics_param_[i], self.tlw_param_)
                else:
                    cost = Reproj2d3dFactor(self.pixels_[i][j], self.pts3d_[i][j])
                    problem.AddResidualBlock(cost, None,
                        self.intrinsics_param_[ic_id], self.extrinsics_param_[i], self.tlw_param_)
    
    def _obtain_refined_camera_params(self, cameras_out, rays_out):
        for i in range(self.num_cams_):
            ic_id = self.shared_ic_ids_[i]
            intr = self.intrinsics_param_[ic_id]
            extr = self.extrinsics_param_[i]
            cameras_out[i].K_ = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]], dtype=np.float64)
            cameras_out[i].R_, _ = cv2.Rodrigues(extr[:3].reshape(3, 1))
            cameras_out[i].t_ = extr[3:6].reshape(3, 1)
            cameras_out[i].dist_ = intr[4:9].reshape(5, 1)
        
        if rays_out is not None:
            rays_out.clear()
            rays_out.extend([[] for _ in range(self.num_cams_)])
            R_l_w, _ = cv2.Rodrigues(self.tlw_param_[:3].reshape(3, 1))
            t_l_w = self.tlw_param_[3:6].reshape(3, 1)
            R_w_l = R_l_w.T
            t_w_l = -R_w_l @ t_l_w
            for track_id, track in self.tracks_.items():
                if track_id not in self.rays_param_:
                    continue
                ray_l = self.rays_param_[track_id].reshape(3, 1)
                ray_w = R_w_l @ ray_l + t_w_l
                for img_id, feat_id in track.items():
                    if img_id >= len(self.features_) or feat_id >= len(self.features_[img_id].keypoints):
                        continue
                    uv = self.features_[img_id].keypoints[feat_id].pt
                    rays_out[img_id].append(Ray(track_id, ray_w, uv))

