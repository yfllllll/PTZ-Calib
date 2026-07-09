"""Incremental PTZ calibration optimizer.

Python port of src/core/ptz_incremental_optimizer.{h,cc}.
Simplified implementation focusing on core functionality.
"""

from typing import Dict, List, Set, Optional
import numpy as np
import logging

from .types import Camera, ImageFeatures, MatchesInfo
from .krt_optimizer import KRTOptimizer, FactorType as KRTFactorType
from .ptzray_optimizer import PTZRayOptimizer, FactorType as PTZFactorType


class PtzIncrementalOptimizer:
    """Incremental bundle adjustment for PTZ camera calibration."""
    
    kMaxNumImages = 100000
    kBaGlobalImagesRatio = 1.1
    
    def __init__(self, features: List[ImageFeatures], matches_info: List[MatchesInfo],
                 cameras: List[Camera], max_iter: int, names: Optional[List[str]] = None):
        self.features_ = features
        self.matches_info_ = matches_info
        self.cameras_ = [cam.clone() for cam in cameras]
        self.max_iter_ = max_iter
        self.names_ = names if names else [f"img_{i}" for i in range(len(cameras))]
        
        self.init_image_pairs_: Set[int] = set()
        self.num_reg_trials_: Dict[int, int] = {}
        self.reg_image_ids_: Set[int] = set()
        self.seed_image_ids_: List[int] = []
    
    def set_seed_image_id(self, image_ids: List[int]):
        """Manually set seed image IDs for initialization."""
        self.seed_image_ids_ = list(image_ids)

    SetSeedImageId = set_seed_image_id
    
    def solve(self, cameras_out: List[Camera]) -> tuple[bool, Set[int]]:
        """Run incremental optimization.
        
        Returns:
            (success, registered_image_ids)
        """
        if not self._check_valid():
            logging.error("Invalid input for PtzIncrementalOptimizer")
            return False, set()
        
        for _ in range(50):
            found, image_id1, image_id2 = self._find_initial_image_pair()
            if not found:
                logging.info("No good initial image pair found")
                return False, set()

            logging.info("Initializing with image pair #%d and #%d", image_id1, image_id2)
            if not self._register_initial_image_pair(image_id1, image_id2):
                logging.info("Initialization failed, trying another initial pair")
                continue

            self._adjust_global_bundle()
            ba_prev_num_reg_images = self._num_reg_images()
            reg_next_success = True

            while reg_next_success:
                reg_next_success = False
                next_image_ids = self._find_next_images()
                if not next_image_ids:
                    break

                for reg_trial, image_id in enumerate(next_image_ids):
                    reg_next_success = self._register_next_image(image_id)
                    logging.info(
                        "Register image #%d %s. Currently registered: %d, total: %d",
                        image_id,
                        "success" if reg_next_success else "failed",
                        self._num_reg_images(),
                        len(self.features_),
                    )

                    if reg_next_success:
                        if self._num_reg_images() >= self.kBaGlobalImagesRatio * ba_prev_num_reg_images:
                            if self._adjust_global_bundle():
                                ba_prev_num_reg_images = self._num_reg_images()
                                break
                            self.reg_image_ids_.discard(image_id)
                            reg_next_success = False

                    if not reg_next_success:
                        k_min_num_initial_reg_trials = 30
                        k_min_model_size = 3
                        if reg_trial >= k_min_num_initial_reg_trials and self._num_reg_images() < k_min_model_size:
                            break

            self._adjust_global_bundle()
            for i in range(len(self.cameras_)):
                cameras_out[i] = self.cameras_[i].clone()
            return True, set(self.reg_image_ids_)

        return False, set()

    Solve = solve
    
    def _check_valid(self) -> bool:
        if not self.features_ or not self.cameras_:
            return False
        if len(self.features_) != len(self.cameras_):
            return False
        return True
    
    def _find_initial_image_pair(self) -> tuple[bool, int, int]:
        """Find the best initial image pair."""
        if self.seed_image_ids_:
            first_candidates = self.seed_image_ids_
        else:
            first_candidates = self._find_first_initial_image()
        
        for img_id1 in first_candidates:
            second_candidates = self._find_second_initial_image(img_id1)
            for img_id2 in second_candidates:
                pair_id = self._image_pair_to_pair_id(img_id1, img_id2)
                if pair_id not in self.init_image_pairs_:
                    self.init_image_pairs_.add(pair_id)
                    return True, img_id1, img_id2
        return False, -1, -1
    
    def _find_first_initial_image(self) -> List[int]:
        """Find candidate first seed images (high feature count, good matches)."""
        confidences = [0.0] * len(self.features_)
        for m in self.matches_info_:
            if m.src_img_idx >= 0 and m.dst_img_idx >= 0:
                confidences[m.src_img_idx] += m.confidence
                confidences[m.dst_img_idx] += m.confidence
        indices = list(range(len(self.features_)))
        indices.sort(key=lambda idx: confidences[idx], reverse=True)
        return [idx for idx in indices if confidences[idx] > 0.0]
    
    def _find_second_initial_image(self, image_id1: int) -> List[int]:
        """Find candidate second images for pairing with image_id1."""
        scores = []
        for m in self.matches_info_:
            if m.src_img_idx == image_id1:
                other_id = m.dst_img_idx
            elif m.dst_img_idx == image_id1:
                other_id = m.src_img_idx
            else:
                continue
            
            if not m.matches:
                continue
            if image_id1 == m.src_img_idx and image_id1 == m.dst_img_idx:
                continue
            if self._cal_pixel_diff(m.src_img_idx, m.dst_img_idx, m.matches) < 50:
                continue
            
            scores.append((other_id, m.confidence))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores if s[1] > 0.0]
    
    def _find_next_images(self) -> List[int]:
        """Find next images to register (connected to registered images)."""
        confidences = [0.0] * len(self.features_)
        for m in self.matches_info_:
            src_idx = m.src_img_idx
            dst_idx = m.dst_img_idx
            if src_idx == dst_idx:
                continue
            if m.H is None:
                continue
            if self.num_reg_trials_.get(src_idx, 0) > 4 or self.num_reg_trials_.get(dst_idx, 0) > 4:
                continue
            src_reg = src_idx in self.reg_image_ids_
            dst_reg = dst_idx in self.reg_image_ids_
            if src_reg and dst_reg:
                continue
            if not src_reg and not dst_reg:
                continue
            if src_reg and not dst_reg:
                confidences[dst_idx] += m.confidence
            elif not src_reg and dst_reg:
                confidences[src_idx] += m.confidence

        indices = list(range(len(self.features_)))
        indices.sort(key=lambda idx: confidences[idx], reverse=True)
        return [idx for idx in indices if confidences[idx] > 0.0]
    
    def _cal_pixel_diff(self, image_id1: int, image_id2: int, matches: List) -> float:
        """Calculate average pixel disparity between matched points."""
        if not matches:
            return 0.0
        
        total_diff = 0.0
        count = 0
        feat1 = self.features_[image_id1]
        feat2 = self.features_[image_id2]
        
        for dm in matches:
            if dm.queryIdx >= len(feat1.keypoints) or dm.trainIdx >= len(feat2.keypoints):
                continue
            pt1 = feat1.keypoints[dm.queryIdx].pt
            pt2 = feat2.keypoints[dm.trainIdx].pt
            diff = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            total_diff += diff
            count += 1
        
        return total_diff / count if count > 0 else 0.0
    
    def _register_initial_image_pair(self, image_id1: int, image_id2: int) -> bool:
        """Register the initial image pair."""
        self.num_reg_trials_[image_id1] = self.num_reg_trials_.get(image_id1, 0) + 1
        self.num_reg_trials_[image_id2] = self.num_reg_trials_.get(image_id2, 0) + 1
        
        self._set_initial_image_pair_parameters(image_id1, image_id2)
        
        cam_ids = {image_id1, image_id2}
        optimizer = PTZRayOptimizer(
            self.features_, self.matches_info_, self.cameras_,
            cam_ids, self.max_iter_, PTZFactorType.PTZRay
        )
        
        cameras_temp = [cam.clone() for cam in self.cameras_]
        success = optimizer.solve(cameras_temp)
        
        if success:
            self.cameras_[image_id1] = cameras_temp[image_id1].clone()
            self.cameras_[image_id2] = cameras_temp[image_id2].clone()
            self.reg_image_ids_.add(image_id1)
            self.reg_image_ids_.add(image_id2)
            return True
        
        return False
    
    def _register_next_image(self, image_id: int) -> bool:
        """Register a single new image."""
        self.num_reg_trials_[image_id] = self.num_reg_trials_.get(image_id, 0) + 1

        for match_info in self.matches_info_:
            i = match_info.src_img_idx
            j = match_info.dst_img_idx
            H_j_i = match_info.H
            if H_j_i is None:
                continue

            if i in self.reg_image_ids_ and j == image_id:
                self.cameras_[j].set_K(self.cameras_[i].K().copy())
                R_j_i = np.linalg.inv(self.cameras_[j].K()) @ H_j_i @ self.cameras_[i].K()
                self.cameras_[j].set_R(R_j_i @ self.cameras_[i].R())

                optimizer = KRTOptimizer(100, 100.0, KRTFactorType.F)
                optimizer.set_init_params(
                    self.cameras_[j].K(),
                    self.cameras_[j].R(),
                    self.cameras_[j].t(),
                    self.cameras_[j].dist(),
                )
                optimizer.add_2d2d_constraints(
                    self.cameras_[i],
                    self.features_[i].keypoints,
                    self.features_[j].keypoints,
                    match_info.matches,
                )
                success, K, R, _t, _dist = optimizer.solve()
                if success:
                    self.cameras_[j].set_K(K)
                    self.cameras_[j].set_R(R)
                    self.reg_image_ids_.add(j)
                    return True

        return False
    
    def _set_initial_image_pair_parameters(self, image_id1: int, image_id2: int):
        ratio = 1.2
        for image_id in (image_id1, image_id2):
            w, h = self.features_[image_id].img_size
            focal = ratio * max(w, h)
            K = self.cameras_[image_id].K().copy()
            K[0, 0] = focal
            K[1, 1] = focal
            K[0, 2] = 0.5 * w
            K[1, 2] = 0.5 * h
            self.cameras_[image_id].set_K(K)
        self.cameras_[image_id1].set_R(np.eye(3, dtype=np.float64))

        for match_info in self.matches_info_:
            i = match_info.src_img_idx
            j = match_info.dst_img_idx
            if i == image_id1 and j == image_id2 and match_info.H is not None:
                R_j_i = np.linalg.inv(self.cameras_[j].K()) @ match_info.H @ self.cameras_[i].K()
                self.cameras_[j].set_R(R_j_i @ self.cameras_[i].R())
                break
    
    def _adjust_global_bundle(self) -> bool:
        """Run global bundle adjustment on all registered images."""
        if len(self.reg_image_ids_) < 2:
            return True
        
        optimizer = PTZRayOptimizer(
            self.features_, self.matches_info_, self.cameras_,
            self.reg_image_ids_, self.max_iter_, PTZFactorType.PTZRay
        )
        
        cameras_temp = [cam.clone() for cam in self.cameras_]
        success = optimizer.solve(cameras_temp)
        
        if success:
            for cid in self.reg_image_ids_:
                self.cameras_[cid] = cameras_temp[cid].clone()
            logging.info(f"Global BA success with {len(self.reg_image_ids_)} images")
            return True
        
        logging.warning("Global BA failed")
        return False
    
    def _image_pair_to_pair_id(self, image_id1: int, image_id2: int) -> int:
        """Convert image pair to unique ID."""
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
        return image_id1 * self.kMaxNumImages + image_id2
    
    def _num_reg_images(self) -> int:
        return len(self.reg_image_ids_)
