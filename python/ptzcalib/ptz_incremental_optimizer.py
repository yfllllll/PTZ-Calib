"""Incremental PTZ calibration optimizer.

Python port of src/core/ptz_incremental_optimizer.{h,cc}.
Simplified implementation focusing on core functionality.
"""

from typing import Dict, List, Set, Optional
import numpy as np
import logging

from .types import Camera, ImageFeatures, MatchesInfo
from .ptzray_optimizer import PTZRayOptimizer, FactorType


class PtzIncrementalOptimizer:
    """Incremental bundle adjustment for PTZ camera calibration."""
    
    kMaxNumImages = 1000
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
    
    def solve(self, cameras_out: List[Camera]) -> tuple[bool, Set[int]]:
        """Run incremental optimization.
        
        Returns:
            (success, registered_image_ids)
        """
        if not self._check_valid():
            logging.error("Invalid input for PtzIncrementalOptimizer")
            return False, set()
        
        # Find and register initial image pair
        image_id1, image_id2 = -1, -1
        if not self._find_initial_image_pair(image_id1, image_id2):
            logging.error("Failed to find initial image pair")
            return False, set()
        
        if not self._register_initial_image_pair(image_id1, image_id2):
            logging.error(f"Failed to register initial pair ({image_id1}, {image_id2})")
            return False, set()
        
        logging.info(f"Successfully registered initial pair: {image_id1}, {image_id2}")
        
        # Incrementally register remaining images
        while self._num_reg_images() < len(self.cameras_):
            next_images = self._find_next_images()
            if not next_images:
                logging.info(f"No more images to register. Total: {self._num_reg_images()}/{len(self.cameras_)}")
                break
            
            registered_any = False
            for img_id in next_images:
                if img_id in self.reg_image_ids_:
                    continue
                if self._register_next_image(img_id):
                    logging.info(f"Registered image {img_id} ({self._num_reg_images()}/{len(self.cameras_)})")
                    registered_any = True
                    break
            
            if not registered_any:
                logging.warning("Failed to register any new image in this round")
                break
            
            # Run global BA periodically
            if self._num_reg_images() % max(1, int(self.kBaGlobalImagesRatio)) == 0:
                self._adjust_global_bundle()
        
        # Final global BA
        self._adjust_global_bundle()
        
        # Copy results
        for i in range(len(self.cameras_)):
            cameras_out[i] = self.cameras_[i].clone()
        
        return True, self.reg_image_ids_
    
    def _check_valid(self) -> bool:
        if not self.features_ or not self.cameras_:
            return False
        if len(self.features_) != len(self.cameras_):
            return False
        return True
    
    def _find_initial_image_pair(self, img_id1_out, img_id2_out) -> tuple[int, int]:
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
                    return img_id1, img_id2
        return -1, -1
    
    def _find_first_initial_image(self) -> List[int]:
        """Find candidate first seed images (high feature count, good matches)."""
        scores = []
        for i in range(len(self.features_)):
            num_features = len(self.features_[i].keypoints)
            num_matches = sum(1 for m in self.matches_info_ 
                            if (m.src_img_idx == i or m.dst_img_idx == i) and m.num_inliers > 0)
            scores.append((i, num_features * num_matches))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:10]]
    
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
            
            if m.num_inliers == 0:
                continue
            
            disparity = self._cal_pixel_diff(image_id1, other_id, m.matches)
            score = m.num_inliers * disparity
            scores.append((other_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:5]]
    
    def _find_next_images(self) -> List[int]:
        """Find next images to register (connected to registered images)."""
        candidates = []
        for m in self.matches_info_:
            if m.num_inliers == 0:
                continue
            
            if m.src_img_idx in self.reg_image_ids_ and m.dst_img_idx not in self.reg_image_ids_:
                candidates.append((m.dst_img_idx, m.num_inliers))
            elif m.dst_img_idx in self.reg_image_ids_ and m.src_img_idx not in self.reg_image_ids_:
                candidates.append((m.src_img_idx, m.num_inliers))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:10]]
    
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
        pair_id = self._image_pair_to_pair_id(image_id1, image_id2)
        self.init_image_pairs_.add(pair_id)
        
        # Set initial parameters (simplified: assume reasonable defaults)
        self._set_initial_image_pair_parameters(image_id1, image_id2)
        
        # Run PTZRay optimizer on this pair
        cam_ids = {image_id1, image_id2}
        optimizer = PTZRayOptimizer(
            self.features_, self.matches_info_, self.cameras_,
            cam_ids, self.max_iter_, FactorType.PTZRayDist
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
        
        if self.num_reg_trials_[image_id] > 3:
            logging.warning(f"Image {image_id} failed registration {self.num_reg_trials_[image_id]} times, skipping")
            return False
        
        # Run PTZRay optimizer including this new image
        cam_ids = self.reg_image_ids_ | {image_id}
        optimizer = PTZRayOptimizer(
            self.features_, self.matches_info_, self.cameras_,
            cam_ids, self.max_iter_, FactorType.PTZRayDist
        )
        
        cameras_temp = [cam.clone() for cam in self.cameras_]
        success = optimizer.solve(cameras_temp)
        
        if success:
            for cid in cam_ids:
                self.cameras_[cid] = cameras_temp[cid].clone()
            self.reg_image_ids_.add(image_id)
            return True
        
        return False
    
    def _set_initial_image_pair_parameters(self, image_id1: int, image_id2: int):
        """Set initial camera parameters for the first pair (simplified)."""
        # In a full implementation, this would use essential matrix decomposition
        # Here we assume cameras are pre-initialized with reasonable values
        pass
    
    def _adjust_global_bundle(self) -> bool:
        """Run global bundle adjustment on all registered images."""
        if len(self.reg_image_ids_) < 2:
            return True
        
        optimizer = PTZRayOptimizer(
            self.features_, self.matches_info_, self.cameras_,
            self.reg_image_ids_, self.max_iter_, FactorType.PTZRayDist
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
