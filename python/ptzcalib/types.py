"""Basic types: Camera, ImageFeatures, MatchesInfo, Ray.

Python port of src/core/types.h and src/core/types.cc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ImageFeatures:
    """Per-image feature detection result.

    Attributes:
        img_idx: Global image index.
        img_size: (width, height) tuple.
        keypoints: List of cv2.KeyPoint.
        descriptors: (N, D) float32 numpy array or None.
    """

    img_idx: int = -1
    img_size: Tuple[int, int] = (0, 0)
    keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None


@dataclass
class MatchesInfo:
    """Pairwise match info between two images.

    Attributes:
        src_img_idx / dst_img_idx: image indices.
        matches: list of cv2.DMatch.
        inliers_mask: same-length list of 0/1 flags.
        num_inliers: count of geometric inliers.
        H: 3x3 homography (dst = H * src) or None.
        confidence: matching confidence in [0, 1].
    """

    src_img_idx: int = -1
    dst_img_idx: int = -1
    matches: List[cv2.DMatch] = field(default_factory=list)
    inliers_mask: List[int] = field(default_factory=list)
    num_inliers: int = 0
    H: Optional[np.ndarray] = None
    confidence: float = 0.0


@dataclass
class Ray:
    """A 3D ray attached to a 2D pixel observation.

    Attributes:
        id_: track id.
        pt3d_: (3,) numpy array ray direction (unit length ideally).
        uv_: (x, y) pixel coordinate as np.ndarray of shape (2,).
    """

    id_: int
    pt3d_: np.ndarray
    uv_: np.ndarray

    def __init__(self, id_: int, pt3d: np.ndarray, uv: Tuple[float, float]):
        self.id_ = id_
        # Accept (3,) or (1,3) or (3,1) shapes and flatten to (3,)
        self.pt3d_ = np.asarray(pt3d, dtype=np.float64).reshape(-1)[:3].copy()
        self.uv_ = np.asarray(uv, dtype=np.float64).reshape(-1)[:2].copy()


class Camera:
    """Pinhole camera model with distortion.

    Members (all numpy arrays, dtype=float64):
        K_ : 3x3 intrinsic matrix
        R_ : 3x3 rotation matrix (world to camera)
        t_ : 3x1 translation vector (world to camera)
        dist_ : 5x1 distortion coefficients [k1, k2, p1, p2, k3]
              (OpenCV convention). Note: in the C++ code the ToVector/FromVector
              orders distortion as [k1, k2, k3, p1, p2] to match Ceres residuals.
              We keep OpenCV order in dist_ but preserve the C++ 15-vector layout
              via to_vector()/from_vector() below.
    """

    __slots__ = ("K_", "R_", "t_", "dist_")

    def __init__(
        self,
        K: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        dist: Optional[np.ndarray] = None,
    ):
        if K is None:
            self.K_ = np.eye(3, dtype=np.float64)
        else:
            self.K_ = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()

        if R is None:
            self.R_ = np.eye(3, dtype=np.float64)
        else:
            self.R_ = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()

        if t is None:
            self.t_ = np.zeros((3, 1), dtype=np.float64)
        else:
            self.t_ = np.asarray(t, dtype=np.float64).reshape(3, 1).copy()

        if dist is None:
            self.dist_ = np.zeros((5, 1), dtype=np.float64)
        else:
            self.dist_ = np.asarray(dist, dtype=np.float64).reshape(5, 1).copy()

    # ---- accessors that match the C++ API loosely ----
    def K(self) -> np.ndarray:
        return self.K_

    def set_K(self, K: np.ndarray) -> None:
        self.K_ = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()

    def R(self) -> np.ndarray:
        return self.R_

    def set_R(self, R: np.ndarray) -> None:
        self.R_ = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()

    def t(self) -> np.ndarray:
        return self.t_

    def set_t(self, t: np.ndarray) -> None:
        self.t_ = np.asarray(t, dtype=np.float64).reshape(3, 1).copy()

    def dist(self) -> np.ndarray:
        return self.dist_

    def set_dist(self, dist: np.ndarray) -> None:
        self.dist_ = np.asarray(dist, dtype=np.float64).reshape(5, 1).copy()

    def rvec(self) -> np.ndarray:
        """Rotation vector (Rodrigues) as (3,1) array."""
        rvec, _ = cv2.Rodrigues(self.R_)
        return rvec

    def t_wc(self) -> np.ndarray:
        """Translation of camera center in world coordinates: -R^{-1} t.

        Returns (3, 1) array.
        """
        return -np.linalg.inv(self.R_) @ self.t_

    def clone(self) -> "Camera":
        return Camera(self.K_.copy(), self.R_.copy(), self.t_.copy(), self.dist_.copy())

    # ---- (de)serialization ----
    def to_vector(self) -> np.ndarray:
        """Serialize to 15-vector as in the C++ Camera::ToVector.

        Layout: [fx, fy, cx, cy, rvec(3), t(3), k1, k2, k3, p1, p2]
        Note the (k1, k2, k3, p1, p2) order, matching the C++ Ceres factor
        layout — distinct from OpenCV's [k1, k2, p1, p2, k3].
        """
        v = np.zeros(15, dtype=np.float64)
        v[0] = self.K_[0, 0]  # fx
        v[1] = self.K_[1, 1]  # fy
        v[2] = self.K_[0, 2]  # cx
        v[3] = self.K_[1, 2]  # cy

        rvec, _ = cv2.Rodrigues(self.R_)
        v[4:7] = rvec.reshape(-1)
        v[7:10] = self.t_.reshape(-1)

        # C++ ToVector packs dist_ elements in their storage order,
        # which corresponds to [k1, k2, k3, p1, p2] as documented in
        # PTZRayDistFactor comments.
        v[10:15] = self.dist_.reshape(-1)
        return v

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "Camera":
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.size != 15:
            raise ValueError(f"Expected camera vector size 15, got {v.size}")

        cam = cls()
        cam.K_ = np.array(
            [
                [v[0], 0.0, v[2]],
                [0.0, v[1], v[3]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rvec = v[4:7].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        cam.R_ = R.astype(np.float64)
        cam.t_ = v[7:10].reshape(3, 1).astype(np.float64)
        cam.dist_ = v[10:15].reshape(5, 1).astype(np.float64)
        return cam


def pt3d_to_pix(camera: Camera, pt3d: np.ndarray) -> Optional[np.ndarray]:
    """Project a 3D point (world) to pixel homogeneous coords.

    Returns a (3, 1) array [u, v, 1]^T scaled so z==1, or None if the point
    is behind the near plane.
    """
    pt3d = np.asarray(pt3d, dtype=np.float64).reshape(3, 1)
    pt_cam = camera.R_ @ pt3d + camera.t_
    near_plane = 1.0
    if pt_cam[2, 0] < near_plane:
        return None
    pt_cam = pt_cam / pt_cam[2, 0]
    uv = camera.K_ @ pt_cam
    return uv
