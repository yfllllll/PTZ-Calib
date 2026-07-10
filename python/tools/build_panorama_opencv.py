#!/usr/bin/env python3
"""Build panoramas with OpenCV's detailed stitching pipeline.

This follows the structure of OpenCV's stitching_detailed.py and the user's
stitching reference repository:

  features -> pairwise matches -> largest component -> camera estimation ->
  bundle adjustment -> wave correction -> warping -> exposure compensation ->
  seam finding -> multi-band blending.

Unlike build_panorama_vismatch.py, this tool does not compose a chain of
pairwise homographies. The geometry is optimized as camera rotations and
intrinsics in OpenCV's stitching module.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

from spatial_gcp_utils import list_images, natural_key

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prepare_data_vismatch import (  # noqa: E402
    _loaded_size,
    _scale_keypoints,
    extract_pair_data,
    load_image_for_matcher,
    load_matcher,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


FEATURES = {
    "orb": cv.ORB.create,
}
if hasattr(cv, "SIFT_create"):
    FEATURES["sift"] = cv.SIFT_create

ESTIMATORS = {
    "homography": cv.detail_HomographyBasedEstimator,
    "affine": cv.detail_AffineBasedEstimator,
}
BA_COSTS = {
    "ray": cv.detail_BundleAdjusterRay,
    "reproj": cv.detail_BundleAdjusterReproj,
    "affine": cv.detail_BundleAdjusterAffinePartial,
    "none": cv.detail_NoBundleAdjuster,
}
WAVE_CORRECT = {
    "horiz": cv.detail.WAVE_CORRECT_HORIZ,
    "vert": cv.detail.WAVE_CORRECT_VERT,
    "no": None,
}
SEAM_FINDERS = {
    "no": cv.detail.SeamFinder_NO,
    "voronoi": cv.detail.SeamFinder_VORONOI_SEAM,
    "dp_color": cv.detail.SeamFinder_DP_SEAM,
}
EXPOSURE = {
    "no": cv.detail.ExposureCompensator_NO,
    "gain": cv.detail.ExposureCompensator_GAIN,
    "gain_blocks": cv.detail.ExposureCompensator_GAIN_BLOCKS,
    "channels": cv.detail.ExposureCompensator_CHANNELS,
    "channels_blocks": cv.detail.ExposureCompensator_CHANNELS_BLOCKS,
}
BLENDERS = {
    "no": cv.detail.Blender_NO,
    "feather": cv.detail.Blender_FEATHER,
    "multiband": cv.detail.Blender_MULTI_BAND,
}


def _scale_from_megapix(img: np.ndarray, megapix: float) -> float:
    if megapix < 0:
        return 1.0
    return min(1.0, float(np.sqrt(megapix * 1e6 / (img.shape[0] * img.shape[1]))))


def _make_feature_mask(shape: Tuple[int, int], exclude_top: float, exclude_bottom: float) -> np.ndarray:
    h, w = shape
    top = int(round(h * exclude_top))
    bottom = int(round(h * (1.0 - exclude_bottom)))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[max(top, 0) : min(bottom, h), :] = 255
    return mask


def _read_image_names(images_dir: str, max_images: int) -> List[str]:
    names = sorted([p.name for p in list_images(images_dir)], key=natural_key)
    if max_images > 0:
        names = names[:max_images]
    if len(names) < 2:
        raise ValueError("At least two images are required")
    return names


def _create_matcher(args, image_names: Optional[Sequence[str]] = None, full_sizes: Optional[Sequence[Tuple[int, int]]] = None):
    if args.matcher == "vismatch":
        return VismatchFeatureMatcher(args, list(image_names or []), list(full_sizes or []))
    match_conf = args.match_conf
    if match_conf < 0:
        match_conf = 0.3 if args.features == "orb" else 0.65
    if args.matcher == "affine":
        return cv.detail_AffineBestOf2NearestMatcher(False, args.try_cuda, match_conf)
    return cv.detail.BestOf2NearestMatcher_create(args.try_cuda, match_conf)


def _empty_match_info(i: int, j: int) -> cv.detail_MatchesInfo:
    match_info = cv.detail.MatchesInfo()
    match_info.src_img_idx = int(i)
    match_info.dst_img_idx = int(j)
    match_info.confidence = 0.0
    match_info.num_inliers = 0
    match_info.inliers_mask = []
    return match_info


def _matched_points_from_vismatch_result(result: dict, size0: Tuple[int, int], size1: Tuple[int, int], load_size0: Tuple[int, int], load_size1: Tuple[int, int]):
    all_kpts0, _desc0, all_kpts1, _desc1, matches, matched0, matched1 = extract_pair_data(result)
    all_kpts0 = _scale_keypoints(all_kpts0, load_size0, size0)
    all_kpts1 = _scale_keypoints(all_kpts1, load_size1, size1)
    matched0 = _scale_keypoints(matched0, load_size0, size0)
    matched1 = _scale_keypoints(matched1, load_size1, size1)
    if matches is not None and all_kpts0 is not None and all_kpts1 is not None:
        valid = (
            (matches[:, 0] >= 0)
            & (matches[:, 0] < len(all_kpts0))
            & (matches[:, 1] >= 0)
            & (matches[:, 1] < len(all_kpts1))
        )
        matches = matches[valid]
        return all_kpts0[matches[:, 0]], all_kpts1[matches[:, 1]]
    if matched0 is not None and matched1 is not None:
        n = min(len(matched0), len(matched1))
        return matched0[:n], matched1[:n]
    return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)


def _feature_scale(feature, full_size: Tuple[int, int]) -> Tuple[float, float]:
    feat_w, feat_h = feature.img_size
    full_w, full_h = full_size
    sx = float(feat_w) / float(max(full_w, 1))
    sy = float(feat_h) / float(max(full_h, 1))
    return sx, sy


def _scale_homography(H: np.ndarray, src_scale: Tuple[float, float], dst_scale: Tuple[float, float]) -> np.ndarray:
    S_src = np.array([[src_scale[0], 0.0, 0.0], [0.0, src_scale[1], 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    S_dst = np.array([[dst_scale[0], 0.0, 0.0], [0.0, dst_scale[1], 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return S_dst @ H @ np.linalg.inv(S_src)


def _nearest_feature_matches(projected: np.ndarray, target: np.ndarray, threshold: float):
    if len(projected) == 0 or len(target) == 0:
        return {}
    matcher = cv.BFMatcher(cv.NORM_L2)
    raw = matcher.match(projected.astype(np.float32), target.astype(np.float32))
    return {int(m.queryIdx): (int(m.trainIdx), float(m.distance)) for m in raw if m.distance <= threshold}


class VismatchFeatureMatcher:
    """Adapt arbitrary vismatch pair matching into OpenCV detail.MatchesInfo.

    The OpenCV stitching estimator still consumes OpenCV ImageFeatures. Vismatch
    is used to estimate a robust pairwise homography, then OpenCV feature
    keypoints are paired by symmetric nearest reprojection under that homography,
    mirroring the RoMAFeatureMatcher pattern in the user's stitching repository.
    """

    def __init__(self, args, image_names: Sequence[str], full_sizes: Sequence[Tuple[int, int]]):
        self.args = args
        self.image_names = list(image_names)
        self.full_sizes = list(full_sizes)
        self.matcher = load_matcher(args.vismatch_matcher, args.device)
        self.pair_reports = []

    def collectGarbage(self):
        return None

    def subset(self, indices: Sequence[int]):
        self.image_names = [self.image_names[int(i)] for i in indices]
        self.full_sizes = [self.full_sizes[int(i)] for i in indices]

    def apply2(self, features):
        num_images = len(features)
        pairwise_matches = []
        self.pair_reports = []
        forward_infos = {}

        for i in range(num_images):
            for j in range(num_images):
                if i == j:
                    match_info = _empty_match_info(i, j)
                elif i < j and (self.args.rangewidth < 0 or abs(i - j) <= self.args.rangewidth):
                    match_info = self._match_forward(i, j, features)
                    forward_infos[(i, j)] = match_info
                elif i > j and (j, i) in forward_infos:
                    match_info = self._reverse_match_info(forward_infos[(j, i)], i, j)
                else:
                    match_info = _empty_match_info(i, j)
                pairwise_matches.append(match_info)
        return pairwise_matches

    def _match_forward(self, i: int, j: int, features):
        path0 = Path(self.args.images) / self.image_names[i]
        path1 = Path(self.args.images) / self.image_names[j]
        size0 = self.full_sizes[i]
        size1 = self.full_sizes[j]
        report = OrderedDict([("image0", self.image_names[i]), ("image1", self.image_names[j])])
        logger.info("Vismatch %s -> %s", self.image_names[i], self.image_names[j])

        try:
            img0 = load_image_for_matcher(self.matcher, str(path0), self.args.resize)
            img1 = load_image_for_matcher(self.matcher, str(path1), self.args.resize)
            load_size0 = _loaded_size(img0, size0)
            load_size1 = _loaded_size(img1, size1)
            result = self.matcher(img0, img1)
            pts0, pts1 = _matched_points_from_vismatch_result(result, size0, size1, load_size0, load_size1)
            report["raw_matches"] = int(len(pts0))
            if len(pts0) < self.args.vismatch_min_matches:
                report["status"] = "not_enough_raw_matches"
                self.pair_reports.append(report)
                return _empty_match_info(i, j)

            H_orig, mask = cv.findHomography(pts0.astype(np.float64), pts1.astype(np.float64), cv.RANSAC, self.args.vismatch_ransac_thresh)
            if H_orig is None or mask is None:
                report["status"] = "homography_failed"
                self.pair_reports.append(report)
                return _empty_match_info(i, j)
            inlier_mask = mask.reshape(-1).astype(bool)
            report["ransac_inliers"] = int(inlier_mask.sum())
            if report["ransac_inliers"] < self.args.vismatch_min_inliers:
                report["status"] = "not_enough_ransac_inliers"
                self.pair_reports.append(report)
                return _empty_match_info(i, j)

            projected = cv.perspectiveTransform(pts0[inlier_mask].reshape(-1, 1, 2).astype(np.float64), H_orig).reshape(-1, 2)
            rmse = float(np.sqrt(np.mean(np.sum((projected - pts1[inlier_mask]) ** 2, axis=1))))
            report["ransac_rmse"] = rmse
            if self.args.vismatch_max_rmse > 0 and rmse > self.args.vismatch_max_rmse:
                report["status"] = "rmse_too_high"
                self.pair_reports.append(report)
                return _empty_match_info(i, j)

            H = _scale_homography(H_orig, _feature_scale(features[i], size0), _feature_scale(features[j], size1))
            match_info = self._matches_info_from_homography(i, j, features, H, report)
            self.pair_reports.append(report)
            return match_info
        except Exception as exc:
            report["status"] = f"error:{type(exc).__name__}"
            report["message"] = str(exc)
            self.pair_reports.append(report)
            logger.warning("Vismatch pair failed %s -> %s: %s", self.image_names[i], self.image_names[j], exc)
            return _empty_match_info(i, j)

    def _matches_info_from_homography(self, i: int, j: int, features, H: np.ndarray, report: OrderedDict):
        kpts_i = features[i].getKeypoints()
        kpts_j = features[j].getKeypoints()
        if not kpts_i or not kpts_j:
            report["status"] = "missing_opencv_features"
            return _empty_match_info(i, j)
        pts_i = np.float32([kp.pt for kp in kpts_i])
        pts_j = np.float32([kp.pt for kp in kpts_j])
        proj_j = cv.perspectiveTransform(pts_i.reshape(-1, 1, 2), H).reshape(-1, 2)
        forward = _nearest_feature_matches(proj_j, pts_j, self.args.vismatch_feature_reproj_thresh)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            report["status"] = "singular_homography"
            return _empty_match_info(i, j)
        proj_i = cv.perspectiveTransform(pts_j.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        reverse = _nearest_feature_matches(proj_i, pts_i, self.args.vismatch_feature_reproj_thresh)

        matches = []
        for idx_i, (idx_j, dist_fwd) in forward.items():
            rev = reverse.get(idx_j)
            if rev is None or rev[0] != idx_i:
                continue
            matches.append(cv.DMatch(int(idx_i), int(idx_j), float(0.5 * (dist_fwd + rev[1]))))

        match_info = _empty_match_info(i, j)
        report["opencv_feature_matches"] = len(matches)
        if len(matches) < self.args.vismatch_min_projected_matches:
            report["confidence"] = 0.0
            report["status"] = "not_enough_projected_feature_matches"
            logger.info(
                "  raw=%d inliers=%d projected_features=%d skipped",
                report.get("raw_matches", 0),
                report.get("ransac_inliers", 0),
                len(matches),
            )
            return match_info
        matched_i = np.float32([pts_i[m.queryIdx] for m in matches])
        matched_j = np.float32([pts_j[m.trainIdx] for m in matches])
        H_cv, cv_mask = cv.findHomography(
            matched_i.astype(np.float64),
            matched_j.astype(np.float64),
            cv.RANSAC,
            self.args.vismatch_feature_reproj_thresh,
        )
        if H_cv is None or cv_mask is None:
            report["confidence"] = 0.0
            report["status"] = "opencv_feature_homography_failed"
            return match_info
        inliers_mask = cv_mask.reshape(-1).astype(bool)
        num_inliers = int(inliers_mask.sum())
        report["opencv_feature_inliers"] = num_inliers
        if num_inliers < self.args.vismatch_min_projected_inliers:
            report["confidence"] = 0.0
            report["status"] = "not_enough_projected_feature_inliers"
            return match_info
        match_info.matches = matches
        match_info.inliers_mask = [int(v) for v in inliers_mask]
        match_info.num_inliers = num_inliers
        match_info.confidence = float(min(num_inliers / max(self.args.vismatch_confidence_norm, 1.0), 1.0))
        match_info.H = H_cv.astype(np.float64)
        report["confidence"] = match_info.confidence
        report["status"] = "ok"
        logger.info(
            "  raw=%d inliers=%d projected_features=%d projected_inliers=%d confidence=%.3f",
            report.get("raw_matches", 0),
            report.get("ransac_inliers", 0),
            len(matches),
            num_inliers,
            match_info.confidence,
        )
        return match_info

    @staticmethod
    def _reverse_match_info(original, i: int, j: int):
        match_info = _empty_match_info(i, j)
        match_info.confidence = original.confidence
        match_info.num_inliers = original.num_inliers
        match_info.inliers_mask = list(original.inliers_mask) if original.inliers_mask is not None else []
        match_info.matches = [cv.DMatch(int(m.trainIdx), int(m.queryIdx), float(m.distance)) for m in original.getMatches()]
        if original.H is not None and len(original.H) > 0:
            match_info.H = np.linalg.inv(original.H)
        return match_info


def _compute_features(args, image_names: Sequence[str]):
    finder = FEATURES[args.features]()
    features = []
    seam_images = []
    full_sizes = []
    masks_full = []
    work_scale = None
    seam_scale = None
    seam_work_aspect = 1.0

    for idx, name in enumerate(image_names):
        path = Path(args.images) / name
        full_img = cv.imread(str(path), cv.IMREAD_COLOR)
        if full_img is None:
            raise ValueError(f"Cannot read image: {path}")
        full_sizes.append((full_img.shape[1], full_img.shape[0]))
        if work_scale is None:
            work_scale = _scale_from_megapix(full_img, args.work_megapix)
        if seam_scale is None:
            seam_scale = _scale_from_megapix(full_img, args.seam_megapix)
            seam_work_aspect = seam_scale / work_scale

        work_img = cv.resize(full_img, None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        mask = None
        if args.exclude_top > 0 or args.exclude_bottom > 0:
            full_mask = _make_feature_mask(full_img.shape[:2], args.exclude_top, args.exclude_bottom)
            masks_full.append(full_mask)
            mask = cv.resize(full_mask, (work_img.shape[1], work_img.shape[0]), interpolation=cv.INTER_NEAREST)
        else:
            masks_full.append(None)

        feat = cv.detail.computeImageFeatures2(finder, work_img, mask) if mask is not None else cv.detail.computeImageFeatures2(finder, work_img)
        feat.img_idx = idx
        features.append(feat)
        seam_img = cv.resize(full_img, None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        seam_images.append(seam_img)
        logger.info("Features %s: %d keypoints", name, len(feat.getKeypoints()))

    return features, seam_images, full_sizes, masks_full, float(work_scale), float(seam_scale), float(seam_work_aspect)


def _subset_after_matching(features, pairwise_matches, image_names, seam_images, full_sizes, masks_full, conf_thresh, matcher):
    indices = cv.detail.leaveBiggestComponent(features, pairwise_matches, conf_thresh)
    indices = [int(i) for i in indices]
    if len(indices) < 2:
        raise RuntimeError("Need at least two images in the largest connected component")
    if len(indices) == len(image_names):
        return features, pairwise_matches, image_names, seam_images, full_sizes, masks_full, indices

    logger.warning("Keeping %d/%d images in the largest connected component", len(indices), len(image_names))
    features = [features[i] for i in indices]
    for new_idx, feat in enumerate(features):
        feat.img_idx = new_idx
    image_names = [image_names[i] for i in indices]
    seam_images = [seam_images[i] for i in indices]
    full_sizes = [full_sizes[i] for i in indices]
    masks_full = [masks_full[i] for i in indices]
    if hasattr(matcher, "subset"):
        matcher.subset(indices)
    pairwise_matches = matcher.apply2(features)
    matcher.collectGarbage()
    return features, pairwise_matches, image_names, seam_images, full_sizes, masks_full, indices


def _estimate_cameras(args, features, pairwise_matches):
    estimator = ESTIMATORS[args.estimator]()
    ok, cameras = estimator.apply(features, pairwise_matches, None)
    if not ok:
        raise RuntimeError("Camera estimation failed")
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
    args._ba_status = "estimator_only"

    adjuster = BA_COSTS[args.ba]()
    if args.ba != "none":
        adjuster.setConfThresh(args.ba_conf_thresh)
        refine_mask = np.zeros((3, 3), np.uint8)
        mask = args.ba_refine_mask
        if len(mask) != 5:
            raise ValueError("--ba_refine_mask must contain exactly 5 characters")
        if mask[0] == "x":
            refine_mask[0, 0] = 1
        if mask[1] == "x":
            refine_mask[0, 1] = 1
        if mask[2] == "x":
            refine_mask[0, 2] = 1
        if mask[3] == "x":
            refine_mask[1, 1] = 1
        if mask[4] == "x":
            refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)

    try:
        ok, adjusted_cameras = adjuster.apply(features, pairwise_matches, cameras)
    except cv.error as exc:
        if not args.ba_fallback:
            raise
        logger.warning("Bundle adjustment failed inside OpenCV; falling back to estimator-only cameras: %s", exc)
        ok = False
        adjusted_cameras = cameras
        args._ba_status = f"{args.ba}_failed_fallback_none"
    if ok:
        cameras = adjusted_cameras
        args._ba_status = args.ba
    elif args.ba == "none":
        args._ba_status = "none"
    elif not args.ba_fallback:
        raise RuntimeError("Bundle adjustment failed")
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    wave_correct = WAVE_CORRECT[args.wave_correct]
    if wave_correct is not None:
        rmats = [np.copy(cam.R) for cam in cameras]
        rmats = cv.detail.waveCorrect(rmats, wave_correct)
        for cam, R in zip(cameras, rmats):
            cam.R = R
    return cameras


def _median_focal(cameras) -> float:
    focals = sorted(float(cam.focal) for cam in cameras)
    if len(focals) % 2:
        return focals[len(focals) // 2]
    return 0.5 * (focals[len(focals) // 2] + focals[len(focals) // 2 - 1])


def _warp_for_seams(args, seam_images, cameras, warped_image_scale, seam_work_aspect):
    warper = cv.PyRotationWarper(args.warp, warped_image_scale * seam_work_aspect)
    corners = []
    sizes = []
    images_warped = []
    masks_warped = []
    for img, cam in zip(seam_images, cameras):
        K = cam.K().astype(np.float32)
        K[0, 0] *= seam_work_aspect
        K[0, 2] *= seam_work_aspect
        K[1, 1] *= seam_work_aspect
        K[1, 2] *= seam_work_aspect
        corner, image_wp = warper.warp(img, K, cam.R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        _, mask_wp = warper.warp(mask, K, cam.R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        corners.append((int(corner[0]), int(corner[1])))
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        masks_warped.append(mask_wp)
    return corners, sizes, images_warped, masks_warped


def _create_seam_finder(name: str):
    if name in ("no", "voronoi"):
        return cv.detail.SeamFinder_createDefault(SEAM_FINDERS[name])
    if name == "dp_color":
        return cv.detail_DpSeamFinder("COLOR")
    if name == "dp_colorgrad":
        return cv.detail_DpSeamFinder("COLOR_GRAD")
    if name == "gc_color":
        return cv.detail_GraphCutSeamFinder("COST_COLOR")
    if name == "gc_colorgrad":
        return cv.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    raise ValueError(f"Unknown seam finder: {name}")


def _result_roi(corners: Sequence[Tuple[int, int]], sizes: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_x = min(int(c[0]) for c in corners)
    min_y = min(int(c[1]) for c in corners)
    max_x = max(int(c[0]) + int(s[0]) for c, s in zip(corners, sizes))
    max_y = max(int(c[1]) + int(s[1]) for c, s in zip(corners, sizes))
    return min_x, min_y, max_x - min_x, max_y - min_y


def _create_blender(args, dst_roi):
    blend_width = np.sqrt(dst_roi[2] * dst_roi[3]) * args.blend_strength / 100.0
    if args.blend == "no" or blend_width < 1:
        blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
    elif args.blend == "feather":
        blender = cv.detail_FeatherBlender()
        blender.setSharpness(1.0 / blend_width)
    elif args.blend == "multiband":
        blender = cv.detail_MultiBandBlender(int(args.try_cuda), max(1, int(np.log2(blend_width) - 1)))
    else:
        blender = cv.detail.Blender_createDefault(BLENDERS[args.blend])
    blender.prepare(tuple(int(v) for v in dst_roi))
    return blender


def _save_stitch_params(
    path,
    args,
    image_names,
    indices,
    cameras,
    warped_image_scale,
    corners,
    crop_box,
    output_shape,
    full_sizes,
    warped_sizes,
    blend_roi,
    work_scale,
    seam_scale,
    compose_scale,
    compose_work_aspect,
):
    data = OrderedDict(
        [
            ("version", "1.0"),
            ("type", "opencv_detailed_stitch_params"),
            ("images_dir", args.images),
            ("images", list(image_names)),
            ("kept_original_indices", list(indices)),
            ("output", args.output),
            ("warp", args.warp),
            ("warped_image_scale", float(warped_image_scale)),
            ("work_scale", float(work_scale)),
            ("seam_scale", float(seam_scale)),
            ("compose_scale", float(compose_scale)),
            ("compose_work_aspect", float(compose_work_aspect)),
            ("blend_roi", [int(v) for v in blend_roi]),
            ("corners_before_crop", [[int(c[0]), int(c[1])] for c in corners]),
            ("crop_box", [int(v) for v in crop_box]),
            ("output_shape", [int(v) for v in output_shape]),
            ("full_sizes", [[int(w), int(h)] for w, h in full_sizes]),
            ("warped_sizes", [[int(w), int(h)] for w, h in warped_sizes]),
            (
                "cameras",
                [
                    OrderedDict(
                        [
                            ("focal", float(cam.focal)),
                            ("aspect", float(cam.aspect)),
                            ("ppx", float(cam.ppx)),
                            ("ppy", float(cam.ppy)),
                            ("K_compose", (cam.K().astype(float) * np.array([[compose_work_aspect, 1.0, compose_work_aspect], [1.0, compose_work_aspect, compose_work_aspect], [1.0, 1.0, 1.0]])).tolist()),
                            ("R", cam.R.astype(float).tolist()),
                            ("t", cam.t.astype(float).reshape(-1).tolist() if hasattr(cam, "t") else []),
                        ]
                    )
                    for cam in cameras
                ],
            ),
        ]
    )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_panorama(args):
    image_names = _read_image_names(args.images, args.max_images)
    features, seam_images, full_sizes, masks_full, work_scale, seam_scale, seam_work_aspect = _compute_features(args, image_names)

    matcher = _create_matcher(args, image_names, full_sizes)
    pairwise_matches = matcher.apply2(features)
    matcher.collectGarbage()
    features, pairwise_matches, image_names, seam_images, full_sizes, masks_full, kept_indices = _subset_after_matching(
        features, pairwise_matches, image_names, seam_images, full_sizes, masks_full, args.conf_thresh, matcher
    )

    cameras = _estimate_cameras(args, features, pairwise_matches)
    warped_image_scale = _median_focal(cameras)
    logger.info("Median warped image scale: %.3f", warped_image_scale)

    corners, sizes, images_warped, masks_warped = _warp_for_seams(args, seam_images, cameras, warped_image_scale, seam_work_aspect)
    compensator = cv.detail.ExposureCompensator_createDefault(EXPOSURE[args.expos_comp])
    compensator.feed(corners, images_warped, masks_warped)

    seam_finder = _create_seam_finder(args.seam)
    seam_images_float = [img.astype(np.float32) for img in images_warped]
    seam_finder.find(seam_images_float, corners, masks_warped)

    compose_scale = None
    compose_work_aspect = 1.0
    final_corners = []
    final_sizes = []
    final_images_warped = []
    final_masks_warped = []
    final_seam_masks = []
    final_warped_scale = warped_image_scale

    for idx, name in enumerate(image_names):
        full_img = cv.imread(str(Path(args.images) / name), cv.IMREAD_COLOR)
        if full_img is None:
            raise ValueError(f"Cannot read image: {Path(args.images) / name}")
        if compose_scale is None:
            compose_scale = _scale_from_megapix(full_img, args.compose_megapix)
            compose_work_aspect = compose_scale / work_scale
            final_warped_scale = warped_image_scale * compose_work_aspect
            logger.info("Compose scale: %.6f, final warped scale: %.3f", compose_scale, final_warped_scale)

        if abs(compose_scale - 1.0) > 1e-12:
            img = cv.resize(full_img, None, fx=compose_scale, fy=compose_scale, interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img

        cam = cameras[idx]
        K = cam.K().astype(np.float32)
        K[0, 0] *= compose_work_aspect
        K[0, 2] *= compose_work_aspect
        K[1, 1] *= compose_work_aspect
        K[1, 2] *= compose_work_aspect
        warper = cv.PyRotationWarper(args.warp, final_warped_scale)
        corner, image_wp = warper.warp(img, K, cam.R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        _, mask_wp = warper.warp(mask, K, cam.R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)

        compensator.apply(idx, corner, image_wp, mask_wp)
        seam_mask = cv.resize(masks_warped[idx], (mask_wp.shape[1], mask_wp.shape[0]), interpolation=cv.INTER_LINEAR_EXACT)
        seam_mask = cv.bitwise_and(seam_mask, mask_wp)

        final_corners.append((int(corner[0]), int(corner[1])))
        final_sizes.append((image_wp.shape[1], image_wp.shape[0]))
        final_images_warped.append(image_wp.astype(np.int16))
        final_masks_warped.append(mask_wp)
        final_seam_masks.append(seam_mask)

    dst_roi = _result_roi(final_corners, final_sizes)
    logger.info("Blend ROI: x=%d y=%d w=%d h=%d", dst_roi[0], dst_roi[1], dst_roi[2], dst_roi[3])
    blender = _create_blender(args, dst_roi)
    for name, corner, image_wp, seam_mask in zip(image_names, final_corners, final_images_warped, final_seam_masks):
        logger.info("Blending %s", name)
        blender.feed(image_wp, seam_mask, corner)
    result, result_mask = blender.blend(None, None)
    result = np.clip(result, 0, 255).astype(np.uint8)

    crop_box = (0, 0, result.shape[1], result.shape[0])
    if args.crop:
        valid = result_mask > 0
        ys, xs = np.where(valid)
        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            result = result[y0:y1, x0:x1].copy()
            crop_box = (x0, y0, x1, y1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(args.output, result)
    params_path = args.params or str(Path(args.output).with_suffix(".opencv_params.json"))
    report_path = args.report or str(Path(args.output).with_suffix(".opencv_report.json"))
    _save_stitch_params(
        params_path,
        args,
        image_names,
        kept_indices,
        cameras,
        final_warped_scale,
        final_corners,
        crop_box,
        result.shape,
        full_sizes,
        final_sizes,
        dst_roi,
        work_scale,
        seam_scale,
        compose_scale,
        compose_work_aspect,
    )
    report = OrderedDict(
        [
            ("version", "1.0"),
            ("type", "opencv_detailed_panorama_report"),
            ("images", image_names),
            ("output", args.output),
            ("params", params_path),
            ("num_images", len(image_names)),
            ("features", args.features),
            ("matcher", args.matcher),
            ("estimator", args.estimator),
            ("ba", args.ba),
            ("ba_status", getattr(args, "_ba_status", args.ba)),
            ("warp", args.warp),
            ("seam", args.seam),
            ("blend", args.blend),
            ("work_scale", work_scale),
            ("seam_scale", seam_scale),
            ("compose_scale", compose_scale),
            ("output_shape", [int(v) for v in result.shape]),
            ("external_match_reports", getattr(matcher, "pair_reports", [])),
        ]
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved panorama: %s", args.output)
    logger.info("Saved params: %s", params_path)
    logger.info("Saved report: %s", report_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Build panorama with OpenCV detailed stitching")
    parser.add_argument("--images", required=True, help="Input image directory")
    parser.add_argument("--output", required=True, help="Output panorama image")
    parser.add_argument("--params", default="", help="Output OpenCV stitch params JSON")
    parser.add_argument("--report", default="", help="Output summary report JSON")
    parser.add_argument("--features", choices=sorted(FEATURES.keys()), default="sift" if "sift" in FEATURES else "orb")
    parser.add_argument("--matcher", choices=["homography", "affine", "vismatch"], default="homography")
    parser.add_argument("--estimator", choices=sorted(ESTIMATORS.keys()), default="homography")
    parser.add_argument("--ba", choices=sorted(BA_COSTS.keys()), default="ray")
    parser.add_argument("--ba_refine_mask", default="xxxxx", help="Five chars for fx,skew,ppx,aspect,ppy refinement")
    parser.add_argument("--ba_conf_thresh", type=float, default=1.0)
    parser.add_argument("--no_ba_fallback", action="store_false", dest="ba_fallback", help="Do not fall back to estimator-only cameras if OpenCV bundle adjustment fails")
    parser.add_argument("--wave_correct", choices=sorted(WAVE_CORRECT.keys()), default="horiz")
    parser.add_argument("--warp", default="spherical", help="OpenCV warper type, e.g. spherical, cylindrical, plane")
    parser.add_argument("--seam", choices=["no", "voronoi", "dp_color", "dp_colorgrad", "gc_color", "gc_colorgrad"], default="gc_color")
    parser.add_argument("--expos_comp", choices=sorted(EXPOSURE.keys()), default="gain")
    parser.add_argument("--blend", choices=sorted(BLENDERS.keys()), default="multiband")
    parser.add_argument("--blend_strength", type=float, default=5.0)
    parser.add_argument("--work_megapix", type=float, default=0.6)
    parser.add_argument("--seam_megapix", type=float, default=0.1)
    parser.add_argument("--compose_megapix", type=float, default=-1.0)
    parser.add_argument("--conf_thresh", type=float, default=0.2)
    parser.add_argument("--match_conf", type=float, default=-1.0, help="Feature match confidence; negative uses OpenCV default")
    parser.add_argument("--rangewidth", type=int, default=-1, help="Limit external matcher to image pairs within this index distance; -1 means all pairs")
    parser.add_argument("--vismatch_matcher", default="superpoint-lightglue", help="vismatch model name used when --matcher=vismatch")
    parser.add_argument("--device", default="auto", help="vismatch device: auto, cpu, cuda, or mps")
    parser.add_argument("--resize", type=int, default=1024, help="vismatch square resize; <=0 keeps original size")
    parser.add_argument("--vismatch_min_matches", type=int, default=12, help="Minimum raw vismatch matches for a pair")
    parser.add_argument("--vismatch_min_inliers", type=int, default=8, help="Minimum RANSAC inliers for a vismatch pair")
    parser.add_argument("--vismatch_min_projected_matches", type=int, default=12, help="Minimum OpenCV feature matches after external homography projection")
    parser.add_argument("--vismatch_min_projected_inliers", type=int, default=8, help="Minimum OpenCV feature RANSAC inliers after projection matching")
    parser.add_argument("--vismatch_ransac_thresh", type=float, default=5.0, help="RANSAC threshold in original image pixels")
    parser.add_argument("--vismatch_max_rmse", type=float, default=0.0, help="Skip external matches above this RMSE; 0 disables")
    parser.add_argument("--vismatch_feature_reproj_thresh", type=float, default=5.0, help="Threshold for pairing OpenCV keypoints after external homography")
    parser.add_argument("--vismatch_confidence_norm", type=float, default=50.0, help="Projected OpenCV matches required for confidence 1.0")
    parser.add_argument("--exclude_top", type=float, default=0.0, help="Fraction of top image area ignored during feature detection")
    parser.add_argument("--exclude_bottom", type=float, default=0.0, help="Fraction of bottom image area ignored during feature detection")
    parser.add_argument("--max_images", type=int, default=0, help="Debug limit; 0 means all images")
    parser.add_argument("--try_cuda", action="store_true")
    parser.add_argument("--no_crop", action="store_false", dest="crop", help="Keep full blended canvas")
    parser.set_defaults(crop=True)
    return parser.parse_args()


def main():
    build_panorama(parse_args())


if __name__ == "__main__":
    main()
