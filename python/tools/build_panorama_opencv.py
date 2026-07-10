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
from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2 as cv
import numpy as np

from spatial_gcp_utils import list_images, natural_key


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


def _create_matcher(args):
    match_conf = args.match_conf
    if match_conf < 0:
        match_conf = 0.3 if args.features == "orb" else 0.65
    if args.matcher == "affine":
        return cv.detail_AffineBestOf2NearestMatcher(False, args.try_cuda, match_conf)
    return cv.detail.BestOf2NearestMatcher_create(args.try_cuda, match_conf)


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

    ok, cameras = adjuster.apply(features, pairwise_matches, cameras)
    if not ok:
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


def _save_stitch_params(path, args, image_names, indices, cameras, warped_image_scale, corners, crop_box, output_shape, full_sizes):
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
            ("corners_before_crop", [[int(c[0]), int(c[1])] for c in corners]),
            ("crop_box", [int(v) for v in crop_box]),
            ("output_shape", [int(v) for v in output_shape]),
            ("full_sizes", [[int(w), int(h)] for w, h in full_sizes]),
            (
                "cameras",
                [
                    OrderedDict(
                        [
                            ("focal", float(cam.focal)),
                            ("aspect", float(cam.aspect)),
                            ("ppx", float(cam.ppx)),
                            ("ppy", float(cam.ppy)),
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

    matcher = _create_matcher(args)
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
            ("warp", args.warp),
            ("seam", args.seam),
            ("blend", args.blend),
            ("work_scale", work_scale),
            ("seam_scale", seam_scale),
            ("compose_scale", compose_scale),
            ("output_shape", [int(v) for v in result.shape]),
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
    parser.add_argument("--matcher", choices=["homography", "affine"], default="homography")
    parser.add_argument("--estimator", choices=sorted(ESTIMATORS.keys()), default="homography")
    parser.add_argument("--ba", choices=sorted(BA_COSTS.keys()), default="ray")
    parser.add_argument("--ba_refine_mask", default="xxxxx", help="Five chars for fx,skew,ppx,aspect,ppy refinement")
    parser.add_argument("--ba_conf_thresh", type=float, default=1.0)
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
