#!/usr/bin/env python3
"""Build an ordered panorama using vismatch pairwise homographies.

This tool is intentionally simple and explicit:

  1. Match adjacent images in their input order with vismatch.
  2. Estimate adjacent image-to-image homographies with RANSAC.
  3. Compose all images into the coordinate system of a center image.
  4. Warp images to a common canvas and average overlapping pixels.

It is a practical panorama bootstrapper for the spatial GCP workflow. For
production-quality panoramas, seam finding and exposure compensation can be
added later without changing the downstream GCP tools.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prepare_data_vismatch import (  # noqa: E402
    _loaded_size,
    _scale_keypoints,
    extract_pair_data,
    list_images,
    load_image_for_matcher,
    load_matcher,
)

from spatial_gcp_utils import natural_key  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _read_images(images_dir: str, max_images: int = 0) -> Tuple[List[str], List[np.ndarray]]:
    names = sorted(list_images(images_dir), key=natural_key)
    if max_images > 0:
        names = names[:max_images]
    if len(names) < 2:
        raise ValueError("At least two images are required to build a panorama")
    imgs = []
    for name in names:
        img = cv2.imread(str(Path(images_dir) / name), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {Path(images_dir) / name}")
        imgs.append(img)
    return names, imgs


def _matched_points(
    all_kpts0: Optional[np.ndarray],
    all_kpts1: Optional[np.ndarray],
    matches: Optional[np.ndarray],
    matched0: Optional[np.ndarray],
    matched1: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
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


def _estimate_pair_homography(matcher, path0: Path, path1: Path, size0, size1, args):
    img0 = load_image_for_matcher(matcher, str(path0), args.resize)
    img1 = load_image_for_matcher(matcher, str(path1), args.resize)
    load_size0 = _loaded_size(img0, size0)
    load_size1 = _loaded_size(img1, size1)

    result = matcher(img0, img1)
    all_kpts0, _desc0, all_kpts1, _desc1, matches, matched0, matched1 = extract_pair_data(result)
    all_kpts0 = _scale_keypoints(all_kpts0, load_size0, size0)
    all_kpts1 = _scale_keypoints(all_kpts1, load_size1, size1)
    matched0 = _scale_keypoints(matched0, load_size0, size0)
    matched1 = _scale_keypoints(matched1, load_size1, size1)
    pts0, pts1 = _matched_points(all_kpts0, all_kpts1, matches, matched0, matched1)

    if len(pts0) < args.min_matches:
        raise RuntimeError(f"not enough matches: {len(pts0)} < {args.min_matches}")

    H, inlier_mask = cv2.findHomography(
        pts0.astype(np.float64),
        pts1.astype(np.float64),
        cv2.RANSAC,
        args.ransac_thresh,
    )
    if H is None or inlier_mask is None:
        raise RuntimeError("homography estimation failed")
    mask = inlier_mask.reshape(-1).astype(bool)
    num_inliers = int(mask.sum())
    if num_inliers < args.min_inliers:
        raise RuntimeError(f"not enough inliers: {num_inliers} < {args.min_inliers}")

    projected = cv2.perspectiveTransform(pts0[mask].reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((projected - pts1[mask]) ** 2, axis=1))))
    if args.max_rmse > 0 and rmse > args.max_rmse:
        raise RuntimeError(f"RANSAC rmse too high: {rmse:.3f} > {args.max_rmse}")

    return H, OrderedDict(
        [
            ("matches", int(len(pts0))),
            ("inliers", num_inliers),
            ("inlier_ratio", float(num_inliers / max(len(pts0), 1))),
            ("rmse", rmse),
        ]
    )


def _compose_to_center(adjacent_h: Sequence[np.ndarray], center: int) -> List[np.ndarray]:
    transforms = []
    for i in range(len(adjacent_h) + 1):
        H = np.eye(3, dtype=np.float64)
        if i < center:
            for j in range(i, center):
                H = adjacent_h[j] @ H
        elif i > center:
            for j in range(center, i):
                H = np.linalg.inv(adjacent_h[j]) @ H
        transforms.append(H)
    return transforms


def _canvas_bounds(images: Sequence[np.ndarray], transforms: Sequence[np.ndarray]):
    warped_corners = []
    for img, H in zip(images, transforms):
        h, w = img.shape[:2]
        corners = np.float64([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
        warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        warped_corners.append(warped)
    pts = np.vstack(warped_corners)
    min_xy = np.floor(pts.min(axis=0)).astype(int)
    max_xy = np.ceil(pts.max(axis=0)).astype(int)
    width = int(max_xy[0] - min_xy[0])
    height = int(max_xy[1] - min_xy[1])
    shift = np.array([[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]])
    return width, height, shift, min_xy, max_xy


def _crop_valid_region(pano: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    ys, xs = np.where(valid_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return pano, (0, 0, pano.shape[1], pano.shape[0])
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return pano[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def build_panorama(args):
    image_names, images = _read_images(args.images, args.max_images)
    sizes = [(img.shape[1], img.shape[0]) for img in images]
    center = args.center if args.center >= 0 else len(images) // 2
    if center < 0 or center >= len(images):
        raise ValueError(f"--center must be in [0, {len(images) - 1}]")

    matcher = load_matcher(args.matcher, args.device)
    adjacent_h = []
    pair_reports = []
    for i in range(len(images) - 1):
        name0, name1 = image_names[i], image_names[i + 1]
        logger.info("[%d/%d] Matching %s -> %s", i + 1, len(images) - 1, name0, name1)
        try:
            H, stats = _estimate_pair_homography(
                matcher,
                Path(args.images) / name0,
                Path(args.images) / name1,
                sizes[i],
                sizes[i + 1],
                args,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to connect adjacent pair {name0} -> {name1}: {exc}") from exc
        adjacent_h.append(H)
        pair_reports.append(OrderedDict([("image0", name0), ("image1", name1), *stats.items()]))
        logger.info("  ok: matches=%d inliers=%d rmse=%.3f", stats["matches"], stats["inliers"], stats["rmse"])

    transforms = _compose_to_center(adjacent_h, center)
    width, height, shift, min_xy, max_xy = _canvas_bounds(images, transforms)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid panorama canvas bounds")
    if args.max_canvas_pixels > 0 and width * height > args.max_canvas_pixels:
        raise RuntimeError(
            f"Panorama canvas is too large: {width}x{height}={width * height} pixels. "
            "Increase --max_canvas_pixels or reduce image count/size."
        )

    acc = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    for name, img, H in zip(image_names, images, transforms):
        logger.info("Warping %s", name)
        warped = cv2.warpPerspective(img, shift @ H, (width, height))
        mask = (warped.sum(axis=2) > 0).astype(np.float32)[..., None]
        acc += warped.astype(np.float32) * mask
        weight += mask

    pano = (acc / np.maximum(weight, 1.0)).astype(np.uint8)
    valid = (weight[..., 0] > 0).astype(np.uint8)
    crop_box = (0, 0, width, height)
    if args.crop:
        pano, crop_box = _crop_valid_region(pano, valid)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, pano)
    report_path = args.report or str(Path(args.output).with_suffix(".report.json"))
    report = OrderedDict(
        [
            ("version", "1.0"),
            ("type", "vismatch_ordered_panorama_report"),
            ("images_dir", args.images),
            ("output", args.output),
            ("matcher", args.matcher),
            ("resize", int(args.resize)),
            ("center_index", int(center)),
            ("center_image", image_names[center]),
            ("canvas_shape_before_crop", [int(height), int(width), 3]),
            ("output_shape", [int(pano.shape[0]), int(pano.shape[1]), int(pano.shape[2])]),
            ("crop_box", list(map(int, crop_box))),
            ("canvas_min_xy", list(map(int, min_xy))),
            ("canvas_max_xy", list(map(int, max_xy))),
            ("pairs", pair_reports),
            ("transforms_image_to_center", [H.tolist() for H in transforms]),
        ]
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved panorama: %s", args.output)
    logger.info("Saved report: %s", report_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Build ordered panorama using vismatch homographies")
    parser.add_argument("--images", required=True, help="Ordered input image directory")
    parser.add_argument("--output", required=True, help="Output panorama image")
    parser.add_argument("--report", default="", help="Output report JSON; defaults to output path with .report.json")
    parser.add_argument("--matcher", default="superpoint-lightglue", help="vismatch matcher name")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--resize", type=int, default=1024, help="Matcher resize square side; <=0 keeps original size")
    parser.add_argument("--min_matches", type=int, default=20, help="Minimum raw matches per adjacent pair")
    parser.add_argument("--min_inliers", type=int, default=12, help="Minimum RANSAC inliers per adjacent pair")
    parser.add_argument("--ransac_thresh", type=float, default=5.0, help="RANSAC reprojection threshold in pixels")
    parser.add_argument("--max_rmse", type=float, default=0.0, help="Skip pairs above this inlier RMSE; 0 disables")
    parser.add_argument("--center", type=int, default=-1, help="Center image index; default is middle image")
    parser.add_argument("--max_images", type=int, default=0, help="Debug limit; 0 means all images")
    parser.add_argument("--max_canvas_pixels", type=int, default=120_000_000, help="Safety limit for panorama canvas pixels")
    parser.add_argument("--no_crop", action="store_false", dest="crop", help="Keep full canvas including empty borders")
    parser.set_defaults(crop=True)
    return parser.parse_args()


def main():
    build_panorama(parse_args())


if __name__ == "__main__":
    main()
