#!/usr/bin/env python3
"""Estimate image-to-panorama homographies with vismatch.

The output maps each PTZ image pixel to the annotated panorama pixel plane.
It is intended for the panorama GCP workflow:

  panorama GCP project + image->panorama homographies -> per-image annotation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

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


def _original_size(path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    return w, h


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


def _homography_rmse(H: np.ndarray, pts0: np.ndarray, pts1: np.ndarray, mask: np.ndarray) -> float:
    inliers0 = pts0[mask]
    inliers1 = pts1[mask]
    if len(inliers0) == 0:
        return float("inf")
    projected = cv2.perspectiveTransform(inliers0.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    residual = projected - inliers1
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def estimate_homographies(args):
    images_dir = Path(args.images)
    panorama_path = Path(args.panorama)
    image_names = sorted(list_images(str(images_dir)), key=natural_key)
    if args.max_images > 0:
        image_names = image_names[: args.max_images]
    if not image_names:
        raise ValueError(f"No images found in {images_dir}")

    pano_size = _original_size(panorama_path)
    matcher = load_matcher(args.matcher, args.device)
    pano_loaded = load_image_for_matcher(matcher, str(panorama_path), args.resize)
    pano_loaded_size = _loaded_size(pano_loaded, pano_size)

    mappings = OrderedDict()
    skipped = OrderedDict()
    for idx, image_name in enumerate(image_names, start=1):
        image_path = images_dir / image_name
        image_size = _original_size(image_path)
        logger.info("[%d/%d] Matching %s -> %s", idx, len(image_names), image_name, panorama_path.name)
        img_loaded = load_image_for_matcher(matcher, str(image_path), args.resize)
        img_loaded_size = _loaded_size(img_loaded, image_size)

        result = matcher(img_loaded, pano_loaded)
        all_kpts0, _desc0, all_kpts1, _desc1, matches, matched0, matched1 = extract_pair_data(result)
        all_kpts0 = _scale_keypoints(all_kpts0, img_loaded_size, image_size)
        all_kpts1 = _scale_keypoints(all_kpts1, pano_loaded_size, pano_size)
        matched0 = _scale_keypoints(matched0, img_loaded_size, image_size)
        matched1 = _scale_keypoints(matched1, pano_loaded_size, pano_size)
        pts_img, pts_pano = _matched_points(all_kpts0, all_kpts1, matches, matched0, matched1)

        if len(pts_img) < args.min_matches:
            skipped[image_name] = f"not enough matches: {len(pts_img)}"
            logger.warning("  skipped: %s", skipped[image_name])
            continue

        H, inlier_mask = cv2.findHomography(
            pts_img.astype(np.float64),
            pts_pano.astype(np.float64),
            cv2.RANSAC,
            args.ransac_thresh,
        )
        if H is None or inlier_mask is None:
            skipped[image_name] = "homography estimation failed"
            logger.warning("  skipped: %s", skipped[image_name])
            continue

        mask = inlier_mask.reshape(-1).astype(bool)
        num_inliers = int(mask.sum())
        if num_inliers < args.min_inliers:
            skipped[image_name] = f"not enough RANSAC inliers: {num_inliers}"
            logger.warning("  skipped: %s", skipped[image_name])
            continue

        rmse = _homography_rmse(H, pts_img, pts_pano, mask)
        if args.max_rmse > 0 and rmse > args.max_rmse:
            skipped[image_name] = f"RANSAC rmse too high: {rmse:.3f}"
            logger.warning("  skipped: %s", skipped[image_name])
            continue

        mappings[image_name] = OrderedDict(
            [
                ("image", image_name),
                ("H_image_to_pano", H.tolist()),
                ("num_matches", int(len(pts_img))),
                ("num_inliers", num_inliers),
                ("inlier_ratio", float(num_inliers / max(len(pts_img), 1))),
                ("rmse", rmse),
                ("image_size", [int(image_size[0]), int(image_size[1])]),
                ("panorama_size", [int(pano_size[0]), int(pano_size[1])]),
            ]
        )
        logger.info("  ok: matches=%d inliers=%d rmse=%.3f", len(pts_img), num_inliers, rmse)

    output = OrderedDict(
        [
            ("version", "1.0"),
            ("type", "pano_homography_mappings"),
            ("panorama", str(panorama_path)),
            ("images", str(images_dir)),
            ("matcher", args.matcher),
            ("resize", int(args.resize)),
            ("ransac_thresh", float(args.ransac_thresh)),
            ("mappings", mappings),
            ("skipped", skipped),
        ]
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved %d homographies to %s", len(mappings), args.output)
    if skipped:
        logger.info("Skipped %d images; inspect the 'skipped' field in the output JSON", len(skipped))


def parse_args():
    parser = argparse.ArgumentParser(description="Build image-to-panorama homographies with vismatch")
    parser.add_argument("--images", required=True, help="Directory containing PTZ images")
    parser.add_argument("--panorama", required=True, help="Panorama image")
    parser.add_argument("--output", required=True, help="Output homography mapping JSON")
    parser.add_argument("--matcher", default="superpoint-lightglue", help="vismatch matcher name")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--resize", type=int, default=1024, help="Matcher resize square side; <=0 keeps original size")
    parser.add_argument("--min_matches", type=int, default=30, help="Minimum raw matches before RANSAC")
    parser.add_argument("--min_inliers", type=int, default=12, help="Minimum RANSAC inliers")
    parser.add_argument("--ransac_thresh", type=float, default=5.0, help="RANSAC reprojection threshold in panorama pixels")
    parser.add_argument("--max_rmse", type=float, default=0.0, help="Skip homographies above this inlier RMSE; 0 disables")
    parser.add_argument("--max_images", type=int, default=0, help="Debug limit; 0 means all images")
    return parser.parse_args()


def main():
    estimate_homographies(parse_args())


if __name__ == "__main__":
    main()
