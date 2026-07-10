#!/usr/bin/env python3
"""Project panorama GCP annotations back to PTZ images."""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

from spatial_gcp_utils import (
    export_annotation,
    list_images,
    load_homography_mappings,
    load_project,
    panorama_point_to_image,
    parse_dist,
    parse_k,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _inside_image(point: Sequence[float], size: Sequence[int], margin: float) -> bool:
    x, y = float(point[0]), float(point[1])
    w, h = int(size[0]), int(size[1])
    return -margin <= x < w + margin and -margin <= y < h + margin


def _read_size(path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    return w, h


def project(args):
    project_data = load_project(args.pano_gcp)
    if project_data.get("target_type") != "panorama":
        logger.warning("Input project target_type is '%s', expected 'panorama'", project_data.get("target_type"))

    points = project_data.get("points", [])
    if not points:
        raise ValueError(f"No GCP points found in {args.pano_gcp}")

    mappings_json = load_homography_mappings(args.mappings)
    mappings = mappings_json.get("mappings", mappings_json)
    image_paths = {p.name: p for p in list_images(args.images)}
    image_points: Dict[str, List[Tuple[Sequence[float], Sequence[float]]]] = OrderedDict()
    summary = OrderedDict()

    for image_name, mapping in mappings.items():
        if image_name not in image_paths:
            summary[image_name] = OrderedDict([("status", "missing_image"), ("used_points", 0)])
            continue
        if args.max_rmse > 0 and float(mapping.get("rmse", 0.0)) > args.max_rmse:
            summary[image_name] = OrderedDict(
                [
                    ("status", "skipped_rmse"),
                    ("rmse", float(mapping.get("rmse", 0.0))),
                    ("used_points", 0),
                ]
            )
            continue

        size = mapping.get("image_size") or list(_read_size(image_paths[image_name]))
        H = mapping["H_image_to_pano"]
        per_image = []
        rejected = 0
        for point in points:
            pano_xy = point["target_pixel"]
            try:
                img_xy = panorama_point_to_image(pano_xy, H)
            except Exception:
                rejected += 1
                continue
            if not _inside_image(img_xy, size, args.margin):
                rejected += 1
                continue
            per_image.append((img_xy, point["world"]))

        if len(per_image) < args.min_points:
            summary[image_name] = OrderedDict(
                [
                    ("status", "not_enough_projected_points"),
                    ("used_points", len(per_image)),
                    ("rejected_points", rejected),
                ]
            )
            continue

        image_points[image_name] = per_image
        summary[image_name] = OrderedDict(
            [
                ("status", "ok"),
                ("used_points", len(per_image)),
                ("rejected_points", rejected),
                ("homography_inliers", int(mapping.get("num_inliers", 0))),
                ("homography_rmse", float(mapping.get("rmse", 0.0))),
            ]
        )

    if not image_points:
        raise RuntimeError("No image has enough projected GCPs. Check homographies, panorama overlap, or --min_points.")

    first_name = next(iter(image_points.keys()))
    first_size = _read_size(image_paths[first_name])
    K = parse_k(args.K, first_size[0], first_size[1]) if args.K else None
    dist = parse_dist(args.dist) if args.dist else None
    export_annotation(image_points, args.images, args.output, K=K, dist=dist)

    summary_path = args.summary or str(Path(args.output).with_suffix(".summary.json"))
    with open(summary_path, "w") as f:
        json.dump(
            OrderedDict(
                [
                    ("version", "1.0"),
                    ("type", "pano_gcp_projection_summary"),
                    ("pano_gcp", args.pano_gcp),
                    ("mappings", args.mappings),
                    ("annotation", args.output),
                    ("total_panorama_points", len(points)),
                    ("exported_images", len(image_points)),
                    ("images", summary),
                ]
            ),
            f,
            indent=2,
        )

    logger.info("Exported annotation for %d images: %s", len(image_points), args.output)
    logger.info("Saved projection summary: %s", summary_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Project panorama GCPs to PTZ images")
    parser.add_argument("--pano_gcp", required=True, help="Spatial GCP project JSON from spatial_gcp_annotator.py")
    parser.add_argument("--mappings", required=True, help="Homography mapping JSON from build_pano_homographies_vismatch.py")
    parser.add_argument("--images", required=True, help="Directory containing PTZ images")
    parser.add_argument("--output", required=True, help="Output PTZ-Calib annotation JSON")
    parser.add_argument("--summary", default="", help="Optional projection summary JSON path")
    parser.add_argument("--min_points", type=int, default=4, help="Minimum projected GCPs per image")
    parser.add_argument("--margin", type=float, default=0.0, help="Allow projected pixels outside image bounds by this many pixels")
    parser.add_argument("--max_rmse", type=float, default=0.0, help="Skip homographies above this RMSE; 0 disables")
    parser.add_argument("--K", default=None, help="Initial K, 9 comma-separated values")
    parser.add_argument("--dist", default=None, help="Initial dist, 5 comma-separated values")
    return parser.parse_args()


def main():
    project(parse_args())


if __name__ == "__main__":
    main()
