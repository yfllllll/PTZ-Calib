#!/usr/bin/env python3
"""Project OpenCV detailed panorama GCP annotations back to source images."""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2 as cv
import numpy as np

from spatial_gcp_utils import export_annotation, list_images, load_project, parse_dist, parse_k


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _inside_image(point: Sequence[float], size: Sequence[int], margin: float) -> bool:
    x, y = float(point[0]), float(point[1])
    w, h = int(size[0]), int(size[1])
    return -margin <= x < w + margin and -margin <= y < h + margin


def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def _camera_k(camera: dict, compose_work_aspect: float) -> np.ndarray:
    if "K_compose" in camera:
        return np.asarray(camera["K_compose"], dtype=np.float32)
    focal = float(camera["focal"])
    aspect = float(camera.get("aspect", 1.0))
    ppx = float(camera["ppx"])
    ppy = float(camera["ppy"])
    K = np.array([[focal, 0.0, ppx], [0.0, focal * aspect, ppy], [0.0, 0.0, 1.0]], dtype=np.float32)
    K[0, 0] *= compose_work_aspect
    K[0, 2] *= compose_work_aspect
    K[1, 1] *= compose_work_aspect
    K[1, 2] *= compose_work_aspect
    return K


def _pano_to_global(point: Sequence[float], blend_roi: Sequence[int], crop_box: Sequence[int]) -> Tuple[float, float]:
    # Output panorama pixels are after cropping the blender result. The blender
    # result itself is local to blend_roi, whose origin is in warped world coords.
    return (
        float(point[0]) + float(crop_box[0]) + float(blend_roi[0]),
        float(point[1]) + float(crop_box[1]) + float(blend_roi[1]),
    )


def _compose_size(size: Sequence[int], compose_scale: float) -> Tuple[int, int]:
    return max(1, int(round(float(size[0]) * compose_scale))), max(1, int(round(float(size[1]) * compose_scale)))


def _sample_map(xmap: np.ndarray, ymap: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    if x < 0 or y < 0 or x > xmap.shape[1] - 1 or y > xmap.shape[0] - 1:
        raise ValueError("Point outside warped map")
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, xmap.shape[1] - 1)
    y1 = min(y0 + 1, xmap.shape[0] - 1)
    dx = float(x - x0)
    dy = float(y - y0)

    def interp(arr):
        return (
            float(arr[y0, x0]) * (1.0 - dx) * (1.0 - dy)
            + float(arr[y0, x1]) * dx * (1.0 - dy)
            + float(arr[y1, x0]) * (1.0 - dx) * dy
            + float(arr[y1, x1]) * dx * dy
        )

    return interp(xmap), interp(ymap)


def _inverse_map_for_image(params, image_idx: int):
    camera = params["cameras"][image_idx]
    K = _camera_k(camera, float(params.get("compose_work_aspect", 1.0)))
    R = np.asarray(camera["R"], dtype=np.float32)
    warper = cv.PyRotationWarper(params.get("warp", "spherical"), float(params["warped_image_scale"]))
    compose_scale = float(params.get("compose_scale", 1.0))
    src_size = _compose_size(params["full_sizes"][image_idx], compose_scale)
    roi, xmap, ymap = warper.buildMaps(src_size, K, R)
    return warper, K, R, roi, xmap, ymap


def _project_point_to_image(global_xy, inverse_map, params, image_idx: int, args) -> Tuple[float, float, float]:
    warper, K, R, roi, xmap, ymap = inverse_map

    warped_local = (float(global_xy[0]) - float(roi[0]), float(global_xy[1]) - float(roi[1]))
    comp_xy = _sample_map(xmap, ymap, warped_local[0], warped_local[1])
    compose_scale = float(params.get("compose_scale", 1.0))
    orig_xy = (float(comp_xy[0]) / compose_scale, float(comp_xy[1]) / compose_scale)

    if args.max_roundtrip_error > 0:
        fwd = warper.warpPoint((float(comp_xy[0]), float(comp_xy[1])), K, R)
        fwd_global = (float(fwd[0]) + float(roi[0]), float(fwd[1]) + float(roi[1]))
        err = float(np.hypot(fwd_global[0] - global_xy[0], fwd_global[1] - global_xy[1]))
    else:
        err = 0.0
    return orig_xy[0], orig_xy[1], err


def project(args):
    project_data = load_project(args.pano_gcp)
    if project_data.get("target_type") != "panorama":
        logger.warning("Input project target_type is '%s', expected 'panorama'", project_data.get("target_type"))
    points = project_data.get("points", [])
    if not points:
        raise ValueError(f"No GCP points found in {args.pano_gcp}")

    params = _read_json(args.params)
    if params.get("type") != "opencv_detailed_stitch_params":
        logger.warning("Params type is '%s', expected 'opencv_detailed_stitch_params'", params.get("type"))
    required = ["cameras", "corners_before_crop", "crop_box", "full_sizes", "warped_image_scale"]
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(f"Missing fields in OpenCV stitch params: {missing}")
    if "blend_roi" not in params:
        raise ValueError("Missing blend_roi in params. Regenerate the panorama with the current build_panorama_opencv.py.")

    images_dir = args.images or params.get("images_dir")
    if not images_dir:
        raise ValueError("--images is required when params does not contain images_dir")
    image_paths = {p.name: p for p in list_images(images_dir)}
    image_points: Dict[str, List[Tuple[Sequence[float], Sequence[float]]]] = OrderedDict()
    summary = OrderedDict()

    blend_roi = params["blend_roi"]
    crop_box = params["crop_box"]
    for image_idx, image_name in enumerate(params["images"]):
        if image_name not in image_paths:
            summary[image_name] = OrderedDict([("status", "missing_image"), ("used_points", 0)])
            continue
        inverse_map = _inverse_map_for_image(params, image_idx)
        size = params["full_sizes"][image_idx]
        per_image = []
        rejected = 0
        roundtrip_errors = []
        for point in points:
            global_xy = _pano_to_global(point["target_pixel"], blend_roi, crop_box)
            try:
                x, y, err = _project_point_to_image(global_xy, inverse_map, params, image_idx, args)
            except Exception:
                rejected += 1
                continue
            if args.max_roundtrip_error > 0 and err > args.max_roundtrip_error:
                rejected += 1
                continue
            if not _inside_image((x, y), size, args.margin):
                rejected += 1
                continue
            per_image.append(((x, y), point["world"]))
            roundtrip_errors.append(err)

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
                ("mean_roundtrip_error", float(np.mean(roundtrip_errors)) if roundtrip_errors else 0.0),
                ("max_roundtrip_error", float(np.max(roundtrip_errors)) if roundtrip_errors else 0.0),
            ]
        )

    if not image_points:
        raise RuntimeError("No image has enough projected GCPs. Check panorama params, GCP positions, or --min_points.")

    first_name = next(iter(image_points.keys()))
    first_img = cv.imread(str(image_paths[first_name]), cv.IMREAD_COLOR)
    if first_img is None:
        raise ValueError(f"Cannot read image: {image_paths[first_name]}")
    h, w = first_img.shape[:2]
    K = parse_k(args.K, w, h) if args.K else None
    dist = parse_dist(args.dist) if args.dist else None
    export_annotation(image_points, images_dir, args.output, K=K, dist=dist)

    summary_path = args.summary or str(Path(args.output).with_suffix(".opencv_pano_summary.json"))
    with open(summary_path, "w") as f:
        json.dump(
            OrderedDict(
                [
                    ("version", "1.0"),
                    ("type", "opencv_pano_gcp_projection_summary"),
                    ("pano_gcp", args.pano_gcp),
                    ("params", args.params),
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
    parser = argparse.ArgumentParser(description="Project OpenCV detailed panorama GCPs to source PTZ images")
    parser.add_argument("--pano_gcp", required=True, help="Spatial GCP project JSON from spatial_gcp_annotator.py")
    parser.add_argument("--params", required=True, help="OpenCV stitch params JSON from build_panorama_opencv.py")
    parser.add_argument("--images", default="", help="Source image directory; defaults to params images_dir")
    parser.add_argument("--output", required=True, help="Output PTZ-Calib annotation JSON")
    parser.add_argument("--summary", default="", help="Optional projection summary JSON path")
    parser.add_argument("--min_points", type=int, default=4, help="Minimum projected GCPs per image")
    parser.add_argument("--margin", type=float, default=0.0, help="Allow projected pixels outside image bounds by this many pixels")
    parser.add_argument("--max_roundtrip_error", type=float, default=0.0, help="Optional diagnostic: reject points whose map/forward warp check exceeds this many panorama pixels; <=0 disables")
    parser.add_argument("--K", default=None, help="Initial K, 9 comma-separated values")
    parser.add_argument("--dist", default=None, help="Initial dist, 5 comma-separated values")
    return parser.parse_args()


def main():
    project(parse_args())


if __name__ == "__main__":
    main()
