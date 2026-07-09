#!/usr/bin/env python3
"""PTZ bundle adjustment application.

Python port of src/app/run_ptz_ba.cc.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Set, Tuple

from ptzcalib.data_io import (
    load_annotation,
    load_imgs_and_features,
    load_matches_info,
    mkdir_ifnot_exist,
    save_registered_cam,
)
from ptzcalib.ptz_incremental_optimizer import PtzIncrementalOptimizer
from ptzcalib.ptzray_optimizer import FactorType, PTZRayOptimizer
from ptzcalib.types import Camera, ImageFeatures, MatchesInfo


def run_ptz_ba(
    fnames: List[str],
    features: List[ImageFeatures],
    matches_info: List[MatchesInfo],
    max_iter: int,
) -> Tuple[bool, List[Camera], Set[int]]:
    cameras = [Camera() for _ in fnames]
    optimizer = PtzIncrementalOptimizer(features, matches_info, cameras, fnames, max_iter)
    cameras_out = [cam.clone() for cam in cameras]
    success, reg_image_ids = optimizer.solve(cameras_out)
    return success, cameras_out, reg_image_ids


def run_georeferencing(
    features: List[ImageFeatures],
    matches_info: List[MatchesInfo],
    pixels,
    pts3d,
    cam_ids: Set[int],
    max_iter: int,
    has_dist: bool,
    cameras: List[Camera],
) -> Tuple[bool, float, float, List[Camera]]:
    factor_type = FactorType.PTZRayDist if has_dist else FactorType.PTZRay
    optimizer = PTZRayOptimizer(features, matches_info, cameras, pixels, pts3d, cam_ids, max_iter, factor_type)
    cameras_out = [cam.clone() for cam in cameras]
    rays = []
    success = optimizer.solve(cameras_out, rays)
    if not success:
        return False, -1.0, -1.0, cameras
    return (
        True,
        optimizer.final_reproj_error_2d2d(),
        optimizer.final_reproj_error_2d3d(),
        cameras_out,
    )


RunPtzBA = run_ptz_ba
RunGeoreferencing = run_georeferencing


def parse_args(argv):
    parser = argparse.ArgumentParser(description="PTZ bundle adjustment")
    parser.add_argument("-i", "--images", required=True, help="Images directory")
    parser.add_argument("-f", "--features", required=True, help="Features and matches directory")
    parser.add_argument("-a", "--annotation", default="", help="Annotation filepath")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--dist", action="store_true", help="Whether images have distortion")
    parser.add_argument("--max_iter", type=int, default=200, help="Max BA iterations")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    ok, fnames, features, _sizes = load_imgs_and_features(args.images, args.features)
    if not ok:
        logging.error("Error loading images and features. Exiting ...")
        return 1

    matches_path = os.path.join(args.features, "pairs_matches.txt")
    ok, matches_info = load_matches_info(matches_path, fnames, features)
    if not ok:
        logging.error("Error loading matches from %s. Exiting ...", matches_path)
        return 1

    logging.info("================== PTZ-IBA Begin ==========================")
    ok, cameras, reg_image_ids = run_ptz_ba(fnames, features, matches_info, args.max_iter)
    if not ok:
        logging.info("================== PTZ-IBA End: failed ==========================")
        return 1
    logging.info("================== PTZ-IBA End: success ==========================")

    pixels = [[] for _ in fnames]
    pts3d = [[] for _ in fnames]
    error_2d2d = -1.0
    error_2d3d = -1.0

    if args.annotation:
        ok, pixels, pts3d = load_annotation(args.annotation, fnames)
        if not ok:
            logging.error("Error loading annotation from %s. Exiting ...", args.annotation)
            return 1

        logging.info("================== Georeferencing Begin ==========================")
        ok, error_2d2d, error_2d3d, cameras = run_georeferencing(
            features,
            matches_info,
            pixels,
            pts3d,
            reg_image_ids,
            args.max_iter,
            args.dist,
            cameras,
        )
        if not ok:
            logging.info("================== Georeferencing End: failed ==========================")
            return 1
        logging.info("================== Georeferencing End: success ==========================")

    cam_id = os.path.basename(os.path.normpath(args.images))
    mkdir_ifnot_exist(args.output)
    out_path = os.path.join(args.output, cam_id + ".json")
    save_registered_cam(cameras, reg_image_ids, fnames, pixels, pts3d, out_path)

    logging.info("================== Summary Begin ==========================")
    logging.info("Registered/Total: %d/%d", len(reg_image_ids), len(fnames))
    logging.info("Error 2d-2d: %s", error_2d2d)
    logging.info("Error 2d-3d: %s", error_2d3d)
    logging.info("==================== Summary End ==========================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
