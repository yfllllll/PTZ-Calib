#!/usr/bin/env python3
"""PTZ relocalization application.

Python port of src/app/run_ptz_reloc.cc.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Tuple

import numpy as np

from ptzcalib.data_io import (
    find_img_index,
    load_imgs_and_features,
    mkdir_ifnot_exist,
    read_cam_from_json,
    read_colmap_matches,
    save_registered_cam,
)
from ptzcalib.krt_optimizer import FactorType, KRTOptimizer
from ptzcalib.types import Camera


def find_best_match(fname: str, img_pairs_name: List[Tuple[str, str]], pairs_matches):
    best_name = ""
    best_matches = []
    for i, pair in enumerate(img_pairs_name):
        if pair[1] != fname:
            continue
        if len(pairs_matches[i]) > len(best_matches):
            best_name = pair[0]
            best_matches = pairs_matches[i]
    return best_name, best_matches


FindBestMatch = find_best_match


def parse_args(argv):
    parser = argparse.ArgumentParser(description="PTZ relocalization")
    parser.add_argument("--ref_images", required=True, help="Reference images directory")
    parser.add_argument("--ref_features", required=True, help="Reference images features directory")
    parser.add_argument("--ref_params", required=True, help="Reference camera parameters filepath")
    parser.add_argument("--test_images", required=True, help="Test images directory")
    parser.add_argument("--test_features", required=True, help="Test images features and matches directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--dist", action="store_true", help="Whether images have distortion")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    ok, ref_fnames, ref_features, _ref_sizes = load_imgs_and_features(args.ref_images, args.ref_features)
    if not ok:
        logging.error("Error loading reference images and features. Exiting ...")
        return 1

    ok, test_fnames, test_features, test_sizes = load_imgs_and_features(args.test_images, args.test_features)
    if not ok:
        logging.error("Error loading test images and features. Exiting ...")
        return 1

    matches_path = os.path.join(args.test_features, "pairs_matches.txt")
    pairs_matches, img_pairs_name = read_colmap_matches(matches_path)

    ok, ref_cameras = read_cam_from_json(args.ref_params, ref_fnames)
    if not ok:
        logging.error("Error loading reference camera parameters. Exiting ...")
        return 1

    test_cameras = [Camera() for _ in test_fnames]
    success_ids = set()

    for test_idx, test_name in enumerate(test_fnames):
        best_ref_name, best_matches = find_best_match(test_name, img_pairs_name, pairs_matches)
        ref_idx = find_img_index(ref_fnames, best_ref_name)
        if ref_idx == -1:
            logging.info("Running ptz-reloc failed: %s", test_name)
            continue

        ref_cam = ref_cameras[ref_idx]
        factor_type = FactorType.FDist if args.dist else FactorType.F
        optimizer = KRTOptimizer(200, 100.0, factor_type)

        f = ref_cam.K()[0, 0]
        width, height = test_sizes[test_idx]
        K = np.array([[f, 0.0, 0.5 * width], [0.0, f, 0.5 * height], [0.0, 0.0, 1.0]], dtype=np.float64)
        R = ref_cam.R().copy()
        t = ref_cam.t().copy()
        dist = ref_cam.dist().copy()
        optimizer.set_init_params(K, R, t, dist)
        optimizer.add_2d2d_constraints(
            ref_cam,
            ref_features[ref_idx].keypoints,
            test_features[test_idx].keypoints,
            best_matches,
        )
        opti_success, K, R, t, dist = optimizer.solve()

        if opti_success:
            test_cameras[test_idx] = Camera(K, R, t, dist)
            success_ids.add(test_idx)
            logging.info("Running ptz-reloc success: %s", test_name)
        else:
            logging.info("Running ptz-reloc failed: %s", test_name)

    cam_id = os.path.basename(os.path.normpath(args.test_images))
    mkdir_ifnot_exist(args.output)
    out_path = os.path.join(args.output, cam_id + ".json")
    pixels = [[] for _ in test_fnames]
    pts3d = [[] for _ in test_fnames]
    save_registered_cam(test_cameras, success_ids, test_fnames, pixels, pts3d, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
