#!/usr/bin/env python3
"""PTZ Bundle Adjustment application.

Python port of src/app/run_ptz_ba.cc.
Simplified command-line interface for PTZ camera calibration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from ptzcalib.data_io import load_features, load_matches, load_cameras, save_cameras
from ptzcalib.ptz_incremental_optimizer import PtzIncrementalOptimizer


def main():
    parser = argparse.ArgumentParser(description='PTZ Bundle Adjustment')
    parser.add_argument('--features_dir', required=True, help='Directory with feature files')
    parser.add_argument('--matches_file', required=True, help='Matches JSON file')
    parser.add_argument('--cameras_in', required=True, help='Input cameras JSON file')
    parser.add_argument('--cameras_out', required=True, help='Output cameras JSON file')
    parser.add_argument('--max_iter', type=int, default=100, help='Max BA iterations')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level),
                       format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Load data
    logging.info(f"Loading features from {args.features_dir}")
    features = load_features(args.features_dir)
    if not features:
        logging.error("Failed to load features")
        return 1
    
    logging.info(f"Loading matches from {args.matches_file}")
    matches_info = load_matches(args.matches_file)
    if not matches_info:
        logging.error("Failed to load matches")
        return 1
    
    logging.info(f"Loading cameras from {args.cameras_in}")
    cameras = load_cameras(args.cameras_in)
    if not cameras:
        logging.error("Failed to load cameras")
        return 1
    
    if len(features) != len(cameras):
        logging.error(f"Feature count ({len(features)}) != camera count ({len(cameras)})")
        return 1
    
    logging.info(f"Loaded {len(features)} images, {len(matches_info)} match pairs")
    
    # Run incremental BA
    logging.info("Starting incremental PTZ bundle adjustment...")
    optimizer = PtzIncrementalOptimizer(features, matches_info, cameras, args.max_iter)
    
    cameras_out = [cam.clone() for cam in cameras]
    success, reg_ids = optimizer.solve(cameras_out)
    
    if not success:
        logging.error("Optimization failed")
        return 1
    
    logging.info(f"Successfully optimized {len(reg_ids)}/{len(cameras)} cameras")
    
    # Save results
    logging.info(f"Saving cameras to {args.cameras_out}")
    if not save_cameras(args.cameras_out, cameras_out):
        logging.error("Failed to save cameras")
        return 1
    
    # Save registration status
    status_file = Path(args.cameras_out).parent / "registration_status.json"
    with open(status_file, 'w') as f:
        json.dump({
            'registered_ids': sorted(list(reg_ids)),
            'total_images': len(cameras),
            'registered_count': len(reg_ids)
        }, f, indent=2)
    
    logging.info(f"Done. Registration status: {status_file}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
