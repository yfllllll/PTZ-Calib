#!/usr/bin/env python3
"""Prepare PTZ-Calib feature and match files from an image folder.

This script uses the optional gmberton/vismatch matcher package to generate the
COLMAP-style text files consumed by run_ptz_ba.py:

  image.jpg.txt
  pairs_matches.txt

Only images are required. 2D-3D annotation is still optional and must be
provided separately if georeferencing is needed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _to_numpy(value):
    """Convert torch/jax/numpy/list values to a numpy array when possible."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _first_array(result: dict, names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in result and result[name] is not None:
            arr = _to_numpy(result[name])
            if arr is not None and arr.size > 0:
                return arr
    return None


def _normalize_keypoints(kpts) -> Optional[np.ndarray]:
    arr = _to_numpy(kpts)
    if arr is None or arr.size == 0:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 2:
        return None
    return arr[:, :2].copy()


def _normalize_descriptors(desc, num_keypoints: int) -> Optional[np.ndarray]:
    arr = _to_numpy(desc)
    if arr is None or arr.size == 0:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(num_keypoints, -1)
    elif arr.shape[0] != num_keypoints and arr.shape[-1] == num_keypoints:
        arr = arr.T
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] != num_keypoints:
        return None
    return arr.copy()


def _loaded_size(image_obj, fallback: Tuple[int, int]) -> Tuple[int, int]:
    arr = _to_numpy(image_obj)
    if arr is None or arr.ndim < 2:
        return fallback
    shape = arr.shape
    if len(shape) == 4:
        shape = shape[-3:]
    if len(shape) == 3:
        # CHW tensors are common in matching libraries; HWC also appears.
        if shape[0] in (1, 3):
            h, w = int(shape[1]), int(shape[2])
        else:
            h, w = int(shape[0]), int(shape[1])
    else:
        h, w = int(shape[0]), int(shape[1])
    return w, h


def _scale_keypoints(kpts: Optional[np.ndarray], src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if kpts is None:
        return None
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0:
        return kpts
    scaled = kpts.copy()
    scaled[:, 0] *= float(dst_w) / float(src_w)
    scaled[:, 1] *= float(dst_h) / float(src_h)
    return scaled


@dataclass
class FeatureBank:
    keypoints: List[np.ndarray] = field(default_factory=list)
    descriptors: List[Optional[np.ndarray]] = field(default_factory=list)
    desc_dim: int = 0

    def add_or_map(
        self,
        kpts: np.ndarray,
        desc: Optional[np.ndarray] = None,
        tol: float = 1.0,
    ) -> np.ndarray:
        mapping = np.full(len(kpts), -1, dtype=np.int64)
        for i, pt in enumerate(kpts):
            existing = self._find_existing(pt, tol)
            if existing is None:
                existing = len(self.keypoints)
                self.keypoints.append(np.asarray(pt, dtype=np.float64).reshape(2))
                if desc is not None and i < len(desc):
                    d = np.asarray(desc[i], dtype=np.float32).reshape(-1)
                    self.desc_dim = max(self.desc_dim, int(d.size))
                    self.descriptors.append(d)
                else:
                    self.descriptors.append(None)
            mapping[i] = existing
        return mapping

    def map_points(self, pts: np.ndarray, tol: float = 2.0) -> np.ndarray:
        mapping = np.full(len(pts), -1, dtype=np.int64)
        for i, pt in enumerate(pts):
            existing = self._find_existing(pt, tol)
            if existing is None:
                existing = len(self.keypoints)
                self.keypoints.append(np.asarray(pt, dtype=np.float64).reshape(2))
                self.descriptors.append(None)
            mapping[i] = existing
        return mapping

    def _find_existing(self, pt: np.ndarray, tol: float) -> Optional[int]:
        if not self.keypoints:
            return None
        pts = np.asarray(self.keypoints, dtype=np.float64)
        d2 = np.sum((pts - pt.reshape(1, 2)) ** 2, axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] <= tol * tol:
            return idx
        return None

    def write(self, filepath: str) -> None:
        desc_dim = max(self.desc_dim, 1)
        with open(filepath, "w") as f:
            f.write(f"{len(self.keypoints)} {desc_dim}\n")
            for i, pt in enumerate(self.keypoints):
                desc = self.descriptors[i] if i < len(self.descriptors) else None
                if desc is None or desc.size == 0:
                    desc_values = np.zeros(desc_dim, dtype=np.float32)
                elif desc.size < desc_dim:
                    desc_values = np.zeros(desc_dim, dtype=np.float32)
                    desc_values[: desc.size] = desc
                else:
                    desc_values = desc[:desc_dim]
                desc_str = " ".join(f"{float(v):.8g}" for v in desc_values)
                f.write(f"{pt[0]:.8f} {pt[1]:.8f} 1.0 0.0 {desc_str}\n")


def list_images(image_dir: str) -> List[str]:
    names = []
    for name in sorted(os.listdir(image_dir), key=_natural_key):
        root, ext = os.path.splitext(name)
        if ext.lower() not in VALID_IMAGE_EXTS:
            continue
        if name == "mask.png":
            continue
        path = os.path.join(image_dir, name)
        if os.path.isfile(path):
            names.append(name)
    return names


def make_pairs(images: List[str], mode: str, window: int, pairs_file: str = "") -> List[Tuple[str, str]]:
    if mode == "all":
        return list(itertools.combinations(images, 2))
    if mode == "window":
        pairs = []
        for i, name0 in enumerate(images):
            for j in range(i + 1, min(len(images), i + window + 1)):
                pairs.append((name0, images[j]))
        return pairs
    if mode == "file":
        pairs = []
        with open(pairs_file, "r") as f:
            for raw in f:
                parts = raw.split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        return pairs
    raise ValueError(f"Unknown pair mode: {mode}")


def load_matcher(name: str, device: str):
    try:
        from vismatch import get_matcher
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import vismatch. Install it with:\n"
            "  pip install git+https://github.com/gmberton/vismatch.git"
        ) from exc
    resolved_device = _resolve_device(device)
    try:
        return get_matcher(name, device=resolved_device)
    except TypeError:
        logging.warning("vismatch.get_matcher() does not accept device=; using its default device")
        return get_matcher(name)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_image_for_matcher(matcher, path: str, resize: int):
    if resize and resize > 0:
        return matcher.load_image(path, resize=[resize, resize])
    return matcher.load_image(path)


def extract_pair_data(result: dict):
    all_kpts0 = _normalize_keypoints(_first_array(result, ("all_kpts0", "keypoints0", "kpts0")))
    all_kpts1 = _normalize_keypoints(_first_array(result, ("all_kpts1", "keypoints1", "kpts1")))
    desc0 = _normalize_descriptors(_first_array(result, ("all_desc0", "descriptors0", "desc0")), len(all_kpts0)) if all_kpts0 is not None else None
    desc1 = _normalize_descriptors(_first_array(result, ("all_desc1", "descriptors1", "desc1")), len(all_kpts1)) if all_kpts1 is not None else None

    matched0 = _normalize_keypoints(_first_array(result, ("inlier_kpts0", "matched_kpts0", "mkpts0", "mkeypoints0")))
    matched1 = _normalize_keypoints(_first_array(result, ("inlier_kpts1", "matched_kpts1", "mkpts1", "mkeypoints1")))
    matches = _first_array(result, ("matches", "matches01", "matched_indices"))
    matches0 = _first_array(result, ("matches0",))

    inliers = _first_array(result, ("inliers", "inlier_mask", "inliers_mask"))
    if inliers is not None:
        inliers = np.asarray(inliers).reshape(-1).astype(bool)

    if matches is not None:
        matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
        matches = matches[(matches[:, 0] >= 0) & (matches[:, 1] >= 0)]
        if inliers is not None and len(inliers) == len(matches):
            matches = matches[inliers]
    elif matches0 is not None:
        matches0 = np.asarray(matches0, dtype=np.int64).reshape(-1)
        idx0 = np.where(matches0 >= 0)[0]
        idx1 = matches0[idx0]
        matches = np.stack([idx0, idx1], axis=1)
        if inliers is not None and len(inliers) == len(matches):
            matches = matches[inliers]

    if matches is None and matched0 is not None and matched1 is not None:
        n = min(len(matched0), len(matched1))
        matched0 = matched0[:n]
        matched1 = matched1[:n]

    if all_kpts0 is None and matched0 is not None:
        all_kpts0 = matched0.copy()
        desc0 = None
        matches = np.stack([np.arange(len(matched0)), np.arange(len(matched0))], axis=1)
    if all_kpts1 is None and matched1 is not None:
        all_kpts1 = matched1.copy()
        desc1 = None

    return all_kpts0, desc0, all_kpts1, desc1, matches, matched0, matched1


def run(args) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    images = list_images(args.images)
    if len(images) < 2:
        raise RuntimeError("Need at least two images")
    if args.pairing == "file" and not args.pairs_file:
        raise RuntimeError("--pairs_file is required when --pairing=file")

    os.makedirs(args.output, exist_ok=True)
    pairs = make_pairs(images, args.pairing, args.window, args.pairs_file)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    logging.info("Images: %d, pairs to match: %d", len(images), len(pairs))

    matcher = load_matcher(args.matcher, args.device)
    banks: Dict[str, FeatureBank] = {name: FeatureBank() for name in images}
    valid_pairs = []
    summary = {"images": images, "pairs": []}

    for pair_id, (name0, name1) in enumerate(pairs, start=1):
        path0 = os.path.join(args.images, name0)
        path1 = os.path.join(args.images, name1)
        original0 = cv2.imread(path0)
        original1 = cv2.imread(path1)
        if original0 is None or original1 is None:
            logging.warning("Skip unreadable pair: %s %s", name0, name1)
            continue
        orig_size0 = (original0.shape[1], original0.shape[0])
        orig_size1 = (original1.shape[1], original1.shape[0])

        img0 = load_image_for_matcher(matcher, path0, args.resize)
        img1 = load_image_for_matcher(matcher, path1, args.resize)
        load_size0 = _loaded_size(img0, (args.resize, args.resize) if args.resize > 0 else orig_size0)
        load_size1 = _loaded_size(img1, (args.resize, args.resize) if args.resize > 0 else orig_size1)

        result = matcher(img0, img1)
        all_kpts0, desc0, all_kpts1, desc1, matches, matched0, matched1 = extract_pair_data(result)
        if all_kpts0 is None or all_kpts1 is None:
            logging.warning("Skip pair without keypoints: %s %s", name0, name1)
            continue

        all_kpts0 = _scale_keypoints(all_kpts0, load_size0, orig_size0)
        all_kpts1 = _scale_keypoints(all_kpts1, load_size1, orig_size1)
        map0 = banks[name0].add_or_map(all_kpts0, desc0, tol=args.feature_tol)
        map1 = banks[name1].add_or_map(all_kpts1, desc1, tol=args.feature_tol)

        if matches is not None:
            pair_matches = []
            for idx0, idx1 in matches.astype(np.int64):
                if 0 <= idx0 < len(map0) and 0 <= idx1 < len(map1):
                    pair_matches.append((int(map0[idx0]), int(map1[idx1])))
        elif matched0 is not None and matched1 is not None:
            matched0 = _scale_keypoints(matched0, load_size0, orig_size0)
            matched1 = _scale_keypoints(matched1, load_size1, orig_size1)
            m0 = banks[name0].map_points(matched0, tol=args.match_tol)
            m1 = banks[name1].map_points(matched1, tol=args.match_tol)
            pair_matches = [(int(a), int(b)) for a, b in zip(m0, m1) if a >= 0 and b >= 0]
        else:
            pair_matches = []

        # Remove duplicate index pairs while preserving order.
        seen = set()
        unique_matches = []
        for m in pair_matches:
            if m not in seen:
                seen.add(m)
                unique_matches.append(m)

        if len(unique_matches) >= args.min_matches:
            valid_pairs.append((name0, name1, unique_matches))
            logging.info("[%d/%d] %s %s: %d matches", pair_id, len(pairs), name0, name1, len(unique_matches))
        else:
            logging.info("[%d/%d] %s %s: skipped (%d matches)", pair_id, len(pairs), name0, name1, len(unique_matches))

        summary["pairs"].append({"image0": name0, "image1": name1, "matches": len(unique_matches)})

    for name, bank in banks.items():
        bank.write(os.path.join(args.output, name + ".txt"))

    matches_path = os.path.join(args.output, "pairs_matches.txt")
    with open(matches_path, "w") as f:
        for name0, name1, matches in valid_pairs:
            f.write(f"{name0} {name1}\n")
            for idx0, idx1 in matches:
                f.write(f"{idx0} {idx1}\n")
            f.write("\n")

    with open(os.path.join(args.output, "prepare_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Wrote %d image feature files and %d matched pairs to %s", len(images), len(valid_pairs), args.output)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Prepare PTZ-Calib input files using vismatch")
    parser.add_argument("--images", required=True, help="Input image directory")
    parser.add_argument("--output", required=True, help="Output feature/match directory")
    parser.add_argument("--matcher", default="superpoint-lightglue", help="vismatch matcher name")
    parser.add_argument("--device", default="auto", help="Matcher device: auto, cuda, or cpu")
    parser.add_argument("--resize", type=int, default=1024, help="Square resize for matching; <=0 keeps original size")
    parser.add_argument("--pairing", choices=["all", "window", "file"], default="window", help="Image-pair generation mode")
    parser.add_argument("--window", type=int, default=5, help="Sliding window size for --pairing window")
    parser.add_argument("--pairs_file", default="", help="Text file with explicit image pairs for --pairing file")
    parser.add_argument("--min_matches", type=int, default=15, help="Drop image pairs with fewer matches")
    parser.add_argument("--max_pairs", type=int, default=0, help="Debug limit; 0 means no limit")
    parser.add_argument("--feature_tol", type=float, default=1.0, help="Pixel tolerance for merging repeated extracted keypoints")
    parser.add_argument("--match_tol", type=float, default=2.0, help="Pixel tolerance for mapping matched coordinates to keypoints")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
