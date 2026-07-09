"""I/O utilities: Colmap features/matches, JSON camera params, annotations.

Python port of src/core/data_io.h and src/core/data_io.cc.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from .types import Camera, ImageFeatures, MatchesInfo

logger = logging.getLogger(__name__)

VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# --------------------------- helpers ---------------------------

def _splitext(fname: str) -> Tuple[str, str]:
    """Return (rootname, ext) mimicking C++ splitext (ext includes the dot).

    Note: C++ splitext uses only the last '.', which matches os.path.splitext.
    """
    return os.path.splitext(fname)


def _basename(path: str) -> str:
    return os.path.basename(path)


def _listdir_sorted(dir_path: str) -> List[str]:
    """Return absolute file paths in dir_path, sorted by filename."""
    if not os.path.isdir(dir_path):
        return []
    names = sorted(os.listdir(dir_path))
    return [os.path.join(dir_path, n) for n in names]


def _has_ending(s: str, ending: str) -> bool:
    return s.lower().endswith(ending.lower())


def find_img_index(fnames: List[str], fname: str) -> int:
    """Find image index by name (ignoring extension). Returns -1 if not found."""
    target_root, _ = _splitext(fname)
    for i, name in enumerate(fnames):
        root_i, _ = _splitext(name)
        if root_i == target_root:
            return i
    return -1


# --------------------------- Colmap features & matches ---------------------------

def read_colmap_features(filepath: str) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Read Colmap-format text features.

    Format:
        num_kpts desc_dim
        x y scale orientation d0 d1 ... d(D-1)
        ...
    """
    if not os.path.isfile(filepath):
        return [], None


    try:
        with open(filepath, "r") as f:
            first = f.readline().split()
            if len(first) < 2:
                return [], None
            num_kpts = int(first[0])
            desc_dim = int(first[1])

            kpts: List[cv2.KeyPoint] = []
            desc = np.zeros((num_kpts, desc_dim), dtype=np.float32)

            for i in range(num_kpts):
                parts = f.readline().split()
                if len(parts) < 4 + desc_dim:
                    logger.debug("Malformed feature line %d in %s", i, filepath)
                    return [], None
                x = float(parts[0])
                y = float(parts[1])
                scale = float(parts[2])
                _orientation = float(parts[3])
                kpts.append(cv2.KeyPoint(x=x, y=y, size=scale))
                desc[i, :] = [float(v) for v in parts[4 : 4 + desc_dim]]
        return kpts, desc
    except Exception as e:
        logger.debug("Cannot read colmap features from %s: %s", filepath, e)
        return [], None


ReadColmapFeatures = read_colmap_features


def read_colmap_matches(
    filepath: str,
) -> Tuple[List[List[cv2.DMatch]], List[Tuple[str, str]]]:
    """Read Colmap-format text matches. Returns (pairs_matches, img_pairs_name)."""
    pairs_matches: List[List[cv2.DMatch]] = []
    img_pairs_name: List[Tuple[str, str]] = []

    if not os.path.isfile(filepath):
        return pairs_matches, img_pairs_name

    matches: List[cv2.DMatch] = []
    img_pair: Tuple[str, str] = ("", "")

    try:
        with open(filepath, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    if matches:
                        pairs_matches.append(matches)
                        img_pairs_name.append(img_pair)
                        matches = []
                        img_pair = ("", "")
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue
                s1, s2 = parts[0], parts[1]

                if any(_has_ending(s1, e) for e in (".png", ".jpg", ".jpeg")):
                    img_pair = (s1, s2)
                else:
                    matches.append(cv2.DMatch(int(s1), int(s2), 0.0))

        # Handle case where file does not end with blank line.
        if matches:
            pairs_matches.append(matches)
            img_pairs_name.append(img_pair)
    except Exception as e:
        logger.debug("Cannot read colmap matches from %s: %s", filepath, e)

    return pairs_matches, img_pairs_name


ReadColmapMatches = read_colmap_matches


# --------------------------- JSON I/O ---------------------------

def save_to_json(
    cameras: List[Camera],
    names: List[str],
    pixels_gt: List[List[Tuple[float, float]]],
    pts3d_gt: List[List[Tuple[float, float, float]]],
    filepath: str,
) -> bool:
    """Save cameras + annotations to JSON (compatible with C++ format)."""
    j_all: "OrderedDict[str, OrderedDict[str, OrderedDict]]" = OrderedDict()
    cams_dict: "OrderedDict[str, OrderedDict]" = OrderedDict()

    for i, cam in enumerate(cameras):
        rootname, _ = _splitext(names[i])
        entry: "OrderedDict" = OrderedDict()

        entry["name"] = rootname
        t_wc = cam.t_wc().reshape(-1)
        entry["pos"] = [float(t_wc[0]), float(t_wc[1]), float(t_wc[2])]

        width = int(2 * cam.K()[0, 2])
        height = int(2 * cam.K()[1, 2])
        entry["res"] = [width, height]

        # Row-major flatten to match cv::Mat begin/end iteration.
        entry["K"] = cam.K().reshape(-1).tolist()
        entry["R"] = cam.R().reshape(-1).tolist()
        entry["t"] = cam.t().reshape(-1).tolist()

        dist_flat = cam.dist().reshape(-1).tolist()
        entry["dist"] = dist_flat
        entry["distType"] = "" if abs(dist_flat[0]) < 1e-5 else "k1"

        pix_list: List[List[float]] = []
        pos_list: List[List[float]] = []
        for k in range(len(pixels_gt[i])):
            px = pixels_gt[i][k]
            p3 = pts3d_gt[i][k]
            # Some entries may be tuples, cv2 Point2f, or np.ndarray.
            px_x = float(px[0]) if not hasattr(px, "x") else float(px.x)
            px_y = float(px[1]) if not hasattr(px, "y") else float(px.y)
            p3_x = float(p3[0]) if not hasattr(p3, "x") else float(p3.x)
            p3_y = float(p3[1]) if not hasattr(p3, "y") else float(p3.y)
            p3_z = float(p3[2]) if not hasattr(p3, "z") else float(p3.z)
            pix_list.append([px_x / width, px_y / height])
            pos_list.append([p3_x, p3_y, p3_z])

        entry["marker"] = OrderedDict([("pix", pix_list), ("pos", pos_list)])
        entry["version"] = "2.0"

        cams_dict[rootname] = entry

    j_all["cameras"] = cams_dict

    try:
        with open(filepath, "w") as f:
            json.dump(j_all, f, indent=4)
        return True
    except OSError as e:
        logger.error("Cannot write JSON to %s: %s", filepath, e)
        return False


SaveToJson = save_to_json


def _read_json_file(filepath: str) -> dict:
    if not filepath or not os.path.isfile(filepath):
        return {}
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Cannot read JSON %s: %s", filepath, e)
        return {}


def read_from_json(
    filepath: str,
) -> Tuple[
    bool,
    List[Camera],
    List[str],
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    List[Tuple[int, int]],
]:
    """Load full camera+annotation JSON.

    Returns (success, cameras, names, pixels, pts3d, sizes).
    pixels[i] is a list of (2,) np.ndarray, pts3d[i] a list of (3,) np.ndarray.
    """
    cameras: List[Camera] = []
    names: List[str] = []
    pixels: List[List[np.ndarray]] = []
    pts3d: List[List[np.ndarray]] = []
    sizes: List[Tuple[int, int]] = []

    j = _read_json_file(filepath)
    if not j:
        logger.error("JSON file not exists or cannot be opened: %s", filepath)
        return False, cameras, names, pixels, pts3d, sizes

    try:
        j_cameras = j["cameras"]
        for name, value in j_cameras.items():
            cam = Camera()
            cam.K_ = np.asarray(value["K"], dtype=np.float64).reshape(3, 3)
            cam.R_ = np.asarray(value["R"], dtype=np.float64).reshape(3, 3)
            cam.t_ = np.asarray(value["t"], dtype=np.float64).reshape(3, 1)
            cam.dist_ = np.asarray(value["dist"], dtype=np.float64).reshape(5, 1)

            width = int(value["res"][0])
            height = int(value["res"][1])

            pix_list = value.get("marker", {}).get("pix", [])
            pos_list = value.get("marker", {}).get("pos", [])

            pixs: List[np.ndarray] = []
            for p in pix_list:
                pixs.append(np.array([width * p[0], height * p[1]], dtype=np.float64))

            pts: List[np.ndarray] = []
            for p in pos_list:
                pts.append(np.array([p[0], p[1], p[2]], dtype=np.float64))

            names.append(name)
            pixels.append(pixs)
            pts3d.append(pts)
            cameras.append(cam)
            sizes.append((width, height))
        return True, cameras, names, pixels, pts3d, sizes
    except Exception as e:
        logger.error("Exception in read_from_json: %s", e)
        return False, cameras, names, pixels, pts3d, sizes


ReadFromJson = read_from_json


def read_cam_from_json(
    filepath: str, names: List[str]
) -> Tuple[bool, List[Camera]]:
    """Read only camera params from JSON, indexed by rootname order in `names`."""
    cameras: List[Camera] = [Camera() for _ in names]

    j = _read_json_file(filepath)
    if not j:
        logger.error("JSON file not exists or cannot be opened: %s", filepath)
        return False, cameras

    try:
        j_cameras = j["cameras"]
        for i, name in enumerate(names):
            rootname, _ = _splitext(name)
            if rootname in j_cameras:
                v = j_cameras[rootname]
                cam = Camera()
                cam.K_ = np.asarray(v["K"], dtype=np.float64).reshape(3, 3)
                cam.R_ = np.asarray(v["R"], dtype=np.float64).reshape(3, 3)
                cam.t_ = np.asarray(v["t"], dtype=np.float64).reshape(3, 1)
                cam.dist_ = np.asarray(v["dist"], dtype=np.float64).reshape(5, 1)
                cameras[i] = cam
            else:
                logger.error("Cannot find %s in file: %s", rootname, filepath)
                return False, cameras
        return True, cameras
    except Exception as e:
        logger.error("Exception in read_cam_from_json: %s", e)
        return False, cameras


ReadCamFromJson = read_cam_from_json


# --------------------------- image + feature loading ---------------------------

def load_imgs_and_features(
    img_dir: str, feature_dir: str
) -> Tuple[bool, List[str], List[ImageFeatures], List[Tuple[int, int]]]:
    """Enumerate images in `img_dir` and read their pre-extracted features.

    Returns (success, fnames, features, sizes).
    """
    fnames: List[str] = []
    features: List[ImageFeatures] = []
    sizes: List[Tuple[int, int]] = []

    fpaths = _listdir_sorted(img_dir)
    for fpath in fpaths:
        fname = _basename(fpath)
        _root, ext = _splitext(fname)
        if ext.lower() not in VALID_IMG_EXTS:
            continue
        if fname == "mask.png":
            continue

        image = cv2.imread(fpath)
        if image is None:
            continue

        h, w = image.shape[:2]
        feature = ImageFeatures(img_idx=len(fnames), img_size=(w, h))
        feature_path = os.path.join(feature_dir, fname + ".txt")
        kpts, desc = read_colmap_features(feature_path)
        feature.keypoints = kpts
        feature.descriptors = desc

        logger.info("Index: %d, image: %s", len(fnames), fname)
        fnames.append(fname)
        features.append(feature)
        sizes.append((w, h))

    if len(fnames) < 2:
        logger.error("Images number not enough (< 2): %d", len(fnames))
        return False, fnames, features, sizes

    return True, fnames, features, sizes


LoadImgsAndFeatures = load_imgs_and_features


def _cal_homography(
    kpts1: List[cv2.KeyPoint],
    kpts2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_thresh: float,
) -> Optional[np.ndarray]:
    if len(matches) < 4:
        return None
    ref_pts = np.array([kpts1[m.queryIdx].pt for m in matches], dtype=np.float32)
    src_pts = np.array([kpts2[m.trainIdx].pt for m in matches], dtype=np.float32)
    H, _mask = cv2.findHomography(ref_pts, src_pts, cv2.RANSAC, ransac_thresh)
    return H


def _cal_matching_score(num_matches: int, max_num_matches: int) -> float:
    assert num_matches >= 0 and max_num_matches >= 0
    if num_matches >= max_num_matches:
        return 1.0
    return float(num_matches) / float(max_num_matches)


def load_matches_info(
    matches_path: str,
    fnames: List[str],
    features: List[ImageFeatures],
) -> Tuple[bool, List[MatchesInfo]]:
    """Load pairwise match info and estimate homographies.

    Returns (success, matches_info) where matches_info is a flat list of
    length num_images*num_images (row-major), only populated cells contain
    non-default entries.
    """
    assert len(fnames) == len(features)

    pairs_matches, img_pairs_name = read_colmap_matches(matches_path)
    num_images = len(fnames)
    matches_info: List[MatchesInfo] = [MatchesInfo() for _ in range(num_images * num_images)]

    RANSAC_THRESH = 4.0
    MAX_NUM_MATCHES = 100

    for i, pair in enumerate(pairs_matches):
        s1, s2 = img_pairs_name[i]
        idx_i = find_img_index(fnames, s1)
        idx_j = find_img_index(fnames, s2)
        if idx_i < 0 or idx_j < 0:
            continue

        H = _cal_homography(features[idx_i].keypoints, features[idx_j].keypoints, pair, RANSAC_THRESH)

        mi = MatchesInfo()
        mi.matches = pair
        mi.H = H
        mi.inliers_mask = [1] * len(pair)
        mi.num_inliers = len(pair)
        mi.confidence = _cal_matching_score(len(pair), MAX_NUM_MATCHES)
        mi.src_img_idx = idx_i
        mi.dst_img_idx = idx_j
        matches_info[idx_i * num_images + idx_j] = mi

    return True, matches_info


LoadMatchesInfo = load_matches_info


def load_annotation(
    annot_path: str, fnames: List[str]
) -> Tuple[bool, List[List[np.ndarray]], List[List[np.ndarray]]]:
    """Load 2D-3D annotations aligned to `fnames`."""
    num_images = len(fnames)
    pixels: List[List[np.ndarray]] = [[] for _ in range(num_images)]
    pts3d: List[List[np.ndarray]] = [[] for _ in range(num_images)]

    ok, _gt_cams, gt_names, gt_pixels, gt_pts3d, _gt_sizes = read_from_json(annot_path)
    if not ok:
        return False, pixels, pts3d

    for i in range(len(gt_names)):
        idx = find_img_index(fnames, gt_names[i])
        if idx == -1:
            continue
        pixels[idx] = gt_pixels[i]
        pts3d[idx] = gt_pts3d[i]

    return True, pixels, pts3d


LoadAnnotation = load_annotation


def save_registered_cam(
    cameras: List[Camera],
    reg_image_ids: Set[int],
    fnames: List[str],
    pixels: List[List[np.ndarray]],
    pts3d: List[List[np.ndarray]],
    out_path: str,
) -> bool:
    """Save only the successfully-registered cameras to JSON."""
    cameras_reg: List[Camera] = []
    names_reg: List[str] = []
    pixels_reg: List[List[np.ndarray]] = []
    pts3d_reg: List[List[np.ndarray]] = []

    for i, cam in enumerate(cameras):
        if i not in reg_image_ids:
            logger.info("Filter out failed image #%d: %s", i, fnames[i])
            continue
        cameras_reg.append(cam)
        names_reg.append(fnames[i])
        pixels_reg.append(pixels[i] if i < len(pixels) else [])
        pts3d_reg.append(pts3d[i] if i < len(pts3d) else [])

    return save_to_json(cameras_reg, names_reg, pixels_reg, pts3d_reg, out_path)


SaveRegisteredCam = save_registered_cam


def mkdir_ifnot_exist(path: str) -> None:
    os.makedirs(path, exist_ok=True)
