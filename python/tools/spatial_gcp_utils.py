#!/usr/bin/env python3
"""Shared helpers for spatial GCP annotation tools."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
WEB_MERCATOR_HALF_WORLD = 20037508.342789244
TILE_SIZE = 256
TILE_TEMPLATES = {
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "google_satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "google_roadmap": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
    "amap_satellite": "https://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
    "amap_roadmap": "https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
}


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def list_images(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    return sorted([p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=lambda p: natural_key(p.name))


def parse_csv_floats(value: str, expected: int, name: str) -> List[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(vals) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated values, got {len(vals)}")
    return vals


def parse_k(value: Optional[str], width: int, height: int) -> List[float]:
    if value:
        return parse_csv_floats(value, 9, "K")
    f = float(max(width, height))
    return [f, 0.0, width / 2.0, 0.0, f, height / 2.0, 0.0, 0.0, 1.0]


def parse_dist(value: Optional[str]) -> List[float]:
    if value:
        return parse_csv_floats(value, 5, "dist")
    return [0.0, 0.0, 0.0, 0.0, 0.0]


@dataclass
class OrthoGrid:
    """North-up raster coordinate transform.

    extent is xmin, ymin, xmax, ymax in the target world coordinate system.
    Pixel (0, 0) maps to (xmin, ymax), which matches common north-up imagery.
    """

    width: int
    height: int
    extent: Tuple[float, float, float, float]
    dsm: Optional[np.ndarray] = None
    dsm_extent: Optional[Tuple[float, float, float, float]] = None
    default_z: float = 0.0

    @classmethod
    def from_image(
        cls,
        ortho_path: str,
        extent: Tuple[float, float, float, float],
        dsm_path: str = "",
        dsm_extent: Optional[Tuple[float, float, float, float]] = None,
        default_z: float = 0.0,
    ) -> "OrthoGrid":
        img = cv2.imread(ortho_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read ortho image: {ortho_path}")
        dsm = np.load(dsm_path) if dsm_path else None
        return cls(img.shape[1], img.shape[0], extent, dsm, dsm_extent or extent, default_z)

    @classmethod
    def from_array(
        cls,
        image: np.ndarray,
        extent: Tuple[float, float, float, float],
        dsm_path: str = "",
        dsm_extent: Optional[Tuple[float, float, float, float]] = None,
        default_z: float = 0.0,
    ) -> "OrthoGrid":
        if image is None or image.size == 0:
            raise ValueError("Empty orthophoto/map image")
        dsm = np.load(dsm_path) if dsm_path else None
        return cls(image.shape[1], image.shape[0], extent, dsm, dsm_extent or extent, default_z)

    def pixel_to_xy(self, pixel: Sequence[float], extent: Optional[Tuple[float, float, float, float]] = None, shape=None) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = extent or self.extent
        if shape is None:
            width, height = self.width, self.height
        else:
            height, width = shape[:2]
        u, v = float(pixel[0]), float(pixel[1])
        denom_x = max(width - 1, 1)
        denom_y = max(height - 1, 1)
        x = xmin + (u / denom_x) * (xmax - xmin)
        y = ymax - (v / denom_y) * (ymax - ymin)
        return x, y

    def xy_to_pixel(self, xy: Sequence[float], extent: Optional[Tuple[float, float, float, float]] = None, shape=None) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = extent or self.extent
        if shape is None:
            width, height = self.width, self.height
        else:
            height, width = shape[:2]
        x, y = float(xy[0]), float(xy[1])
        denom_x = xmax - xmin if xmax != xmin else 1.0
        denom_y = ymax - ymin if ymax != ymin else 1.0
        u = (x - xmin) / denom_x * max(width - 1, 1)
        v = (ymax - y) / denom_y * max(height - 1, 1)
        return u, v

    def sample_z(self, xy: Sequence[float]) -> float:
        if self.dsm is None:
            return float(self.default_z)
        u, v = self.xy_to_pixel(xy, self.dsm_extent, self.dsm.shape)
        col = int(round(u))
        row = int(round(v))
        if row < 0 or col < 0 or row >= self.dsm.shape[0] or col >= self.dsm.shape[1]:
            return float(self.default_z)
        z = float(self.dsm[row, col])
        if not np.isfinite(z):
            return float(self.default_z)
        return z

    def pixel_to_world(self, pixel: Sequence[float]) -> Tuple[float, float, float]:
        x, y = self.pixel_to_xy(pixel)
        return x, y, self.sample_z((x, y))


def make_project(
    target_type: str,
    target_image: str,
    target_size: Sequence[int],
    coordinate_system: str = "",
) -> OrderedDict:
    return OrderedDict(
        [
            ("version", "1.0"),
            ("type", "spatial_gcp_project"),
            ("target_type", target_type),
            ("target_image", target_image),
            ("target_size", [int(target_size[0]), int(target_size[1])]),
            ("coordinate_system", coordinate_system),
            ("points", []),
        ]
    )


def load_project(path: str) -> OrderedDict:
    with open(path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def save_project(project: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(project, f, indent=2)


def add_project_point(project: dict, target_pixel: Sequence[float], world: Sequence[float], source: str, note: str = "") -> None:
    point_id = len(project.setdefault("points", [])) + 1
    project["points"].append(
        OrderedDict(
            [
                ("id", point_id),
                ("target_pixel", [float(target_pixel[0]), float(target_pixel[1])]),
                ("world", [float(world[0]), float(world[1]), float(world[2])]),
                ("source", source),
                ("note", note),
            ]
        )
    )


def image_point_to_panorama(point: Sequence[float], H_image_to_pano: Sequence[Sequence[float]]) -> Tuple[float, float]:
    H = np.asarray(H_image_to_pano, dtype=np.float64).reshape(3, 3)
    p = H @ np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-12:
        raise ValueError("Degenerate homography projection")
    return float(p[0] / p[2]), float(p[1] / p[2])


def panorama_point_to_image(point: Sequence[float], H_image_to_pano: Sequence[Sequence[float]]) -> Tuple[float, float]:
    H = np.asarray(H_image_to_pano, dtype=np.float64).reshape(3, 3)
    Hinv = np.linalg.inv(H)
    p = Hinv @ np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-12:
        raise ValueError("Degenerate homography projection")
    return float(p[0] / p[2]), float(p[1] / p[2])


def export_annotation(
    image_points: Dict[str, List[Tuple[Sequence[float], Sequence[float]]]],
    images_dir: str,
    output_path: str,
    K: Optional[Sequence[float]] = None,
    dist: Optional[Sequence[float]] = None,
) -> None:
    image_paths = {p.name: p for p in list_images(images_dir)}
    cameras = OrderedDict()
    for image_name in sorted(image_points.keys(), key=natural_key):
        if image_name not in image_paths:
            continue
        img = cv2.imread(str(image_paths[image_name]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        root = Path(image_name).stem
        pix = []
        pos = []
        for point2d, point3d in image_points[image_name]:
            x, y = float(point2d[0]), float(point2d[1])
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            pix.append([x / w, y / h])
            pos.append([float(point3d[0]), float(point3d[1]), float(point3d[2])])
        if not pix:
            continue
        cameras[root] = OrderedDict(
            [
                ("K", list(K) if K is not None else parse_k(None, w, h)),
                ("R", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                ("t", [0.0, 0.0, 0.0]),
                ("dist", list(dist) if dist is not None else parse_dist(None)),
                ("res", [w, h]),
                ("marker", OrderedDict([("pix", pix), ("pos", pos)])),
            ]
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(OrderedDict([("cameras", cameras)]), f, indent=2)


def load_homography_mappings(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def lonlat_to_webmercator(lon: float, lat: float) -> Tuple[float, float]:
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    lon = float(lon)
    x = lon * WEB_MERCATOR_HALF_WORLD / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) * WEB_MERCATOR_HALF_WORLD / np.pi
    return float(x), float(y)


def webmercator_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    lon = float(x) / WEB_MERCATOR_HALF_WORLD * 180.0
    lat = float(y) / WEB_MERCATOR_HALF_WORLD * 180.0
    lat = 180.0 / np.pi * (2.0 * np.arctan(np.exp(lat * np.pi / 180.0)) - np.pi / 2.0)
    return float(lon), float(lat)


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    n = 2**int(zoom)
    x = int(np.floor((lon + 180.0) / 360.0 * n))
    lat_rad = np.radians(lat)
    y = int(np.floor((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n))
    return int(np.clip(x, 0, n - 1)), int(np.clip(y, 0, n - 1))


def tile_bounds_webmercator(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    n = 2**int(zoom)
    tile_span = 2.0 * WEB_MERCATOR_HALF_WORLD / n
    xmin = -WEB_MERCATOR_HALF_WORLD + x * tile_span
    xmax = xmin + tile_span
    ymax = WEB_MERCATOR_HALF_WORLD - y * tile_span
    ymin = ymax - tile_span
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _tile_url(source: str, template: str, x: int, y: int, z: int, key: str) -> str:
    url_template = template or TILE_TEMPLATES.get(source)
    if not url_template:
        raise ValueError(f"Unknown map source '{source}'. Use --map_template for custom tiles.")
    return url_template.format(x=x, y=y, z=z, key=key)


def _read_tile_from_cache(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _download_tile(url: str, cache_path: Path, timeout: float) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "PTZ-Calib spatial GCP annotator"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Downloaded tile is not an image: {url}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def load_web_map(
    source: str,
    template: str,
    bbox_lonlat: Tuple[float, float, float, float],
    zoom: int,
    cache_dir: str,
    key: str = "",
    timeout: float = 10.0,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Load online XYZ map tiles and return a cropped BGR image plus EPSG:3857 extent."""

    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("--map_bbox must be lon_min,lat_min,lon_max,lat_max")
    x0, y0 = lonlat_to_tile(lon_min, lat_max, zoom)
    x1, y1 = lonlat_to_tile(lon_max, lat_min, zoom)
    xmin_tile, xmax_tile = sorted((x0, x1))
    ymin_tile, ymax_tile = sorted((y0, y1))
    rows = []
    cache_root = Path(cache_dir).expanduser()
    source_slug = source or "custom"
    for ty in range(ymin_tile, ymax_tile + 1):
        row = []
        for tx in range(xmin_tile, xmax_tile + 1):
            cache_path = cache_root / source_slug / str(zoom) / str(tx) / f"{ty}.tile"
            tile = _read_tile_from_cache(cache_path)
            if tile is None:
                tile = _download_tile(_tile_url(source, template, tx, ty, zoom, key), cache_path, timeout)
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                tile = cv2.resize(tile, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
            row.append(tile)
        rows.append(np.hstack(row))
    mosaic = np.vstack(rows)

    tile_extent_min = tile_bounds_webmercator(xmin_tile, ymax_tile, zoom)
    tile_extent_max = tile_bounds_webmercator(xmax_tile, ymin_tile, zoom)
    mosaic_extent = (tile_extent_min[0], tile_extent_min[1], tile_extent_max[2], tile_extent_max[3])
    bbox_xmin, bbox_ymin = lonlat_to_webmercator(lon_min, lat_min)
    bbox_xmax, bbox_ymax = lonlat_to_webmercator(lon_max, lat_max)
    crop_extent = (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        xmin, ymin, xmax, ymax = mosaic_extent
        u = int(round((x - xmin) / (xmax - xmin) * mosaic.shape[1]))
        v = int(round((ymax - y) / (ymax - ymin) * mosaic.shape[0]))
        return u, v

    left, top = to_px(bbox_xmin, bbox_ymax)
    right, bottom = to_px(bbox_xmax, bbox_ymin)
    left = int(np.clip(left, 0, mosaic.shape[1] - 1))
    right = int(np.clip(right, left + 1, mosaic.shape[1]))
    top = int(np.clip(top, 0, mosaic.shape[0] - 1))
    bottom = int(np.clip(bottom, top + 1, mosaic.shape[0]))
    return mosaic[top:bottom, left:right].copy(), crop_extent


def ensure_same_dir_imports():
    import sys

    tools_dir = str(Path(__file__).resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
