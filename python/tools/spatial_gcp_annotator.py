#!/usr/bin/env python3
"""Unified spatial GCP annotator for panorama or single-image workflows.

The tool stores annotations in an intermediate project JSON:

  target pixel on panorama/single image <-> world XYZ

World XYZ can be entered manually, picked from a local orthophoto (+ optional
DSM), or picked from an Open3D mesh.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from spatial_gcp_utils import (
    OrthoGrid,
    add_project_point,
    export_annotation,
    load_web_map,
    load_project,
    make_project,
    parse_csv_floats,
    parse_dist,
    parse_k,
    save_project,
)

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SpatialGCPAnnotator:
    def __init__(self, args):
        self.args = args
        self.target_path = Path(args.target)
        self.target = cv2.imread(str(self.target_path), cv2.IMREAD_COLOR)
        if self.target is None:
            raise ValueError(f"Cannot read target image: {self.target_path}")
        self.target_h, self.target_w = self.target.shape[:2]

        if args.load:
            self.project = load_project(args.load)
        else:
            self.project = make_project(
                args.target_type,
                str(self.target_path),
                (self.target_w, self.target_h),
                args.coordinate_system,
            )

        self.ortho = None
        self.ortho_img = None
        self.map_source_label = ""
        if args.ortho:
            if not args.ortho_extent:
                raise ValueError("--ortho_extent is required with --ortho")
            extent = tuple(parse_csv_floats(args.ortho_extent, 4, "ortho_extent"))
            dsm_extent = tuple(parse_csv_floats(args.dsm_extent, 4, "dsm_extent")) if args.dsm_extent else extent
            self.ortho = OrthoGrid.from_image(args.ortho, extent, args.dsm or "", dsm_extent, args.z)
            self.ortho_img = cv2.imread(args.ortho, cv2.IMREAD_COLOR)
            if self.ortho_img is None:
                raise ValueError(f"Cannot read ortho image: {args.ortho}")
            self.map_source_label = "ortho_dsm"
        elif args.map_source or args.map_template:
            if not args.map_bbox:
                raise ValueError("--map_bbox is required with --map_source or --map_template")
            bbox = tuple(parse_csv_floats(args.map_bbox, 4, "map_bbox"))
            map_img, extent = load_web_map(
                args.map_source,
                args.map_template,
                bbox,
                args.map_zoom,
                args.map_cache,
                args.map_key,
                args.map_timeout,
            )
            dsm_extent = tuple(parse_csv_floats(args.dsm_extent, 4, "dsm_extent")) if args.dsm_extent else extent
            self.ortho = OrthoGrid.from_array(map_img, extent, args.dsm or "", dsm_extent, args.z)
            self.ortho_img = map_img
            self.map_source_label = f"web_map:{args.map_source or 'custom'}"
            if not args.coordinate_system:
                self.project["coordinate_system"] = "EPSG:3857"

        self.mesh = None
        if args.model:
            if not HAS_OPEN3D:
                raise RuntimeError("Open3D is required for --model. Install with: pip install open3d")
            self.mesh = self._load_mesh(args.model)

        self.pending_target = None
        self.window_target = "Target image / panorama"
        self.window_ortho = "Orthophoto / map source"

    def _load_mesh(self, path: str):
        mesh_path = Path(path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.has_triangles():
            raise ValueError(f"No triangles in mesh: {mesh_path}")
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        logger.info("Loaded mesh: %s (%d vertices)", mesh_path, len(mesh.vertices))
        return mesh

    def run(self):
        self._print_help()
        cv2.namedWindow(self.window_target, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_target, self._on_target_click)
        cv2.resizeWindow(self.window_target, min(self.target_w, 1200), min(self.target_h, 800))
        if self.ortho_img is not None:
            cv2.namedWindow(self.window_ortho, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_ortho, self._on_ortho_click)
            cv2.resizeWindow(self.window_ortho, min(self.ortho_img.shape[1], 1200), min(self.ortho_img.shape[0], 800))

        while True:
            cv2.imshow(self.window_target, self._render_target())
            if self.ortho_img is not None:
                cv2.imshow(self.window_ortho, self._render_ortho())
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                self._save()
                break
            if key == ord("s"):
                self._save()
            elif key == ord("u"):
                self._undo()
            elif key == ord("m"):
                self._add_manual_world()
            elif key == ord("3"):
                self._add_model_world()
            elif key == ord("e"):
                self._export_single_image_annotation()
            elif key == ord("h"):
                self._print_help()
        cv2.destroyAllWindows()

    def _print_help(self):
        logger.info("")
        logger.info("Spatial GCP Annotator")
        logger.info("  Left-click target image/panorama: choose target pixel")
        logger.info("  Left-click orthophoto/map: assign XY and Z from DSM/default z")
        logger.info("  m: type XYZ manually for pending target point")
        logger.info("  3: pick XYZ from 3D mesh for pending target point")
        logger.info("  u: undo last point")
        logger.info("  s: save project")
        logger.info("  e: export annotation directly (only for --target_type image)")
        logger.info("  q: save and quit")
        logger.info("")

    def _render_target(self):
        canvas = self.target.copy()
        for p in self.project.get("points", []):
            x, y = p["target_pixel"]
            cv2.circle(canvas, (int(round(x)), int(round(y))), 7, (0, 255, 0), 2)
            cv2.putText(canvas, str(p["id"]), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.pending_target is not None:
            x, y = self.pending_target
            cv2.drawMarker(canvas, (int(round(x)), int(round(y))), (0, 165, 255), cv2.MARKER_CROSS, 24, 2)
        status = f"{self.args.target_type} | points: {len(self.project.get('points', []))}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(canvas, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return canvas

    def _render_ortho(self):
        canvas = self.ortho_img.copy()
        if self.ortho is not None:
            for p in self.project.get("points", []):
                u, v = self.ortho.xy_to_pixel(p["world"][:2])
                if 0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]:
                    cv2.circle(canvas, (int(round(u)), int(round(v))), 7, (255, 128, 0), 2)
                    cv2.putText(canvas, str(p["id"]), (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        label = "click to assign world XY/Z to pending target point"
        if self.map_source_label.startswith("web_map"):
            label += " (EPSG:3857)"
        cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return canvas

    def _on_target_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending_target = (float(x), float(y))
            logger.info("Target pixel selected: %.1f %.1f", x, y)

    def _on_ortho_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.pending_target is None:
            logger.warning("Select a target point first")
            return
        if self.ortho is None:
            return
        world = self.ortho.pixel_to_world((x, y))
        add_project_point(self.project, self.pending_target, world, self.map_source_label or "ortho_dsm")
        logger.info("Added point #%d: target=(%.1f, %.1f), world=(%.3f, %.3f, %.3f)", len(self.project["points"]), self.pending_target[0], self.pending_target[1], world[0], world[1], world[2])
        self.pending_target = None

    def _add_manual_world(self):
        if self.pending_target is None:
            logger.warning("Select a target point first")
            return
        raw = input("World XYZ (x y z): ").strip()
        vals = [float(x) for x in raw.replace(",", " ").split()]
        if len(vals) != 3:
            logger.error("Expected exactly 3 values")
            return
        add_project_point(self.project, self.pending_target, vals, "manual")
        logger.info("Added manual point #%d", len(self.project["points"]))
        self.pending_target = None

    def _add_model_world(self):
        if self.pending_target is None:
            logger.warning("Select a target point first")
            return
        if self.mesh is None:
            logger.warning("No --model provided")
            return
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Pick 3D point", width=1280, height=720)
        vis.add_geometry(self.mesh)
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        vis.run()
        picked = vis.get_picked_points()
        vis.destroy_window()
        if not picked:
            logger.warning("No 3D point picked")
            return
        idx = picked[-1].index if hasattr(picked[-1], "index") else int(picked[-1])
        world = np.asarray(self.mesh.vertices)[idx].astype(float)
        add_project_point(self.project, self.pending_target, world, "model")
        logger.info("Added model point #%d", len(self.project["points"]))
        self.pending_target = None

    def _undo(self):
        if self.project.get("points"):
            removed = self.project["points"].pop()
            logger.info("Removed point #%s", removed.get("id"))
            for i, p in enumerate(self.project["points"], start=1):
                p["id"] = i

    def _save(self):
        save_project(self.project, self.args.output)
        logger.info("Saved project: %s", self.args.output)

    def _export_single_image_annotation(self):
        if self.args.target_type != "image":
            logger.warning("Direct annotation export is only available for --target_type image")
            return
        if not self.args.annotation_output:
            logger.warning("Set --annotation_output to export annotation")
            return
        image_name = self.args.image_name or self.target_path.name
        image_points = {image_name: [(p["target_pixel"], p["world"]) for p in self.project.get("points", [])]}
        K = parse_k(self.args.K, self.target_w, self.target_h)
        dist = parse_dist(self.args.dist)
        images_dir = self.args.images_dir or str(self.target_path.parent)
        export_annotation(image_points, images_dir, self.args.annotation_output, K, dist)
        logger.info("Exported annotation: %s", self.args.annotation_output)


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial GCP annotator for panorama or single image")
    parser.add_argument("--target", required=True, help="Panorama or single PTZ image to annotate")
    parser.add_argument("--target_type", choices=["panorama", "image"], default="panorama")
    parser.add_argument("--output", required=True, help="Output spatial GCP project JSON")
    parser.add_argument("--load", default="", help="Load existing project JSON")
    parser.add_argument("--coordinate_system", default="", help="Coordinate system label, e.g. EPSG:32650 or local_enu")
    parser.add_argument("--ortho", default="", help="Local orthophoto image used as map source")
    parser.add_argument("--ortho_extent", default="", help="xmin,ymin,xmax,ymax for orthophoto")
    parser.add_argument("--dsm", default="", help="Optional DSM .npy raster aligned with --dsm_extent")
    parser.add_argument("--dsm_extent", default="", help="xmin,ymin,xmax,ymax for DSM; defaults to ortho extent")
    parser.add_argument("--z", type=float, default=0.0, help="Default Z when no DSM is supplied or sampling is invalid")
    parser.add_argument("--map_source", default=None, choices=["osm", "google_satellite", "google_roadmap", "amap_satellite", "amap_roadmap"], help="Online XYZ tile source")
    parser.add_argument("--map_template", default="", help="Custom XYZ tile URL template with {z}, {x}, {y}, optional {key}")
    parser.add_argument("--map_bbox", default="", help="lon_min,lat_min,lon_max,lat_max for online map view")
    parser.add_argument("--map_zoom", type=int, default=18, help="Online map XYZ zoom level")
    parser.add_argument("--map_cache", default="~/.cache/ptzcalib_map_tiles", help="Tile cache directory")
    parser.add_argument("--map_key", default="", help="Optional API key substituted into --map_template as {key}")
    parser.add_argument("--map_timeout", type=float, default=10.0, help="Tile download timeout in seconds")
    parser.add_argument("--model", default="", help="Optional OBJ/PLY mesh for 3D picking")
    parser.add_argument("--annotation_output", default="", help="Direct annotation output for --target_type image")
    parser.add_argument("--images_dir", default="", help="Image directory used for direct annotation export")
    parser.add_argument("--image_name", default="", help="Image filename for direct annotation export")
    parser.add_argument("--K", default=None, help="Initial K, 9 comma-separated values")
    parser.add_argument("--dist", default=None, help="Initial dist, 5 comma-separated values")
    return parser.parse_args()


def main():
    annotator = SpatialGCPAnnotator(parse_args())
    annotator.run()


if __name__ == "__main__":
    main()
