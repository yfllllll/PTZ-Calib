#!/usr/bin/env python3
"""Ground Control Point (GCP) Annotator for PTZ-Calib.

Interactive tool to annotate ground control points by:
1. Selecting 2D points on PTZ camera images
2. Selecting corresponding 3D points on an oblique photogrammetry model
3. Generating annotation.json in PTZ-Calib format for run_ptz_ba

The generated annotation.json can be directly used with:
    python run_ptz_ba.py --annotation annotation.json ...

Usage:
    python gcp_annotator.py --images /path/to/images/ \
                            --model /path/to/model.obj \
                            --output annotation.json
    
    # With initial camera parameters
    python gcp_annotator.py --images /path/to/images/ \
                            --model /path/to/model.obj \
                            --K "fx,0,cx,0,fy,cy,0,0,1" \
                            --output annotation.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class GCPAnnotator:
    """Ground Control Point annotator with 2D image + 3D model dual-view."""
    
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, images_dir: str, model_path: str, output_path: str,
                 K: Optional[np.ndarray] = None, dist: Optional[np.ndarray] = None):
        self.images_dir = Path(images_dir)
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        
        # Default intrinsics (rough guess)
        self.K_init = K if K is not None else np.array([
            [1000.0, 0, 960.0],
            [0, 1000.0, 540.0],
            [0, 0, 1]
        ])
        self.dist_init = dist if dist is not None else np.zeros(5)
        
        # Load image list
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTS
        ])
        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")
        logger.info(f"Found {len(self.image_files)} images")
        
        # Load 3D model
        self.mesh = None
        self._load_mesh()
        
        # Annotations: {image_name: [(pt2d_xy, pt3d_xyz), ...]}
        self.annotations: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        
        # UI state
        self.current_image_idx = 0
        self.current_image = None
        self.current_image_display = None
        self.current_image_name = ""
        self.image_size = (0, 0)  # (width, height)
        self.pending_2d = None  # 2D point waiting for 3D match
        
        self.window_2d = "Image (2D)"
        self.window_3d = "Model (3D)"
    
    def _load_mesh(self):
        """Load 3D mesh."""
        if not HAS_OPEN3D:
            raise RuntimeError("Open3D not installed. Run: pip install open3d")
        
        ext = self.model_path.suffix.lower()
        if ext == '.osgb':
            # Look for converted sibling
            for alt_ext in ['.obj', '.ply']:
                alt_path = self.model_path.with_suffix(alt_ext)
                if alt_path.exists():
                    logger.info(f"OSGB not directly supported, using {alt_path}")
                    self.model_path = alt_path
                    break
            else:
                raise ValueError(
                    f"OSGB not supported directly. Convert to OBJ/PLY first:\n"
                    f"  osgconv {self.model_path} {self.model_path.with_suffix('.obj')}\n"
                    f"Or use CloudCompare."
                )
        
        logger.info(f"Loading 3D model: {self.model_path}")
        self.mesh = o3d.io.read_triangle_mesh(str(self.model_path))
        if not self.mesh.has_triangles():
            raise ValueError(f"No triangles in mesh: {self.model_path}")
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
        logger.info(f"Loaded mesh: {len(self.mesh.vertices):,} vertices, "
                   f"{len(self.mesh.triangles):,} triangles")
    
    def run(self):
        """Run annotation loop."""
        self._print_help()
        
        cv2.namedWindow(self.window_2d, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_2d, self._on_2d_click)
        cv2.resizeWindow(self.window_2d, 960, 540)
        
        # Load first image
        self._load_current_image()
        
        while True:
            self._render_2d()
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                self._save_and_quit()
                break
            elif key == ord('n'):
                self._next_image()
            elif key == ord('p'):
                self._prev_image()
            elif key == ord('3'):
                self._pick_3d_point()
            elif key == ord('u'):
                self._undo_last_point()
            elif key == ord('s'):
                self._save_annotation()
            elif key == ord('h'):
                self._print_help()
        
        cv2.destroyAllWindows()
    
    def _print_help(self):
        logger.info("\n" + "="*70)
        logger.info("Ground Control Point Annotator - Instructions")
        logger.info("="*70)
        logger.info("Workflow:")
        logger.info("  1. Left-click on IMAGE to select a 2D point")
        logger.info("  2. Press '3' to open 3D viewer and pick corresponding 3D point")
        logger.info("  3. In 3D viewer: Shift+LeftClick to pick, press 'Q' to close")
        logger.info("  4. Point pair is automatically saved")
        logger.info("")
        logger.info("Keys:")
        logger.info("  Left-click   : Select 2D point on image")
        logger.info("  '3'          : Open 3D viewer to pick corresponding 3D point")
        logger.info("  'n'          : Next image")
        logger.info("  'p'          : Previous image")
        logger.info("  'u'          : Undo last point pair on current image")
        logger.info("  's'          : Save annotation JSON")
        logger.info("  'h'          : Show this help")
        logger.info("  'q'          : Save and quit")
        logger.info("="*70 + "\n")
    
    def _load_current_image(self):
        """Load current image into memory."""
        img_path = self.image_files[self.current_image_idx]
        self.current_image_name = img_path.name
        self.current_image = cv2.imread(str(img_path))
        if self.current_image is None:
            logger.error(f"Failed to load image: {img_path}")
            return
        
        h, w = self.current_image.shape[:2]
        self.image_size = (w, h)
        
        # Initialize annotations for this image if not exist
        if self.current_image_name not in self.annotations:
            self.annotations[self.current_image_name] = []
        
        logger.info(f"\n[{self.current_image_idx + 1}/{len(self.image_files)}] "
                   f"{self.current_image_name} ({w}x{h}) - "
                   f"{len(self.annotations[self.current_image_name])} points annotated")
        self.pending_2d = None
    
    def _render_2d(self):
        """Render current image with annotations."""
        if self.current_image is None:
            return
        
        display = self.current_image.copy()
        
        # Draw existing annotations
        pts = self.annotations.get(self.current_image_name, [])
        for i, (pt2d, pt3d) in enumerate(pts):
            x, y = int(pt2d[0]), int(pt2d[1])
            cv2.circle(display, (x, y), 8, (0, 255, 0), 2)
            cv2.circle(display, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(display, f"{i+1}", (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw pending 2D point
        if self.pending_2d is not None:
            x, y = int(self.pending_2d[0]), int(self.pending_2d[1])
            cv2.circle(display, (x, y), 10, (0, 165, 255), 2)
            cv2.drawMarker(display, (x, y), (0, 165, 255),
                          cv2.MARKER_CROSS, 20, 2)
            cv2.putText(display, "Press '3' to pick 3D", (x + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Status bar
        status = (f"[{self.current_image_idx + 1}/{len(self.image_files)}] "
                 f"{self.current_image_name} | Points: {len(pts)}")
        cv2.rectangle(display, (0, 0), (display.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(display, status, (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        self.current_image_display = display
        cv2.imshow(self.window_2d, display)
    
    def _on_2d_click(self, event, x, y, flags, param):
        """Mouse callback for 2D image window."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending_2d = np.array([float(x), float(y)])
            logger.info(f"  2D selected: ({x}, {y}) - Press '3' to pick 3D point")
    
    def _pick_3d_point(self):
        """Open 3D viewer to pick corresponding 3D point."""
        if self.pending_2d is None:
            logger.warning("Select a 2D point first (left-click on image)")
            return
        
        logger.info("  Opening 3D viewer... (Shift+LeftClick to pick, 'Q' to close)")
        
        # Use Open3D's picking visualizer
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=self.window_3d, width=1280, height=720)
        vis.add_geometry(self.mesh)
        
        # Show existing 3D points as spheres for reference
        pts = self.annotations.get(self.current_image_name, [])
        for i, (pt2d, pt3d) in enumerate(pts):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            sphere.translate(pt3d)
            sphere.paint_uniform_color([0, 1, 0])  # Green
            vis.add_geometry(sphere)
        
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        
        vis.run()  # User picks point, then closes window
        picked = vis.get_picked_points()
        vis.destroy_window()
        
        if len(picked) == 0:
            logger.warning("  No 3D point picked (cancelled)")
            return
        
        # Get last picked point
        idx = picked[-1]
        pt3d = np.asarray(self.mesh.vertices)[idx].copy()
        
        # Save point pair
        self.annotations[self.current_image_name].append(
            (self.pending_2d.copy(), pt3d)
        )
        
        n = len(self.annotations[self.current_image_name])
        logger.info(f"  ✓ Saved point pair #{n}")
        logger.info(f"    2D: ({self.pending_2d[0]:.1f}, {self.pending_2d[1]:.1f})")
        logger.info(f"    3D: ({pt3d[0]:.3f}, {pt3d[1]:.3f}, {pt3d[2]:.3f})")
        
        self.pending_2d = None
    
    def _undo_last_point(self):
        """Remove last point pair on current image."""
        pts = self.annotations.get(self.current_image_name, [])
        if not pts:
            logger.warning("No points to undo on current image")
            return
        removed = pts.pop()
        logger.info(f"  Removed point pair: 2D={removed[0]}, 3D={removed[1]}")
    
    def _next_image(self):
        """Go to next image."""
        if self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self._load_current_image()
        else:
            logger.info("Already at last image")
    
    def _prev_image(self):
        """Go to previous image."""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self._load_current_image()
        else:
            logger.info("Already at first image")
    
    def _save_annotation(self):
        """Save annotations in PTZ-Calib JSON format."""
        cameras_dict = {}
        
        for img_name, pt_pairs in self.annotations.items():
            if not pt_pairs:
                continue  # Skip images with no annotations
            
            # Get image size (reload if not current)
            if img_name == self.current_image_name:
                w, h = self.image_size
            else:
                img_path = self.images_dir / img_name
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
            
            # Rootname (no extension) as key
            rootname = Path(img_name).stem
            
            # Normalized 2D pixels (divide by width/height)
            pix = [[float(p2d[0]) / w, float(p2d[1]) / h] for p2d, _ in pt_pairs]
            # 3D world coordinates
            pos = [[float(p3d[0]), float(p3d[1]), float(p3d[2])] for _, p3d in pt_pairs]
            
            cameras_dict[rootname] = {
                "K": self.K_init.flatten().tolist(),
                "R": np.eye(3).flatten().tolist(),
                "t": [0.0, 0.0, 0.0],
                "dist": self.dist_init.flatten().tolist(),
                "res": [w, h],
                "marker": {
                    "pix": pix,
                    "pos": pos
                }
            }
        
        output = {"cameras": cameras_dict}
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        n_imgs = len(cameras_dict)
        n_pts = sum(len(c["marker"]["pix"]) for c in cameras_dict.values())
        logger.info(f"\n✓ Saved annotation: {self.output_path}")
        logger.info(f"  {n_imgs} images, {n_pts} total point pairs")
    
    def _save_and_quit(self):
        """Save and exit."""
        self._save_annotation()
        logger.info("Bye!")
    
    def _load_existing(self, path: str):
        """Load existing annotation.json to continue editing."""
        with open(path) as f:
            data = json.load(f)
        
        # Build rootname -> filename map
        name_map = {Path(f.name).stem: f.name for f in self.image_files}
        
        loaded = 0
        for rootname, cam_data in data.get("cameras", {}).items():
            if rootname not in name_map:
                logger.warning(f"Skipping {rootname}: image not found")
                continue
            
            img_name = name_map[rootname]
            w, h = cam_data["res"]
            pix_list = cam_data["marker"]["pix"]
            pos_list = cam_data["marker"]["pos"]
            
            pt_pairs = []
            for pix, pos in zip(pix_list, pos_list):
                p2d = np.array([pix[0] * w, pix[1] * h])
                p3d = np.array(pos)
                pt_pairs.append((p2d, p3d))
            
            self.annotations[img_name] = pt_pairs
            loaded += len(pt_pairs)
        
        logger.info(f"Loaded {loaded} existing point pairs from {path}")


def parse_K(K_str: str) -> np.ndarray:
    """Parse K matrix from comma-separated string."""
    vals = [float(x) for x in K_str.split(',')]
    if len(vals) != 9:
        raise ValueError(f"K must have 9 values, got {len(vals)}")
    return np.array(vals).reshape(3, 3)


def main():
    parser = argparse.ArgumentParser(
        description="Ground Control Point Annotator for PTZ-Calib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing PTZ camera images')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to 3D model (OBJ/PLY, or OSGB with converted sibling)')
    parser.add_argument('--output', type=str, default='annotation.json',
                       help='Output annotation JSON path (default: annotation.json)')
    parser.add_argument('--K', type=str, default=None,
                       help='Initial K matrix, 9 comma-separated values '
                            '(default: fx=fy=1000, cx=960, cy=540)')
    parser.add_argument('--load', type=str, default=None,
                       help='Load existing annotation.json to continue editing')
    
    args = parser.parse_args()
    
    K = parse_K(args.K) if args.K else None
    
    annotator = GCPAnnotator(
        images_dir=args.images,
        model_path=args.model,
        output_path=args.output,
        K=K
    )
    
    # Load existing annotations to continue
    if args.load:
        annotator._load_existing(args.load)
    
    annotator.run()


if __name__ == '__main__':
    main()
