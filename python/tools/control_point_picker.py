#!/usr/bin/env python3
"""Control Point Picker Tool for 3D Models.

Interactive tool to pick control points on oblique photogrammetry 3D models (OSGB/OBJ/PLY)
and solve similarity transformation (7-DOF: R, t, s) between world and local coordinate systems.

Usage:
    python control_point_picker.py --model path/to/model.osgb
    
    # Or use converted OBJ/PLY
    python control_point_picker.py --model path/to/model.obj
    
Controls:
    - Left click: Pick point on 3D model
    - 'n': Next point (save current and start new)
    - 's': Solve transformation
    - 'e': Export control points
    - 'q': Quit
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ControlPointPicker:
    """Interactive control point picker for 3D models."""
    
    def __init__(self, mesh_path: str):
        """Initialize picker with 3D model."""
        self.mesh_path = Path(mesh_path)
        self.mesh = None
        self.vis = None
        
        # Control points: list of (local_xyz, world_xyz) tuples
        self.control_points = []
        
        # Current picking state
        self.current_local_pt = None
        self.picked_points = []  # For visualization
        
        # Transformation result
        self.R = None  # 3x3 rotation
        self.t = None  # 3x1 translation
        self.s = None  # scalar scale
        
        self._load_mesh()
    
    def _load_mesh(self):
        """Load 3D mesh from file."""
        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Model not found: {self.mesh_path}")
        
        ext = self.mesh_path.suffix.lower()
        
        if ext == '.osgb':
            # OSGB needs conversion - try to find OBJ/PLY sibling
            obj_path = self.mesh_path.with_suffix('.obj')
            ply_path = self.mesh_path.with_suffix('.ply')
            
            if obj_path.exists():
                logger.info(f"Loading converted OBJ: {obj_path}")
                self.mesh = o3d.io.read_triangle_mesh(str(obj_path))
            elif ply_path.exists():
                logger.info(f"Loading converted PLY: {ply_path}")
                self.mesh = o3d.io.read_triangle_mesh(str(ply_path))
            else:
                raise ValueError(
                    f"OSGB file found but no OBJ/PLY conversion available.\n"
                    f"Please convert {self.mesh_path} to OBJ or PLY first using:\n"
                    f"  - CloudCompare: File > Open > Export to OBJ/PLY\n"
                    f"  - FME Desktop: OSGB Reader > OBJ/PLY Writer\n"
                    f"  - osgconv (OpenSceneGraph): osgconv input.osgb output.obj"
                )
        else:
            logger.info(f"Loading mesh: {self.mesh_path}")
            self.mesh = o3d.io.read_triangle_mesh(str(self.mesh_path))
        
        if not self.mesh.has_triangles():
            raise ValueError(f"No triangles found in mesh: {self.mesh_path}")
        
        # Compute normals if missing
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
        
        logger.info(f"Loaded mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.triangles)} triangles")
    
    def run(self):
        """Run interactive picker."""
        logger.info("\n" + "="*60)
        logger.info("Control Point Picker - Instructions:")
        logger.info("="*60)
        logger.info("1. LEFT CLICK on model to pick a point")
        logger.info("2. Enter world coordinates in terminal")
        logger.info("3. Press 'n' to save and pick next point")
        logger.info("4. Press 's' to solve transformation (need ≥3 points)")
        logger.info("5. Press 'e' to export control points")
        logger.info("6. Press 'q' to quit")
        logger.info("="*60 + "\n")
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithEditing()
        self.vis.create_window(window_name="Control Point Picker", width=1280, height=720)
        self.vis.add_geometry(self.mesh)
        
        # Set rendering options
        opt = self.vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.light_on = True
        
        # Register callback
        self.vis.register_key_callback(ord('N'), self._on_next_point)
        self.vis.register_key_callback(ord('S'), self._on_solve)
        self.vis.register_key_callback(ord('E'), self._on_export)
        self.vis.register_key_callback(ord('Q'), self._on_quit)
        
        # Run
        self.vis.run()
        self.vis.destroy_window()
    
    def _on_next_point(self, vis):
        """Callback: save current point and start new one."""
        # Get picked points
        picked = vis.get_picked_points()
        
        if len(picked) == 0:
            logger.warning("No point picked! Click on the model first.")
            return False
        
        # Get last picked point
        idx = picked[-1].index
        local_pt = np.asarray(self.mesh.vertices)[idx]
        
        logger.info(f"\nPicked local point: {local_pt}")
        logger.info("Enter corresponding world coordinates (x y z):")
        
        try:
            world_str = input("World (x y z): ")
            world_pt = np.array([float(x) for x in world_str.strip().split()])
            
            if len(world_pt) != 3:
                logger.error("Please enter exactly 3 coordinates!")
                return False
            
            # Save control point pair
            self.control_points.append((local_pt.copy(), world_pt.copy()))
            logger.info(f"✓ Saved control point #{len(self.control_points)}")
            logger.info(f"  Local:  {local_pt}")
            logger.info(f"  World:  {world_pt}")
            
            # Visualize picked point
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            sphere.translate(local_pt)
            sphere.paint_uniform_color([1, 0, 0])  # Red
            vis.add_geometry(sphere)
            self.picked_points.append(sphere)
            
        except (ValueError, EOFError) as e:
            logger.error(f"Invalid input: {e}")
            return False
        
        return False
    
    def _on_solve(self, vis):
        """Callback: solve similarity transformation."""
        if len(self.control_points) < 3:
            logger.warning(f"Need at least 3 control points (have {len(self.control_points)})")
            return False
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Solving similarity transformation with {len(self.control_points)} points...")
        logger.info(f"{'='*60}")
        
        # Solve 7-DOF similarity transformation
        success = self._solve_similarity_transform()
        
        if success:
            logger.info("✓ Transformation solved successfully!")
            self._print_transformation()
            self._evaluate_residuals()
        else:
            logger.error("✗ Transformation solving failed!")
        
        return False
    
    def _solve_similarity_transform(self) -> bool:
        """Solve 7-DOF similarity transformation: world = s * R * local + t."""
        local_pts = np.array([cp[0] for cp in self.control_points])  # Nx3
        world_pts = np.array([cp[1] for cp in self.control_points])  # Nx3
        
        # Center the point sets
        local_center = local_pts.mean(axis=0)
        world_center = world_pts.mean(axis=0)
        
        local_centered = local_pts - local_center
        world_centered = world_pts - world_center
        
        # Compute scale
        local_scale = np.sqrt((local_centered ** 2).sum())
        world_scale = np.sqrt((world_centered ** 2).sum())
        self.s = world_scale / local_scale
        
        # Normalize
        local_normalized = local_centered / local_scale
        world_normalized = world_centered / world_scale
        
        # Solve rotation using SVD (Procrustes)
        H = local_normalized.T @ world_normalized
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        self.R = R
        
        # Solve translation
        self.t = world_center - self.s * (self.R @ local_center)
        
        return True
    
    def _print_transformation(self):
        """Print transformation parameters."""
        logger.info("\n" + "="*60)
        logger.info("SIMILARITY TRANSFORMATION (7-DOF)")
        logger.info("="*60)
        logger.info(f"Formula: world = s * R * local + t\n")
        
        logger.info(f"Scale (s): {self.s:.6f}\n")
        
        logger.info("Rotation matrix (R):")
        for i in range(3):
            logger.info(f"  [{self.R[i, 0]:9.6f}, {self.R[i, 1]:9.6f}, {self.R[i, 2]:9.6f}]")
        
        # Convert to Euler angles (ZYX convention)
        rot = Rotation.from_matrix(self.R)
        euler_deg = rot.as_euler('zyx', degrees=True)
        logger.info(f"\nEuler angles (Z-Y-X, degrees): [{euler_deg[0]:.3f}, {euler_deg[1]:.3f}, {euler_deg[2]:.3f}]")
        
        logger.info(f"\nTranslation (t):")
        logger.info(f"  [{self.t[0]:12.6f}, {self.t[1]:12.6f}, {self.t[2]:12.6f}]")
        logger.info("="*60 + "\n")
    
    def _evaluate_residuals(self):
        """Evaluate transformation residuals."""
        logger.info("Residual Analysis:")
        logger.info("-" * 60)
        
        residuals = []
        for i, (local_pt, world_pt) in enumerate(self.control_points):
            # Transform local to world
            world_pred = self.s * (self.R @ local_pt) + self.t
            
            # Compute residual
            residual = np.linalg.norm(world_pred - world_pt)
            residuals.append(residual)
            
            logger.info(f"Point {i+1}: residual = {residual:.6f} m")
            logger.info(f"  Local:         {local_pt}")
            logger.info(f"  World (true):  {world_pt}")
            logger.info(f"  World (pred):  {world_pred}")
        
        residuals = np.array(residuals)
        logger.info("-" * 60)
        logger.info(f"RMSE: {np.sqrt((residuals**2).mean()):.6f} m")
        logger.info(f"Mean: {residuals.mean():.6f} m")
        logger.info(f"Max:  {residuals.max():.6f} m")
        logger.info("="*60 + "\n")
    
    def _on_export(self, vis):
        """Callback: export control points and transformation."""
        output_dir = Path("control_points_output")
        output_dir.mkdir(exist_ok=True)
        
        # Export control points
        cp_file = output_dir / "control_points.json"
        cp_data = {
            "num_points": len(self.control_points),
            "points": [
                {
                    "id": i + 1,
                    "local": local_pt.tolist(),
                    "world": world_pt.tolist()
                }
                for i, (local_pt, world_pt) in enumerate(self.control_points)
            ]
        }
        
        with open(cp_file, 'w') as f:
            json.dump(cp_data, f, indent=2)
        logger.info(f"✓ Control points saved: {cp_file}")
        
        # Export transformation (if solved)
        if self.R is not None:
            transform_file = output_dir / "transformation.json"
            rot = Rotation.from_matrix(self.R)
            
            transform_data = {
                "type": "similarity_7dof",
                "formula": "world = s * R * local + t",
                "scale": float(self.s),
                "rotation_matrix": self.R.tolist(),
                "rotation_euler_zyx_deg": rot.as_euler('zyx', degrees=True).tolist(),
                "rotation_quaternion_xyzw": rot.as_quat().tolist(),
                "translation": self.t.tolist()
            }
            
            with open(transform_file, 'w') as f:
                json.dump(transform_data, f, indent=2)
            logger.info(f"✓ Transformation saved: {transform_file}")
        
        logger.info(f"\nAll outputs in: {output_dir.absolute()}")
        
        return False
    
    def _on_quit(self, vis):
        """Callback: quit application."""
        logger.info("\nQuitting...")
        vis.close()
        return True


def main():
    parser = argparse.ArgumentParser(description="Control Point Picker for 3D Models")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to 3D model (OSGB/OBJ/PLY)')
    
    args = parser.parse_args()
    
    try:
        picker = ControlPointPicker(args.model)
        picker.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
