"""
HYBRID SVG RENDERER
===================

Renders multi-voltage-level substation diagrams by stitching individual
topology SVGs with transformers connecting the levels.

Works with JSON from hybrid_json_builder.py and calls individual renderers
in hybrid mode.

Usage:
    renderer = HybridSVGRenderer()
    renderer.render(
        merged_json_path="output/hybrid.json",
        individual_json_paths=["json/220kV.json", "json/132kV.json"],
        topology_types=["SINGLE_BUS", "SINGLE_BUS"],
        svg_output="output/hybrid.svg",
        yolo_output="output/hybrid.txt"
    )
"""

import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import all renderers
from core.render_brk_hlf import render_breaker_and_half_svg
from core.render_dbl_bs import render_double_bus_svg
from core.render_mn_trfr import render_main_and_transfer_svg
from core.render_mesh import render_mesh_svg
from core.render_ring_bs import render_ring_bus_svg
from core.render_sngl_bs import render_single_bus_svg


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

CANVAS_MARGIN_X = 10              # Left/right margin
CANVAS_MARGIN_Y = 10             # Top/bottom margin
TRANSFORMER_SPACE_ABOVE = 0     # Vertical space from bottom of upper level to top of transformer symbol
TRANSFORMER_SPACE_BELOW = 0     # Vertical space from bottom of transformer symbol to top of lower level
TRANSFORMER_SIZE = 32             # 32x32 viewBox
LINE_WIDTH = 1                    # Connection line width
LINE_COLOR = "#000"               # Black

# IEC Transformer Symbol
IEC_TRANSFORMER_SYMBOL = (
    '<line x1="16" y1="0" x2="16" y2="6" stroke="#000" stroke-width="1"/>'
    '<circle cx="16" cy="11.5" r="6" fill="none" stroke="#000" stroke-width="1"/>'
    '<circle cx="16" cy="20.5" r="6" fill="none" stroke="#000" stroke-width="1"/>'
    '<line x1="16" y1="26" x2="16" y2="32" stroke="#000" stroke-width="1"/>'
)

# YOLO class IDs
YOLO_CLASS_IDS = {
    "GridSource": 0,
    "Transformer": 1,
    "Breaker": 2,
    "Feeder": 3,
    "Capacitor": 4,
    "Isolator": 5,
    "Bus": 6,
}


# ============================================================================
# RENDERER MAPPING
# ============================================================================

RENDERER_MAP = {
    "SINGLE_BUS": render_single_bus_svg,
    "DOUBLE_BUS": render_double_bus_svg,
    "BREAKER_AND_HALF": render_breaker_and_half_svg,
    "MAIN_AND_TRANSFER": render_main_and_transfer_svg,
    "RING_BUS": render_ring_bus_svg,
    "MESH": render_mesh_svg,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RenderResult:
    """Result from rendering a single level"""
    boundary_pts: Dict[str, Dict]
    svg_content: str
    yolo_lines: List[str]
    width: float
    height: float


@dataclass
class TransformerPlacement:
    """Transformer placement info"""
    id: str
    label: str
    center: Tuple[float, float]
    from_point: Tuple[float, float]
    to_point: Tuple[float, float]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_svg_dimensions(svg_content: str) -> Tuple[float, float]:
    """Extract width and height from SVG"""
    width_match = re.search(r'width="([\d.]+)"', svg_content)
    height_match = re.search(r'height="([\d.]+)"', svg_content)
    width = float(width_match.group(1)) if width_match else 800
    height = float(height_match.group(1)) if height_match else 600
    return width, height


def extract_svg_inner_content(svg_content: str) -> str:
    """Extract content between <svg> and </svg> tags"""
    if '<svg' in svg_content and '</svg>' in svg_content:
        svg_start = svg_content.find('>')
        if svg_start != -1:
            svg_end = svg_content.rfind('</svg>')
            if svg_end != -1:
                return svg_content[svg_start + 1:svg_end]
    return svg_content


def manhattan_path(from_point: Tuple[float, float], to_point: Tuple[float, float]) -> str:
    """Generate Manhattan routing path"""
    x1, y1 = from_point
    x2, y2 = to_point
    mid_y = (y1 + y2) / 2
    return f"M {x1:.1f},{y1:.1f} L {x1:.1f},{mid_y:.1f} L {x2:.1f},{mid_y:.1f} L {x2:.1f},{y2:.1f}"


def generate_transformer_symbol(x: float, y: float, label: str) -> str:
    """Generate IEC transformer symbol SVG with label to the right (matching render_brk_hlf style)"""
    transform_x = x - TRANSFORMER_SIZE / 2
    transform_y = y - TRANSFORMER_SIZE / 2
    # Label positioned to the right of the symbol at 60% of symbol height, monospace size 7
    label_x = x + TRANSFORMER_SIZE / 2 + 2
    label_y = transform_y + TRANSFORMER_SIZE * 0.6

    return (
        f'\n    <g id="transformer_{label.replace("/", "_").replace("-", "_")}"'
        f' transform="translate({transform_x:.1f},{transform_y:.1f})">\n'
        f'        <g transform="scale({TRANSFORMER_SIZE/32:.4f})">\n'
        f'            {IEC_TRANSFORMER_SYMBOL}\n'
        f'        </g>\n'
        f'    </g>\n'
        f'    <text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="start"'
        f' font-size="7" font-family="monospace" fill="{LINE_COLOR}">{label}</text>\n'
    )


# ============================================================================
# MAIN RENDERER CLASS
# ============================================================================

class HybridSVGRenderer:
    """
    Renders multi-voltage SVG diagrams from merged JSON
    """
    
    def __init__(self):
        self.merged_json = None
        self.level_results: List[RenderResult] = []
        self.level_positions: Dict[int, Tuple[float, float]] = {}
        self.transformers: List[TransformerPlacement] = []
        self.canvas_width = 0
        self.canvas_height = 0
    
    def render(
        self,
        merged_json_path: str,
        individual_json_paths: List[str],
        topology_types: List[str],
        svg_output: str,
        yolo_output: str
    ):
        """
        Main rendering pipeline
        
        Args:
            merged_json_path: Path to merged JSON from hybrid_json_builder
            individual_json_paths: Paths to individual level JSONs (in order)
            topology_types: Topology type for each level (in same order)
            svg_output: Output path for stitched SVG
            yolo_output: Output path for YOLO annotations
        """
        print(f"\n{'='*70}")
        print(f"HYBRID SVG RENDERER")
        print(f"{'='*70}")
        
        # Load merged JSON
        with open(merged_json_path, 'r') as f:
            self.merged_json = json.load(f)
        
        print(f"Rendering {len(individual_json_paths)} voltage levels...")
        
        # Step 1: Render each level
        self._render_levels(individual_json_paths, topology_types)
        
        # Step 2: Calculate layout
        self._calculate_layout()
        
        # Step 3: Place transformers
        self._place_transformers()
        
        # Step 4: Generate SVG
        self._generate_svg(svg_output)
        
        # Step 5: Generate YOLO
        self._generate_yolo(yolo_output)
        
        print(f"\n{'='*70}")
        print(f"RENDERING COMPLETE!")
        print(f"{'='*70}")
        print(f"  SVG:  {svg_output}")
        print(f"  YOLO: {yolo_output}")
        print()
    
    def _render_levels(self, json_paths: List[str], topology_types: List[str]):
        """Render each level in hybrid mode"""
        for i, (json_path, topology_type) in enumerate(zip(json_paths, topology_types)):
            voltage_label = self.merged_json["layout"]["voltage_levels"][i]["name"]
            print(f"  - Level {i}: {voltage_label} ({topology_type})")
            
            renderer = RENDERER_MAP[topology_type]
            
            # Call renderer in hybrid mode
            boundary_pts, full_svg, yolo_lines = renderer(json_path, "", mode="hybrid")
            
            # Extract dimensions
            width, height = extract_svg_dimensions(full_svg)
            
            # Store result
            self.level_results.append(RenderResult(
                boundary_pts=boundary_pts,
                svg_content=full_svg,
                yolo_lines=yolo_lines,
                width=width,
                height=height
            ))
    
    def _calculate_layout(self):
        """Calculate positions for each level"""
        print(f"\nCalculating layout...")

        inter_level_gap = TRANSFORMER_SPACE_ABOVE + TRANSFORMER_SIZE + TRANSFORMER_SPACE_BELOW

        max_width = max(r.width for r in self.level_results)
        current_y = CANVAS_MARGIN_Y
        
        for i, result in enumerate(self.level_results):
            x_offset = CANVAS_MARGIN_X + (max_width - result.width) / 2
            y_offset = current_y
            
            self.level_positions[i] = (x_offset, y_offset)
            current_y += result.height + inter_level_gap
        
        self.canvas_width = max_width + 2 * CANVAS_MARGIN_X
        self.canvas_height = current_y - inter_level_gap + CANVAS_MARGIN_Y
    
    def _place_transformers(self):
        """Place transformers between levels"""
        print(f"Placing transformers...")
        
        unified_transformers = self.merged_json.get("unified_transformers", {})
        transformer_connections = self.merged_json.get("transformer_connections", {})
        
        for tr_id, levels in transformer_connections.items():
            if len(levels) != 2:
                continue
            
            from_level_idx, to_level_idx = levels
            tr_data = unified_transformers[tr_id]
            
            # Get original device IDs
            from_device_id = tr_data["from_device_id"]
            to_device_id = tr_data["to_device_id"]
            
            # Extract original IDs (without voltage prefix)
            from_original = from_device_id.split('_', 1)[1] if '_' in from_device_id else from_device_id
            to_original = to_device_id.split('_', 1)[1] if '_' in to_device_id else to_device_id
            
            # Get boundary points from renderers
            from_result = self.level_results[from_level_idx]
            to_result = self.level_results[to_level_idx]
            
            from_pt_data = from_result.boundary_pts.get(from_original, {})
            to_pt_data = to_result.boundary_pts.get(to_original, {})
            
            if not from_pt_data or not to_pt_data:
                continue
            
            # Calculate absolute canvas positions
            from_offset = self.level_positions[from_level_idx]
            to_offset = self.level_positions[to_level_idx]
            
            from_point = (
                from_pt_data["x"] + from_offset[0],
                from_pt_data["y"] + from_offset[1]
            )
            
            to_point = (
                to_pt_data["x"] + to_offset[0],
                to_pt_data["y"] + to_offset[1]
            )
            
            # Calculate center:
            # x is the midpoint of the two connection x-coords.
            # y is derived from layout: bottom of upper level + TRANSFORMER_SPACE_ABOVE + half symbol.
            upper_level_bottom = (
                self.level_positions[from_level_idx][1]
                + self.level_results[from_level_idx].height
            )
            center = (
                (from_point[0] + to_point[0]) / 2,
                upper_level_bottom + TRANSFORMER_SPACE_ABOVE + TRANSFORMER_SIZE / 2,
            )
            
            self.transformers.append(TransformerPlacement(
                id=tr_id,
                label=tr_data["label"],
                center=center,
                from_point=from_point,
                to_point=to_point
            ))
        
        print(f"  ✓ Placed {len(self.transformers)} transformers")
    
    def _generate_svg(self, output_path: str):
        """Generate final stitched SVG"""
        print(f"Generating SVG...")
        
        svg_parts = []
        
        # Header
        svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.canvas_width:.0f}" height="{self.canvas_height:.0f}" '
            f'viewBox="0 0 {self.canvas_width:.0f} {self.canvas_height:.0f}">'
        )
        svg_parts.append('<rect width="100%" height="100%" fill="white"/>')
        
        # Add each level
        for i, (result, (x_offset, y_offset)) in enumerate(zip(self.level_results, self.level_positions.values())):
            voltage_label = self.merged_json["layout"]["voltage_levels"][i]["name"]
            svg_parts.append(f'<!-- Level {i}: {voltage_label} -->')
            svg_parts.append(f'<g transform="translate({x_offset:.1f},{y_offset:.1f})">')
            svg_parts.append(extract_svg_inner_content(result.svg_content))
            svg_parts.append('</g>')
        
        # Add transformers
        svg_parts.append('<!-- Inter-Level Connections -->')
        svg_parts.append('<g id="inter_level_connections">')
        
        for tr in self.transformers:
            tr_x, tr_y = tr.center
            tr_top_y = tr_y - TRANSFORMER_SIZE / 2
            tr_bottom_y = tr_y + TRANSFORMER_SIZE / 2
            
            tr_top_point = (tr_x, tr_top_y)
            tr_bottom_point = (tr_x, tr_bottom_y)
            
            # Line from upper level to transformer
            path1 = manhattan_path(tr.from_point, tr_top_point)
            svg_parts.append(
                f'<path d="{path1}" stroke="{LINE_COLOR}" stroke-width="{LINE_WIDTH}" '
                f'fill="none" stroke-linecap="square"/>'
            )
            
            # Transformer symbol
            svg_parts.append(generate_transformer_symbol(tr_x, tr_y, tr.label))
            
            # Line from transformer to lower level
            path2 = manhattan_path(tr_bottom_point, tr.to_point)
            svg_parts.append(
                f'<path d="{path2}" stroke="{LINE_COLOR}" stroke-width="{LINE_WIDTH}" '
                f'fill="none" stroke-linecap="square"/>'
            )
        
        svg_parts.append('</g>')
        svg_parts.append('</svg>')
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(svg_parts))
        
        print(f"  ✓ Saved SVG")
    
    def _generate_yolo(self, output_path: str):
        """Generate merged YOLO annotations"""
        print(f"Generating YOLO annotations...")
        
        annotations = []
        
        # Process each level's YOLO
        for i, (result, (x_offset, y_offset)) in enumerate(zip(self.level_results, self.level_positions.values())):
            for line in result.yolo_lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id, x_norm, y_norm, w_norm, h_norm = map(float, parts)
                
                # Denormalize
                x_abs = x_norm * result.width
                y_abs = y_norm * result.height
                w_abs = w_norm * result.width
                h_abs = h_norm * result.height
                
                # Translate
                x_canvas = x_abs + x_offset
                y_canvas = y_abs + y_offset
                
                # Renormalize
                x_canvas_norm = x_canvas / self.canvas_width
                y_canvas_norm = y_canvas / self.canvas_height
                w_canvas_norm = w_abs / self.canvas_width
                h_canvas_norm = h_abs / self.canvas_height
                
                annotations.append(
                    f"{int(class_id)} {x_canvas_norm:.6f} {y_canvas_norm:.6f} "
                    f"{w_canvas_norm:.6f} {h_canvas_norm:.6f}"
                )
        
        # Add transformers
        for tr in self.transformers:
            tr_x, tr_y = tr.center
            x_norm = tr_x / self.canvas_width
            y_norm = tr_y / self.canvas_height
            w_norm = TRANSFORMER_SIZE / self.canvas_width
            h_norm = TRANSFORMER_SIZE / self.canvas_height
            
            annotations.append(
                f"{YOLO_CLASS_IDS['Transformer']} {x_norm:.6f} {y_norm:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}"
            )
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        print(f"  ✓ Saved YOLO ({len(annotations)} annotations)")


# Example usage
if __name__ == "__main__":
    renderer = HybridSVGRenderer()
    
    renderer.render(
        merged_json_path="output_hybrid/json/hybrid.json",
        individual_json_paths=[
            "output_hybrid/temp_json/220kV_0.json",                   
            "output_hybrid/temp_json/132kV_1.json",
            "output_hybrid/temp_json/66kV_2.json",
            "output_hybrid/temp_json/33kV_3.json",
            "output_hybrid/temp_json/11kV_4.json",
        ],
        topology_types=[
            "MESH", 
            "RING_BUS", 
            "BREAKER_AND_HALF", 
            "MAIN_AND_TRANSFER", 
            "DOUBLE_BUS"
        ],
        svg_output="output_hybrid/svg/hybrid.svg",
        yolo_output="output_hybrid/yolo/hybrid.txt"
    )