"""
HYBRID SVG BUILDER
==================

Builds multi-voltage-level substation diagrams by stitching individual topology SVGs
with transformers connecting the levels.

Works with updated renderers that return: (boundary_pts, full_svg, yolo_lines) in hybrid mode.
"""

import json
import re
from typing import Dict, List, Tuple, Any
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

# Canvas margins
CANVAS_MARGIN_X = 50  # Left/right margin
CANVAS_MARGIN_Y = 100  # Top/bottom margin

# Spacing between topology diagrams
INTER_TOPOLOGY_SPACING = 150  # Vertical space between voltage levels

# Transformer symbol size (32x32 viewBox)
TRANSFORMER_SIZE = 32

# Line styling
LINE_WIDTH = 1
LINE_COLOR = "#000"

# IEC Transformer Symbol (32x32 viewBox)
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
# DATA STRUCTURES
# ============================================================================

@dataclass
class LevelConfig:
    """Configuration for a single voltage level"""
    json_path: str
    topology_type: str
    voltage_label: str  # e.g., "220kV", "132kV"


@dataclass
class LevelRenderResult:
    """Result from rendering a single level"""
    boundary_pts: Dict[str, Dict]  # {device_id: {"label": str, "x": float, "y": float}}
    svg_content: str
    yolo_lines: List[str]
    width: float
    height: float


@dataclass
class TransformerConnection:
    """Represents a transformer connecting two levels"""
    id: str
    label: str
    from_level_idx: int
    to_level_idx: int
    from_device_id: str
    to_device_id: str
    from_point: Tuple[float, float]  # Absolute canvas coordinates
    to_point: Tuple[float, float]    # Absolute canvas coordinates
    center: Tuple[float, float]      # Transformer symbol center


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
# HELPER FUNCTIONS
# ============================================================================

def extract_svg_dimensions(svg_content: str) -> Tuple[float, float]:
    """Extract width and height from SVG content"""
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
    """
    Generate SVG path for Manhattan routing (vertical → horizontal → vertical)
    
    Args:
        from_point: Starting point (x, y)
        to_point: Ending point (x, y)
    
    Returns:
        SVG path d attribute
    """
    x1, y1 = from_point
    x2, y2 = to_point
    
    # Calculate midpoint for horizontal segment
    mid_y = (y1 + y2) / 2
    
    # Path: start → down to mid → across to x2 → down to end
    return f"M {x1:.1f},{y1:.1f} L {x1:.1f},{mid_y:.1f} L {x2:.1f},{mid_y:.1f} L {x2:.1f},{y2:.1f}"


def generate_transformer_symbol(x: float, y: float, label: str) -> str:
    """
    Generate SVG for IEC transformer symbol
    
    Args:
        x, y: Center position of transformer
        label: Transformer label to display
    
    Returns:
        SVG markup
    """
    # Center the 32x32 symbol at (x, y)
    transform_x = x - TRANSFORMER_SIZE / 2
    transform_y = y - TRANSFORMER_SIZE / 2
    
    return f'''
    <g id="transformer_{label.replace('/', '_').replace('-', '_')}" transform="translate({transform_x:.1f},{transform_y:.1f})">
        <g transform="scale({TRANSFORMER_SIZE/32:.4f})">
            {IEC_TRANSFORMER_SYMBOL}
        </g>
    </g>
    <text x="{x:.1f}" y="{transform_y - 5:.1f}" text-anchor="middle" font-size="10" font-family="Arial" fill="{LINE_COLOR}">{label}</text>
    '''


# ============================================================================
# MAIN HYBRID BUILDER CLASS
# ============================================================================

class HybridSVGBuilder:
    """
    Builds hybrid multi-voltage SVG diagrams
    """
    
    def __init__(self):
        self.levels: List[LevelConfig] = []
        self.level_results: List[LevelRenderResult] = []
        self.level_positions: Dict[int, Tuple[float, float]] = {}
        self.transformers: List[TransformerConnection] = []
        self.canvas_width = 0
        self.canvas_height = 0
    
    def add_level(self, json_path: str, topology_type: str, voltage_label: str):
        """Add a voltage level to the hybrid diagram"""
        self.levels.append(LevelConfig(json_path, topology_type, voltage_label))
    
    def render_levels(self):
        """Render all voltage levels in hybrid mode"""
        print(f"Rendering {len(self.levels)} voltage levels...")
        
        for i, level in enumerate(self.levels):
            print(f"  - Rendering {level.voltage_label} ({level.topology_type})...")
            
            # Get the appropriate renderer
            renderer = RENDERER_MAP[level.topology_type]
            
            # Call renderer in hybrid mode (it loads JSON internally)
            boundary_pts, full_svg, yolo_lines = renderer(level.json_path, "", mode="hybrid")
            
            # Extract dimensions
            width, height = extract_svg_dimensions(full_svg)
            
            # Store result
            self.level_results.append(LevelRenderResult(
                boundary_pts=boundary_pts,
                svg_content=full_svg,
                yolo_lines=yolo_lines,
                width=width,
                height=height
            ))
    
    def calculate_layout(self):
        """Calculate positions for each level on the canvas"""
        print("Calculating layout...")
        
        # Find maximum width
        max_width = max(result.width for result in self.level_results)
        
        # Calculate positions (top to bottom)
        current_y = CANVAS_MARGIN_Y
        
        for i, result in enumerate(self.level_results):
            # Center horizontally
            x_offset = CANVAS_MARGIN_X + (max_width - result.width) / 2
            y_offset = current_y
            
            self.level_positions[i] = (x_offset, y_offset)
            
            # Move down for next level
            current_y += result.height + INTER_TOPOLOGY_SPACING
        
        # Calculate total canvas size
        self.canvas_width = max_width + 2 * CANVAS_MARGIN_X
        self.canvas_height = current_y - INTER_TOPOLOGY_SPACING + CANVAS_MARGIN_Y
    
    def identify_transformers(self):
        """Identify and create transformer connections between levels"""
        print("Identifying transformer connections...")
        
        transformer_counter = 1
        
        for i in range(len(self.levels) - 1):
            from_level_idx = i
            to_level_idx = i + 1
            
            from_level = self.levels[from_level_idx]
            to_level = self.levels[to_level_idx]
            
            from_result = self.level_results[from_level_idx]
            to_result = self.level_results[to_level_idx]
            
            from_offset = self.level_positions[from_level_idx]
            to_offset = self.level_positions[to_level_idx]
            
            # Find downstream transformers in from_level
            downstream_trs = [
                (dev_id, data) for dev_id, data in from_result.boundary_pts.items()
                if "TR_DN" in dev_id
            ]
            
            # Find upstream transformers in to_level
            upstream_trs = [
                (dev_id, data) for dev_id, data in to_result.boundary_pts.items()
                if "TR_UP" in dev_id
            ]
            
            # Match transformers by index
            num_connections = min(len(downstream_trs), len(upstream_trs))
            
            for idx in range(num_connections):
                dn_id, dn_data = downstream_trs[idx]
                up_id, up_data = upstream_trs[idx]
                
                # Calculate absolute canvas positions
                from_point = (
                    dn_data["x"] + from_offset[0],
                    dn_data["y"] + from_offset[1]
                )
                
                to_point = (
                    up_data["x"] + to_offset[0],
                    up_data["y"] + to_offset[1]
                )
                
                # Calculate transformer center (midpoint)
                center_x = (from_point[0] + to_point[0]) / 2
                center_y = (from_point[1] + to_point[1]) / 2
                
                # Create unified transformer
                tr_id = f"{from_level.voltage_label}_{to_level.voltage_label}_TR_{transformer_counter:02d}"
                tr_label = f"{from_level.voltage_label}/{to_level.voltage_label}-TR-{transformer_counter:02d}"
                
                self.transformers.append(TransformerConnection(
                    id=tr_id,
                    label=tr_label,
                    from_level_idx=from_level_idx,
                    to_level_idx=to_level_idx,
                    from_device_id=dn_id,
                    to_device_id=up_id,
                    from_point=from_point,
                    to_point=to_point,
                    center=(center_x, center_y)
                ))
                
                transformer_counter += 1
        
        print(f"  - Found {len(self.transformers)} transformer connections")
    
    def generate_svg(self, output_path: str):
        """Generate the final stitched SVG"""
        print("Generating final SVG...")
        
        svg_parts = []
        
        # SVG header
        svg_parts.append(f'<?xml version="1.0" encoding="UTF-8"?>')
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.canvas_width:.0f}" height="{self.canvas_height:.0f}" '
            f'viewBox="0 0 {self.canvas_width:.0f} {self.canvas_height:.0f}">'
        )
        svg_parts.append(f'<rect width="100%" height="100%" fill="white"/>')
        
        # Add each level's SVG
        for i, (result, (x_offset, y_offset)) in enumerate(zip(self.level_results, self.level_positions.values())):
            svg_parts.append(f'<!-- Voltage Level {i}: {self.levels[i].voltage_label} -->')
            svg_parts.append(f'<g transform="translate({x_offset:.1f},{y_offset:.1f})">')
            svg_parts.append(extract_svg_inner_content(result.svg_content))
            svg_parts.append('</g>')
        
        # Add transformer connections
        svg_parts.append('<!-- Inter-Level Transformer Connections -->')
        svg_parts.append('<g id="inter_level_connections">')
        
        for tr in self.transformers:
            # Calculate transformer symbol boundaries
            tr_x, tr_y = tr.center
            tr_top_y = tr_y - TRANSFORMER_SIZE / 2
            tr_bottom_y = tr_y + TRANSFORMER_SIZE / 2
            
            # Connection points on transformer
            tr_top_point = (tr_x, tr_top_y)
            tr_bottom_point = (tr_x, tr_bottom_y)
            
            # Draw line from upper level to transformer top
            path1 = manhattan_path(tr.from_point, tr_top_point)
            svg_parts.append(
                f'<path d="{path1}" stroke="{LINE_COLOR}" stroke-width="{LINE_WIDTH}" '
                f'fill="none" stroke-linecap="square"/>'
            )
            
            # Draw transformer symbol
            svg_parts.append(generate_transformer_symbol(tr_x, tr_y, tr.label))
            
            # Draw line from transformer bottom to lower level
            path2 = manhattan_path(tr_bottom_point, tr.to_point)
            svg_parts.append(
                f'<path d="{path2}" stroke="{LINE_COLOR}" stroke-width="{LINE_WIDTH}" '
                f'fill="none" stroke-linecap="square"/>'
            )
        
        svg_parts.append('</g>')
        svg_parts.append('</svg>')
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(svg_parts))
        
        print(f"  - Saved SVG: {output_path}")
    
    def generate_yolo(self, output_path: str):
        """Generate merged YOLO annotations"""
        print("Generating YOLO annotations...")
        
        annotations = []
        
        # Process each level's YOLO annotations
        for i, (result, (x_offset, y_offset)) in enumerate(zip(self.level_results, self.level_positions.values())):
            for line in result.yolo_lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id, x_norm, y_norm, w_norm, h_norm = map(float, parts)
                
                # Denormalize to level coordinates
                x_abs = x_norm * result.width
                y_abs = y_norm * result.height
                w_abs = w_norm * result.width
                h_abs = h_norm * result.height
                
                # Translate to canvas coordinates
                x_canvas = x_abs + x_offset
                y_canvas = y_abs + y_offset
                
                # Normalize to canvas
                x_canvas_norm = x_canvas / self.canvas_width
                y_canvas_norm = y_canvas / self.canvas_height
                w_canvas_norm = w_abs / self.canvas_width
                h_canvas_norm = h_abs / self.canvas_height
                
                annotations.append(
                    f"{int(class_id)} {x_canvas_norm:.6f} {y_canvas_norm:.6f} "
                    f"{w_canvas_norm:.6f} {h_canvas_norm:.6f}"
                )
        
        # Add transformer annotations
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
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        print(f"  - Saved YOLO: {output_path} ({len(annotations)} annotations)")
    
    def generate_merged_json(self, output_path: str):
        """Generate merged JSON with unified transformers"""
        print("Generating merged JSON...")
        
        merged = {
            "devices": [],
            "connections": [],
            "exposed_terminals": [],
            "layout": {
                "topology_mode": "HYBRID",
                "voltage_levels": []
            },
            "transformer_connections": {}
        }
        
        # Add voltage level info
        for i, level in enumerate(self.levels):
            merged["layout"]["voltage_levels"].append({
                "name": level.voltage_label,
                "topology": level.topology_type,
                "index": i
            })
        
        # Load and merge devices from each level JSON
        for i, level in enumerate(self.levels):
            with open(level.json_path, 'r') as f:
                level_json = json.load(f)
            
            # Get exposed terminals
            exposed_data = level_json.get("layout", {}).get("exposed_terminals", {})
            if isinstance(exposed_data, dict):
                upstream = exposed_data.get("upstream", [])
                downstream = exposed_data.get("downstream", [])
            else:
                upstream = [t for t in exposed_data if "TR_UP" in t]
                downstream = [t for t in exposed_data if "TR_DN" in t]
            
            # Add devices (skip transformers that will be unified)
            for device in level_json.get("devices", []):
                dev_id = device["id"]
                
                # Skip transformers that are part of inter-level connections
                if dev_id in upstream or dev_id in downstream:
                    continue
                
                # Add with prefixed ID
                new_device = device.copy()
                new_device["id"] = f"{level.voltage_label}_{dev_id}"
                new_device["original_id"] = dev_id
                new_device["voltage_level_index"] = i
                merged["devices"].append(new_device)
            
            # Add connections (will update with unified transformer IDs later)
            for conn in level_json.get("connections", []):
                # Skip for now, will handle after adding unified transformers
                pass
        
        # Add unified transformers
        for tr in self.transformers:
            merged["devices"].append({
                "id": tr.id,
                "label": tr.label,
                "type": "Transformer",
                "voltage_from": self.levels[tr.from_level_idx].voltage_label,
                "voltage_to": self.levels[tr.to_level_idx].voltage_label,
                "connects_levels": [tr.from_level_idx, tr.to_level_idx]
            })
            merged["exposed_terminals"].append(tr.id)
            merged["transformer_connections"][tr.id] = [tr.from_level_idx, tr.to_level_idx]
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(merged, f, indent=2)
        
        print(f"  - Saved JSON: {output_path}")
    
    def build(self, svg_path: str, json_path: str, yolo_path: str):
        """
        Main build pipeline
        
        Args:
            svg_path: Output path for stitched SVG
            json_path: Output path for merged JSON
            yolo_path: Output path for YOLO annotations
        """
        print("\n" + "="*70)
        print("HYBRID SVG BUILDER")
        print("="*70)
        
        # Step 1: Render all levels
        self.render_levels()
        
        # Step 2: Calculate layout
        self.calculate_layout()
        
        # Step 3: Identify transformers
        self.identify_transformers()
        
        # Step 4: Generate outputs
        self.generate_svg(svg_path)
        self.generate_yolo(yolo_path)
        self.generate_merged_json(json_path)
        
        print("\n" + "="*70)
        print("BUILD COMPLETE!")
        print("="*70)
        print(f"  SVG:  {svg_path}")
        print(f"  JSON: {json_path}")
        print(f"  YOLO: {yolo_path}")
        print()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create builder
    builder = HybridSVGBuilder()
    
    # Add voltage levels (top to bottom, highest to lowest voltage)
    builder.add_level(
        json_path="hybrid_output/json/220kv_0.json",
        topology_type="SINGLE_BUS",
        voltage_label="220kV"
    )
    
    builder.add_level(
        json_path="hybrid_output/json/132kv_1.json",
        topology_type="SINGLE_BUS",
        voltage_label="132kV"
    )
    
    # Build hybrid diagram
    builder.build(
        svg_path="hybrid_output/two_level.svg",
        json_path="hybrid_output/two_level.json",
        yolo_path="hybrid_output/two_level.txt"
    )