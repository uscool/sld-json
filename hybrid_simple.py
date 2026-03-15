"""
Hybrid Topology Generator
--------------------------
Generates multi-voltage-level substation diagrams by stitching standalone topologies
through transformers. Handles JSON merging, SVG rendering, layout optimization,
and YOLO annotation export.
"""

import core.topology as t
from core.grammar import Voltage
import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import copy

# Import all renderers
from core.render_brk_hlf import render_breaker_and_half_svg
from core.render_dbl_bs import render_double_bus_svg
from core.render_mn_trfr import render_main_and_transfer_svg
from core.render_mesh import render_mesh_svg
from core.render_ring_bs import render_ring_bus_svg
from core.render_sngl_bs import render_single_bus_svg


@dataclass
class TopologyConfig:
    """Configuration for a single topology in the hybrid system"""
    topology_type: str  # "SINGLE_BUS", "DOUBLE_BUS", etc.
    voltage_level: Voltage
    params: Dict[str, Any]  # Parameters for the topology generator function
    level_name: str  # e.g., "400kV", "220kV", "132kV"


@dataclass
class RendererResult:
    """Result from rendering a topology in hybrid mode"""
    svg_content: str
    connection_points: Dict[str, Tuple[float, float]]  # transformer_id -> (x, y)
    width: float
    height: float
    yolo_annotations: List[str]  # YOLO format annotations


class HybridTopologyGenerator:
    """
    Generates hybrid multi-voltage topologies by:
    1. Building standalone topology JSONs
    2. Rendering each in hybrid mode (extending transformer stubs to boundary)
    3. Stitching JSONs together
    4. Placing SVGs hierarchically on canvas
    5. Drawing transformers to connect levels
    6. Exporting final SVG + YOLO annotations
    """
    
    TOPOLOGY_FUNCTIONS = {
        "SINGLE_BUS": t.single_bus,
        "DOUBLE_BUS": t.double_bus,
        "BREAKER_AND_HALF": t.breaker_and_half,
        "MAIN_AND_TRANSFER": t.main_and_transfer,
        "RING_BUS": t.ring_bus,
        "MESH": t.mesh,
    }
    
    RENDERER_FUNCTIONS = {
        "SINGLE_BUS": render_single_bus_svg,
        "DOUBLE_BUS": render_double_bus_svg,
        "BREAKER_AND_HALF": render_breaker_and_half_svg,
        "MAIN_AND_TRANSFER": render_main_and_transfer_svg,
        "RING_BUS": render_ring_bus_svg,
        "MESH": render_mesh_svg,
    }
    
    def __init__(self, output_dir: str = "hybrid_output"):
        """Initialize generator with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/json", exist_ok=True)
        os.makedirs(f"{output_dir}/svg", exist_ok=True)
        os.makedirs(f"{output_dir}/yolo", exist_ok=True)
        
    def generate_topology_json(self, config: TopologyConfig, index: int) -> Tuple[str, Dict]:
        """
        Generate JSON for a single topology
        
        Returns:
            Tuple of (json_filepath, json_data)
        """
        topology_func = self.TOPOLOGY_FUNCTIONS[config.topology_type]
        
        # Generate JSON file path
        json_path = f"{self.output_dir}/json/{config.level_name}_{index}.json"
        
        # Call topology generator with parameters + filepath
        # The function now handles JSON export internally and returns the filepath
        topology_func(voltage_level=config.voltage_level, filepath=json_path, **config.params)
        
        # Load and return JSON data
        with open(json_path) as f:
            json_data = json.load(f)
        
        return json_path, json_data
    
    def render_topology_hybrid(
        self, 
        config: TopologyConfig, 
        json_data: Dict,
        index: int
    ) -> RendererResult:
        """
        Render a topology in hybrid mode
        
        Returns:
            RendererResult with SVG content, connection points, dimensions, and YOLO data
        """
        svg_path = f"{self.output_dir}/svg/{config.level_name}_{index}_hybrid.svg"
        yolo_path = f"{self.output_dir}/yolo/{config.level_name}_{index}.txt"
        
        renderer = self.RENDERER_FUNCTIONS[config.topology_type]
        
        # Call renderer in hybrid mode
        # Your renderers return boundary_pts dict: {device_id: {"label": str, "x": float, "y": float}}
        boundary_pts = renderer(json_data, svg_path, mode="hybrid")
        
        # Read the generated SVG file
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        
        # Extract width and height from SVG
        import re
        width_match = re.search(r'width="([\d.]+)"', svg_content)
        height_match = re.search(r'height="([\d.]+)"', svg_content)
        width = float(width_match.group(1)) if width_match else 800
        height = float(height_match.group(1)) if height_match else 600
        
        # Convert boundary_pts to connection_points format
        # Your format: {device_id: {"label": str, "x": float, "y": float}}
        # Expected format: {device_id: (x, y)}
        connection_points = {}
        if boundary_pts:
            for device_id, pt_data in boundary_pts.items():
                connection_points[device_id] = (pt_data["x"], pt_data["y"])
        
        # Read YOLO annotations if generated
        yolo_annotations = []
        if os.path.exists(yolo_path):
            with open(yolo_path) as f:
                yolo_annotations = [line.strip() for line in f if line.strip()]
        
        return RendererResult(
            svg_content=svg_content,
            connection_points=connection_points,
            width=width,
            height=height,
            yolo_annotations=yolo_annotations
        )
    
    def merge_jsons(self, json_datas: List[Dict], configs: List[TopologyConfig]) -> Dict:
        """
        Merge multiple topology JSONs into a single unified JSON
        Transformers connecting levels are deduplicated and relabeled
        """
        merged = {
            "devices": [],
            "connections": [],
            "exposed_terminals": [],
            "layout": {
                "topology_mode": "HYBRID",
                "voltage_levels": []
            }
        }
        
        # Track transformers that connect levels
        # Maps: (from_level, to_level, index) -> unified_transformer_info
        unified_transformers = {}
        transformer_id_mapping = {}  # old_prefixed_id -> new_unified_id
        
        # First pass: collect all devices and identify transformer pairs
        all_upstream_transformers = {}  # level -> [transformer_ids]
        all_downstream_transformers = {}  # level -> [transformer_ids]
        
        for i, (json_data, config) in enumerate(zip(json_datas, configs)):
            level_prefix = config.level_name
            
            # Add voltage level info
            merged["layout"]["voltage_levels"].append({
                "name": level_prefix,
                "topology": config.topology_type,
                "voltage": str(config.voltage_level),
                "index": i
            })
            
            # Get exposed terminals - handle both formats
            exposed_data = json_data.get("layout", {}).get("exposed_terminals", [])
            
            # Check if it's the new dict format or old list format
            if isinstance(exposed_data, dict):
                upstream = exposed_data.get("upstream", [])
                downstream = exposed_data.get("downstream", [])
            else:
                # Old format - flat list, infer from device IDs
                upstream = [t for t in exposed_data if "TR_UP" in t]
                downstream = [t for t in exposed_data if "TR_DN" in t]
            
            all_upstream_transformers[i] = upstream
            all_downstream_transformers[i] = downstream
        
        # Create unified transformers connecting adjacent levels
        transformer_counter = 1
        for i in range(len(configs) - 1):
            # Level i downstream transformers connect to level i+1 upstream transformers
            from_level = i
            to_level = i + 1
            
            from_voltage = configs[from_level].level_name
            to_voltage = configs[to_level].level_name
            
            downstream_trs = all_downstream_transformers.get(from_level, [])
            upstream_trs = all_upstream_transformers.get(to_level, [])
            
            # Match transformers by index
            num_connections = min(len(downstream_trs), len(upstream_trs))
            
            for idx in range(num_connections):
                dn_id = downstream_trs[idx]
                up_id = upstream_trs[idx]
                
                # Create unified transformer ID
                unified_id = f"{from_voltage}_{to_voltage}_TR_{transformer_counter:02d}"
                unified_label = f"{from_voltage}/{to_voltage}-TR-{transformer_counter:02d}"
                
                # Store mapping
                transformer_id_mapping[f"{from_voltage}_{dn_id}"] = unified_id
                transformer_id_mapping[f"{to_voltage}_{up_id}"] = unified_id
                
                # Find the transformer devices to get labels
                from_json = json_datas[from_level]
                to_json = json_datas[to_level]
                
                dn_device = next((d for d in from_json["devices"] if d["id"] == dn_id), None)
                up_device = next((d for d in to_json["devices"] if d["id"] == up_id), None)
                
                # Create unified transformer device
                unified_transformers[unified_id] = {
                    "id": unified_id,
                    "label": unified_label,
                    "type": "Transformer",
                    "voltage_from": str(configs[from_level].voltage_level),
                    "voltage_to": str(configs[to_level].voltage_level),
                    "connects_levels": [from_level, to_level],
                    "from_device_id": f"{from_voltage}_{dn_id}",
                    "to_device_id": f"{to_voltage}_{up_id}",
                }
                
                transformer_counter += 1
        
        # Second pass: add all non-transformer devices with prefixed IDs
        for i, (json_data, config) in enumerate(zip(json_datas, configs)):
            level_prefix = config.level_name
            
            for device in json_data["devices"]:
                original_id = device["id"]
                prefixed_id = f"{level_prefix}_{original_id}"
                
                # Skip transformers that are part of inter-level connections
                if prefixed_id in transformer_id_mapping:
                    continue
                
                # Add regular device
                new_device = device.copy()
                new_device["id"] = prefixed_id
                new_device["original_id"] = original_id
                new_device["voltage_level_index"] = i
                merged["devices"].append(new_device)
        
        # Add unified transformers
        for unified_tr in unified_transformers.values():
            merged["devices"].append(unified_tr)
            merged["exposed_terminals"].append(unified_tr["id"])
        
        # Third pass: add connections, replacing transformer IDs with unified IDs
        for i, (json_data, config) in enumerate(zip(json_datas, configs)):
            level_prefix = config.level_name
            
            for conn in json_data["connections"]:
                from_id = f"{level_prefix}_{conn['from']}"
                to_id = f"{level_prefix}_{conn['to']}"
                
                # Replace with unified transformer IDs if applicable
                from_id = transformer_id_mapping.get(from_id, from_id)
                to_id = transformer_id_mapping.get(to_id, to_id)
                
                # Skip internal transformer connections (they're now unified)
                if from_id == to_id:
                    continue
                
                new_conn = {
                    "from": from_id,
                    "to": to_id
                }
                merged["connections"].append(new_conn)
        
        # Build transformer_connections map for layout
        merged["transformer_connections"] = {
            tr_id: tr_data["connects_levels"]
            for tr_id, tr_data in unified_transformers.items()
        }
        
        # Store transformer mapping for reference
        merged["transformer_id_mapping"] = transformer_id_mapping
        merged["unified_transformers"] = unified_transformers
        
        return merged
    
    def calculate_layout(
        self, 
        renderer_results: List[RendererResult],
        configs: List[TopologyConfig]
    ) -> Dict[int, Tuple[float, float]]:
        """
        Calculate positions for each voltage level on the canvas
        Layouts hierarchically from top (highest voltage) to bottom (lowest voltage)
        
        Returns:
            Dict mapping level_index -> (x_offset, y_offset)
        """
        layout = {}
        current_y = 100  # Start with top margin
        max_width = max(r.width for r in renderer_results)
        
        VERTICAL_SPACING = 150  # Space between levels for transformers
        
        for i, (result, config) in enumerate(zip(renderer_results, configs)):
            # Center horizontally
            x_offset = (max_width - result.width) / 2 + 50
            y_offset = current_y
            
            layout[i] = (x_offset, y_offset)
            
            # Move down for next level
            current_y += result.height + VERTICAL_SPACING
        
        return layout
    
    def optimize_transformer_placement(
        self,
        transformer_map: Dict[str, List[int]],
        layout_positions: Dict[int, Tuple[float, float]],
        connection_points: Dict[int, Dict[str, Tuple[float, float]]],
        merged_json: Dict
    ) -> Dict[str, Dict]:
        """
        Optimize transformer placement to minimize wire crossings
        
        Returns:
            Dict mapping transformer_id -> {
                'x': x_position,
                'y': y_position,
                'from_point': (x, y),
                'to_point': (x, y),
                'label': str
            }
        """
        transformer_placements = {}
        unified_transformers = merged_json.get("unified_transformers", {})
        
        for tr_id, levels in transformer_map.items():
            if len(levels) != 2:
                continue
            
            from_level, to_level = levels
            
            # Get transformer data
            tr_data = unified_transformers.get(tr_id, {})
            from_device_id = tr_data.get("from_device_id", "")
            to_device_id = tr_data.get("to_device_id", "")
            
            # Extract original IDs (without level prefix)
            from_original = from_device_id.split('_', 1)[1] if '_' in from_device_id else from_device_id
            to_original = to_device_id.split('_', 1)[1] if '_' in to_device_id else to_device_id
            
            # Get connection points from renderers
            from_points = connection_points.get(from_level, {})
            to_points = connection_points.get(to_level, {})
            
            from_point = from_points.get(from_original, (400, 0))
            to_point = to_points.get(to_original, (400, 0))
            
            # Adjust for layout offsets
            from_x_offset, from_y_offset = layout_positions[from_level]
            to_x_offset, to_y_offset = layout_positions[to_level]
            
            from_abs = (from_point[0] + from_x_offset, from_point[1] + from_y_offset)
            to_abs = (to_point[0] + to_x_offset, to_point[1] + to_y_offset)
            
            # Place transformer in the middle
            tr_x = (from_abs[0] + to_abs[0]) / 2
            tr_y = (from_abs[1] + to_abs[1]) / 2
            
            label = tr_data.get("label", tr_id)
            
            transformer_placements[tr_id] = {
                'x': tr_x,
                'y': tr_y,
                'from_point': from_abs,
                'to_point': to_abs,
                'label': label
            }
        
        return transformer_placements
    
    def generate_transformer_svg(
        self,
        placement: Dict,
        width: float = 60,
        height: float = 80
    ) -> str:
        """Generate SVG for a transformer symbol"""
        x = placement['x'] - width / 2
        y = placement['y'] - height / 2
        
        svg = f"""
        <!-- Transformer {placement['label']} -->
        <g id="transformer_{placement['label']}">
            <!-- Transformer body -->
            <rect x="{x}" y="{y}" width="{width}" height="{height}" 
                  fill="none" stroke="black" stroke-width="2"/>
            <!-- Windings -->
            <circle cx="{x + width/4}" cy="{y + height/3}" r="{width/6}" 
                    fill="none" stroke="black" stroke-width="1.5"/>
            <circle cx="{x + 3*width/4}" cy="{y + 2*height/3}" r="{width/6}" 
                    fill="none" stroke="black" stroke-width="1.5"/>
            <!-- Label -->
            <text x="{placement['x']}" y="{y - 10}" 
                  text-anchor="middle" font-size="12" font-family="Arial">{placement['label']}</text>
        </g>
        """
        return svg
    
    def generate_connection_line_svg(
        self,
        from_point: Tuple[float, float],
        to_point: Tuple[float, float]
    ) -> str:
        """Generate SVG for a connection line"""
        return f"""
        <line x1="{from_point[0]}" y1="{from_point[1]}" 
              x2="{to_point[0]}" y2="{to_point[1]}" 
              stroke="black" stroke-width="2"/>
        """
    
    def stitch_svg(
        self,
        renderer_results: List[RendererResult],
        layout_positions: Dict[int, Tuple[float, float]],
        transformer_placements: Dict[str, Dict],
        output_path: str
    ) -> None:
        """
        Stitch all SVGs together into final hybrid SVG
        """
        # Calculate canvas size
        max_width = max(
            pos[0] + result.width 
            for pos, result in zip(layout_positions.values(), renderer_results)
        ) + 100
        max_height = max(
            pos[1] + result.height 
            for pos, result in zip(layout_positions.values(), renderer_results)
        ) + 100
        
        # Start SVG
        svg_parts = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_width}" height="{max_height}" viewBox="0 0 {max_width} {max_height}">',
            f'<rect width="100%" height="100%" fill="white"/>',
        ]
        
        # Add each topology SVG
        for i, (result, (x_offset, y_offset)) in enumerate(zip(renderer_results, layout_positions.values())):
            svg_parts.append(f'<!-- Voltage Level {i} -->')
            svg_parts.append(f'<g transform="translate({x_offset},{y_offset})">')
            # Extract inner content from SVG (remove outer <svg> tags)
            content = result.svg_content
            # Simple extraction - in production, use XML parser
            if '<svg' in content and '</svg>' in content:
                start = content.find('>') + 1
                end = content.rfind('</svg>')
                content = content[start:end]
            svg_parts.append(content)
            svg_parts.append('</g>')
        
        # Add transformer symbols and connections
        svg_parts.append('<!-- Transformers and Connections -->')
        for tr_id, placement in transformer_placements.items():
            # Draw connection lines
            svg_parts.append(self.generate_connection_line_svg(
                placement['from_point'],
                (placement['x'], placement['y'])
            ))
            svg_parts.append(self.generate_connection_line_svg(
                (placement['x'], placement['y']),
                placement['to_point']
            ))
            # Draw transformer
            svg_parts.append(self.generate_transformer_svg(placement))
        
        svg_parts.append('</svg>')
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(svg_parts))
    
    def merge_yolo_annotations(
        self,
        renderer_results: List[RendererResult],
        layout_positions: Dict[int, Tuple[float, float]],
        transformer_placements: Dict[str, Dict],
        canvas_width: float,
        canvas_height: float,
        output_path: str
    ) -> None:
        """
        Merge YOLO annotations from all levels and add transformer annotations
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All coordinates normalized to [0, 1]
        """
        annotations = []
        
        # Class IDs
        CLASS_IDS = {
            "gridsource": 0,
            "transformer": 1,
            "breaker": 2,
            "feeder": 3,
            "capacitor": 4,
            "isolator": 5,
            "bus": 6
        }
        
        # Process each level's YOLO annotations
        for i, (result, (x_offset, y_offset)) in enumerate(zip(renderer_results, layout_positions.values())):
            for annotation in result.yolo_annotations:
                parts = annotation.split()
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
                x_canvas_norm = x_canvas / canvas_width
                y_canvas_norm = y_canvas / canvas_height
                w_canvas_norm = w_abs / canvas_width
                h_canvas_norm = h_abs / canvas_height
                
                annotations.append(
                    f"{int(class_id)} {x_canvas_norm:.6f} {y_canvas_norm:.6f} "
                    f"{w_canvas_norm:.6f} {h_canvas_norm:.6f}"
                )
        
        # Add transformer annotations
        TR_WIDTH = 60
        TR_HEIGHT = 80
        
        for tr_id, placement in transformer_placements.items():
            x_norm = placement['x'] / canvas_width
            y_norm = placement['y'] / canvas_height
            w_norm = TR_WIDTH / canvas_width
            h_norm = TR_HEIGHT / canvas_height
            
            annotations.append(
                f"{CLASS_IDS['transformer']} {x_norm:.6f} {y_norm:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}"
            )
        
        # Write annotations
        with open(output_path, 'w') as f:
            f.write('\n'.join(annotations))
    
    def generate(
        self,
        configs: List[TopologyConfig],
        output_name: str = "hybrid_substation"
    ) -> str:
        """
        Main generation pipeline
        
        Args:
            configs: List of TopologyConfig, ordered from highest to lowest voltage
            output_name: Base name for output files
            
        Returns:
            Path to generated SVG file
        """
        print(f"Generating hybrid topology with {len(configs)} voltage levels...")
        
        # Step 1: Generate JSONs
        print("Step 1: Generating topology JSONs...")
        json_datas = []
        for i, config in enumerate(configs):
            print(f"  - {config.level_name} ({config.topology_type})")
            json_path, json_data = self.generate_topology_json(config, i)
            json_datas.append(json_data)
        
        # Step 2: Merge JSONs
        print("Step 2: Merging JSONs...")
        merged_json = self.merge_jsons(json_datas, configs)
        merged_json_path = f"{self.output_dir}/json/{output_name}_merged.json"
        with open(merged_json_path, 'w') as f:
            json.dump(merged_json, f, indent=2)
        print(f"  - Saved merged JSON: {merged_json_path}")
        
        # Step 3: Render each topology in hybrid mode
        print("Step 3: Rendering topologies in hybrid mode...")
        renderer_results = []
        connection_points_by_level = {}
        for i, config in enumerate(configs):
            print(f"  - Rendering {config.level_name}...")
            result = self.render_topology_hybrid(config, json_datas[i], i)
            renderer_results.append(result)
            connection_points_by_level[i] = result.connection_points
        
        # Step 4: Calculate layout
        print("Step 4: Calculating layout...")
        layout_positions = self.calculate_layout(renderer_results, configs)
        
        # Step 5: Optimize transformer placement
        print("Step 5: Optimizing transformer placement...")
        transformer_placements = self.optimize_transformer_placement(
            merged_json.get("transformer_connections", {}),
            layout_positions,
            connection_points_by_level,
            merged_json
        )
        
        # Step 6: Stitch SVGs
        print("Step 6: Stitching SVGs...")
        svg_output_path = f"{self.output_dir}/svg/{output_name}.svg"
        self.stitch_svg(
            renderer_results,
            layout_positions,
            transformer_placements,
            svg_output_path
        )
        print(f"  - Saved final SVG: {svg_output_path}")
        
        # Step 7: Merge YOLO annotations
        print("Step 7: Generating YOLO annotations...")
        canvas_width = max(
            pos[0] + result.width 
            for pos, result in zip(layout_positions.values(), renderer_results)
        ) + 100
        canvas_height = max(
            pos[1] + result.height 
            for pos, result in zip(layout_positions.values(), renderer_results)
        ) + 100
        
        yolo_output_path = f"{self.output_dir}/yolo/{output_name}.txt"
        self.merge_yolo_annotations(
            renderer_results,
            layout_positions,
            transformer_placements,
            canvas_width,
            canvas_height,
            yolo_output_path
        )
        print(f"  - Saved YOLO annotations: {yolo_output_path}")
        
        print(f"\nHybrid topology generation complete!")
        print(f"  - SVG: {svg_output_path}")
        print(f"  - JSON: {merged_json_path}")
        print(f"  - YOLO: {yolo_output_path}")
        
        return svg_output_path


# Example usage
if __name__ == "__main__":
    
    # Define a 3-level hybrid substation
    configs = [
        # 220kV level (highest voltage)
        TopologyConfig(
            topology_type="DOUBLE_BUS",
            voltage_level=Voltage.V220KV,
            params={
                "breaker_per_bay": 1,
                "num_upstream_transformers": 2,
                "num_generators": 0,
                "num_downstream_transformers": 2,
                "num_feeders": 4,
            },
            level_name="220kV"
        ),
        
        # 132kV level (middle voltage)
        TopologyConfig(
            topology_type="BREAKER_AND_HALF",
            voltage_level=Voltage.V132KV,
            params={
                "bayCount": 4,
                "num_upstream_transformers": 2,
                "num_generators": 0,
                "num_downstream_transformers": 2,
                "num_feeders": 2,
            },
            level_name="132kV"
        ),
        
        # 33kV level (lowest voltage)
        TopologyConfig(
            topology_type="RING_BUS",
            voltage_level=Voltage.V33KV,
            params={
                "num_upstream_transformers": 2,
                "num_generators": 0,
                "num_downstream_transformers": 1,
                "num_feeders": 3,
            },
            level_name="33kV"
        ),
    ]
    
    # Generate hybrid topology
    generator = HybridTopologyGenerator(output_dir="hybrid_output")
    svg_path = generator.generate(configs, output_name="three_level_substation")