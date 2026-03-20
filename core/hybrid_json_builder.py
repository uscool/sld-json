"""
HYBRID JSON BUILDER
===================

Generates and merges topology JSONs for multi-voltage-level substations.
Creates unified transformers connecting adjacent voltage levels.

Usage:
    builder = HybridJSONBuilder()
    builder.add_level(Voltage.V220KV, "SINGLE_BUS", {...params...}, "220kV")
    builder.add_level(Voltage.V132KV, "SINGLE_BUS", {...params...}, "132kV")
    merged_json_path = builder.generate("output/merged.json")
"""

import core.topology as t
from core.grammar import Voltage
import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class LevelConfig:
    """Configuration for a single voltage level"""
    voltage: Voltage
    topology_type: str
    params: Dict[str, Any]
    voltage_label: str  # e.g., "220kV"


class HybridJSONBuilder:
    """
    Builds merged JSON for multi-voltage substations with unified transformers
    """
    
    TOPOLOGY_FUNCTIONS = {
        "SINGLE_BUS": t.single_bus,
        "DOUBLE_BUS": t.double_bus,
        "BREAKER_AND_HALF": t.breaker_and_half,
        "MAIN_AND_TRANSFER": t.main_and_transfer,
        "RING_BUS": t.ring_bus,
        "MESH": t.mesh,
    }
    
    def __init__(self):
        self.levels: List[LevelConfig] = []
        self.level_json_paths: List[str] = []
        self.level_jsons: List[Dict] = []
    
    def add_level(
        self, 
        voltage: Voltage, 
        topology_type: str, 
        params: Dict[str, Any],
        voltage_label: str
    ):
        """
        Add a voltage level to the hybrid substation
        
        Args:
            voltage: Voltage level (Voltage.V220KV, etc.)
            topology_type: Type of topology ("SINGLE_BUS", "DOUBLE_BUS", etc.)
            params: Parameters for topology generator (num_upstream_transformers, etc.)
            voltage_label: Human-readable label ("220kV", "132kV", etc.)
        """
        self.levels.append(LevelConfig(voltage, topology_type, params, voltage_label))
    
    def generate(self, output_path: str, temp_dir: str = "temp_json") -> str:
        """
        Generate individual JSONs and merge them
        
        Args:
            output_path: Path for merged JSON output
            temp_dir: Directory for temporary individual JSONs
            
        Returns:
            Path to merged JSON file
        """
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # print(f"\n{'='*70}")
        # print(f"HYBRID JSON BUILDER")
        # print(f"{'='*70}")
        # print(f"Generating {len(self.levels)} voltage levels...")
        
        # Step 1: Generate individual JSONs
        for i, level in enumerate(self.levels):
            json_path = f"{temp_dir}/{level.voltage_label}_{i}.json"
            
            # print(f"  - Level {i}: {level.voltage_label} ({level.topology_type})")
            
            # Call topology generator
            topology_func = self.TOPOLOGY_FUNCTIONS[level.topology_type]
            topology_func(
                voltage_level=level.voltage,
                filepath=json_path,
                **level.params
            )
            
            self.level_json_paths.append(json_path)
            
            # Load JSON
            with open(json_path, 'r') as f:
                self.level_jsons.append(json.load(f))
        
        # Step 2: Merge JSONs
        # print(f"\nMerging JSONs with unified transformers...")
        merged_json = self._merge_jsons()
        
        # Step 3: Save merged JSON
        with open(output_path, 'w') as f:
            json.dump(merged_json, f, indent=2)
        
        # print(f"  ✓ Saved merged JSON: {output_path}")
        # print(f"  ✓ Total devices: {len(merged_json['devices'])}")
        # print(f"  ✓ Unified transformers: {len(merged_json['transformer_connections'])}")
        
        return output_path
    
    def _merge_jsons(self) -> Dict:
        """
        Merge all level JSONs into single unified JSON with merged transformers
        """
        merged = {
            "devices": [],
            "connections": [],
            "exposed_terminals": [],
            "layout": {
                "topology_mode": "HYBRID",
                "voltage_levels": []
            },
            "transformer_connections": {},
            "unified_transformers": {}
        }
        
        # Track transformer mapping
        transformer_id_mapping = {}  # old_prefixed_id -> new_unified_id
        
        # First pass: Collect exposed terminals
        all_upstream_transformers = {}  # level -> [transformer_ids]
        all_downstream_transformers = {}  # level -> [transformer_ids]
        
        for i, (level_json, level) in enumerate(zip(self.level_jsons, self.levels)):
            # Add voltage level info
            merged["layout"]["voltage_levels"].append({
                "name": level.voltage_label,
                "topology": level.topology_type,
                "voltage": str(level.voltage),
                "index": i
            })
            
            # Get exposed terminals
            exposed_data = level_json.get("layout", {}).get("exposed_terminals", {})
            if isinstance(exposed_data, dict):
                upstream = exposed_data.get("upstream", [])
                downstream = exposed_data.get("downstream", [])
            else:
                upstream = [t for t in exposed_data if "TR_UP" in t]
                downstream = [t for t in exposed_data if "TR_DN" in t]
            
            all_upstream_transformers[i] = upstream
            all_downstream_transformers[i] = downstream
        
        # Second pass: Create unified transformers
        transformer_counter = 1
        
        for i in range(len(self.levels) - 1):
            from_level_idx = i
            to_level_idx = i + 1
            
            from_voltage = self.levels[from_level_idx].voltage_label
            to_voltage = self.levels[to_level_idx].voltage_label
            
            downstream_trs = all_downstream_transformers.get(from_level_idx, [])
            upstream_trs = all_upstream_transformers.get(to_level_idx, [])
            
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
                
                # Create unified transformer device
                merged["unified_transformers"][unified_id] = {
                    "id": unified_id,
                    "label": unified_label,
                    "type": "Transformer",
                    "voltage_from": str(self.levels[from_level_idx].voltage),
                    "voltage_to": str(self.levels[to_level_idx].voltage),
                    "connects_levels": [from_level_idx, to_level_idx],
                    "from_device_id": f"{from_voltage}_{dn_id}",
                    "to_device_id": f"{to_voltage}_{up_id}",
                }
                
                merged["transformer_connections"][unified_id] = [from_level_idx, to_level_idx]
                
                transformer_counter += 1
        
        # Third pass: Add all non-transformer devices
        for i, (level_json, level) in enumerate(zip(self.level_jsons, self.levels)):
            level_prefix = level.voltage_label
            
            for device in level_json["devices"]:
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
        
        # Add unified transformers to devices
        for unified_tr in merged["unified_transformers"].values():
            merged["devices"].append(unified_tr)
            merged["exposed_terminals"].append(unified_tr["id"])
        
        # Fourth pass: Add connections
        for i, (level_json, level) in enumerate(zip(self.level_jsons, self.levels)):
            level_prefix = level.voltage_label
            
            for conn in level_json["connections"]:
                from_id = f"{level_prefix}_{conn['from']}"
                to_id = f"{level_prefix}_{conn['to']}"
                
                # Replace with unified transformer IDs if applicable
                from_id = transformer_id_mapping.get(from_id, from_id)
                to_id = transformer_id_mapping.get(to_id, to_id)
                
                # Skip internal transformer connections (they're now unified)
                if from_id == to_id:
                    continue
                
                merged["connections"].append({
                    "from": from_id,
                    "to": to_id
                })
        
        return merged


# Example usage
if __name__ == "__main__":
    builder = HybridJSONBuilder()
    
    # Add voltage levels (highest to lowest)

    builder.add_level(
        voltage=Voltage.V220KV,
        topology_type="MESH",
        params={
            "num_buses": 6,
            "num_upstream_transformers": 0,
            "num_generators": 3,
            "num_downstream_transformers": 3,
            "num_feeders": 3
        },
        voltage_label="220kV"
    )

    builder.add_level(
        voltage=Voltage.V132KV,
        topology_type="RING_BUS",
        params={
            "num_upstream_transformers": 3,
            "num_generators": 0,
            "num_downstream_transformers": 2,
            "num_feeders": 2
        },
        voltage_label="132kV"
    )

    builder.add_level(
        voltage=Voltage.V66KV,
        topology_type="BREAKER_AND_HALF",
        params={
            "bayCount": 3,
            "num_upstream_transformers": 2,
            "num_generators": 0,
            "num_downstream_transformers": 1,
            "num_feeders": 2
        },
        voltage_label="66kV"
    )

    builder.add_level(
        voltage=Voltage.V33KV,
        topology_type="MAIN_AND_TRANSFER",
        params={
            "num_upstream_transformers": 1,
            "num_generators": 1,
            "num_downstream_transformers": 2,
            "num_feeders": 3
        },
        voltage_label="33kV"
    )

    builder.add_level(
        voltage=Voltage.V11KV,
        topology_type="DOUBLE_BUS",
        params={
            "breaker_per_bay": 2,
            "num_upstream_transformers": 2,
            "num_generators": 1,
            "num_downstream_transformers": 0,
            "num_feeders": 4
        },
        voltage_label="11kV"
    )
    # Generate merged JSON
    merged_path = builder.generate(
        output_path="output_hybrid/json/hybrid.json", 
        temp_dir="output_hybrid/temp_json" 
    )
    # print(f"\n✓ Done! Merged JSON saved to: {merged_path}")