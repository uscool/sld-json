"""
full_pipeline.py
================
End-to-end pipeline: Generate → Detect Symbols → Extract Lines

Orchestrates three stages:
  1. Generate synthetic SLD diagrams (standalone + hybrid)
  2. Detect symbols and text in generated diagrams
  3. Extract lines and connectivity
  4. Aggregate and visualize all results

Usage:
    python full_pipeline.py [--diagrams 10] [--output pipeline_output]
    python full_pipeline.py --standalone-only --diagrams 5
    python full_pipeline.py --hybrid-only --diagrams 3
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from collections import defaultdict

# Import pipeline modules
from diagram_service import DiagramService
from main import (
    preprocess_image_symbols,
    preprocess_image_text,
    detect_symbols_from_yolo,
    match_symbols_to_text_ray,
    export_to_json,
    draw_matches
)
from line_extraction import (
    detect_components,
    load_symbol_masks,
    mask_all_artifacts,
    detect_lines,
    build_connectivity_graph,
    visualize_results as visualize_lines,
    export_results as export_line_results,
    merge_connectivity_with_symbol_json
)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────────────────────────────────────────

class FullPipeline:
    """Orchestrates generation, detection, and line extraction."""
    
    def __init__(self, output_dir: str = "pipeline_output"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory structure
        self.stage1_dir = os.path.join(output_dir, "01_generated_diagrams")
        self.stage2_dir = os.path.join(output_dir, "02_symbol_detection")
        self.stage3_dir = os.path.join(output_dir, "03_line_extraction")
        self.results_dir = os.path.join(output_dir, "results")
        
        for d in [self.stage1_dir, self.stage2_dir, self.stage3_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Results tracking
        self.generation_results = {}
        self.symbol_detection_results = {}
        self.line_extraction_results = {}
        self.pipeline_summary = {
            "timestamp": self.timestamp,
            "stages_completed": [],
            "total_diagrams": 0,
            "errors": []
        }
    
    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 1: GENERATION
    # ───────────────────────────────────────────────────────────────────────────
    
    def stage1_generate(self, 
                       standalone_samples: int = 3,
                       hybrid_samples: int = 3,
                       include_png: bool = True) -> int:
        """
        Stage 1: Generate synthetic SLD diagrams.
        
        Returns: number of diagrams generated
        """
        print("\n" + "=" * 80)
        print("STAGE 1: GENERATING SYNTHETIC DIAGRAMS")
        print("=" * 80)
        
        # Create service with custom output directories
        service = DiagramService(
            standalone_dir=os.path.join(self.stage1_dir, "standalone"),
            hybrid_dir=os.path.join(self.stage1_dir, "hybrid"),
            standalone_samples=standalone_samples,
            hybrid_samples=hybrid_samples,
        )
        
        try:
            # Generate both standalone and hybrid
            service.generate_standalone()
            service.generate_hybrid()
            
            # Collect generated files
            standalone_dir = os.path.join(self.stage1_dir, "standalone")
            hybrid_dir = os.path.join(self.stage1_dir, "hybrid")
            
            # Find all PNG/SVG files
            diagrams = []
            for root, dirs, files in os.walk(standalone_dir):
                for f in files:
                    if f.endswith(".png"):
                        diagrams.append(os.path.join(root, f))
            
            for root, dirs, files in os.walk(hybrid_dir):
                for f in files:
                    if f.endswith(".png"):
                        diagrams.append(os.path.join(root, f))
            
            total = len(diagrams)
            self.generation_results = {
                "diagrams": diagrams,
                "count": total,
                "standalone": len([d for d in diagrams if "standalone" in d]),
                "hybrid": len([d for d in diagrams if "hybrid" in d]),
            }
            
            self.pipeline_summary["stages_completed"].append("generation")
            self.pipeline_summary["total_diagrams"] = total
            
            print(f"\n✓ Stage 1 Complete")
            print(f"  Standalone diagrams: {self.generation_results['standalone']}")
            print(f"  Hybrid diagrams: {self.generation_results['hybrid']}")
            print(f"  Total: {total}")
            
            return total
            
        except Exception as e:
            error_msg = f"Stage 1 Error: {str(e)}"
            print(f"\n✗ {error_msg}")
            self.pipeline_summary["errors"].append(error_msg)
            return 0
    
    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 2: SYMBOL DETECTION
    # ───────────────────────────────────────────────────────────────────────────
    
    def stage2_detect_symbols(self, 
                             model_path: str = None,
                             conf: float = 0.2,
                             skip_if_no_generation: bool = True) -> int:
        """
        Stage 2: Detect symbols and match to text in generated diagrams.
        
        Returns: number of diagrams successfully processed
        """
        print("\n" + "=" * 80)
        print("STAGE 2: SYMBOL DETECTION & TEXT MATCHING")
        print("=" * 80)
        
        if not self.generation_results:
            if skip_if_no_generation:
                print("\n⚠ No diagrams generated. Run Stage 1 first.")
                return 0
            else:
                print("\n⚠ No diagrams from Stage 1.")
                return 0
        
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..", "models", "yolo_model", "runs", "detect", "train", "weights", "best.pt"
            )
        
        if not os.path.exists(model_path):
            error_msg = f"YOLO model not found: {model_path}"
            print(f"\n✗ {error_msg}")
            self.pipeline_summary["errors"].append(error_msg)
            return 0
        
        diagrams = self.generation_results.get("diagrams", [])
        processed = 0
        
        for diagram_idx, image_path in enumerate(diagrams, 1):
            diagram_name = os.path.basename(image_path)
            diagram_output = os.path.join(self.stage2_dir, diagram_name.replace(".png", ""))
            os.makedirs(diagram_output, exist_ok=True)
            
            try:
                print(f"\n[{diagram_idx}/{len(diagrams)}] Processing {diagram_name}...")
                
                # Step 1: Preprocess for symbols
                symbol_ready = preprocess_image_symbols(image_path)
                symbol_output_path = os.path.join(diagram_output, "preprocessed_symbols.png")
                cv2.imwrite(symbol_output_path, symbol_ready)
                
                # Step 2: Detect symbols
                symbols = detect_symbols_from_yolo(
                    image_path=symbol_output_path,
                    model_path=model_path,
                    conf=conf
                )
                
                if not symbols:
                    print(f"  ⚠ No symbols detected")
                    self.symbol_detection_results[diagram_name] = {
                        "success": False,
                        "symbols": 0,
                        "text_boxes": 0,
                        "matches": 0,
                        "error": "No symbols detected"
                    }
                    continue
                
                # Step 3: Preprocess for text
                text_ready, text_scale = preprocess_image_text(image_path)
                text_output_path = os.path.join(diagram_output, "preprocessed_text.png")
                cv2.imwrite(text_output_path, text_ready)
                
                # Step 4: Match symbols to text
                matched_results = match_symbols_to_text_ray(
                    base_image_path=image_path,
                    ocr_image_path=text_output_path,
                    ocr_scale=text_scale,
                    symbols=symbols,
                    max_distance=300,
                    beam_width=80
                )
                
                # Step 5: Export JSON (all symbols, not just matched)
                json_output_path = os.path.join(diagram_output, "sld_nodes.json")
                
                # Export all detected symbols as devices
                devices = []
                class_names = {0: "Generator", 1: "Transformer", 2: "Circuit_Breaker", 3: "Load", 4: "Capacitor", 5: "Switch"}
                for idx, sym in enumerate(symbols):
                    class_id = sym.get("class", 0)
                    class_name = class_names.get(class_id, "Unknown")
                    devices.append({
                        "id": f"device_{idx}",
                        "label": f"{class_name}_{idx}",
                        "type": class_name,
                        "voltage": "220kV",  # Default, would be matched from text
                        "bus_role": None,
                        "bay_id": None,
                        "detection": sym  # Store YOLO detection info
                    })
                
                # Also export matched results for reference
                with open(json_output_path, "w") as f:
                    json.dump({"devices": devices}, f, indent=2)
                
                # Step 6: Visualize
                vis_output_path = os.path.join(diagram_output, "matches_visualization.png")
                draw_matches(image_path, matched_results, vis_output_path)
                
                matched_count = sum(1 for r in matched_results if r["matched_text"] is not None)
                
                self.symbol_detection_results[diagram_name] = {
                    "success": True,
                    "symbols": len(symbols),
                    "text_boxes": len(matched_results),
                    "matches": matched_count,
                    "output_dir": diagram_output,
                    "json_output": json_output_path
                }
                
                print(f"  ✓ {len(symbols)} symbols, {matched_count} matches")
                processed += 1
                
            except Exception as e:
                error_msg = f"Symbol detection failed for {diagram_name}: {str(e)}"
                print(f"  ✗ {error_msg}")
                self.symbol_detection_results[diagram_name] = {
                    "success": False,
                    "error": str(e)
                }
                self.pipeline_summary["errors"].append(error_msg)
        
        self.pipeline_summary["stages_completed"].append("symbol_detection")
        
        print(f"\n✓ Stage 2 Complete")
        print(f"  Successfully processed: {processed}/{len(diagrams)}")
        
        return processed
    
    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 3: LINE EXTRACTION
    # ───────────────────────────────────────────────────────────────────────────
    
    def stage3_extract_lines(self,
                            model_path: str = None,
                            conf: float = 0.2) -> int:
        """
        Stage 3: Extract lines and connectivity from diagrams.
        
        Returns: number of diagrams successfully processed
        """
        print("\n" + "=" * 80)
        print("STAGE 3: LINE EXTRACTION & CONNECTIVITY")
        print("=" * 80)
        
        if not self.generation_results:
            print("\n⚠ No diagrams generated. Run Stage 1 first.")
            return 0
        
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..", "models", "yolo_model", "runs", "detect", "train", "weights", "best.pt"
            )
        
        if not os.path.exists(model_path):
            error_msg = f"YOLO model not found: {model_path}"
            print(f"\n✗ {error_msg}")
            self.pipeline_summary["errors"].append(error_msg)
            return 0
        
        diagrams = self.generation_results.get("diagrams", [])
        processed = 0
        
        for diagram_idx, image_path in enumerate(diagrams, 1):
            diagram_name = os.path.basename(image_path)
            diagram_output = os.path.join(self.stage3_dir, diagram_name.replace(".png", ""))
            os.makedirs(diagram_output, exist_ok=True)
            
            try:
                print(f"\n[{diagram_idx}/{len(diagrams)}] Processing {diagram_name}...")
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Could not load image: {image_path}")
                
                # Detect components
                boxes = detect_components(image_path, model_path, conf=conf)
                
                # Mask components AND text for clean line detection
                masked_image = mask_all_artifacts(image, boxes, mask_text=True)
                cv2.imwrite(os.path.join(diagram_output, "masked_image.png"), masked_image)
                
                # Detect lines
                segments = detect_lines(masked_image, threshold=50, min_length=30, max_gap=20)
                
                # Build connectivity graph
                connections = build_connectivity_graph(segments, boxes, endpoint_tolerance=10)
                
                # Visualize
                visualize_lines(image, masked_image, boxes, segments, connections, diagram_output)
                
                # Export
                export_line_results(boxes, segments, connections, diagram_output)
                
                self.line_extraction_results[diagram_name] = {
                    "success": True,
                    "components": len(boxes),
                    "line_segments": len(segments),
                    "connections": len(connections),
                    "output_dir": diagram_output
                }
                
                print(f"  ✓ {len(boxes)} components, {len(segments)} lines, {len(connections)} connections")
                processed += 1
                
            except Exception as e:
                error_msg = f"Line extraction failed for {diagram_name}: {str(e)}"
                print(f"  ✗ {error_msg}")
                self.line_extraction_results[diagram_name] = {
                    "success": False,
                    "error": str(e)
                }
                self.pipeline_summary["errors"].append(error_msg)
        
        self.pipeline_summary["stages_completed"].append("line_extraction")
        
        print(f"\n✓ Stage 3 Complete")
        print(f"  Successfully processed: {processed}/{len(diagrams)}")
        
        return processed
    
    def stage4_merge_connectivity(self) -> int:
        """
        Stage 4: Merge line extraction connectivity with symbol detection JSON.
        Directly updates sld_nodes.json files with detected connections.
        
        Returns: number of diagrams successfully merged
        """
        print("\n" + "=" * 80)
        print("STAGE 4: MERGING CONNECTIVITY WITH SYMBOL DETECTION")
        print("=" * 80)
        
        if not self.symbol_detection_results:
            print("\n⚠ No symbol detection results available")
            return 0
        
        if not self.line_extraction_results:
            print("\n⚠ No line extraction results available")
            return 0
        
        print(f"\nFound {len(self.symbol_detection_results)} symbol detection results")
        print(f"Found {len(self.line_extraction_results)} line extraction results")
        
        merged_count = 0
        
        # Iterate over diagrams processed by line extraction
        for diagram_idx, (diagram_name, line_result) in enumerate(
            self.line_extraction_results.items(), 1
        ):
            # Check if both symbol detection and line extraction succeeded
            if not line_result.get("success"):
                print(f"\n[{diagram_idx}] {diagram_name}: line extraction failed, skipping")
                continue
            
            det_result = self.symbol_detection_results.get(diagram_name, {})
            if not det_result.get("success"):
                print(f"\n[{diagram_idx}] {diagram_name}: symbol detection failed, skipping")
                continue
            
            symbol_json_path = det_result.get("json_output")
            if not symbol_json_path:
                print(f"\n[{diagram_idx}] {diagram_name}: no json_output path")
                continue
            
            if not os.path.exists(symbol_json_path):
                print(f"\n[{diagram_idx}] {diagram_name}: json file not found at {symbol_json_path}")
                continue
            
            diagram_output = line_result.get("output_dir")
            if not diagram_output:
                print(f"\n[{diagram_idx}] {diagram_name}: no line extraction output dir")
                continue
            
            try:
                print(f"\n[{diagram_idx}] {diagram_name}...", end=" ")
                
                # Load line extraction results  
                line_results_path = os.path.join(diagram_output, "line_extraction_results.json")
                if not os.path.exists(line_results_path):
                    print("(skip: no line data)")
                    continue
                
                with open(line_results_path, "r") as f:
                    line_data = json.load(f)
                
                # Pass connection dicts directly (no need to reconstruct objects)
                connections = line_data.get("connections", [])
                
                # Merge with symbol JSON (overwrites original)
                merge_connectivity_with_symbol_json(
                    symbol_json_path,
                    connections,
                    boxes=[],
                    output_path=None  # Overwrites original
                )
                
                merged_count += 1
                print("✓")
                
            except Exception as e:
                print(f"✗ ({str(e)[:60]})")
                import traceback
                self.pipeline_summary["errors"].append(
                    f"Merge failed for {diagram_name}: {str(e)}\n{traceback.format_exc()}"
                )
        
        self.pipeline_summary["stages_completed"].append("connectivity_merge")
        print(f"\n✓ Stage 4 Complete: Merged {merged_count}/{len(self.line_extraction_results)} diagrams")
        
        return merged_count
    
    # ───────────────────────────────────────────────────────────────────────────
    # RESULTS AGGREGATION & EXPORT
    # ───────────────────────────────────────────────────────────────────────────
    
    def aggregate_results(self) -> None:
        """Aggregate all pipeline results into summary JSON."""
        
        summary = {
            "metadata": {
                "timestamp": self.timestamp,
                "stages_completed": self.pipeline_summary["stages_completed"],
                "total_diagrams": self.pipeline_summary["total_diagrams"],
                "errors_count": len(self.pipeline_summary["errors"]),
            },
            "generation": {
                "total": self.generation_results.get("count", 0),
                "standalone": self.generation_results.get("standalone", 0),
                "hybrid": self.generation_results.get("hybrid", 0),
            },
            "symbol_detection": {
                "total_processed": sum(1 for r in self.symbol_detection_results.values() if r.get("success")),
                "total_failed": sum(1 for r in self.symbol_detection_results.values() if not r.get("success")),
                "total_symbols": sum(r.get("symbols", 0) for r in self.symbol_detection_results.values()),
                "total_matches": sum(r.get("matches", 0) for r in self.symbol_detection_results.values()),
            },
            "line_extraction": {
                "total_processed": sum(1 for r in self.line_extraction_results.values() if r.get("success")),
                "total_failed": sum(1 for r in self.line_extraction_results.values() if not r.get("success")),
                "total_components": sum(r.get("components", 0) for r in self.line_extraction_results.values()),
                "total_lines": sum(r.get("line_segments", 0) for r in self.line_extraction_results.values()),
                "total_connections": sum(r.get("connections", 0) for r in self.line_extraction_results.values()),
            },
            "errors": self.pipeline_summary["errors"]
        }
        
        # Save summary
        summary_path = os.path.join(self.results_dir, "pipeline_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        details_path = os.path.join(self.results_dir, "pipeline_details.json")
        with open(details_path, "w") as f:
            json.dump({
                "symbol_detection_detailed": self.symbol_detection_results,
                "line_extraction_detailed": self.line_extraction_results,
            }, f, indent=2)
        
        print(f"\n✓ Results aggregated and saved")
        return summary
    
    def print_summary(self, summary: dict) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        
        meta = summary["metadata"]
        print(f"\nTimestamp: {meta['timestamp']}")
        print(f"Stages completed: {', '.join(meta['stages_completed'])}")
        print(f"Total diagrams: {meta['total_diagrams']}")
        
        gen = summary["generation"]
        print(f"\n📊 Generation:")
        print(f"  Total: {gen['total']}")
        print(f"  Standalone: {gen['standalone']}")
        print(f"  Hybrid: {gen['hybrid']}")
        
        sym = summary["symbol_detection"]
        print(f"\n🔍 Symbol Detection:")
        print(f"  Processed: {sym['total_processed']}")
        print(f"  Failed: {sym['total_failed']}")
        print(f"  Total symbols detected: {sym['total_symbols']}")
        print(f"  Total matches: {sym['total_matches']}")
        
        lines = summary["line_extraction"]
        print(f"\n📍 Line Extraction:")
        print(f"  Processed: {lines['total_processed']}")
        print(f"  Failed: {lines['total_failed']}")
        print(f"  Total components: {lines['total_components']}")
        print(f"  Total line segments: {lines['total_lines']}")
        print(f"  Total connections: {lines['total_connections']}")
        
        if summary["errors"]:
            print(f"\n⚠ Errors ({len(summary['errors'])}):")
            for err in summary["errors"]:
                print(f"  - {err}")
        
        print("\n" + "=" * 80)
        print(f"Output directory: {self.output_dir}")
        print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full SLD analysis pipeline: Generate → Detect → Extract Lines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--diagrams",
        type=int,
        default=3,
        help="Total diagrams to generate (split between standalone and hybrid)",
    )
    parser.add_argument(
        "--standalone-only",
        action="store_true",
        help="Generate only standalone diagrams",
    )
    parser.add_argument(
        "--hybrid-only",
        action="store_true",
        help="Generate only hybrid diagrams",
    )
    parser.add_argument(
        "--output",
        default="pipeline_output",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--skip-symbol-detection",
        action="store_true",
        help="Skip symbol detection stage",
    )
    parser.add_argument(
        "--skip-line-extraction",
        action="store_true",
        help="Skip line extraction stage",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to YOLO model (uses default if not specified)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Detection confidence threshold",
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FullPipeline(output_dir=args.output)
    
    # Stage 1: Generation
    half = args.diagrams // 2
    standalone_samples = half if not args.hybrid_only else (args.diagrams if args.standalone_only else 0)
    hybrid_samples = half if not args.standalone_only else (args.diagrams if args.hybrid_only else 0)
    
    total_diagrams = pipeline.stage1_generate(
        standalone_samples=standalone_samples,
        hybrid_samples=hybrid_samples
    )
    
    if total_diagrams == 0:
        print("\n✗ Pipeline failed at generation stage")
        return 1
    
    # Stage 2: Symbol Detection
    if not args.skip_symbol_detection:
        pipeline.stage2_detect_symbols(model_path=args.model, conf=args.conf)
    
    # Stage 3: Line Extraction
    if not args.skip_line_extraction:
        pipeline.stage3_extract_lines(model_path=args.model, conf=args.conf)
    
    # Stage 4: Merge Connectivity
    if not args.skip_line_extraction and not args.skip_symbol_detection:
        pipeline.stage4_merge_connectivity()
    
    # Aggregate and display results
    summary = pipeline.aggregate_results()
    pipeline.print_summary(summary)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
