"""
test_line_extraction_debug.py
=============================
Standalone test script for line extraction with detailed debug output.
Shows intermediate images at each processing step.

Usage:
    python test_line_extraction_debug.py <image_path> [--output debug_output]
"""

import os
import sys
import cv2
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from line_extraction import (
    detect_components,
    load_symbol_masks,
    mask_components_with_symbols,
    detect_and_mask_text,
    detect_lines,
    build_connectivity_graph,
)


def save_debug_image(image, filename, output_dir):
    """Save debug image with text overlay."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, image)
    print(f"  ✓ Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug line extraction with intermediate visualizations"
    )
    parser.add_argument("image", help="Path to SLD image")
    parser.add_argument("--model", default=None, help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence")
    parser.add_argument("--output", default="debug_line_extraction", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("LINE EXTRACTION DEBUG - STEP BY STEP")
    print("=" * 70)
    
    # Load image
    print("\n[STEP 0] Loading image...")
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    print(f"  Image shape: {image.shape}")
    save_debug_image(image, "00_original.png", output_dir)
    
    # Default model
    if args.model is None:
        args.model = os.path.join(
            os.path.dirname(__file__),
            "..", "models", "yolo_model", "runs", "detect", "train", "weights", "best.pt"
        )
    
    # Step 1: Detect components
    print("\n[STEP 1] Detecting components with YOLO...")
    boxes = detect_components(args.image, args.model, conf=args.conf)
    
    # Visualize boxes
    img_with_boxes = image.copy()
    for i, box in enumerate(boxes):
        cv2.rectangle(img_with_boxes, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, f"{i}", (box.x1, box.y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    save_debug_image(img_with_boxes, "01_detected_boxes.png", output_dir)
    
    # Step 2: Load symbol masks
    print("\n[STEP 2] Loading symbol masks...")
    symbol_masks = load_symbol_masks()
    
    # Step 3: Mask components using symbols
    print("\n[STEP 3] Masking components with symbol shapes...")
    masked_comp = mask_components_with_symbols(image, boxes, symbol_masks)
    save_debug_image(masked_comp, "02_masked_components.png", output_dir)
    
    # Show difference
    diff = cv2.absdiff(image, masked_comp)
    save_debug_image(diff, "02b_component_mask_diff.png", output_dir)
    
    # Step 4: Mask text
    print("\n[STEP 4] Masking text regions...")
    masked_text = detect_and_mask_text(masked_comp.copy(), margin=0)
    save_debug_image(masked_text, "03_masked_text.png", output_dir)
    
    # Show text mask difference
    text_diff = cv2.absdiff(masked_comp, masked_text)
    save_debug_image(text_diff, "03b_text_mask_diff.png", output_dir)
    
    # Step 5: Preprocess for line detection
    print("\n[STEP 5] Preprocessing for line detection...")
    if len(masked_text.shape) == 3:
        gray = cv2.cvtColor(masked_text, cv2.COLOR_BGR2GRAY)
    else:
        gray = masked_text.copy()
    save_debug_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "04_grayscale.png", output_dir)
    
    # Apply bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    save_debug_image(cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR), "04b_bilateral_filtered.png", output_dir)
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel, iterations=1)
    save_debug_image(cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR), "04c_morphological_closed.png", output_dir)
    
    # Invert
    inverted = 255 - closed
    save_debug_image(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR), "04d_inverted.png", output_dir)
    
    # Canny edges
    edges = cv2.Canny(inverted, 50, 150)
    save_debug_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "04e_canny_edges.png", output_dir)
    
    # Dilate edges
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel2, iterations=1)
    save_debug_image(cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR), "04f_dilated_edges.png", output_dir)
    
    # Step 6: Detect lines
    print("\n[STEP 6] Detecting lines...")
    segments = detect_lines(masked_text, threshold=20, min_length=30, max_gap=20)
    
    # Visualize lines
    img_lines = image.copy()
    for seg in segments:
        cv2.line(img_lines, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 0, 255), 2)
    save_debug_image(img_lines, "05_detected_lines.png", output_dir)
    
    # Step 7: Build connectivity
    print("\n[STEP 7] Building connectivity graph...")
    connections = build_connectivity_graph(segments, boxes, endpoint_tolerance=10)
    
    # Visualize connectivity
    img_conn = image.copy()
    for i, box in enumerate(boxes):
        cv2.circle(img_conn, box.center, 6, (0, 255, 0), -1)
    
    colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (128, 0, 128)]
    for i, conn in enumerate(connections):
        color = colors[i % len(colors)]
        cv2.line(img_conn, conn.from_point, conn.to_point, color, 3)
        for idx, (box_id, box_class) in enumerate(conn.connected_boxes):
            cv2.putText(img_conn, f"Box{box_id}", 
                       (conn.from_point[0] + 10, conn.from_point[1] + 15 * (idx + 1)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    save_debug_image(img_conn, "06_connectivity_graph.png", output_dir)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Components detected: {len(boxes)}")
    print(f"Line segments: {len(segments)}")
    print(f"Connections: {len(connections)}")
    print(f"\nAll debug images saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
