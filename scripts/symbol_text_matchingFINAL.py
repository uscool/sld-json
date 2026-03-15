import cv2
import numpy as np
import os
import math
import json
import re
from paddleocr import PaddleOCR

# ==========================================
# CONFIGURATION
# ==========================================
IMG_PATH = 'image_label_matching_FINAL/double_bus_1.png'
LBL_PATH = 'image_label_matching_FINAL/double_bus_1.txt'

# Update these to match your specific YOLO classes
CLASS_MAP = {
    1: "Transformer",
    2: "Circuit_Breaker",
    3: "Feeder",
    5: "Disconnect_Switch",
    6: "Other_Switch_or_Bus"
}

# ==========================================
# 1. OCR EXTRACTION & FILTERING (PaddleOCR)
# ==========================================
def get_clean_text(image_path, min_chars=3):
    # Initialize PaddleOCR
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    print("Running PaddleOCR...")
    
    result = ocr_engine.ocr(image_path, cls=True)
    if not result or result[0] is None:
        return []

    valid_text_boxes = []
    
    for line in result[0]:
        bbox = line[0]
        text = line[1][0]
        
        # Filter: Must have at least 3 alphabet letters to ignore hallucinated symbols
        letters = [char for char in text if char.isalpha()]
        if len(text.strip()) >= min_chars and len(letters) >= 3:
            # Convert bounding box to integers
            int_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
            valid_text_boxes.append((int_bbox, text))

    return valid_text_boxes

# ==========================================
# 2. YOLO SYMBOL PARSING
# ==========================================
def get_symbol_centers(label_path, img_width, img_height):
    symbols = []
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found at {label_path}")
        return symbols

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5: continue
            
            cls_id = int(parts[0])
            cx = int(float(parts[1]) * img_width)
            cy = int(float(parts[2]) * img_height)
            symbols.append({"class": cls_id, "center": (cx, cy)})
            
    return symbols

# ==========================================
# 3. RAY CASTING MATCHING LOGIC
# ==========================================
def match_symbols_to_text_ray(image_path, label_path, max_distance=150, beam_width=30):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    h, w = img.shape[:2]

    text_data = get_clean_text(image_path)
    
    # Convert polygons to Axis-Aligned Bounding Boxes
    text_boxes = []
    for bbox, text in text_data:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        text_boxes.append({
            "text": text,
            "xmin": min(x_coords), "xmax": max(x_coords),
            "ymin": min(y_coords), "ymax": max(y_coords),
            "center": (int(np.mean(x_coords)), int(np.mean(y_coords)))
        })

    print("Parsing YOLO symbols...")
    symbols = get_symbol_centers(label_path, w, h)

    print("Matching symbols to text...")
    matched_results = []

    for sym in symbols:
        cx, cy = sym["center"]
        best_match = None
        best_match_center = None
        min_dist = float('inf')

        for tb in text_boxes:
            hit_dist = float('inf')

            # Horizontal Beam Check (Left/Right)
            if (tb["ymin"] - beam_width) <= cy <= (tb["ymax"] + beam_width):
                if tb["xmax"] < cx:   # Text is Left
                    hit_dist = min(hit_dist, cx - tb["xmax"])
                elif tb["xmin"] > cx: # Text is Right
                    hit_dist = min(hit_dist, tb["xmin"] - cx)

            # Vertical Beam Check (Up/Down)
            if (tb["xmin"] - beam_width) <= cx <= (tb["xmax"] + beam_width):
                if tb["ymax"] < cy:   # Text is Above
                    hit_dist = min(hit_dist, cy - tb["ymax"])
                elif tb["ymin"] > cy: # Text is Below
                    hit_dist = min(hit_dist, tb["ymin"] - cy)

            if hit_dist < min_dist and hit_dist <= max_distance:
                min_dist = hit_dist
                best_match = tb["text"]
                best_match_center = tb["center"]

        matched_results.append({
            "symbol_class": sym["class"],
            "symbol_center": sym["center"],
            "matched_text": best_match,
            "text_center": best_match_center,
            "distance": round(min_dist, 2) if best_match else None
        })

    return matched_results

# ==========================================
# 4. DATA CLEANING & EXPORT
# ==========================================
def clean_label(text):
    if not text:
        return "UNKNOWN"
    
    # Fix common OCR quirks and spacing
    text = text.replace("220k V", "220kV").replace("22OkV", "220kV")
    text = re.sub(r'\s+', '_', text.strip()) # Replace spaces with underscores
    return text

def export_to_json(matched_results, output_filename="sld_nodes.json"):
    json_data = {
        "document_type": "Single Line Diagram",
        "nodes": []
    }
    
    for idx, pair in enumerate(matched_results):
        cleaned_label = clean_label(pair.get('matched_text'))
        class_id = pair.get('symbol_class')
        
        json_data["nodes"].append({
            "node_id": f"Node_{idx}",
            "class_id": class_id,
            "component_type": CLASS_MAP.get(class_id, "UNKNOWN"),
            "label": cleaned_label,
            "coordinates": {
                "x": pair.get('symbol_center')[0],
                "y": pair.get('symbol_center')[1]
            }
        })
        
    with open(output_filename, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"\nSaved {len(matched_results)} nodes to {output_filename}")

def draw_matches(image_path, results, output_path="ray_match_results.png"):
    img = cv2.imread(image_path)
    for match in results:
        sym_pt = match["symbol_center"]
        cv2.circle(img, sym_pt, 6, (0, 0, 255), -1) 
        
        if match["matched_text"] and match["text_center"]:
            cv2.line(img, sym_pt, match["text_center"], (255, 0, 0), 2)
            cv2.putText(img, clean_label(match["matched_text"]), 
                        (sym_pt[0] + 10, sym_pt[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)
    print(f"Saved visual map to {output_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Starting Phase 2: Symbol-to-Text Mapping ---")
    
    # 1. Run matching
    final_pairs = match_symbols_to_text_ray(IMG_PATH, LBL_PATH, max_distance=150, beam_width=30)

    # 2. Export JSON
    export_to_json(final_pairs, "sld_nodes.json")
    
    # 3. Create visual map
    draw_matches(IMG_PATH, final_pairs, "ray_match_results.png")