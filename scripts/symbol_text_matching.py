# SEMI HUMAN CODE DO NOT TRUST (1/3rd human code, Ishaan counts as robot)

import cv2
import numpy as np
import os
import re
import json
from paddleocr import PaddleOCR
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================
CLASS_MAP = {
    0: "Generator",
    1: "Transformer",
    2: "Circuit_Breaker",
    3: "Load",
    4: "Capacitor",
    5: "Switch"
}

MAX_DISTANCE_BY_CLASS = {
    0: 600,   # Generator — labels can be far, raise to reduce UNKNOWNs
    1: 600,   # Transformer — same
    2: 300,   # Circuit_Breaker
    3: 400,   # Load
    4: 300,   # Capacitor
    5: 300,   # Switch
}

MAX_ACCEPTABLE_SCORE = 2000

DEBUG = False


# ==========================================
# OCR ENGINE — initialized once at module load
# ==========================================
_ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_db_thresh=0.1,
    det_db_box_thresh=0.4,
    det_db_unclip_ratio=2.0,
    det_limit_side_len=4000,  # lowered from 4800 to avoid resize/oneDNN crash
    det_limit_type='max',
    # rec_char_type removed — no longer supported in PaddleOCR 3.x
)


# ==========================================
# OCR
# ==========================================
def get_clean_text(image_path, ocr_scale, min_chars=3):
    """
    Run OCR on an already-preprocessed image and return filtered text boxes.

    Parameters:
        image_path (str)   : path to the preprocessed image (written by main.py)
        ocr_scale  (float) : scale factor used during preprocessing; OCR bbox
                             coords are divided by this to map back to original
                             image coordinate space
        min_chars  (int)   : minimum character length to keep a text box

    Returns:
        list of (bbox, text) where bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        in original image coordinates
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"OCR image not found: {image_path}")

    print("Running PaddleOCR...")

    try:
        results = _ocr_engine.predict(image_path)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return []

    if not results:
        return []

    valid_text_boxes = []

    for res in results:
        # PaddleOCR 3.x returns result objects — try dict access first,
        # then attribute access, to handle minor version differences
        polys = None
        texts = None

        if isinstance(res, dict):
            polys = res.get("boxes") or res.get("dt_boxes")
            texts = res.get("rec_texts") or res.get("rec_text")
        else:
            polys = getattr(res, "boxes", None) or getattr(res, "dt_boxes", None)
            texts = getattr(res, "rec_texts", None) or getattr(res, "rec_text", None)

        if polys is None or texts is None:
            if DEBUG:
                print(f"  [OCR] Could not extract boxes/texts from result: {type(res)}, attrs: {dir(res)}")
            continue

        for bbox, text in zip(polys, texts):
            text = text.strip()
            if len(text) >= min_chars and re.search(r'[A-Z]', text.upper()):
                int_bbox = [[int(pt[0] / ocr_scale), int(pt[1] / ocr_scale)]
                            for pt in bbox]
                valid_text_boxes.append((int_bbox, text))

    return valid_text_boxes


# ==========================================
# YOLO SYMBOL DETECTION
# ==========================================
def detect_symbols_from_yolo(image_path, model_path, conf=0.25):
    """
    Detect symbols from a YOLO model directly from an image.
    Returns:
    [
        {
            "class": int,
            "confidence": float,
            "center": (cx, cy),
            "bbox": (x1, y1, x2, y2)
        },
        ...
    ]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found for YOLO detection: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)

    symbols = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            symbols.append({
                "class": cls_id,
                "confidence": round(confidence, 4),
                "center": (cx, cy),
                "bbox": (x1, y1, x2, y2)
            })

    print(f"Detected {len(symbols)} symbols from YOLO.")
    return symbols


# ==========================================
# LABEL HELPERS
# ==========================================
def is_bus_label(label):
    return "BUS" in label.upper()

def is_feeder_label(label):
    return bool(re.search(r'FDR', label.upper()))

def get_label_type(label):
    label = label.upper()
    if "BUS" in label:
        return "Bus"
    elif re.search(r'CB', label):
        return "Circuit_Breaker"
    elif re.search(r'DS[AB]?|SW', label):
        return "Switch"
    elif re.search(r'GEN', label):
        return "Generator"
    elif re.search(r'TR', label):
        return "Transformer"
    elif re.search(r'FDR|LOD', label):
        return "Load"
    else:
        return "Unknown"

def infer_voltage_prefix(label, symbol_y, img_height):
    """
    Inject missing voltage prefix for labels like '_03_CB' that OCR
    failed to read, based on the symbol's y-coordinate zone.
    Tune the ratio thresholds to match your diagram's bus y-positions.
    """
    if not label.startswith('_'):
        return label
    ratio = symbol_y / img_height
    prefix = "66kV" if ratio < 0.35 else "33kV"
    return prefix + label

def compute_score(cx, cy, tb, beam_width):
    """
    Compute match score between a symbol center and a text box.
    - dy is the primary distance signal
    - dx contributes a soft normalised penalty — NOT a hard gate,
      so rotated labels far to the side are still considered
    - reuse_penalty set to 5000 to strongly discourage duplicate matches
    """
    dx = abs(cx - tb["center"][0])
    dy = abs(cy - tb["center"][1])
    horizontal_penalty = (dx / beam_width) * 100 if beam_width > 0 else dx
    reuse_penalty = 5000 if tb.get("used", False) else 0
    return dy + horizontal_penalty + reuse_penalty


# ==========================================
# SYMBOL-TEXT MATCHING
# ==========================================
def match_symbols_to_text_ray(base_image_path, ocr_image_path, ocr_scale,
                               symbols, max_distance, beam_width):
    """
    Match already-detected symbols to OCR text using ray/beam logic.

    Parameters:
      - base_image_path : original image path (used for dimensions + visualisation)
      - ocr_image_path  : preprocessed image path written by main.py — OCR runs
                          directly on this, no further preprocessing inside
      - ocr_scale       : scale factor from preprocessing; passed to get_clean_text
                          so OCR coords are correctly mapped to original image space
      - symbols         : output of detect_symbols_from_yolo(...)
      - max_distance    : default vertical cutoff (px); overridden per class
                          by MAX_DISTANCE_BY_CLASS
      - beam_width      : soft horizontal scale — used in scoring, NOT a hard gate
    """
    if not os.path.exists(base_image_path):
        raise FileNotFoundError(f"Base image not found: {base_image_path}")
    if not os.path.exists(ocr_image_path):
        raise FileNotFoundError(f"OCR image not found: {ocr_image_path}")

    img = cv2.imread(base_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load base image at {base_image_path}")

    img_height = img.shape[0]

    text_data = get_clean_text(ocr_image_path, ocr_scale)

    text_boxes = []
    for bbox, text in text_data:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        text_boxes.append({
            "text": text,
            "xmin": min(x_coords),
            "xmax": max(x_coords),
            "ymin": min(y_coords),
            "ymax": max(y_coords),
            "center": (int(np.mean(x_coords)), int(np.mean(y_coords)))
        })

    print(f"Loaded {len(text_boxes)} OCR text boxes.")
    print("Matching symbols to text...")

    matched_results = []

    for sym in symbols:
        x1, y1, x2, y2 = sym["bbox"]
        pad = 40
        cx = int((max(0, x1 - pad) + (x2 + pad)) / 2)
        cy = int((max(0, y1 - pad) + (y2 + pad)) / 2)

        best_match = None
        best_match_center = None
        best_match_text_raw = None
        min_dist = float("inf")

        symbol_type = CLASS_MAP.get(sym["class"], "Unknown")
        effective_max_dist = MAX_DISTANCE_BY_CLASS.get(sym["class"], max_distance)

        # Primary pass: type-aware matching
        for tb in text_boxes:
            label = clean_label(tb["text"])

            if is_bus_label(label):
                continue
            if is_feeder_label(label) and symbol_type != "Load":
                continue

            label_type = get_label_type(label)

            if label_type != symbol_type:
                if label_type == "Unknown":
                    pass
                else:
                    continue

            dy = abs(cy - tb["center"][1])
            if dy > effective_max_dist:
                continue

            score = compute_score(cx, cy, tb, beam_width)

            if DEBUG:
                print(f"  [PRIMARY] sym={symbol_type} @ ({cx},{cy}) | "
                      f"label={label} @ {tb['center']} | "
                      f"dx={abs(cx-tb['center'][0])} dy={dy} score={score:.1f}")

            if score < min_dist:
                min_dist = score
                best_match = label
                best_match_text_raw = tb["text"]
                best_match_center = tb["center"]

        # Fallback pass: relax type constraint, keep distance constraint
        if best_match is None:
            for tb in text_boxes:
                label = clean_label(tb["text"])

                if is_bus_label(label):
                    continue
                if is_feeder_label(label) and symbol_type != "Load":
                    continue

                label_type = get_label_type(label)

                if label_type != symbol_type:
                    if label_type == "Unknown":
                        pass
                    else:
                        continue

                dy = abs(cy - tb["center"][1])
                if dy > effective_max_dist:
                    continue

                score = compute_score(cx, cy, tb, beam_width)

                if DEBUG:
                    print(f"  [FALLBACK] sym={symbol_type} @ ({cx},{cy}) | "
                          f"label={label} @ {tb['center']} | score={score:.1f}")

                if score < min_dist:
                    min_dist = score
                    best_match = label
                    best_match_text_raw = tb["text"]
                    best_match_center = tb["center"]

        # Reject if score too high
        if MAX_ACCEPTABLE_SCORE is not None and min_dist > MAX_ACCEPTABLE_SCORE:
            if DEBUG:
                print(f"  [REJECTED] sym={symbol_type} @ ({cx},{cy}) best_score={min_dist:.1f}")
            best_match = None
            best_match_text_raw = None
            best_match_center = None

        # Infer voltage prefix if label starts with '_'
        if best_match is not None:
            best_match = infer_voltage_prefix(best_match, sym["center"][1], img_height)

        # Mark text box as used
        if best_match_text_raw:
            for tb in text_boxes:
                if tb["text"] == best_match_text_raw:
                    tb["used"] = True
                    break

        if DEBUG:
            print(f"  => MATCHED: {best_match} (score={min_dist:.1f})\n")

        matched_results.append({
            "symbol_class": sym["class"],
            "symbol_center": sym["center"],
            "symbol_bbox": sym.get("bbox"),
            "confidence": sym.get("confidence"),
            "matched_text": best_match,
            "text_center": best_match_center,
            "distance": round(min_dist, 2) if best_match else None
        })

    unmatched = sum(1 for r in matched_results if r["matched_text"] is None)
    print(f"Matching complete. {len(matched_results) - unmatched}/{len(matched_results)} symbols matched.")

    return matched_results


# ==========================================
# DATA CLEANING
# ==========================================
def clean_label(text):
    if not text:
        return "UNKNOWN"

    text = text.upper().strip()

    text = text.replace("CE", "CB")
    text = text.replace("DS/", "DSA")
    text = re.sub(r'(?<=\d)O', '0', text)
    text = re.sub(r'(?<=\d)I', '1', text)
    text = re.sub(r'(?<=[A-Z_])O(?=\d)', '0', text)
    text = text.replace("l32", "132")

    text = re.sub(r'DS[ECF](?=_|$)', 'DSB', text)

    text = re.sub(r'\b3KV\b', '33kV', text)
    text = re.sub(r'\b6KV\b', '66kV', text)
    text = re.sub(r'\b132KV\b', '132kV', text)
    text = re.sub(r'\bl32KV\b', '132kV', text)
    text = re.sub(r'\b32KV\b', '132kV', text)

    text = re.sub(r'(\d+)\s*K\s*V', r'\1kV', text)
    text = re.sub(r'(\d+)KV\b', r'\1kV', text)

    text = re.sub(r'\b3V\b', '33kV', text)
    text = re.sub(r'\b33V\b', '33kV', text)
    text = re.sub(r'\b6V\b', '66kV', text)
    text = re.sub(r'\b66V\b', '66kV', text)
    text = re.sub(r'\b132V\b', '132kV', text)

    text = re.sub(r'\b220KV\b', '220kV', text)
    text = re.sub(r'\b22OKV\b', '220kV', text)

    parts = text.split('kV')
    cleaned_parts = [re.sub(r"[^A-Z0-9_/\\-]", "", part) for part in parts]
    text = 'kV'.join(cleaned_parts)

    text = re.sub(
        r'((?:33|66|132)kV)(\d{1,3})((?:DSA|DSB|DSC|CB\d*|GEN|FDR|TR))',
        r'\1_\2_\3',
        text
    )

    text = re.sub(
        r'((?:33|66|132)kV_\d{1,3})(CB\d+|DSA|DSB|DSC)',
        r'\1_\2',
        text
    )

    return text if len(text) > 1 else "UNKNOWN"


def export_to_json(matched_results, output_filename="sld_nodes.json"):
    json_data = {
        "document_type": "Single Line Diagram",
        "nodes": []
    }

    for idx, pair in enumerate(matched_results):
        cleaned_label = clean_label(pair.get("matched_text"))
        class_id = pair.get("symbol_class")
        center = pair.get("symbol_center", (None, None))

        json_data["nodes"].append({
            "node_id": f"Node_{idx}",
            "class_id": class_id,
            "component_type": CLASS_MAP.get(class_id, "UNKNOWN"),
            "label": cleaned_label,
            "coordinates": {
                "x": center[0],
                "y": center[1]
            }
        })

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    print(f"Saved {len(matched_results)} nodes to {output_filename}")


def draw_matches(image_path, results, output_path="ray_match_results.png"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image for visualization: {image_path}")

    for match in results:
        sym_pt = match["symbol_center"]
        cv2.circle(img, sym_pt, 6, (0, 0, 255), -1)

        if match.get("matched_text") and match.get("text_center"):
            cv2.line(img, sym_pt, match["text_center"], (255, 0, 0), 2)
            cv2.putText(
                img,
                clean_label(match["matched_text"]),
                (sym_pt[0] + 10, sym_pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imwrite(output_path, img)
    print(f"Saved visual map to {output_path}")
