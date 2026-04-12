"""
line_extraction.py
==================
Extract lines and connectivity from SLD diagrams.

Pipeline:
  1. Detect electrical components using YOLO
  2. Mask them with white boxes to hide symbols
  3. Mask small text blobs using connected components analysis
  4. Detect straight lines using Hough transform
  5. Track line connectivity:
     - Line-to-line intersection = continuous connection
     - Line-to-box intersection = terminal point
  6. Build connectivity graph
  7. Visualize connections and export results

Usage:
    python line_extraction.py <image_path> [--conf 0.2] [--output output_dir]
"""

import os
import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict
from ultralytics import YOLO
import json


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class Box:
    """Bounding box for a detected component."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float
    center: Tuple[int, int]
    
    def contains_point(self, x: int, y: int, margin: int = 0) -> bool:
        """Check if point is inside box (with optional margin)."""
        return (self.x1 - margin <= x <= self.x2 + margin and 
                self.y1 - margin <= y <= self.y2 + margin)
    
    def distance_to_point(self, x: int, y: int) -> float:
        """Compute distance from box center to point."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        return np.sqrt(dx**2 + dy**2)
    
    def to_dict(self):
        return {
            "bbox": (self.x1, self.y1, self.x2, self.y2),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "center": self.center,
        }


@dataclass
class LineSegment:
    """Detected line segment."""
    x1: int
    y1: int
    x2: int
    y2: int
    strength: float  # Hough accumulator value
    
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    def midpoint(self) -> Tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))
    
    def slope(self) -> float:
        """Compute slope (handle vertical lines)."""
        if self.x2 == self.x1:
            return float('inf')
        return (self.y2 - self.y1) / (self.x2 - self.x1)
    
    def to_dict(self):
        return {
            "endpoints": [(self.x1, self.y1), (self.x2, self.y2)],
            "length": round(self.length(), 2),
            "strength": round(self.strength, 2),
        }


@dataclass
class Connection:
    """Connection between two points (box centers or line endpoints)."""
    from_point: Tuple[int, int]
    to_point: Tuple[int, int]
    line_segments: List[LineSegment]
    connected_boxes: List[Tuple[int, str]]  # (box_index, class_name)
    
    def to_dict(self):
        return {
            "from": self.from_point,
            "to": self.to_point,
            "line_count": len(self.line_segments),
            "connected_boxes": self.connected_boxes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_components(image_path: str, model_path: str, conf: float = 0.2) -> List[Box]:
    """Detect electrical components using YOLO."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    
    CLASS_MAP = {
        0: "Generator",
        1: "Transformer",
        2: "Circuit_Breaker",
        3: "Load",
        4: "Capacitor",
        5: "Switch"
    }
    
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)
    
    boxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            boxes.append(Box(
                x1=x1, y1=y1, x2=x2, y2=y2,
                class_id=cls_id,
                class_name=CLASS_MAP.get(cls_id, "Unknown"),
                confidence=confidence,
                center=(cx, cy)
            ))
    
    print(f"[DETECT] Found {len(boxes)} components")
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# MASKING WITH ACTUAL SYMBOLS
# ─────────────────────────────────────────────────────────────────────────────

def load_symbol_masks(symbols_dir: str = "../symbols") -> dict:
    """Load symbol PNG masks for all component types and rotations."""
    symbol_masks = {}
    class_map = {
        "Generator": "generator",
        "Transformer": "transformer",
        "Circuit_Breaker": "breaker",
        "Load": None,  # No symbol, will use box
        "Capacitor": None,
        "Switch": "isolator"
    }
    
    for class_name, symbol_name in class_map.items():
        if symbol_name is None:
            symbol_masks[class_name] = {}
            continue
            
        symbol_masks[class_name] = {}
        for angle in [0, 90, 180, 270]:
            symbol_path = os.path.join(symbols_dir, f"{symbol_name}_{angle}.png")
            if os.path.exists(symbol_path):
                mask = cv2.imread(symbol_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Invert: make black pixels (symbols) white (255)
                    symbol_masks[class_name][angle] = 255 - mask
            
    print(f"[SYMBOL_LOAD] Loaded symbol masks")
    return symbol_masks


def detect_best_rotation(image_region: np.ndarray, symbol_masks: dict, class_name: str) -> int:
    """Detect best rotation for a component by template matching."""
    if class_name not in symbol_masks or not symbol_masks[class_name]:
        return 0  # Default to 0 if no symbols available
    
    if len(image_region) < 30 or len(image_region[0]) < 30:
        return 0  # Region too small
    
    best_angle = 0
    best_score = -1
    
    # Convert to grayscale for matching
    gray_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
    
    for angle, symbol_mask in symbol_masks[class_name].items():
        # Resize symbol to match region size
        h, w = gray_region.shape
        resized_symbol = cv2.resize(symbol_mask, (w, h), interpolation=cv2.INTER_AREA)
        
        # Compute correlation
        correlation = cv2.matchTemplate(gray_region, resized_symbol, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation)
        
        if max_val > best_score:
            best_score = max_val
            best_angle = angle
    
    return best_angle


def mask_components_with_symbols(image: np.ndarray, boxes: List[Box], symbol_masks: dict) -> np.ndarray:
    """Replace detected components with actual symbol masks to preserve lines."""
    masked = image.copy()
    
    for box in boxes:
        # Extract region around component
        x1, y1 = max(0, box.x1), max(0, box.y1)
        x2, y2 = min(masked.shape[1], box.x2), min(masked.shape[0], box.y2)
        
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            continue
        
        # Detect best rotation for this component
        rotation = detect_best_rotation(region, symbol_masks, box.class_name)
        
        # Get symbol mask
        if box.class_name in symbol_masks and rotation in symbol_masks[box.class_name]:
            symbol_mask = symbol_masks[box.class_name][rotation]
            
            # Resize symbol to match bounding box
            h, w = y2 - y1, x2 - x1
            if symbol_mask.shape != (h, w):
                symbol_mask = cv2.resize(symbol_mask, (w, h), interpolation=cv2.INTER_AREA)
            
            # Create 3-channel mask if needed
            if len(masked.shape) == 3:
                symbol_mask_3ch = np.stack([symbol_mask] * 3, axis=2)
            else:
                symbol_mask_3ch = symbol_mask
            
            # Apply mask: use symbol_mask where > 128, keep original elsewhere
            mask_pixels = symbol_mask_3ch > 128
            masked[y1:y2, x1:x2][mask_pixels] = 255
        else:
            # Fallback to simple box masking for unknown types
            masked[y1:y2, x1:x2] = 255
    
    print(f"[MASK] Masked {len(boxes)} components with symbol shapes")
    return masked


def detect_and_mask_text(image: np.ndarray, margin: int = 5) -> np.ndarray:
    """
    Detect text regions using threshold/morphology and mask them with white boxes.
    Much simpler than OCR - just removes small blobs that look like text.
    """
    masked = image.copy()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Invert (white background → black foreground)
    gray_inv = 255 - gray
    
    # Find connected components (blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_inv, connectivity=8)
    
    text_regions = 0
    
    # Stats columns: [left, top, width, height, area]
    for i in range(1, num_labels):  # Skip background (label 0)
        left, top, w, h, area = stats[i]
        
        # Criteria for text: small blobs with certain aspect ratio
        # Text typically has area < 10000 and width/height ratio between 0.1 and 10
        aspect_ratio = w / h if h > 0 else 0
        is_likely_text = (
            area < 8000 and 
            area > 20 and
            0.1 < aspect_ratio < 10
        )
        
        if is_likely_text:
            # Mask this region with white
            x1 = max(0, left - margin)
            y1 = max(0, top - margin)
            x2 = min(masked.shape[1], left + w + margin)
            y2 = min(masked.shape[0], top + h + margin)
            
            masked[y1:y2, x1:x2] = 255
            text_regions += 1
    
    print(f"[TEXT] Masked {text_regions} text regions using morphology")
    return masked


def mask_all_artifacts(image: np.ndarray, 
                       boxes: List[Box],
                       symbol_masks: dict = None,
                       mask_text: bool = True,
                       text_margin: int = 0) -> np.ndarray:
    """
    Combined masking: remove components using symbol shapes and text.
    Symbol-based masking preserves lines adjacent to components.
    """
    # Load symbol masks if not provided
    if symbol_masks is None:
        symbol_masks = load_symbol_masks()
    
    # Mask components using actual symbol shapes
    masked = mask_components_with_symbols(image, boxes, symbol_masks)
    
    # Then mask text with minimal margin
    if mask_text:
        masked = detect_and_mask_text(masked, margin=text_margin)
        
        # Additional morphological cleaning to remove text artifacts
        print("[MORPH] Applying morphological operations to remove text remnants...")
        kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3))
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return masked


# ─────────────────────────────────────────────────────────────────────────────
# LINE DETECTION (HOUGH TRANSFORM)
# ─────────────────────────────────────────────────────────────────────────────

def detect_lines(image: np.ndarray, 
                 rho: int = 1, 
                 theta: float = np.pi / 180,
                 threshold: int = 50,
                 min_length: int = 30,
                 max_gap: int = 20,
                 aggressively_clean: bool = True) -> List[LineSegment]:
    """
    Detect lines using Hough transform.
    
    Parameters:
    -----------
    rho : int
        Distance resolution of accumulator in pixels
    theta : float
        Angle resolution of accumulator in radians
    threshold : int
        Minimum number of votes to detect a line
    min_length : int
        Minimum line length
    max_gap : int
        Maximum gap between line segments to be merged
    aggressively_clean : bool
        If True, apply additional cleaning to remove text noise
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply aggressive cleaning to remove text
    if aggressively_clean:
        # Use bilateral filter to smooth small details (text) while preserving edges (lines)
        print("[LINE_DETECT] Applying bilateral filtering to remove text noise...")
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply morphological closing to fill small text gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Invert (white background → black lines)
    gray = 255 - gray
    
    # Apply Canny edge detection with adaptive thresholds
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect broken lines from text removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Detect lines using HoughLinesP (probabilistic)
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap
    )
    
    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Estimate strength from line length (rough proxy)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            segments.append(LineSegment(x1, y1, x2, y2, strength=length))
    
    print(f"[LINE_DETECT] Found {len(segments)} line segments")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# LINE CONNECTIVITY & TERMINATION
# ─────────────────────────────────────────────────────────────────────────────

def point_on_line(px: int, py: int, 
                  x1: int, y1: int, x2: int, y2: int,
                  tolerance: int = 3) -> bool:
    """Check if point (px, py) lies on line segment (x1, y1) -> (x2, y2)."""
    # Perpendicular distance from point to line
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    if den == 0:
        return False
    dist = num / den
    
    # Also check if point is within line bounds
    is_within = (min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance and
                 min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance)
    
    return dist <= tolerance and is_within


def line_intersects_box(segment: LineSegment, box: Box, margin: int = 5) -> bool:
    """Check if line segment intersects expanded box."""
    x1, y1, x2, y2 = segment.x1, segment.y1, segment.x2, segment.y2
    bx1 = box.x1 - margin
    by1 = box.y1 - margin
    bx2 = box.x2 + margin
    by2 = box.y2 + margin
    
    # Check if any endpoint is in box
    if (bx1 <= x1 <= bx2 and by1 <= y1 <= by2) or \
       (bx1 <= x2 <= bx2 and by1 <= y2 <= by2):
        return True
    
    # Check if line crosses box boundaries
    # Simplified: check if line passes through box
    if x1 == x2:  # Vertical line
        if bx1 <= x1 <= bx2:
            if (min(y1, y2) <= by2 and max(y1, y2) >= by1):
                return True
    elif y1 == y2:  # Horizontal line
        if by1 <= y1 <= by2:
            if (min(x1, x2) <= bx2 and max(x1, x2) >= bx1):
                return True
    else:
        # Check corners of box against line
        for cx in [bx1, bx2]:
            for cy in [by1, by2]:
                if point_on_line(cx, cy, x1, y1, x2, y2, tolerance=5):
                    return True
    
    return False


def find_line_endpoints(segments: List[LineSegment], 
                       boxes: List[Box],
                       endpoint_tolerance: int = 10) -> Dict[int, List[Tuple[int, int]]]:
    """
    For each line segment, find where it terminates.
    Returns: {segment_index: [start_point, end_point]}
    
    Termination = hits a box or end of line.
    """
    endpoints = {}
    
    for i, seg in enumerate(segments):
        start = (seg.x1, seg.y1)
        end = (seg.x2, seg.y2)
        
        # Check if start/end hit boxes
        start_hits_box = any(box.contains_point(start[0], start[1], margin=endpoint_tolerance)
                            for box in boxes)
        end_hits_box = any(box.contains_point(end[0], end[1], margin=endpoint_tolerance)
                          for box in boxes)
        
        endpoints[i] = (start, end, start_hits_box, end_hits_box)
    
    return endpoints


def build_connectivity_graph(segments: List[LineSegment],
                            boxes: List[Box],
                            endpoint_tolerance: int = 10,
                            connection_tolerance: int = 5) -> List[Connection]:
    """
    Build connectivity graph by merging line segments that connect end-to-end
    and tracking which boxes they touch.
    """
    endpoints = find_line_endpoints(segments, boxes, endpoint_tolerance)
    
    # Union-find to merge connected segments
    parent = list(range(len(segments)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Merge segments that form a continuous line
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            seg_i, seg_j = segments[i], segments[j]
            
            # Check if segments connect end-to-end
            endpoints_i = endpoints[i]
            endpoints_j = endpoints[j]
            
            # Get termination points
            start_i, end_i = endpoints_i[0], endpoints_i[1]
            start_j, end_j = endpoints_j[0], endpoints_j[1]
            
            # Check if end of i connects to start of j
            dist = np.sqrt((end_i[0] - start_j[0])**2 + (end_i[1] - start_j[1])**2)
            if dist <= connection_tolerance:
                union(i, j)
                continue
            
            dist = np.sqrt((end_i[0] - end_j[0])**2 + (end_i[1] - end_j[1])**2)
            if dist <= connection_tolerance:
                union(i, j)
                continue
            
            dist = np.sqrt((start_i[0] - start_j[0])**2 + (start_i[1] - start_j[1])**2)
            if dist <= connection_tolerance:
                union(i, j)
                continue
            
            dist = np.sqrt((start_i[0] - end_j[0])**2 + (start_i[1] - end_j[1])**2)
            if dist <= connection_tolerance:
                union(i, j)
    
    # Group segments by connected component
    groups = defaultdict(list)
    for i in range(len(segments)):
        groups[find(i)].append(i)
    
    # Build connections
    connections = []
    for group_indices in groups.values():
        if not group_indices:
            continue
        
        # Collect all segments in this connection
        group_segments = [segments[i] for i in group_indices]
        
        # Find start and end points of the connected chain
        # (simplification: use min/max coordinates)
        all_x = [s.x1 for s in group_segments] + [s.x2 for s in group_segments]
        all_y = [s.y1 for s in group_segments] + [s.y2 for s in group_segments]
        
        start_point = (min(all_x), min(all_y))
        end_point = (max(all_x), max(all_y))
        
        # Find which boxes this connection touches
        connected_box_indices = set()
        for seg_idx in group_indices:
            seg = segments[seg_idx]
            for box_idx, box in enumerate(boxes):
                if line_intersects_box(seg, box, margin=endpoint_tolerance):
                    connected_box_indices.add(box_idx)
        
        connected_boxes = [(i, boxes[i].class_name) for i in connected_box_indices]
        
        connections.append(Connection(
            from_point=start_point,
            to_point=end_point,
            line_segments=group_segments,
            connected_boxes=connected_boxes
        ))
    
    print(f"[CONNECTIVITY] Built {len(connections)} connections")
    return connections


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_results(image: np.ndarray,
                     masked_image: np.ndarray,
                     boxes: List[Box],
                     segments: List[LineSegment],
                     connections: List[Connection],
                     output_dir: str) -> None:
    """Create visualizations of detection results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualization 1: Boxes on original
    img_boxes = image.copy()
    for i, box in enumerate(boxes):
        cv2.rectangle(img_boxes, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        cv2.putText(img_boxes, f"{box.class_name}", 
                   (box.x1, box.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, "01_detected_boxes.png"), img_boxes)
    print(f"  Saved: 01_detected_boxes.png")
    
    # Visualization 2: Masked image
    cv2.imwrite(os.path.join(output_dir, "02_masked_image.png"), masked_image)
    print(f"  Saved: 02_masked_image.png")
    
    # Visualization 3: Detected lines
    img_lines = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR) if len(masked_image.shape) == 2 \
                else masked_image.copy()
    for seg in segments:
        cv2.line(img_lines, (seg.x1, seg.y1), (seg.x2, seg.y2), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "03_detected_lines.png"), img_lines)
    print(f"  Saved: 03_detected_lines.png")
    
    # Visualization 4: Connectivity graph
    img_connectivity = image.copy()
    
    # Draw boxes
    for box in boxes:
        cv2.circle(img_connectivity, box.center, 8, (0, 255, 0), -1)
    
    # Draw connections
    colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    for i, conn in enumerate(connections):
        color = colors[i % len(colors)]
        cv2.line(img_connectivity, conn.from_point, conn.to_point, color, 3)
        cv2.circle(img_connectivity, conn.from_point, 5, color, -1)
        cv2.circle(img_connectivity, conn.to_point, 5, color, -1)
    
    cv2.imwrite(os.path.join(output_dir, "04_connectivity_graph.png"), img_connectivity)
    print(f"  Saved: 04_connectivity_graph.png")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_results(boxes: List[Box],
                  segments: List[LineSegment],
                  connections: List[Connection],
                  output_dir: str) -> None:
    """Export detection and connectivity results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "components": [box.to_dict() for box in boxes],
        "line_segments": [seg.to_dict() for seg in segments],
        "connections": [conn.to_dict() for conn in connections],
        "summary": {
            "num_components": len(boxes),
            "num_line_segments": len(segments),
            "num_connections": len(connections),
        }
    }
    
    output_path = os.path.join(output_dir, "line_extraction_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"[EXPORT] Saved results to {output_path}")


def merge_connectivity_with_symbol_json(symbol_json_path: str,
                                       connections: List[Connection],
                                       boxes: List[Box],
                                       output_path: str = None) -> dict:
    """
    Merge line extraction connectivity with symbol detection JSON.
    Updates device connections based on detected electrical lines.
    
    Args:
        symbol_json_path: Path to symbol detection JSON (sld_nodes.json)
        connections: List of detected connections (Connection objects or dicts)
        boxes: List of detected components (Box objects)
        output_path: Output path for merged JSON (if None, overwrites original)
    """
    if not os.path.exists(symbol_json_path):
        print(f"[MERGE] Warning: Symbol JSON not found at {symbol_json_path}")
        return {"devices": []}
    
    # Load symbol detection JSON
    with open(symbol_json_path, "r") as f:
        symbol_data = json.load(f)
    
    devices = symbol_data.get("devices", [])
    
    # Initialize connectivity field for all devices
    for device in devices:
        device["connected_devices"] = []
    
    # For each connection, connect all boxes that touch it
    for conn in connections:
        # Handle both Connection objects and plain dicts
        if isinstance(conn, dict):
            connected_box_list = conn.get("connected_boxes", [])
        else:
            # Connection object
            connected_box_list = conn.connected_boxes if hasattr(conn, 'connected_boxes') else []
        
        if not connected_box_list:
            continue
        
        # Extract box indices from connected_boxes
        # Format is either [(idx, classname), ...] or [[idx, classname], ...]
        box_indices = []
        for item in connected_box_list:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                box_indices.append(item[0])  # Get first element (the index)
        
        if len(box_indices) < 2:
            continue  # Need at least 2 boxes to form a connection
        
        # Connect all pairs of boxes in this connection
        for i in range(len(box_indices)):
            for j in range(i + 1, len(box_indices)):
                box_i_idx = box_indices[i]
                box_j_idx = box_indices[j]
                
                if box_i_idx < len(devices) and box_j_idx < len(devices):
                    dev_i_id = devices[box_i_idx].get("id", f"device_{box_i_idx}")
                    dev_j_id = devices[box_j_idx].get("id", f"device_{box_j_idx}")
                    
                    # Add bidirectional connections
                    if dev_j_id not in devices[box_i_idx].get("connected_devices", []):
                        if "connected_devices" not in devices[box_i_idx]:
                            devices[box_i_idx]["connected_devices"] = []
                        devices[box_i_idx]["connected_devices"].append(dev_j_id)
                    
                    if dev_i_id not in devices[box_j_idx].get("connected_devices", []):
                        if "connected_devices" not in devices[box_j_idx]:
                            devices[box_j_idx]["connected_devices"] = []
                        devices[box_j_idx]["connected_devices"].append(dev_i_id)
    
    # Save merged result (overwrite or write to new file)
    if output_path is None:
        output_path = symbol_json_path  # Overwrite original
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(symbol_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"[MERGE] Saved merged JSON to {output_path}")
    devices_with_connections = sum(1 for d in devices if d.get("connected_devices"))
    print(f"[MERGE] Updated {devices_with_connections} of {len(devices)} devices with connections")
    return symbol_data


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main(image_path: str, 
         model_path: str = None,
         conf: float = 0.2,
         output_dir: str = "output_line_extraction"):
    """
    Main pipeline for line extraction and connectivity analysis.
    """
    print("=" * 70)
    print("LINE EXTRACTION & CONNECTIVITY ANALYSIS")
    print("=" * 70)
    
    # Default model path
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..", "models", "yolo_model", "runs", "detect", "train", "weights", "best.pt"
        )
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    print(f"\n[LOAD] Image shape: {image.shape}")
    
    # Step 1: Detect components
    print("\n[STEP 1] Detecting electrical components...")
    boxes = detect_components(image_path, model_path, conf=conf)
    
    # Step 2: Mask components and text
    print("\n[STEP 2] Masking detected components and text...")
    masked_image = mask_all_artifacts(image, boxes, mask_text=True)
    
    # Step 3: Detect lines
    print("\n[STEP 3] Detecting lines in masked image...")
    segments = detect_lines(masked_image, threshold=50, min_length=30, max_gap=20)
    
    # Step 4: Build connectivity graph
    print("\n[STEP 4] Building connectivity graph...")
    connections = build_connectivity_graph(segments, boxes, endpoint_tolerance=10)
    
    # Step 5: Visualize
    print("\n[STEP 5] Creating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    visualize_results(image, masked_image, boxes, segments, connections, output_dir)
    
    # Step 6: Export
    print("\n[STEP 6] Exporting results...")
    export_results(boxes, segments, connections, output_dir)
    
    print("\n" + "=" * 70)
    print(f"Pipeline complete. Results in: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract lines and connectivity from SLD diagrams"
    )
    parser.add_argument("image", help="Path to SLD image")
    parser.add_argument("--model", default=None, help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--output", default="output_line_extraction", help="Output directory")
    
    args = parser.parse_args()
    main(args.image, model_path=args.model, conf=args.conf, output_dir=args.output)
