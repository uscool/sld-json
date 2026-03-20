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
from .render_brk_hlf import render_breaker_and_half_svg
from .render_dbl_bs import render_double_bus_svg
from .render_mn_trfr import render_main_and_transfer_svg
from .render_mesh import render_mesh_svg
from .render_ring_bs import render_ring_bus_svg
from .render_sngl_bs import render_single_bus_svg


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

CANVAS_MARGIN_X = 10              # Left/right margin
CANVAS_MARGIN_Y = 10             # Top/bottom margin
TRANSFORMER_SPACE_ABOVE = 15    # Vertical space from bottom of upper level to top of transformer symbol
TRANSFORMER_SPACE_BELOW = 15    # Vertical space from bottom of transformer symbol to top of lower level
TRANSFORMER_SIZE = 32             # 32x32 viewBox
LINE_WIDTH = 1                    # Connection line width
LINE_COLOR = "#000"               # Black
WIRE_SEPARATION = 4              # Min visual gap (px) between parallel elbow rows
                                 # when staggering is triggered by detected overlaps.

# Transformer label
TR_LABEL_OFFSET_X  = -6           # Horizontal gap: symbol right edge → label start (px)
TR_LABEL_OFFSET_Y  = 0.6         # Vertical position as fraction of symbol height (0=top, 1=bottom)
TR_LABEL_FONT_SIZE = 7           # Label font size (px)
TR_LABEL_FONT_W    = 4.2         # Estimated px per character at TR_LABEL_FONT_SIZE (monospace)

# Transformer separation
TR_MIN_SEPARATION  = 10          # Extra minimum pixel gap between adjacent transformer centres.
                                 # Even at 0 the label-clearance rule always applies:
                                 # each pair is spaced at least
                                 # TRANSFORMER_SIZE + TR_LABEL_OFFSET_X + len(label)*TR_LABEL_FONT_W
                                 # so the left label never overlaps the right symbol.

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


def manhattan_path(
    from_point: Tuple[float, float],
    to_point: Tuple[float, float],
    elbow_y: float | None = None,
) -> str:
    """
    Generate Manhattan (elbow) routing path: vertical → horizontal → vertical.
    If elbow_y is supplied it is used directly; otherwise defaults to 1/3 of span.
    """
    x1, y1 = from_point
    x2, y2 = to_point
    if elbow_y is None:
        span    = abs(y2 - y1)
        elbow_y = y1 + max(span / 3, 4) if y2 >= y1 else y1 - max(span / 3, 4)
    return f"M {x1:.1f},{y1:.1f} L {x1:.1f},{elbow_y:.1f} L {x2:.1f},{elbow_y:.1f} L {x2:.1f},{y2:.1f}"


def generate_transformer_symbol(x: float, y: float, label: str) -> str:
    """Generate IEC transformer symbol SVG with label to the right (matching render_brk_hlf style)"""
    transform_x = x - TRANSFORMER_SIZE / 2
    transform_y = y - TRANSFORMER_SIZE / 2
    label_x = x + TRANSFORMER_SIZE / 2 + TR_LABEL_OFFSET_X
    label_y = transform_y + TRANSFORMER_SIZE * TR_LABEL_OFFSET_Y

    return (
        f'\n    <g id="transformer_{label.replace("/", "_").replace("-", "_")}"'
        f' transform="translate({transform_x:.1f},{transform_y:.1f})">\n'
        f'        <g transform="scale({TRANSFORMER_SIZE/32:.4f})">\n'
        f'            {IEC_TRANSFORMER_SYMBOL}\n'
        f'        </g>\n'
        f'    </g>\n'
        f'    <text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="start"'
        f' font-size="{TR_LABEL_FONT_SIZE}" font-family="monospace" fill="{LINE_COLOR}">{label}</text>\n'
    )


def _enforce_transformer_separation(
    centers: List[float], labels: List[str]
) -> List[float]:
    """
    Shift transformer centres rightward so each adjacent pair satisfies both
    TR_MIN_SEPARATION and the label-clearance rule:
        gap >= TRANSFORMER_SIZE + TR_LABEL_OFFSET_X + len(left_label) * TR_LABEL_FONT_W
    This guarantees the left transformer's label never overlaps the right symbol.
    centres and labels must already be in left-to-right order.
    """
    if len(centers) <= 1:
        return list(centers)
    result = list(centers)
    for i in range(1, len(result)):
        label_px = len(labels[i - 1]) * TR_LABEL_FONT_W
        needed = max(
            TR_MIN_SEPARATION,
            TRANSFORMER_SIZE + TR_LABEL_OFFSET_X + label_px,
        )
        if result[i] - result[i - 1] < needed:
            result[i] = result[i - 1] + needed
    return result


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
        # (from_i, to_i) -> (space_above, space_below) — set by _calculate_layout
        self._pair_spaces: Dict[tuple, tuple] = {}
    
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
        
        # Load merged JSON
        with open(merged_json_path, 'r') as f:
            self.merged_json = json.load(f)
        
        
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
        
    
    def _render_levels(self, json_paths: List[str], topology_types: List[str]):
        """Render each level in hybrid mode"""
        for i, (json_path, topology_type) in enumerate(zip(json_paths, topology_types)):
            voltage_label = self.merged_json["layout"]["voltage_levels"][i]["name"]
            
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
        """
        Two-phase layout.

        Phase 1 — x-only: compute horizontal offsets for each level.
        Phase 2 — conflict detection: for each inter-level pair, check whether
            any two wires will share the same natural elbow row AND have
            overlapping horizontal extents.  X-coordinates are independent of
            the vertical gap, so this can be done before y positions are known.
            If conflicts are detected the vertical gap for that pair is expanded
            to N_conflict * WIRE_SEPARATION so the stagger step in _generate_svg
            always has enough room to produce visually distinct rows.
        Phase 3 — y layout: stack levels using per-pair gaps.
        """

        from collections import defaultdict

        # ── Phase 1: x offsets ───────────────────────────────────────────
        max_width = max(r.width for r in self.level_results)
        x_offsets = [
            CANVAS_MARGIN_X + (max_width - r.width) / 2
            for r in self.level_results
        ]

        # ── Phase 2: per-pair conflict detection ─────────────────────────
        # Gather wire x-extents per level pair.
        # Because all wires in a pair share the same natural elbow_y (same
        # from_y and tr_top_y), two wires conflict iff their horizontal
        # extents [min(from_x,tr_x), max(from_x,tr_x)] overlap.
        unified_transformers = self.merged_json.get("unified_transformers", {})
        transformer_connections = self.merged_json.get("transformer_connections", {})

        pair_extents: dict = defaultdict(list)  # (from_i,to_i) -> [(from_x, tr_x)]
        for tr_id, levels in transformer_connections.items():
            if len(levels) != 2:
                continue
            from_i, to_i = levels
            td = unified_transformers[tr_id]
            from_orig = td["from_device_id"].split("_", 1)[1] if "_" in td["from_device_id"] else td["from_device_id"]
            to_orig   = td["to_device_id"].split("_", 1)[1]   if "_" in td["to_device_id"]   else td["to_device_id"]
            fp = self.level_results[from_i].boundary_pts.get(from_orig, {})
            tp = self.level_results[to_i].boundary_pts.get(to_orig, {})
            if not fp or not tp:
                continue
            from_x = fp["x"] + x_offsets[from_i]
            to_x   = tp["x"] + x_offsets[to_i]
            tr_x   = (from_x + to_x) / 2  # natural centre before separation enforcement
            pair_extents[(from_i, to_i)].append((from_x, tr_x))

        def _extents_overlap(a0, a1, b0, b1):
            lo_a, hi_a = min(a0, a1), max(a0, a1)
            lo_b, hi_b = min(b0, b1), max(b0, b1)
            return lo_a < hi_b and lo_b < hi_a

        n_levels = len(self.level_results)
        for i in range(n_levels - 1):
            wires = pair_extents.get((i, i + 1), [])
            # Count wires involved in at least one overlap
            conflict_set = set()
            for a in range(len(wires)):
                for b in range(a + 1, len(wires)):
                    if _extents_overlap(*wires[a], *wires[b]):
                        conflict_set.add(a)
                        conflict_set.add(b)
            n_conflict = len(conflict_set)
            sa = max(TRANSFORMER_SPACE_ABOVE, n_conflict * WIRE_SEPARATION)
            sb = max(TRANSFORMER_SPACE_BELOW, n_conflict * WIRE_SEPARATION)
            self._pair_spaces[(i, i + 1)] = (sa, sb)

        # ── Phase 3: y layout ────────────────────────────────────────────
        current_y = CANVAS_MARGIN_Y
        for i, result in enumerate(self.level_results):
            self.level_positions[i] = (x_offsets[i], current_y)
            current_y += result.height
            if i < n_levels - 1:
                sa, sb = self._pair_spaces[(i, i + 1)]
                current_y += sa + TRANSFORMER_SIZE + sb

        self.canvas_width  = max_width + 2 * CANVAS_MARGIN_X
        self.canvas_height = current_y + CANVAS_MARGIN_Y
    
    def _place_transformers(self):
        """
        Place transformers between levels, re-pairing by x-position so wires never cross.

        The JSON builder pairs transformers by list index, which does not guarantee
        that the left-to-right order of downstream terminals on the upper level matches
        the left-to-right order of upstream terminals on the lower level.  When they
        are out of order the connecting wires cross.

        Fix: for every (from_level, to_level) pair, resolve all from/to canvas
        coordinates first, sort each side independently by x, then zip them in
        sorted order so the leftmost upper terminal always connects to the leftmost
        lower terminal.
        """

        unified_transformers = self.merged_json.get("unified_transformers", {})
        transformer_connections = self.merged_json.get("transformer_connections", {})

        # Group transformer metadata by (from_level_idx, to_level_idx)
        from collections import defaultdict
        level_pair_groups: dict = defaultdict(list)

        for tr_id, levels in transformer_connections.items():
            if len(levels) != 2:
                continue

            from_level_idx, to_level_idx = levels
            tr_data = unified_transformers[tr_id]

            from_device_id = tr_data["from_device_id"]
            to_device_id   = tr_data["to_device_id"]

            # Strip voltage prefix to get the key used in boundary_pts
            from_original = from_device_id.split('_', 1)[1] if '_' in from_device_id else from_device_id
            to_original   = to_device_id.split('_', 1)[1]   if '_' in to_device_id   else to_device_id

            from_result = self.level_results[from_level_idx]
            to_result   = self.level_results[to_level_idx]

            from_pt_data = from_result.boundary_pts.get(from_original, {})
            to_pt_data   = to_result.boundary_pts.get(to_original, {})

            if not from_pt_data or not to_pt_data:
                continue

            from_offset = self.level_positions[from_level_idx]
            to_offset   = self.level_positions[to_level_idx]

            from_point = (
                from_pt_data["x"] + from_offset[0],
                from_pt_data["y"] + from_offset[1],
            )
            to_point = (
                to_pt_data["x"] + to_offset[0],
                to_pt_data["y"] + to_offset[1],
            )

            level_pair_groups[(from_level_idx, to_level_idx)].append({
                "tr_id":      tr_id,
                "label":      tr_data["label"],
                "from_point": from_point,
                "to_point":   to_point,
            })

        # For each level pair, sort both sides by x then re-zip to eliminate crossings
        for (from_level_idx, to_level_idx), group in level_pair_groups.items():
            upper_level_bottom = (
                self.level_positions[from_level_idx][1]
                + self.level_results[from_level_idx].height
            )
            sa, _sb = self._pair_spaces.get(
                (from_level_idx, to_level_idx),
                (TRANSFORMER_SPACE_ABOVE, TRANSFORMER_SPACE_BELOW),
            )
            tr_center_y = upper_level_bottom + sa + TRANSFORMER_SIZE / 2

            # Sort from-points and to-points independently by x
            from_side = sorted(
                [(entry["from_point"], entry["tr_id"], entry["label"]) for entry in group],
                key=lambda t: t[0][0]
            )
            to_side = sorted(
                [entry["to_point"] for entry in group],
                key=lambda pt: pt[0]
            )

            # Compute natural centre x values, then enforce minimum separation
            natural_cx = [(fp[0] + tp[0]) / 2 for (fp, _, _), tp in zip(from_side, to_side)]
            labels_lr  = [lbl for (_, _, lbl) in from_side]
            adjusted_cx = _enforce_transformer_separation(natural_cx, labels_lr)

            for ((from_point, tr_id, label), to_point, cx) in zip(from_side, to_side, adjusted_cx):
                center = (cx, tr_center_y)
                self.transformers.append(TransformerPlacement(
                    id=tr_id,
                    label=label,
                    center=center,
                    from_point=from_point,
                    to_point=to_point,
                ))

    
    def _generate_svg(self, output_path: str):
        """Generate final stitched SVG"""
        
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
        
        # ── Compute elbow map (detect-then-stagger) ──────────────────────
        #
        # For each level pair group (same tr_top_y), compute the natural
        # elbow_y for every wire using the 1/3-span rule.
        # Then check whether any two wires share the same elbow row AND have
        # overlapping horizontal extents — a genuine visual overlap.
        # If NO conflicts are found the natural elbows are kept as-is.
        # If ANY conflict is found, ALL wires in the pair are redistributed
        # evenly across the available gap (which _calculate_layout already
        # expanded to n_conflict * WIRE_SEPARATION).
        from collections import defaultdict as _ddict

        def _nat_elbow(y_start, y_end):
            span = abs(y_end - y_start)
            return y_start + max(span / 3, 4) if y_end >= y_start else y_start - max(span / 3, 4)

        def _horiz_overlap(ax0, ax1, bx0, bx1):
            return min(ax0, ax1) < max(bx0, bx1) and min(bx0, bx1) < max(ax0, ax1)

        # Group transformers by level pair key (tr_top_y)
        pair_groups: dict = _ddict(list)
        for tr in self.transformers:
            key = round(tr.center[1] - TRANSFORMER_SIZE / 2, 1)
            pair_groups[key].append(tr)

        elbow_map: dict = {}  # tr.id -> (ey1, ey2)

        for tr_top_key, group in pair_groups.items():
            # Natural elbows for this group
            nat: dict = {}
            for tr in group:
                tr_top_y    = tr.center[1] - TRANSFORMER_SIZE / 2
                tr_bottom_y = tr.center[1] + TRANSFORMER_SIZE / 2
                nat[tr.id] = (
                    _nat_elbow(tr.from_point[1], tr_top_y),
                    _nat_elbow(tr_bottom_y, tr.to_point[1]),
                )

            # Detect path1 conflicts (same ey1 + overlapping from→tr_x extents)
            conflict1 = False
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    ti, tj = group[i], group[j]
                    if (abs(nat[ti.id][0] - nat[tj.id][0]) < 0.5 and
                            _horiz_overlap(ti.from_point[0], ti.center[0],
                                           tj.from_point[0], tj.center[0])):
                        conflict1 = True
                        break
                if conflict1:
                    break

            # Detect path2 conflicts (same ey2 + overlapping tr_x→to extents)
            conflict2 = False
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    ti, tj = group[i], group[j]
                    if (abs(nat[ti.id][1] - nat[tj.id][1]) < 0.5 and
                            _horiz_overlap(ti.center[0], ti.to_point[0],
                                           tj.center[0], tj.to_point[0])):
                        conflict2 = True
                        break
                if conflict2:
                    break

            if not conflict1 and not conflict2:
                # Clean — keep natural elbows
                for tr in group:
                    elbow_map[tr.id] = nat[tr.id]
            else:
                # Stagger with direction-aware ranking.
                #
                # Invariant: if wire j's horizontal [from_x_j, tr_x_j] spans
                # tr_x_i (wire i's right vertical leg), then frac_j < frac_i.
                # This places j's elbow above i's, so j's horizontal segment
                # is above the start of i's right leg — no intersection.
                #
                # Sort key per wire: threatened - threatening
                #   threatened:  others whose horizontals span this wire's tr_x
                #                → needs HIGH frac (low elbow, short right leg)
                #   threatening: others' tr_x values this horizontal spans
                #                → needs LOW frac (high elbow, clears right legs below)
                #
                # Sorting ascending by (threatened - threatening) assigns low
                # frac to threatening wires and high frac to threatened wires,
                # satisfying the invariant for all wire pairs.

                def _threatened(tr, grp):
                    tx = tr.center[0]
                    return sum(
                        1 for o in grp if o.id != tr.id
                        and min(o.from_point[0], o.center[0]) < tx < max(o.from_point[0], o.center[0])
                    )

                def _threatening(tr, grp):
                    lo, hi = min(tr.from_point[0], tr.center[0]), max(tr.from_point[0], tr.center[0])
                    return sum(1 for o in grp if o.id != tr.id and lo < o.center[0] < hi)

                scored = sorted(group, key=lambda t: _threatened(t, group) - _threatening(t, group))
                n = len(scored)
                for rank, tr in enumerate(scored):
                    tr_top_y    = tr.center[1] - TRANSFORMER_SIZE / 2
                    tr_bottom_y = tr.center[1] + TRANSFORMER_SIZE / 2
                    frac = (rank + 1) / (n + 1)
                    ey1 = tr.from_point[1] + frac * (tr_top_y    - tr.from_point[1])
                    ey2 = tr_bottom_y      + frac * (tr.to_point[1] - tr_bottom_y)
                    elbow_map[tr.id] = (ey1, ey2)

        # ── Interval-based segment collision resolution ──────────────────
        from collections import defaultdict
        horizontal_segments = defaultdict(list)  # y -> [(x1, x2)]
        vertical_segments = defaultdict(list)    # x -> [(y1, y2)]

        def normalize_range(a, b):
            return (min(a, b), max(a, b))

        def overlaps(a1, a2, b1, b2):
            return max(a1, b1) < min(a2, b2)

        def resolve_horizontal(p1, p2):
            y = p1[1]
            x1, x2 = normalize_range(p1[0], p2[0])
            k = 0
            while True:
                y_shifted = y + k * WIRE_SEPARATION
                conflict = False
                for ex1, ex2 in horizontal_segments[y_shifted]:
                    if overlaps(x1, x2, ex1, ex2):
                        conflict = True
                        break
                if not conflict:
                    horizontal_segments[y_shifted].append((x1, x2))
                    return (p1[0], y_shifted), (p2[0], y_shifted)
                k += 1
                if k > 100:
                    horizontal_segments[y_shifted].append((x1, x2))
                    return (p1[0], y_shifted), (p2[0], y_shifted)

        def resolve_vertical(p1, p2):
            x = p1[0]
            y1, y2 = normalize_range(p1[1], p2[1])
            k = 0
            while True:
                x_shifted = x + k * WIRE_SEPARATION
                conflict = False
                for ey1, ey2 in vertical_segments[x_shifted]:
                    if overlaps(y1, y2, ey1, ey2):
                        conflict = True
                        break
                if not conflict:
                    vertical_segments[x_shifted].append((y1, y2))
                    return (x_shifted, p1[1]), (x_shifted, p2[1])
                k += 1
                if k > 100:
                    vertical_segments[x_shifted].append((y1, y2))
                    return (x_shifted, p1[1]), (x_shifted, p2[1])

        # ── Draw transformers ─────────────────────────────────────────────
        svg_parts.append('<!-- Inter-Level Connections -->')
        svg_parts.append('<g id="inter_level_connections">')

        for tr in self.transformers:
            tr_x, tr_y = tr.center
            tr_top_y    = tr_y - TRANSFORMER_SIZE / 2
            tr_bottom_y = tr_y + TRANSFORMER_SIZE / 2
            ey1, ey2 = elbow_map[tr.id]

            # Path1: from_point → (tr_x, tr_top_y) with elbow at ey1
            # Decompose into 3 segments
            start1 = tr.from_point
            mid1_1 = (start1[0], ey1)
            mid1_2 = (tr_x, ey1)
            end1 = (tr_x, tr_top_y)

            segments1 = [
                (start1, mid1_1),  # vertical
                (mid1_1, mid1_2),  # horizontal
                (mid1_2, end1),    # vertical
            ]

            # Resolve each segment
            resolved1 = []
            for p1, p2 in segments1:
                if abs(p1[1] - p2[1]) < 0.1:  # horizontal
                    s, e = resolve_horizontal(p1, p2)
                else:  # vertical
                    s, e = resolve_vertical(p1, p2)
                resolved1.append((s, e))

            # Reconstruct path1
            path1_d = f"M {resolved1[0][0][0]:.1f},{resolved1[0][0][1]:.1f}"
            for _, end_pt in resolved1:
                path1_d += f" L {end_pt[0]:.1f},{end_pt[1]:.1f}"

            svg_parts.append(
                f'<path d="{path1_d}" stroke="{LINE_COLOR}" stroke-width="{LINE_WIDTH}" '
                f'fill="none" stroke-linecap="square"/>'
            )

            svg_parts.append(generate_transformer_symbol(tr_x, tr_y, tr.label))

            # Path2: (tr_x, tr_bottom_y) → to_point with elbow at ey2
            # Decompose into 3 segments
            start2 = (tr_x, tr_bottom_y)
            mid2_1 = (start2[0], ey2)
            mid2_2 = (tr.to_point[0], ey2)
            end2 = tr.to_point

            segments2 = [
                (start2, mid2_1),  # vertical
                (mid2_1, mid2_2),  # horizontal
                (mid2_2, end2),    # vertical
            ]

            # Resolve each segment
            resolved2 = []
            for p1, p2 in segments2:
                if abs(p1[1] - p2[1]) < 0.1:  # horizontal
                    s, e = resolve_horizontal(p1, p2)
                else:  # vertical
                    s, e = resolve_vertical(p1, p2)
                resolved2.append((s, e))

            # Reconstruct path2
            path2_d = f"M {resolved2[0][0][0]:.1f},{resolved2[0][0][1]:.1f}"
            for _, end_pt in resolved2:
                path2_d += f" L {end_pt[0]:.1f},{end_pt[1]:.1f}"

            svg_parts.append(
                f'<path d="{path2_d}" stroke="{LINE_COLOR}" stroke-width="{LINE_WIDTH}" '
                f'fill="none" stroke-linecap="square"/>'
            )

        svg_parts.append('</g>')
        svg_parts.append('</svg>')
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(svg_parts))
        
    
    def _generate_yolo(self, output_path: str):
        """Generate merged YOLO annotations"""
        
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