"""
breaker_and_half_renderer.py
============================
Three-function renderer for BREAKER_AND_HALF topology.

  _render_core(...)        → draws buses, bay stacks, connecting wires,
                             horizontal+vertical stub wires to exposed terminal
                             endpoints. Returns layout geometry + svg list.

  _cap_standalone(...)     → stamps symbol + label at every exposed wire end.
                             Labels are nudged vertically to avoid overlap.

  _cap_hybrid(...)         → stamps feeder/generator symbols same as standalone,
                             but for transformers extends the wire to the SVG
                             boundary and leaves it open (no symbol).

  render_breaker_and_half_svg(topology_json, output_path, mode="standalone")
                           → orchestrates the above, writes SVG file.
"""

from collections import defaultdict


# ── Global IEC symbol definitions (32×32 viewBox, stroke-width 1) ─────────────
IEC_SYMBOLS = {
    "isolator": (
        '<line x1="16" y1="0" x2="16" y2="9.5" stroke="#000" stroke-width="1"/>'
        '<polyline points="11,9.5 16,22.5 16,32" fill="none" stroke="#000" stroke-width="1"/>'
    ),
    "breaker": (
        '<line x1="16" y1="0" x2="16" y2="8" stroke="#000" stroke-width="1"/>'
        '<line x1="16" y1="32" x2="16" y2="24" stroke="#000" stroke-width="1"/>'
        '<rect x="12" y="8" width="8" height="16" fill="none" stroke="#000" stroke-width="1"/>'
        '<line x1="12" y1="20" x2="20" y2="16" stroke="#000" stroke-width="1"/>'
    ),
    "transformer": (
        '<line x1="16" y1="0" x2="16" y2="6" stroke="#000" stroke-width="1"/>'
        '<circle cx="16" cy="11.5" r="6" fill="none" stroke="#000" stroke-width="1"/>'
        '<circle cx="16" cy="20.5" r="6" fill="none" stroke="#000" stroke-width="1"/>'
        '<line x1="16" y1="26" x2="16" y2="32" stroke="#000" stroke-width="1"/>'
    ),
    "generator": (
        '<circle cx="16" cy="8" r="7" fill="none" stroke="#000" stroke-width="1"/>'
        '<line x1="16" y1="15" x2="16" y2="32" stroke="#000" stroke-width="1"/>'
        '<text x="16" y="10.5" text-anchor="middle" font-size="7" font-weight="bold" '
        'fill="#000" font-family="Arial">G</text>'
    ),
    "feeder": (
        '<polyline points="16,0 16,5 11,11 22,16 11,21 22,26 16,26" '
        'fill="none" stroke="#000" stroke-width="1"/>'
    ),
}


def _goes_up(dev_id: str, dev_type: str) -> bool:
    if dev_type == "GridSource":
        return True
    if dev_type == "Transformer" and "UP" in dev_id.upper():
        return True
    return False


def _sym_key(dev_type: str) -> str:
    return {"Feeder": "feeder", "Transformer": "transformer",
            "GridSource": "generator"}.get(dev_type, "feeder")


# ── SVG primitive helpers (stateless, operate on an external list) ─────────────

def _ln(svg, offset, x1, y1, x2, y2):
    svg.append(
        f'<line x1="{x1+offset:.1f}" y1="{y1:.1f}" '
        f'x2="{x2+offset:.1f}" y2="{y2:.1f}" '
        f'stroke="#000" stroke-width="1" stroke-linecap="round"/>'
    )

def _dot(svg, offset, x, y, r=1.5):
    svg.append(f'<circle cx="{x+offset:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#000"/>')

def _sym(svg, offset, name, x, y, size=32.0):
    scale = size / 32.0
    svg.append(
        f'<g transform="translate({x+offset:.1f},{y:.1f}) scale({scale:.4f})">'
        + IEC_SYMBOLS[name] + '</g>'
    )

def _lbl(svg, offset, text, x, y, anchor="start", size=7):
    svg.append(
        f'<text x="{x+offset:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="monospace" font-size="{size}" fill="#000">{text}</text>'
    )

def _bus_line(svg, offset, x1, y, x2, w=2.0):
    svg.append(
        f'<line x1="{x1+offset:.1f}" y1="{y:.1f}" '
        f'x2="{x2+offset:.1f}" y2="{y:.1f}" '
        f'stroke="#000" stroke-width="{w}" stroke-linecap="square"/>'
    )


# ── CORE ───────────────────────────────────────────────────────────────────────

def _render_core(topology_json: dict) -> dict:
    """
    Parse topology, compute geometry, draw buses + bay stacks + stub wires.
    Returns a geometry/state dict consumed by the cappers.
    """
    # ── Constants ───────────────────────────────────────────────────────────
    MARGIN             = 0.0
    TOP_MARGIN         = 20.0    # space above top bus (for bus label + clearance)
    BOT_MARGIN         = 20.0    # space below bottom bus
    COL_W_MIN          = 60.0
    FONT_W             = 4.3     # px per char at font-size 7 monospace
    LABEL_PAD_L        = -4.0    # padding between label right-edge and symbol left-edge
    LABEL_PAD_R        = 32.0    # padding to the right of symbol (controls inter-bay gap)
    BUS_OVERHANG_L     = 80.0    # how far bus extends left of first bay symbol centre
    BUS_OVERHANG_R     = 50.0    # how far bus extends right of last bay symbol centre
    SYM                = 32.0
    BUS_SPINE_W        = 2.0
    JOIN               = 0.0
    FEEDER_LEFT        = 20.0    # horizontal clearance outside bus for UP wires
    FEEDER_BEYOND      = 20.0    # vertical stub length past bus for feeders
    TRANSFORMER_BEYOND = 20.0    # vertical stub length past bus for transformers
    UPSTREAM_BUS_GAP   = -15.0     # gap between top bus and upstream symbol bottom
    NODE_R             = 0.0
    DOWN_STAGGER       = 20.0    # horizontal separation between DOWN wires in same bay
    DOWN_V_STAGGER     = 20.0    # vertical stagger between DOWN symbols in same bay

    # ── Parse ────────────────────────────────────────────────────────────────
    devices = {d["id"]: d for d in topology_json["devices"]}

    def dev_label(dev_id):
        return devices[dev_id].get("label") or dev_id

    adj = defaultdict(set)
    for c in topology_json["connections"]:
        adj[c["from"]].add(c["to"])
        adj[c["to"]].add(c["from"])

    bays_raw = defaultdict(list)
    for d in devices.values():
        if d.get("bay_id"):
            bays_raw[d["bay_id"]].append(d["id"])

    bay_ids   = sorted(bays_raw.keys())
    first_bay = bay_ids[0]
    last_bay  = bay_ids[-1]

    # ── Bus identification ───────────────────────────────────────────────────
    bus_ids    = sorted(d["id"] for d in devices.values() if d["type"] == "Bus")
    bus_top_id = bus_ids[0]
    bus_bot_id = bus_ids[1]

    # ── Bay stacks (connection-driven) ───────────────────────────────────────
    bay_stacks = {}
    for bay_id in bay_ids:
        dev_ids   = set(bays_raw[bay_id])
        isolators = [d for d in dev_ids if devices[d]["type"] == "Isolator"]
        breakers  = [d for d in dev_ids if devices[d]["type"] == "Breaker"]

        iso_top = iso_bot = None
        for iso in isolators:
            bus_nbs = [n for n in adj[iso] if devices[n]["type"] == "Bus"]
            if bus_top_id in bus_nbs:
                iso_top = iso
            elif bus_bot_id in bus_nbs:
                iso_bot = iso

        brk1 = next(b for b in breakers if iso_top in adj[b])
        brk3 = next(b for b in breakers if iso_bot in adj[b])
        brk2 = next(b for b in breakers if b not in (brk1, brk3))
        bay_stacks[bay_id] = dict(iso_top=iso_top, brk1=brk1, brk2=brk2,
                                  brk3=brk3, iso_bot=iso_bot)

    # ── Per-bay column widths ────────────────────────────────────────────────
    def _bay_label_w(bay_id):
        stack = bay_stacks[bay_id]
        ids = [stack[k] for k in ("iso_top", "brk1", "brk2", "brk3", "iso_bot")]
        return max(len(dev_label(i)) for i in ids) * FONT_W

    # Column width: label on left, symbol, right pad
    # Left side:  label_w + LABEL_PAD_L (between label and symbol)
    # Symbol:     SYM
    # Right side: LABEL_PAD_R
    bay_col_w = {b: max(COL_W_MIN, _bay_label_w(b) + LABEL_PAD_L + SYM + LABEL_PAD_R)
                 for b in bay_ids}

    # cx: col_left + label_w + LABEL_PAD_L + SYM/2
    bay_cx = {}
    _x = MARGIN
    for b in bay_ids:
        label_w = _bay_label_w(b)
        bay_cx[b] = _x + label_w + LABEL_PAD_L + SYM / 2
        _x += bay_col_w[b]

    # ── Exposed terminal routing ─────────────────────────────────────────────
    exposed_set = set(topology_json.get("layout", {}).get("terminals", []))
    term_info = {}
    for et in exposed_set:
        d   = devices[et]
        brk = next(iter(adj[et]))
        bay_id = devices[brk]["bay_id"]
        stack  = bay_stacks[bay_id]
        if brk == stack["brk1"]:   brk_pos = 1
        elif brk == stack["brk2"]: brk_pos = 2
        else:                      brk_pos = 3
        term_info[et] = dict(
            brk=brk, brk_pos=brk_pos, bay_id=bay_id,
            b_idx=bay_ids.index(bay_id),
            goes_up=_goes_up(et, d["type"]),
        )

    # ── Vertical geometry ────────────────────────────────────────────────────
    STACK     = 5 * SYM + 6 * JOIN
    BUS_Y_TOP = TOP_MARGIN
    BUS_Y_BOT = TOP_MARGIN + STACK

    y_iso_top = BUS_Y_TOP + JOIN
    y_brk1    = y_iso_top + SYM + JOIN
    y_brk2    = y_brk1   + SYM + JOIN
    y_brk3    = y_brk2   + SYM + JOIN
    y_iso_bot = y_brk3   + SYM + JOIN

    tap_y = {1: y_brk1 + SYM, 2: y_brk2 + SYM, 3: y_brk3 + SYM}

    # ── Canvas X bounds ──────────────────────────────────────────────────────
    first_cx    = bay_cx[bay_ids[0]]
    last_cx     = bay_cx[bay_ids[-1]]
    # Bus overhangs SYM/2 beyond the outermost symbol centres so it
    # visually brackets the bays symmetrically, independent of label widths.
    BUS_X1   = first_cx - BUS_OVERHANG_L
    BUS_X2   = last_cx  + BUS_OVERHANG_R

    # ── DOWN stagger assignments ─────────────────────────────────────────────
    bay_down_terms = defaultdict(list)
    for et, ti in term_info.items():
        if not ti["goes_up"]:
            bay_down_terms[ti["bay_id"]].append(et)

    down_reach  = {}
    down_sym_dy = {}
    for bay_id, terms in bay_down_terms.items():
        cx = bay_cx[bay_id]
        # Sort DESCENDING: higher brk_pos (closer to bot bus) gets shorter reach
        # and sits lower (no sym_dy shift). Lower brk_pos gets longer reach and
        # sits higher (negative sym_dy). This keeps left=lower, right=higher.
        terms_sorted = sorted(terms, key=lambda t: term_info[t]["brk_pos"], reverse=True)
        for i, et in enumerate(terms_sorted):
            down_reach[et]  = cx + DOWN_STAGGER * (i + 1)
            down_sym_dy[et] = -DOWN_V_STAGGER * i

    # ── Wire endpoints (where stubs end, capper stamps symbols here) ──────────
    # end_pts: terminal_id → (rx, wire_end_y, goes_up, sym_key, beyond)
    end_pts = {}
    for et, ti in term_info.items():
        d      = devices[et]
        sk     = _sym_key(d["type"])
        beyond = TRANSFORMER_BEYOND if d["type"] in ("Transformer", "GridSource") \
                 else FEEDER_BEYOND
        cx     = bay_cx[ti["bay_id"]]
        ty     = tap_y[ti["brk_pos"]]

        if ti["goes_up"]:
            rx         = BUS_X1 - FEEDER_LEFT if ti["bay_id"] == first_bay \
                         else BUS_X2 + FEEDER_LEFT
            wire_end_y = BUS_Y_TOP - UPSTREAM_BUS_GAP - beyond
        else:
            rx         = down_reach[et]
            wire_end_y = BUS_Y_BOT + beyond

        end_pts[et] = dict(
            rx=rx, wire_end_y=wire_end_y,
            goes_up=ti["goes_up"], sk=sk,
            dev_type=d["type"], beyond=beyond,
            tap_cx=cx, tap_y=ty,
            sym_dy=down_sym_dy.get(et, 0.0),
            is_first_bay=(ti["bay_id"] == first_bay),
        )

    # ── Compute canvas extents ────────────────────────────────────────────────
    all_rx = [BUS_X1, BUS_X2]
    for ep in end_pts.values():
        all_rx.append(ep["rx"])

    # ── Compute canvas extents ────────────────────────────────────────────────
    # canvas_right is the max of:
    #   - bus right edge (BUS_X2) + small margin
    #   - each DOWN terminal's rx + SYM/2 + label_gap + label_width
    right_candidates = [BUS_X2 + MARGIN]
    for et, ep in end_pts.items():
        if not ep["goes_up"]:
            lw = len(dev_label(et)) * FONT_W
            right_candidates.append(ep["rx"] + SYM / 2 + 2 + lw + MARGIN)
    canvas_left  = min(all_rx) - SYM / 2 - MARGIN
    canvas_right = max(right_candidates)
    offset       = -canvas_left

    # Canvas height determined after capper decides TOP/BOT extents.
    # Core returns everything needed; height finalised in render_*.

    # ── Draw core SVG elements ────────────────────────────────────────────────
    svg = []
    bbox_list = []  # (class_id, x_left, y_top, w, h) in internal coords

    # Bus spines
    _bus_line(svg, offset, BUS_X1, BUS_Y_TOP, BUS_X2, BUS_SPINE_W)
    _bus_line(svg, offset, BUS_X1, BUS_Y_BOT, BUS_X2, BUS_SPINE_W)
    bbox_list.append((6, BUS_X1, BUS_Y_TOP - BUS_SPINE_W / 2, BUS_X2 - BUS_X1, BUS_SPINE_W))
    bbox_list.append((6, BUS_X1, BUS_Y_BOT - BUS_SPINE_W / 2, BUS_X2 - BUS_X1, BUS_SPINE_W))

    # Bus labels: left end of each bus, top bus above spine, bottom bus below
    bus_lbl_x = BUS_X1 + 2
    _lbl(svg, offset, dev_label(bus_top_id), bus_lbl_x, BUS_Y_TOP - 3, anchor="start")
    _lbl(svg, offset, dev_label(bus_bot_id), bus_lbl_x, BUS_Y_BOT + 9, anchor="start")

    # Bay stacks
    for bay_id in bay_ids:
        cx    = bay_cx[bay_id]
        sx    = cx - SYM / 2
        stack = bay_stacks[bay_id]

        # Connecting wires
        _ln(svg, offset, cx, BUS_Y_TOP,       cx, y_iso_top)
        _dot(svg, offset, cx, BUS_Y_TOP,       NODE_R)
        _ln(svg, offset, cx, y_iso_top + SYM,  cx, y_brk1)
        _dot(svg, offset, cx, y_iso_top + SYM, NODE_R)
        _ln(svg, offset, cx, y_brk1 + SYM,    cx, y_brk2)
        _dot(svg, offset, cx, y_brk1 + SYM,   NODE_R)
        _ln(svg, offset, cx, y_brk2 + SYM,    cx, y_brk3)
        _dot(svg, offset, cx, y_brk2 + SYM,   NODE_R)
        _ln(svg, offset, cx, y_brk3 + SYM,    cx, y_iso_bot)
        _dot(svg, offset, cx, y_brk3 + SYM,   NODE_R)
        _ln(svg, offset, cx, y_iso_bot + SYM,  cx, BUS_Y_BOT)
        _dot(svg, offset, cx, BUS_Y_BOT,       NODE_R)

        # Symbols + bboxes
        _sym(svg, offset, "isolator", sx, y_iso_top)
        bbox_list.append((5, sx, y_iso_top, SYM, SYM))
        _sym(svg, offset, "breaker",  sx, y_brk1)
        bbox_list.append((2, sx, y_brk1, SYM, SYM))
        _sym(svg, offset, "breaker",  sx, y_brk2)
        bbox_list.append((2, sx, y_brk2, SYM, SYM))
        _sym(svg, offset, "breaker",  sx, y_brk3)
        bbox_list.append((2, sx, y_brk3, SYM, SYM))
        _sym(svg, offset, "isolator", sx, y_iso_bot)
        bbox_list.append((5, sx, y_iso_bot, SYM, SYM))

        # Bay device labels (left of symbol)
        lx = cx - SYM / 2 - LABEL_PAD_L
        _lbl(svg, offset, dev_label(stack["iso_top"]), lx, y_iso_top + SYM * 0.6, "end")
        _lbl(svg, offset, dev_label(stack["brk1"]),    lx, y_brk1   + SYM * 0.6, "end")
        _lbl(svg, offset, dev_label(stack["brk2"]),    lx, y_brk2   + SYM * 0.6, "end")
        _lbl(svg, offset, dev_label(stack["brk3"]),    lx, y_brk3   + SYM * 0.6, "end")
        _lbl(svg, offset, dev_label(stack["iso_bot"]), lx, y_iso_bot + SYM * 0.6, "end")

    # Stub wires to exposed terminal endpoints
    for et, ep in end_pts.items():
        _ln(svg, offset, ep["tap_cx"], ep["tap_y"], ep["rx"],         ep["tap_y"])
        _ln(svg, offset, ep["rx"],     ep["tap_y"], ep["rx"],         ep["wire_end_y"] + ep["sym_dy"])
        _dot(svg, offset, ep["tap_cx"], ep["tap_y"], NODE_R)

    return dict(
        svg=svg, offset=offset,
        BUS_Y_TOP=BUS_Y_TOP, BUS_Y_BOT=BUS_Y_BOT,
        BUS_X1=BUS_X1, BUS_X2=BUS_X2,
        canvas_left=canvas_left, canvas_right=canvas_right,
        MARGIN=MARGIN, SYM=SYM, FONT_W=FONT_W,
        end_pts=end_pts, devices=devices,
        dev_label=dev_label,
        bbox_list=bbox_list,
    )


# ── LABELLER (shared by both cappers) ─────────────────────────────────────────

def _place_labels(svg, offset, label_reqs, SYM, FONT_W):
    """
    label_reqs: list of (x, y, text, anchor)
    Nudges vertically any labels whose y-positions are within 8px of each other
    at the same x column (same rx).
    """
    # Group by rx bucket (rounded to nearest px)
    from collections import defaultdict
    by_x = defaultdict(list)
    for i, (x, y, text, anchor) in enumerate(label_reqs):
        by_x[round(x)].append((y, i))

    nudged = list(label_reqs)
    MIN_DY = 8.0
    for col_items in by_x.values():
        col_items.sort()
        for j in range(1, len(col_items)):
            prev_y, prev_i = col_items[j - 1]
            cur_y,  cur_i  = col_items[j]
            if cur_y - prev_y < MIN_DY:
                new_y = prev_y + MIN_DY
                x, _, text, anchor = nudged[cur_i]
                nudged[cur_i] = (x, new_y, text, anchor)
                col_items[j] = (new_y, cur_i)

    for (x, y, text, anchor) in nudged:
        _lbl(svg, offset, text, x, y, anchor)


# ── CAPPER (standalone + hybrid) ─────────────────────────────────────────────

def _cap(core: dict, mode: str, y_offset: float, H: float) -> dict:
    """
    Stamps symbols and labels at every exposed wire end.

    standalone: all terminals get symbol + label.
    hybrid:     Feeder/GridSource → symbol + label (same as standalone).
                Transformer       → wire extended to SVG boundary, no symbol.

    Returns dict of transformer boundary points (only populated in hybrid mode):
      { device_id: {"label": str, "x": float, "y": float} }
    The x/y are in final SVG coordinates (after y_offset shift).
    """
    svg       = core["svg"]
    offset    = core["offset"]
    SYM       = core["SYM"]
    FONT_W    = core["FONT_W"]
    MARGIN    = core["MARGIN"]
    end_pts   = core["end_pts"]
    dev_label = core["dev_label"]

    label_reqs    = []
    boundary_pts  = {}   # returned for hybrid transformer ends
    bbox_list     = core["bbox_list"]
    TERM_CLASS    = {"GridSource": 0, "Transformer": 1, "Feeder": 3}

    for et, ep in end_pts.items():
        rx         = ep["rx"]
        wire_end_y = ep["wire_end_y"]
        sk         = ep["sk"]
        goes_up    = ep["goes_up"]
        sym_dy     = ep["sym_dy"]
        dev_type   = ep["dev_type"]

        if mode == "hybrid" and dev_type == "Transformer":
            # Extend wire to the SVG boundary edge and record the endpoint
            boundary_y = -y_offset + MARGIN if goes_up else -y_offset + H - MARGIN
            _ln(svg, offset, rx, wire_end_y, rx, boundary_y)
            # Convert to final SVG coordinate space
            final_x = rx + offset
            final_y = boundary_y + y_offset
            boundary_pts[et] = {
                "label": dev_label(et),
                "x": round(final_x, 1),
                "y": round(final_y, 1),
            }
        else:
            # Stamp symbol for all modes / all non-transformer devices
            cls = TERM_CLASS.get(dev_type, 3)
            if goes_up:
                sym_top_y = wire_end_y - SYM
                _sym(svg, offset, sk, rx - SYM / 2, sym_top_y)
                bbox_list.append((cls, rx - SYM / 2, sym_top_y, SYM, SYM))
                if ep.get("is_first_bay"):
                    lx, anchor = rx + SYM / 2 + 2, "start"
                else:
                    lx, anchor = rx - SYM / 2 - 2, "end"
                label_reqs.append((lx, sym_top_y + SYM * 0.6, dev_label(et), anchor))
            else:
                sym_top_y = wire_end_y + sym_dy
                _sym(svg, offset, sk, rx - SYM / 2, sym_top_y)
                bbox_list.append((cls, rx - SYM / 2, sym_top_y, SYM, SYM))
                lx = rx + SYM / 2 + 2
                label_reqs.append((lx, sym_top_y + SYM * 0.6, dev_label(et), "start"))

    _place_labels(svg, offset, label_reqs, SYM, FONT_W)
    return boundary_pts


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def render_breaker_and_half_svg(
    topology_path: str,
    output_path: str,
    mode: str = "standalone",   # "standalone" | "hybrid"
) -> dict | None:

    import json

    with open(topology_path) as f:
        topology_json = json.load(f)

    layout = topology_json.get("layout", {})
    topo_type = layout.get("topology_type") or layout.get("mode", "")

    if topo_type != "BREAKER_AND_HALF":
        raise ValueError(f"Expected BREAKER_AND_HALF, got '{topo_type}'")

    core = _render_core(topology_json)

    svg          = core["svg"]
    offset       = core["offset"]
    BUS_Y_TOP    = core["BUS_Y_TOP"]
    BUS_Y_BOT    = core["BUS_Y_BOT"]
    canvas_left  = core["canvas_left"]
    canvas_right = core["canvas_right"]
    MARGIN       = core["MARGIN"]
    SYM          = core["SYM"]
    end_pts      = core["end_pts"]

    # Determine vertical canvas extent from wire endpoints + symbol heights
    up_ys   = [ep["wire_end_y"] - SYM for ep in end_pts.values() if ep["goes_up"]]
    down_ys = [ep["wire_end_y"] + SYM  for ep in end_pts.values() if not ep["goes_up"]]
    top_y   = min([BUS_Y_TOP] + up_ys)   - MARGIN
    bot_y   = max([BUS_Y_BOT] + down_ys) + MARGIN

    # Shift everything so top_y maps to 0
    y_offset = -top_y
    W = canvas_right - canvas_left
    H = bot_y - top_y

    # Apply y_offset to all existing elements by rewriting y coordinates.
    # Simpler: just set a viewBox that starts at top_y.
    # We'll use a <g> translate for the y shift.
    svg_body = '\n'.join(svg)
    # Wrap in a group that shifts y
    full_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" '
        f'style="background:#fff">\n'
        f'<g transform="translate(0,{y_offset:.1f})">\n'
        + svg_body + '\n'
    )

    boundary_pts = _cap(core, mode, y_offset, H)

    # YOLO export: .txt file alongside the SVG
    # Transform internal coords → final SVG coords: x += offset, y += y_offset
    yolo_lines = []
    for (cls, x_left, y_top, w, h) in core["bbox_list"]:
        fx = x_left + offset
        fy = y_top  + y_offset
        xc = (fx + w / 2) / W
        yc = (fy + h / 2) / H
        nw = w / W
        nh = h / H
        yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")


    full_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" '
        f'style="background:#fff">\n'
        f'<g transform="translate(0,{y_offset:.1f})">\n'
        + '\n'.join(svg) + '\n</g>\n</svg>'
    )

    if mode == "hybrid":
        return boundary_pts, full_svg, yolo_lines
    else:   
        with open(output_path, "w") as fh:
            fh.write(full_svg)
        
        yolo_path = output_path.rsplit(".", 1)[0] + ".txt"
        with open(yolo_path, "w") as fh:
            fh.write('\n'.join(yolo_lines))
  
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 3:
        print("Usage: python breaker_and_half_renderer.py <topology.json> <output.svg> [standalone|hybrid]")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        topo = json.load(fh)
    m = sys.argv[3] if len(sys.argv) > 3 else "standalone"
    render_breaker_and_half_svg(topo, sys.argv[2], mode=m)
    print(f"Written: {sys.argv[2]} ({m})")