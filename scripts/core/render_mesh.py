"""
mesh_renderer.py
================
Three-function renderer for MESH topology.

  _render_core(topology_json)
      → Parses the topology, computes geometry, draws bus bars, mesh-ring
        segment breakers, and external-leg stacks (breaker + isolator + stub
        wire to the terminal endpoint).  Returns a geometry/state dict
        consumed by _cap.

  _cap(core, mode, y_offset, H)
      → Stamps terminal symbols and labels at every exposed wire end.
        standalone : all terminals get symbol + label.
        hybrid     : Feeder/GridSource → symbol + label.
                     Transformer       → wire extended to SVG boundary,
                                         no symbol.
        Returns transformer boundary points (hybrid mode only):
          { device_id: {"label": str, "x": float, "y": float} }

  render_mesh_svg(topology_json, output_path, mode="standalone")
      → Orchestrates the above, writes the SVG file (and a parallel YOLO
        .txt file).  Returns transformer boundary points in hybrid mode,
        else None.
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
    return {
        "Feeder":      "feeder",
        "Transformer": "transformer",
        "GridSource":  "generator",
    }.get(dev_type, "feeder")


# ── SVG primitive helpers ──────────────────────────────────────────────────────

def _ln(svg, x1, y1, x2, y2, w=1):
    svg.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="#000" stroke-width="{w}" stroke-linecap="round"/>'
    )

def _dot(svg, x, y, r=1.5):
    svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#000"/>')

def _sym(svg, name, cx, cy_top, size=32.0, rotate_deg=0):
    """Place IEC symbol centred at cx, top at cy_top.

    rotate_deg:
        0    – upright (default)
        180  – flipped 180° around symbol centre (for upward isolators)
        -90  – rotated CCW around symbol centre (for horizontal ring breakers)
    """
    scale = size / 32.0
    if rotate_deg == 0:
        svg.append(
            f'<g transform="translate({cx - size/2:.1f},{cy_top:.1f}) scale({scale:.4f})">'
            + IEC_SYMBOLS[name] + '</g>'
        )
    else:
        svg.append(
            f'<g transform="translate({cx - size/2:.1f},{cy_top:.1f}) scale({scale:.4f})">'
            f'<g transform="rotate({rotate_deg},16,16)">'
            + IEC_SYMBOLS[name] + '</g></g>'
        )

def _lbl(svg, text, x, y, anchor="start", size=7):
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="monospace" font-size="{size}" fill="#000">{text}</text>'
    )

def _lbl_rotated(svg, text, x, y, anchor="middle", size=7):
    """Label rotated +90° — runs top-to-bottom, placed to the right of a
    vertical wire or symbol."""
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="monospace" font-size="{size}" fill="#000" '
        f'transform="rotate(90,{x:.1f},{y:.1f})">{text}</text>'
    )

def _bus_bar(svg, cx, cy, half_w, w=2.0):
    svg.append(
        f'<line x1="{cx - half_w:.1f}" y1="{cy:.1f}" '
        f'x2="{cx + half_w:.1f}" y2="{cy:.1f}" '
        f'stroke="#000" stroke-width="{w}" stroke-linecap="square"/>'
    )


# ── CORE ───────────────────────────────────────────────────────────────────────

def _render_core(topology_json: dict) -> dict:
    """
    Parse topology, compute geometry, draw bus bars + mesh-ring segments +
    external-leg stacks (breaker + isolator + stub wire).

    Terminal symbols are NOT drawn here; _cap stamps those.
    Returns a geometry/state dict consumed by _cap.
    """
    # ── Layout constants ─────────────────────────────────────────────────────
    MARGIN_X    = 0.0    # left/right canvas margin
    MARGIN_Y    = 20.0    # top/bottom canvas margin
    SYM         = 32.0    # IEC symbol size (px)
    BUS_SPINE_W = 2.0
    NODE_R      = 0.0
    BUS_HALF_W  = 48.0    # half-width of each bus bar
    WIRE_GAP    = 2.0     # gap between successive stack elements
    COL_OFFSET  = 22.0    # horizontal pitch between legs on the same bus
    LEG_REACH   = 75      # Max reach of a bay
    FONT_W      = 0.0     # px per char at font-size 7 monospace
    BUS_LBL_X_OFF     =  10.0   # x shift from bar left-end  (0 = centred on left end)
    BUS_LBL_Y_OFF     =  2.0   # gap above spine top edge (positive = further above)
    RING_VT_LBL_X_OFF =  -8.0   # gap right of symbol right edge
    RING_VT_LBL_Y_OFF =  0.0   # vertical shift from symbol centre
    RING_HZ_LBL_Y_OFF =  -4.0   # gap below symbol bottom edge
    LBL_X_OFF         =  -8.0   # gap between symbol edge and label (left or right side)
    LBL_Y_OFF         =  0.0   # baseline offset from symbol vertical centre
      
    # ── Parse ────────────────────────────────────────────────────────────────
    devices = {d["id"]: d for d in topology_json["devices"]}

    def dev_label(dev_id):
        return devices[dev_id].get("label") or dev_id

    adj: dict = defaultdict(set)
    for c in topology_json["connections"]:
        adj[c["from"]].add(c["to"])
        adj[c["to"]].add(c["from"])

    raw_bus_pos = topology_json["layout"]["bus_positions"]
    bus_pos = {bid: (float(xy[0]), float(xy[1])) for bid, xy in raw_bus_pos.items()}

    xs = [p[0] for p in bus_pos.values()]
    ys = [p[1] for p in bus_pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    CANVAS_W = (max_x - min_x) + 2 * MARGIN_X + 2 * LEG_REACH
    CANVAS_H = (max_y - min_y) + 2 * MARGIN_Y + 2 * LEG_REACH

    # Canvas origin offset: bus at (bx, by) draws at (bx + ox, by + oy).
    ox = MARGIN_X + LEG_REACH - min_x
    oy = MARGIN_Y + LEG_REACH - min_y

    # Whitelist from JSON — only these device ids are valid terminal endpoints.
    exposed_set = set(topology_json.get("layout", {}).get("terminals", []))

    # ── Categorise all legs from every bus ────────────────────────────────────
    mesh_segs: list[dict] = []                    # ring segments between two buses
    ext_legs_by_bus: dict = defaultdict(list)     # external (terminal) legs
    seen_segs: set = set()

    for bus_id in sorted(bus_pos.keys()):
        for brk_id in sorted(adj[bus_id]):
            if devices[brk_id]["type"] != "Breaker":
                continue
            for far in sorted([nb for nb in adj[brk_id] if nb != bus_id]):
                far_type = devices[far]["type"]
                if far_type == "Bus":
                    key = frozenset({bus_id, far})
                    if key not in seen_segs:
                        seen_segs.add(key)
                        mesh_segs.append(dict(bus_a=bus_id, bus_b=far, brk=brk_id))
                elif far_type == "Isolator":
                    # Only accept neighbours of the isolator that are in exposed_set.
                    term_id = next(
                        (nb for nb in adj[far] if nb != brk_id and nb in exposed_set),
                        None,
                    )
                    if term_id:
                        goes_up = _goes_up(term_id, devices[term_id]["type"])
                        ext_legs_by_bus[bus_id].append(dict(
                            brk=brk_id, iso=far, terminal=term_id, goes_up=goes_up
                        ))

    # ── x values occupied by vertical ring wires at each bus ─────────────────
    occupied_cx: dict = defaultdict(set)
    for seg in mesh_segs:
        ax  = bus_pos[seg["bus_a"]][0]
        bxs = bus_pos[seg["bus_b"]][0]
        ay  = bus_pos[seg["bus_a"]][1]
        bys = bus_pos[seg["bus_b"]][1]
        if abs(bys - ay) >= abs(bxs - ax):       # vertical segment
            occupied_cx[seg["bus_a"]].add(ax + ox)
            occupied_cx[seg["bus_b"]].add(ax + ox)

    # ── Assign x positions to external legs per bus ───────────────────────────
    def _assign_leg_xs(bus_id: str, n: int) -> list:
        """Centre n legs around bus cx; nudge right if any would land on an
        occupied ring-wire x-coordinate."""
        bcx = bus_pos[bus_id][0] + ox
        candidates = [bcx + (i - (n - 1) / 2.0) * COL_OFFSET for i in range(n)]
        occ = occupied_cx.get(bus_id, set())
        if occ:
            for occ_x in occ:
                if any(abs(c - occ_x) < 4.0 for c in candidates):
                    candidates = [c + COL_OFFSET / 2.0 for c in candidates]
                    break
        return candidates

    # ── Init SVG state ────────────────────────────────────────────────────────
    svg: list[str] = []
    # bbox_list: (class_id, x_left, y_top, width, height)
    # class_id: 0=gridsource  1=transformer  2=breaker  3=feeder  4=capacitor  5=isolator  6=bus
    bbox_list: list[tuple] = []
    end_pts: dict = {}    # terminal_id → geometry dict, consumed by _cap

    # ── Draw bus bars + labels ────────────────────────────────────────────────
    # Label: text-anchor="middle", x at bar left-end + BUS_LBL_X_OFF.
    # Top-row buses (by == min_y): label below the spine (inside the mesh).
    # All other buses: label above the spine.
    for bus_id in sorted(bus_pos.keys()):
        bx, by = bus_pos[bus_id]
        cx = bx + ox
        cy = by + oy
        _bus_bar(svg, cx, cy, BUS_HALF_W, BUS_SPINE_W)
        bbox_list.append((6, cx - BUS_HALF_W, cy - BUS_SPINE_W / 2,
                          BUS_HALF_W * 2, BUS_SPINE_W))
        lbl_x = cx - BUS_HALF_W + BUS_LBL_X_OFF
        is_top_row = (by == min_y)
        if is_top_row:
            # Below spine: baseline = spine bottom + gap + cap-height (≈ font-size 7)
            lbl_y = cy + BUS_SPINE_W / 2 + BUS_LBL_Y_OFF + 7
        else:
            lbl_y = cy - BUS_SPINE_W / 2 - BUS_LBL_Y_OFF
        _lbl(svg, dev_label(bus_id), lbl_x, lbl_y, anchor="middle")

    # ── Draw mesh ring segments: BUS_A ── BRK ── BUS_B ───────────────────────
    for seg in mesh_segs:
        ax  = bus_pos[seg["bus_a"]][0] + ox
        ay  = bus_pos[seg["bus_a"]][1] + oy
        bx2 = bus_pos[seg["bus_b"]][0] + ox
        by2 = bus_pos[seg["bus_b"]][1] + oy
        dx  = bx2 - ax
        dy  = by2 - ay
        brk_lbl = dev_label(seg["brk"])

        if abs(dy) >= abs(dx):
            # ── Vertical ring segment ─────────────────────────────────────────
            # Wire at wx = bus_a.x; breaker centred at mid_y.
            mid_y   = (ay + by2) / 2.0
            wx      = ax
            brk_top = mid_y - SYM / 2.0
            brk_bot = mid_y + SYM / 2.0

            _ln(svg, wx, ay, wx, brk_top)
            _dot(svg, wx, ay, NODE_R)
            _dot(svg, wx, brk_top, NODE_R)
            _sym(svg, "breaker", wx, brk_top)
            bbox_list.append((2, wx - SYM / 2, brk_top, SYM, SYM))
            # Rotated label to the right of the breaker body
            _lbl_rotated(svg, brk_lbl,
                         wx + SYM / 2 + RING_VT_LBL_X_OFF,
                         mid_y + RING_VT_LBL_Y_OFF,
                         anchor="middle")
            _dot(svg, wx, brk_bot, NODE_R)
            if abs(wx - bx2) > 0.5:
                _ln(svg, wx, brk_bot, wx, by2)
                _ln(svg, wx, by2, bx2, by2)
            else:
                _ln(svg, wx, brk_bot, bx2, by2)
            _dot(svg, bx2, by2, NODE_R)

        else:
            # ── Horizontal ring segment ───────────────────────────────────────
            # Wire at wy = bus_a.y; breaker rotated CCW, centred at mid_x.
            sign      = 1.0 if dx > 0 else -1.0
            mid_x     = (ax + bx2) / 2.0
            wy        = ay
            tap_a_x   = ax  + sign * BUS_HALF_W
            tap_b_x   = bx2 - sign * BUS_HALF_W
            brk_left  = mid_x - SYM / 2.0
            brk_right = mid_x + SYM / 2.0

            _ln(svg, tap_a_x, wy, brk_left, wy)
            _dot(svg, tap_a_x, wy, NODE_R)
            _dot(svg, brk_left, wy, NODE_R)
            _sym(svg, "breaker", mid_x, wy - SYM / 2, rotate_deg=-90)
            bbox_list.append((2, brk_left, wy - SYM / 2, SYM, SYM))
            # Horizontal label centred below the rotated breaker
            _lbl(svg, brk_lbl,
                 mid_x, wy + SYM / 2 + RING_HZ_LBL_Y_OFF,
                 anchor="middle")
            _dot(svg, brk_right, wy, NODE_R)
            if abs(wy - by2) > 0.5:
                _ln(svg, brk_right, wy, tap_b_x, wy)
                _ln(svg, tap_b_x, wy, tap_b_x, by2)
            else:
                _ln(svg, brk_right, wy, tap_b_x, by2)
            _dot(svg, tap_b_x, by2, NODE_R)

    # ── Draw external legs: BUS ── BRK ── ISO ── terminal stub ───────────────
    for bus_id, legs in ext_legs_by_bus.items():
        bcx = bus_pos[bus_id][0] + ox
        bcy = bus_pos[bus_id][1] + oy
        leg_xs = _assign_leg_xs(bus_id, len(legs))
        single = (len(legs) == 1)

        for leg_idx, (leg, lx) in enumerate(zip(legs, leg_xs)):
            term_id = leg["terminal"]
            goes_up = leg["goes_up"]
            brk_lbl = dev_label(leg["brk"])
            iso_lbl = dev_label(leg["iso"])

            if single:
                lx = bcx

            # ── Bay label side: left leg → label on left; right leg → label on right ──
            # For a single leg (centred on bus): always label on the right.
            # For multiple legs: leftmost (lx < bcx) gets label on the left,
            # rightmost and any others (lx >= bcx) get label on the right.
            lbl_on_left = (not single) and (lx < bcx)
            if lbl_on_left:
                lbl_anchor = "end"
                lbl_x_fn = lambda _lx=lx: _lx - SYM / 2 - LBL_X_OFF
            else:
                lbl_anchor = "start"
                lbl_x_fn = lambda _lx=lx: _lx + SYM / 2 + LBL_X_OFF

            # ── Horizontal bay label helper ───────────────────────────────────
            # text-anchor driven by lbl_anchor; x from lbl_x_fn;
            # y baseline at sym vertical centre + LBL_Y_OFF.
            def _bay_lbl(text, sym_top):
                _lbl(svg, text,
                     lbl_x_fn(),
                     sym_top + SYM / 2 + LBL_Y_OFF,
                     anchor=lbl_anchor)

            if not goes_up:
                # ── Downward leg ──────────────────────────────────────────────
                # Stack below bus: BRK (upright) → ISO (upright) → terminal stub.
                brk_top  = bcy + WIRE_GAP
                iso_top  = brk_top + SYM + WIRE_GAP
                term_top = iso_top  + SYM + WIRE_GAP    # wire_end_y for _cap

                if single:
                    _dot(svg, lx, bcy, NODE_R)
                    _ln(svg, lx, bcy, lx, brk_top)
                else:
                    _dot(svg, bcx, bcy, NODE_R)
                    if abs(bcx - lx) > 0.5:
                        _ln(svg, bcx, bcy, lx, bcy)
                    _dot(svg, lx, bcy, NODE_R)
                    _ln(svg, lx, bcy, lx, brk_top)

                _sym(svg, "breaker", lx, brk_top)
                bbox_list.append((2, lx - SYM / 2, brk_top, SYM, SYM))
                _bay_lbl(brk_lbl, brk_top)
                _dot(svg, lx, brk_top, NODE_R)
                _dot(svg, lx, brk_top + SYM, NODE_R)
                _ln(svg, lx, brk_top + SYM, lx, iso_top)
                _sym(svg, "isolator", lx, iso_top)
                bbox_list.append((5, lx - SYM / 2, iso_top, SYM, SYM))
                _bay_lbl(iso_lbl, iso_top)
                _dot(svg, lx, iso_top + SYM, NODE_R)
                _ln(svg, lx, iso_top + SYM, lx, term_top)

                end_pts[term_id] = dict(
                    mid_x=lx, wire_end_y=term_top, goes_up=False,
                    sk=_sym_key(devices[term_id]["type"]),
                    dev_type=devices[term_id]["type"],
                    lbl_on_left=lbl_on_left,
                )

            else:
                # ── Upward leg ────────────────────────────────────────────────
                # Stack above bus: BRK (upright) → ISO (flipped 180°) → terminal stub.
                # Breaker body sits entirely above the bus; entry at brk_bot, exit at brk_top.
                # Isolator is flipped so its entry faces downward.
                brk_bot  = bcy - WIRE_GAP           # breaker bottom (entry)
                brk_top  = brk_bot - SYM             # breaker top  (exit)
                iso_bot  = brk_top - WIRE_GAP        # isolator bottom (entry, flipped sym)
                iso_top  = iso_bot  - SYM             # isolator top  (exit)
                term_bot = iso_top  - WIRE_GAP        # wire_end_y for _cap

                if single:
                    _dot(svg, lx, bcy, NODE_R)
                    _ln(svg, lx, bcy, lx, brk_bot)
                else:
                    _dot(svg, bcx, bcy, NODE_R)
                    if abs(bcx - lx) > 0.5:
                        _ln(svg, bcx, bcy, lx, bcy)
                    _dot(svg, lx, bcy, NODE_R)
                    _ln(svg, lx, bcy, lx, brk_bot)

                _sym(svg, "breaker", lx, brk_top)
                bbox_list.append((2, lx - SYM / 2, brk_top, SYM, SYM))
                _bay_lbl(brk_lbl, brk_top)
                _dot(svg, lx, brk_top, NODE_R)
                _dot(svg, lx, brk_bot, NODE_R)
                _ln(svg, lx, brk_top, lx, iso_bot)
                _sym(svg, "isolator", lx, iso_top, rotate_deg=180)
                bbox_list.append((5, lx - SYM / 2, iso_top, SYM, SYM))
                _bay_lbl(iso_lbl, iso_top)
                _dot(svg, lx, iso_bot, NODE_R)
                _dot(svg, lx, iso_top, NODE_R)
                _ln(svg, lx, iso_top, lx, term_bot)

                end_pts[term_id] = dict(
                    mid_x=lx, wire_end_y=term_bot, goes_up=True,
                    sk=_sym_key(devices[term_id]["type"]),
                    dev_type=devices[term_id]["type"],
                    lbl_on_left=lbl_on_left,
                )

    return dict(
        svg=svg, bbox_list=bbox_list, end_pts=end_pts,
        CANVAS_W=CANVAS_W, CANVAS_H=CANVAS_H,
        devices=devices, dev_label=dev_label,
        SYM=SYM, FONT_W=FONT_W,
        LBL_X_OFF=LBL_X_OFF,
        LBL_Y_OFF=LBL_Y_OFF,
    )


# ── LABELLER ──────────────────────────────────────────────────────────────────

def _place_labels(svg, label_reqs):
    """Nudge labels vertically to avoid overlap within the same x-column,
    then emit them as SVG text elements."""
    by_x: dict = defaultdict(list)
    for i, (x, y, text, anchor) in enumerate(label_reqs):
        by_x[round(x)].append((y, i))

    nudged  = list(label_reqs)
    MIN_DY  = 8.0
    for col_items in by_x.values():
        col_items.sort()
        for j in range(1, len(col_items)):
            prev_y, prev_i = col_items[j - 1]
            cur_y,  cur_i  = col_items[j]
            if cur_y - prev_y < MIN_DY:
                new_y = prev_y + MIN_DY
                x, _, text, anchor = nudged[cur_i]
                nudged[cur_i]  = (x, new_y, text, anchor)
                col_items[j]   = (new_y, cur_i)

    for (x, y, text, anchor) in nudged:
        _lbl(svg, text, x, y, anchor)


# ── CAPPER ─────────────────────────────────────────────────────────────────────

def _cap(core: dict, mode: str, y_offset: float, H: float) -> dict:
    """
    Stamps terminal symbols and labels at every exposed wire end.

    standalone : all terminals get symbol + label.
    hybrid     : Feeder/GridSource → symbol + label.
                 Transformer       → wire extended to the SVG boundary,
                                     no symbol.

    y_offset / H encode the SVG coordinate space:
      top boundary in raw coords    = -y_offset
      bottom boundary in raw coords = -y_offset + H
    For mesh (coords baked in, no <g transform>): y_offset=0, H=CANVAS_H.

    Returns { device_id: {"label", "x", "y"} } for transformers in hybrid mode.
    """
    svg            = core["svg"]
    SYM            = core["SYM"]
    end_pts        = core["end_pts"]
    dev_label      = core["dev_label"]
    LBL_X_OFF      = core["LBL_X_OFF"]
    LBL_Y_OFF      = core["LBL_Y_OFF"]
    bbox_list      = core["bbox_list"]

    # YOLO class ids — must match canonical mapping:
    #   0=gridsource  1=transformer  2=breaker  3=load  4=capacitor  5=isolator  6=bus
    # YOLO class ids — must match spec exactly:
    # 0=gridsource  1=transformer  2=breaker  3=feeder  4=capacitor  5=isolator  6=bus
    TERM_CLASS = {
        "GridSource":  0,
        "Transformer": 1,
        "Feeder":      3,
        "Load":        3,   # Load treated as feeder class
    }

    label_reqs   = []
    boundary_pts = {}

    for et, ep in end_pts.items():
        mid_x      = ep["mid_x"]
        wire_end_y = ep["wire_end_y"]
        sk         = ep["sk"]
        goes_up    = ep["goes_up"]
        dev_type   = ep["dev_type"]

        if mode == "hybrid" and dev_type == "Transformer":
            boundary_y = -y_offset if goes_up else -y_offset + H
            _ln(svg, mid_x, wire_end_y, mid_x, boundary_y)
            boundary_pts[et] = {
                "label": dev_label(et),
                "x":     round(mid_x, 1),
                "y":     round(boundary_y + y_offset, 1),
            }
        else:
            cls = TERM_CLASS.get(dev_type, 3)
            sym_top_y = (wire_end_y - SYM) if goes_up else wire_end_y
            _sym(svg, sk, mid_x, sym_top_y)
            bbox_list.append((cls, mid_x - SYM / 2, sym_top_y, SYM, SYM))
            lbl_on_left = ep.get("lbl_on_left", False)
            if lbl_on_left:
                lx = mid_x - SYM / 2 - LBL_X_OFF
                anchor = "end"
            else:
                lx = mid_x + SYM / 2 + LBL_X_OFF
                anchor = "start"
            ly = sym_top_y + SYM / 2 + LBL_Y_OFF
            label_reqs.append((lx, ly, dev_label(et), anchor))

    _place_labels(svg, label_reqs)
    return boundary_pts


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def render_mesh_svg(
    topology_path: str,
    output_path: str,
    mode: str = "standalone",   # "standalone" | "hybrid"
) -> dict | None:

    import json

    with open(topology_path) as f:
        topology_json = json.load(f)

    layout = topology_json.get("layout", {})
    topo_type = layout.get("topology_type") or layout.get("mode", "")

    if topo_type != "MESH":
        raise ValueError(f"Expected MESH, got '{topo_type}'")

    core = _render_core(topology_json)

    svg      = core["svg"]
    CANVAS_W = core["CANVAS_W"]
    CANVAS_H = core["CANVAS_H"]

    boundary_pts = _cap(core, mode, y_offset=0, H=CANVAS_H)

    # ── YOLO bounding-box export ──────────────────────────────────────────────
    yolo_lines = []
    for (cls, x_left, y_top, w, h) in core["bbox_list"]:
        xc = (x_left + w / 2) / CANVAS_W
        yc = (y_top  + h / 2) / CANVAS_H
        nw = w / CANVAS_W
        nh = h / CANVAS_H
        yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    # ── Assemble SVG ─────────────────────────────────────────────────────────
    full_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {CANVAS_W:.0f} {CANVAS_H:.0f}" '
        f'width="{CANVAS_W:.0f}" height="{CANVAS_H:.0f}" '
        f'style="background:#fff">\n'
        + '\n'.join(svg) + '\n</svg>'
    )

    if mode == "hybrid":
        return boundary_pts, full_svg, yolo_lines
    else:   
        with open(output_path, "w") as fh:
            fh.write(full_svg)
        
        yolo_path = output_path.rsplit(".", 1)[0] + ".txt"
        with open(yolo_path, "w") as fh:
            fh.write('\n'.join(yolo_lines))

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys

    if len(sys.argv) < 3:
        print("Usage: python mesh_renderer.py <topology.json> <output.svg> [standalone|hybrid]")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        topo = json.load(fh)
    m = sys.argv[3] if len(sys.argv) > 3 else "standalone"
    render_mesh_svg(topo, sys.argv[2], mode=m)
    print(f"Written: {sys.argv[2]}  ({m})")