"""
double_bus_renderer.py
======================
Three-function renderer for DOUBLE_BUS topology.

  _render_core(...)     → draws buses, bay stacks (isolators, breakers, wires,
                          junction dots). Returns layout geometry + svg list.

  _cap(core, mode)      → stamps terminal symbols + labels at every exposed
                          wire end.
                          standalone: all terminals get symbol + label.
                          hybrid:     Feeder/GridSource → symbol + label.
                                      Transformer       → wire to SVG boundary,
                                                          no symbol.

  render_double_bus_svg(topology_json, output_path, mode="standalone")
                        → orchestrates the above, writes SVG file.
                          Returns transformer boundary points in hybrid mode.
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

def _sym(svg, name, cx, cy_top, size=32.0, rotate180=False):
    scale = size / 32.0
    if rotate180:
        svg.append(
            f'<g transform="translate({cx - size/2:.1f},{cy_top:.1f}) scale({scale:.4f})">'
            f'<g transform="rotate(180,16,16)">'
            + IEC_SYMBOLS[name] + '</g></g>'
        )
    else:
        svg.append(
            f'<g transform="translate({cx - size/2:.1f},{cy_top:.1f}) scale({scale:.4f})">'
            + IEC_SYMBOLS[name] + '</g>'
        )

def _lbl(svg, text, x, y, anchor="start", size=7):
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="monospace" font-size="{size}" fill="#000">{text}</text>'
    )

def _lbl_rotated(svg, text, x, y, anchor="middle", size=7):
    """Label rotated +90° around (x,y) — runs top-to-bottom along a vertical wire, placed to the right."""
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="monospace" font-size="{size}" fill="#000" '
        f'transform="rotate(90,{x:.1f},{y:.1f})">{text}</text>'
    )

def _bus_line(svg, x1, y, x2, w=2.0):
    svg.append(
        f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
        f'stroke="#000" stroke-width="{w}" stroke-linecap="square"/>'
    )


# ── CORE ───────────────────────────────────────────────────────────────────────

def _render_core(topology_json: dict) -> dict:
    """
    Parse topology, compute geometry, draw buses + bay stacks.
    Returns a geometry/state dict consumed by _cap.
    """
    # ── Constants ───────────────────────────────────────────────────────────
    MARGIN_X      = 30.0   # left/right canvas margin
    MARGIN_Y      = 0.0   # top/bottom canvas margin
    BUS_LBL_PAD   = 20.0   # how far bus extends left of BUS_X1 for label stub
    COL_W         = 90.0   # horizontal pitch per logical column
    SYM           = 32.0   # symbol size px
    BUS_GAP       = 12.0   # vertical gap between Bus A and Bus B
    ISO_Y_OFF     = 20.0   # wire from bus tap to nearest iso terminal
    ISO_JCT_GAP   = 30.0   # gap between iso edge and junction/U-bar
    U_W           = 50.0   # x-distance between the two isolators in a bay
    COL_X_OFF     = 0.0    # x offset of UP bay leftmost iso from column centre
    STAGGER       = 5.0    # extra x offset of DOWN bay leftmost iso vs UP
    BRK_GAP       = 10.0   # gap between junction and breaker edge
    TERM_BEYOND   = 10.0   # wire length from junction/brk to terminal wire end
    BUS_SPINE_W   = 2.0
    NODE_R        = 0.0
    FONT_W        = 4.3    # px per char at font-size 7 monospace
    LBL_X_PAD     = -8.0   # horizontal gap between symbol edge and rotated label
    LBL_Y_PAD     = 0.0    # vertical offset of label from symbol centre

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

    bay_ids = sorted(bays_raw.keys())

    bus_ids  = sorted(d["id"] for d in devices.values() if d["type"] == "Bus")
    bus_a_id = bus_ids[0]   # top bus
    bus_b_id = bus_ids[1]   # bottom bus

    breakers_per_bay = topology_json.get("layout", {}).get("breakersPerBay", 1)
    exposed_set = set(topology_json.get("layout", {}).get("terminals", []))

    # ── Per-bay topology ─────────────────────────────────────────────────────
    bay_info = {}
    for bay_id in bay_ids:
        dev_ids   = set(bays_raw[bay_id])
        isolators = [d for d in dev_ids if devices[d]["type"] == "Isolator"]
        breakers  = [d for d in dev_ids if devices[d]["type"] == "Breaker"]

        iso_a = iso_b = None
        for iso in isolators:
            bus_nbs = [n for n in adj[iso] if devices[n]["type"] == "Bus"]
            if bus_a_id in bus_nbs:   iso_a = iso
            elif bus_b_id in bus_nbs: iso_b = iso

        dual = (breakers_per_bay == 2)
        brk_a = brk_b = brk = None
        if not dual:
            brk = breakers[0] if breakers else None
        else:
            for b in breakers:
                if iso_a and iso_a in adj[b]: brk_a = b
                if iso_b and iso_b in adj[b]: brk_b = b

        terminal = None
        for b in ([brk] if brk else [b for b in [brk_a, brk_b] if b]):
            for nb in adj[b]:
                if nb in exposed_set:
                    terminal = nb
                    break
            if terminal:
                break

        goes_up = _goes_up(terminal, devices[terminal]["type"]) if terminal else False

        bay_info[bay_id] = dict(
            iso_a=iso_a, iso_b=iso_b,
            brk=brk, brk_a=brk_a, brk_b=brk_b,
            dual=dual, terminal=terminal, goes_up=goes_up,
        )

    # ── Column assignment ────────────────────────────────────────────────────
    # Incomers bookend: incomer[0]+feeder[0] → col 0,
    #                   incomer[1]+feeder[-1] → last col,
    #                   remaining feeders fill cols 1…n-2.
    up_bays   = [b for b in bay_ids if bay_info[b]["goes_up"]]
    down_bays = [b for b in bay_ids if not bay_info[b]["goes_up"]]

    columns = []

    max_len = max(len(up_bays), len(down_bays))

    for i in range(max_len):
        up = up_bays[i] if i < len(up_bays) else None
        dn = down_bays[i] if i < len(down_bays) else None
        columns.append((up, dn))

    n_cols_actual = len(columns)

    def cx_of(col_idx):
        return MARGIN_X + col_idx * COL_W + COL_W / 2

    # ── Vertical geometry ────────────────────────────────────────────────────
    any_dual = any(bay_info[b]["dual"] for b in bay_ids)
    if any_dual:
        # iso → ISO_JCT_GAP → brk(SYM) → BRK_GAP → jct → TERM_BEYOND → wire_end
        UP_REACH   = ISO_Y_OFF + SYM + ISO_JCT_GAP + SYM + BRK_GAP + TERM_BEYOND
        DOWN_REACH = ISO_Y_OFF + SYM + ISO_JCT_GAP + SYM + BRK_GAP + TERM_BEYOND
    else:
        # iso → ISO_JCT_GAP → jct → BRK_GAP → brk(SYM) → TERM_BEYOND → wire_end
        UP_REACH   = ISO_Y_OFF + SYM + ISO_JCT_GAP + BRK_GAP + SYM + TERM_BEYOND
        DOWN_REACH = ISO_Y_OFF + SYM + ISO_JCT_GAP + BRK_GAP + SYM + TERM_BEYOND

    BUS_A_Y = UP_REACH
    BUS_B_Y = BUS_A_Y + BUS_GAP

    # ── Canvas X bounds ──────────────────────────────────────────────────────
    left_edge  = cx_of(0) + COL_X_OFF - 8.0
    right_edge = cx_of(n_cols_actual - 1) + COL_X_OFF + STAGGER + U_W + 8.0
    BUS_X1     = left_edge
    BUS_X2     = right_edge
    CANVAS_X1  = BUS_X1 - BUS_LBL_PAD - MARGIN_X   # canvas left includes stub + margin

    # ── Wire endpoints (where terminal stubs end; capper stamps symbols here) ─
    # end_pts: terminal_id → dict of geometry + metadata
    end_pts = {}
    for col_idx, (up_bay, dn_bay) in enumerate(columns):
        cx = cx_of(col_idx)
        for bay_id, goes_up in [(up_bay, True), (dn_bay, False)]:
            if bay_id is None:
                continue
            info = bay_info[bay_id]
            if info["terminal"] is None:
                continue
            et   = info["terminal"]
            d    = devices[et]
            dual = info["dual"]

            if goes_up:
                x_a   = cx + COL_X_OFF
                x_b   = cx + COL_X_OFF + U_W
                mid_x = (x_a + x_b) / 2
                if dual:
                    jct_y      = BUS_A_Y - ISO_Y_OFF - SYM - ISO_JCT_GAP - SYM - BRK_GAP
                    wire_end_y = jct_y - TERM_BEYOND
                else:
                    jct_y      = BUS_A_Y - ISO_Y_OFF - SYM - ISO_JCT_GAP
                    wire_end_y = jct_y - BRK_GAP - SYM - TERM_BEYOND
            else:
                x_a   = cx + COL_X_OFF + STAGGER
                x_b   = cx + COL_X_OFF + STAGGER + U_W
                mid_x = (x_a + x_b) / 2
                if dual:
                    jct_y      = BUS_B_Y + ISO_Y_OFF + SYM + ISO_JCT_GAP + SYM + BRK_GAP
                    wire_end_y = jct_y + TERM_BEYOND
                else:
                    jct_y      = BUS_B_Y + ISO_Y_OFF + SYM + ISO_JCT_GAP
                    wire_end_y = jct_y + BRK_GAP + SYM + TERM_BEYOND

            end_pts[et] = dict(
                rx=mid_x, wire_end_y=wire_end_y,
                jct_y=jct_y, goes_up=goes_up,
                sk=_sym_key(d["type"]), dev_type=d["type"],
                bay_id=bay_id, col_idx=col_idx,
                x_a=x_a, x_b=x_b, mid_x=mid_x, dual=dual,
            )

    # ── Draw core SVG elements ────────────────────────────────────────────────
    svg = []

    # bbox_list: raw (pre-transform) symbol bounding boxes for YOLO export
    # Each entry: (class_id, x_left, y_top, width, height)
    # class_id: 0=gridsource, 1=transformer, 2=breaker, 3=load/feeder, 5=isolator, 6=bus
    bbox_list = []

    # Bus spines — extended left by BUS_LBL_PAD to provide a stub for the label
    _bus_line(svg, BUS_X1 - BUS_LBL_PAD, BUS_A_Y, BUS_X2, BUS_SPINE_W)
    _bus_line(svg, BUS_X1 - BUS_LBL_PAD, BUS_B_Y, BUS_X2, BUS_SPINE_W)
    # Bus bboxes: treat full spine as bounding box, height = BUS_SPINE_W
    bus_w = BUS_X2 - (BUS_X1 - BUS_LBL_PAD)
    bbox_list.append((6, BUS_X1 - BUS_LBL_PAD, BUS_A_Y - BUS_SPINE_W / 2, bus_w, BUS_SPINE_W))
    bbox_list.append((6, BUS_X1 - BUS_LBL_PAD, BUS_B_Y - BUS_SPINE_W / 2, bus_w, BUS_SPINE_W))

    # Bus labels: Bus A above its spine, Bus B below its spine
    _lbl(svg, dev_label(bus_a_id), BUS_X1 - 2, BUS_A_Y - 3, anchor="end")
    _lbl(svg, dev_label(bus_b_id), BUS_X1 - 2, BUS_B_Y + 9, anchor="end")

    # Bay stacks
    for col_idx, (up_bay, dn_bay) in enumerate(columns):
        cx = cx_of(col_idx)

        for bay_id, goes_up in [(up_bay, True), (dn_bay, False)]:
            if bay_id is None:
                continue
            info = bay_info[bay_id]
            dual = info["dual"]

            if goes_up:
                x_a   = cx + COL_X_OFF
                x_b   = cx + COL_X_OFF + U_W
                mid_x = (x_a + x_b) / 2

                iso_top_y = BUS_A_Y - ISO_Y_OFF - SYM
                iso_bot_y = iso_top_y + SYM

                # Bus taps → wires up to iso bottoms
                _dot(svg, x_a, BUS_A_Y, NODE_R)
                _ln(svg, x_a, BUS_A_Y, x_a, iso_bot_y)
                _dot(svg, x_b, BUS_B_Y, NODE_R)
                _ln(svg, x_b, BUS_B_Y, x_b, iso_bot_y)

                _sym(svg, "isolator", x_a, iso_top_y)
                _sym(svg, "isolator", x_b, iso_top_y)
                bbox_list.append((5, x_a - SYM/2, iso_top_y, SYM, SYM))
                bbox_list.append((5, x_b - SYM/2, iso_top_y, SYM, SYM))
                # Isolator terminal dots (top = bus side y=0, bottom = junction side y=32)
                _dot(svg, x_a, iso_top_y, NODE_R)
                _dot(svg, x_a, iso_bot_y, NODE_R)
                _dot(svg, x_b, iso_top_y, NODE_R)
                _dot(svg, x_b, iso_bot_y, NODE_R)
                if info["iso_a"]:
                    _lbl_rotated(svg, dev_label(info["iso_a"]), x_a + SYM / 2 + LBL_X_PAD, iso_top_y + SYM / 2 + LBL_Y_PAD, anchor="middle")
                if info["iso_b"]:
                    _lbl_rotated(svg, dev_label(info["iso_b"]), x_b + SYM / 2 + LBL_X_PAD, iso_top_y + SYM / 2 + LBL_Y_PAD, anchor="middle")

                if dual:
                    brk_bot_y = iso_top_y - ISO_JCT_GAP
                    brk_top_y = brk_bot_y - SYM
                    jct_y     = brk_top_y - BRK_GAP

                    _ln(svg, x_a, iso_top_y, x_a, brk_bot_y)
                    _ln(svg, x_b, iso_top_y, x_b, brk_bot_y)
                    _dot(svg, x_a, brk_bot_y, NODE_R)
                    _dot(svg, x_b, brk_bot_y, NODE_R)
                    _sym(svg, "breaker", x_a, brk_top_y)
                    _sym(svg, "breaker", x_b, brk_top_y)
                    bbox_list.append((2, x_a - SYM/2, brk_top_y, SYM, SYM))
                    bbox_list.append((2, x_b - SYM/2, brk_top_y, SYM, SYM))
                    _dot(svg, x_a, brk_top_y, NODE_R)
                    _dot(svg, x_b, brk_top_y, NODE_R)
                    _ln(svg, x_a, brk_top_y, x_a, jct_y)
                    _ln(svg, x_b, brk_top_y, x_b, jct_y)
                    _ln(svg, x_a, jct_y, x_b, jct_y)
                    _dot(svg, mid_x, jct_y, NODE_R)
                    # Breaker labels: rotated 90°, tight to the right of each symbol
                    if info["brk_a"]:
                        _lbl_rotated(svg, dev_label(info["brk_a"]), x_a + SYM / 2 + LBL_X_PAD, brk_top_y + SYM / 2 + LBL_Y_PAD, anchor="middle")
                    if info["brk_b"]:
                        _lbl_rotated(svg, dev_label(info["brk_b"]), x_b + SYM / 2 + LBL_X_PAD, brk_top_y + SYM / 2 + LBL_Y_PAD, anchor="middle")
                    # wire from junction to wire_end (capper stamps symbol there)
                    wire_end_y = jct_y - TERM_BEYOND
                    _ln(svg, mid_x, jct_y, mid_x, wire_end_y)
                else:
                    jct_y = iso_top_y - ISO_JCT_GAP
                    _ln(svg, x_a, iso_top_y, x_a, jct_y)
                    _ln(svg, x_b, iso_top_y, x_b, jct_y)
                    _ln(svg, x_a, jct_y, x_b, jct_y)
                    _dot(svg, mid_x, jct_y, NODE_R)

                    brk_top = jct_y - BRK_GAP - SYM
                    _ln(svg, mid_x, jct_y, mid_x, brk_top + SYM)
                    _dot(svg, mid_x, brk_top + SYM, NODE_R)
                    _sym(svg, "breaker", mid_x, brk_top)
                    bbox_list.append((2, mid_x - SYM/2, brk_top, SYM, SYM))
                    _dot(svg, mid_x, brk_top, NODE_R)
                    # Breaker label: rotated 90°, tight to the right of symbol
                    if info["brk"]:
                        _lbl_rotated(svg, dev_label(info["brk"]), mid_x + SYM / 2 + LBL_X_PAD, brk_top + SYM / 2 + LBL_Y_PAD, anchor="middle")
                    wire_end_y = brk_top - TERM_BEYOND
                    _ln(svg, mid_x, brk_top, mid_x, wire_end_y)

            else:
                x_a   = cx + COL_X_OFF + STAGGER
                x_b   = cx + COL_X_OFF + STAGGER + U_W
                mid_x = (x_a + x_b) / 2

                iso_top_y = BUS_B_Y + ISO_Y_OFF
                iso_bot_y = iso_top_y + SYM

                # Bus taps → wires down to iso tops
                _dot(svg, x_a, BUS_A_Y, NODE_R)
                _ln(svg, x_a, BUS_A_Y, x_a, iso_top_y)
                _dot(svg, x_b, BUS_B_Y, NODE_R)
                _ln(svg, x_b, BUS_B_Y, x_b, iso_top_y)

                _sym(svg, "isolator", x_a, iso_top_y, rotate180=True)
                _sym(svg, "isolator", x_b, iso_top_y, rotate180=True)
                bbox_list.append((5, x_a - SYM/2, iso_top_y, SYM, SYM))
                bbox_list.append((5, x_b - SYM/2, iso_top_y, SYM, SYM))
                # Isolator terminal dots (top = bus side, bottom = junction side)
                _dot(svg, x_a, iso_top_y, NODE_R)
                _dot(svg, x_a, iso_bot_y, NODE_R)
                _dot(svg, x_b, iso_top_y, NODE_R)
                _dot(svg, x_b, iso_bot_y, NODE_R)
                if info["iso_a"]:
                    _lbl_rotated(svg, dev_label(info["iso_a"]), x_a + SYM / 2 + LBL_X_PAD, iso_top_y + SYM / 2 + LBL_Y_PAD, anchor="middle")
                if info["iso_b"]:
                    _lbl_rotated(svg, dev_label(info["iso_b"]), x_b + SYM / 2 + LBL_X_PAD, iso_top_y + SYM / 2 + LBL_Y_PAD, anchor="middle")

                if dual:
                    brk_top_y = iso_bot_y + ISO_JCT_GAP
                    brk_bot_y = brk_top_y + SYM
                    jct_y     = brk_bot_y + BRK_GAP

                    _ln(svg, x_a, iso_bot_y, x_a, brk_top_y)
                    _ln(svg, x_b, iso_bot_y, x_b, brk_top_y)
                    _dot(svg, x_a, brk_top_y, NODE_R)
                    _dot(svg, x_b, brk_top_y, NODE_R)
                    _sym(svg, "breaker", x_a, brk_top_y)
                    _sym(svg, "breaker", x_b, brk_top_y)
                    bbox_list.append((2, x_a - SYM/2, brk_top_y, SYM, SYM))
                    bbox_list.append((2, x_b - SYM/2, brk_top_y, SYM, SYM))
                    _dot(svg, x_a, brk_bot_y, NODE_R)
                    _dot(svg, x_b, brk_bot_y, NODE_R)
                    _ln(svg, x_a, brk_bot_y, x_a, jct_y)
                    _ln(svg, x_b, brk_bot_y, x_b, jct_y)
                    _ln(svg, x_a, jct_y, x_b, jct_y)
                    _dot(svg, mid_x, jct_y, NODE_R)
                    # Breaker labels: rotated 90°, tight to the right of each symbol
                    if info["brk_a"]:
                        _lbl_rotated(svg, dev_label(info["brk_a"]), x_a + SYM / 2 + LBL_X_PAD, brk_top_y + SYM / 2 + LBL_Y_PAD, anchor="middle")
                    if info["brk_b"]:
                        _lbl_rotated(svg, dev_label(info["brk_b"]), x_b + SYM / 2 + LBL_X_PAD, brk_top_y + SYM / 2 + LBL_Y_PAD, anchor="middle")
                    wire_end_y = jct_y + TERM_BEYOND
                    _ln(svg, mid_x, jct_y, mid_x, wire_end_y)
                else:
                    jct_y = iso_bot_y + ISO_JCT_GAP
                    _ln(svg, x_a, iso_bot_y, x_a, jct_y)
                    _ln(svg, x_b, iso_bot_y, x_b, jct_y)
                    _ln(svg, x_a, jct_y, x_b, jct_y)
                    _dot(svg, mid_x, jct_y, NODE_R)

                    brk_top = jct_y + BRK_GAP
                    _ln(svg, mid_x, jct_y, mid_x, brk_top)
                    _dot(svg, mid_x, brk_top, NODE_R)
                    _sym(svg, "breaker", mid_x, brk_top)
                    bbox_list.append((2, mid_x - SYM/2, brk_top, SYM, SYM))
                    _dot(svg, mid_x, brk_top + SYM, NODE_R)
                    # Breaker label: rotated 90°, tight to the right of symbol
                    if info["brk"]:
                        _lbl_rotated(svg, dev_label(info["brk"]), mid_x + SYM / 2 + LBL_X_PAD, brk_top + SYM / 2 + LBL_Y_PAD, anchor="middle")
                    wire_end_y = brk_top + SYM + TERM_BEYOND
                    _ln(svg, mid_x, brk_top + SYM, mid_x, wire_end_y)

    return dict(
        svg=svg,
        BUS_A_Y=BUS_A_Y, BUS_B_Y=BUS_B_Y,
        BUS_X1=BUS_X1, BUS_X2=BUS_X2,
        CANVAS_X1=CANVAS_X1,
        end_pts=end_pts,
        devices=devices, dev_label=dev_label,
        SYM=SYM, FONT_W=FONT_W, MARGIN_X=MARGIN_X, MARGIN_Y=MARGIN_Y,
        LBL_X_PAD=LBL_X_PAD, LBL_Y_PAD=LBL_Y_PAD,
        bbox_list=bbox_list,
    )


# ── LABELLER ──────────────────────────────────────────────────────────────────

def _place_labels(svg, label_reqs):
    by_x = defaultdict(list)
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

    standalone: all terminals get symbol + label.
    hybrid:     Feeder/GridSource → symbol + label.
                Transformer       → wire extended to SVG boundary, no symbol.

    Returns transformer boundary points (hybrid mode only):
      { device_id: {"label": str, "x": float, "y": float} }
    """
    svg       = core["svg"]
    SYM       = core["SYM"]
    FONT_W    = core["FONT_W"]
    MARGIN_Y  = core["MARGIN_Y"]
    CANVAS_X1 = core["CANVAS_X1"]
    end_pts   = core["end_pts"]
    dev_label = core["dev_label"]
    LBL_X_PAD = core["LBL_X_PAD"]
    LBL_Y_PAD = core["LBL_Y_PAD"]
    bbox_list  = core["bbox_list"]

    # class_id map for terminal device types
    TERM_CLASS = {"GridSource": 0, "Transformer": 1, "Feeder": 3}

    label_reqs   = []
    boundary_pts = {}

    for et, ep in end_pts.items():
        mid_x      = ep["mid_x"]
        wire_end_y = ep["wire_end_y"]
        sk         = ep["sk"]
        goes_up    = ep["goes_up"]
        dev_type   = ep["dev_type"]

        if mode == "hybrid" and dev_type == "Transformer":
            # Extend stub to the actual SVG boundary edge (y=0 or y=H in final coords)
            boundary_y = -y_offset if goes_up else -y_offset + H
            _ln(svg, mid_x, wire_end_y, mid_x, boundary_y)
            boundary_pts[et] = {
                "label": dev_label(et),
                "x": round(mid_x - CANVAS_X1, 1),   # apply canvas x-translation
                "y": round(boundary_y + y_offset, 1), # 0 (top edge) or H (bottom edge)
            }
        else:
            cls = TERM_CLASS.get(dev_type, 3)
            if goes_up:
                sym_top_y = wire_end_y - SYM
                _sym(svg, sk, mid_x, sym_top_y)
                bbox_list.append((cls, mid_x - SYM/2, sym_top_y, SYM, SYM))
                lx     = mid_x + SYM / 2 + LBL_X_PAD
                label_reqs.append((lx, sym_top_y + SYM / 2 + LBL_Y_PAD, dev_label(et), "start"))
            else:
                sym_top_y = wire_end_y
                _sym(svg, sk, mid_x, sym_top_y)
                bbox_list.append((cls, mid_x - SYM/2, sym_top_y, SYM, SYM))
                lx     = mid_x + SYM / 2 + LBL_X_PAD
                label_reqs.append((lx, sym_top_y + SYM / 2 + LBL_Y_PAD, dev_label(et), "start"))

    _place_labels(svg, label_reqs)
    return boundary_pts


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def render_double_bus_svg(
    topology_path: str,
    output_path: str,
    mode: str = "standalone",   # "standalone" | "hybrid"
) -> dict | None:

    import json

    with open(topology_path) as f:
        topology_json = json.load(f)

    layout = topology_json.get("layout", {})
    topo_type = layout.get("topology_type") or layout.get("mode", "")

    if topo_type != "DOUBLE_BUS":
        raise ValueError(f"Expected DOUBLE_BUS, got '{topo_type}'")

    core = _render_core(topology_json)

    svg    = core["svg"]
    SYM    = core["SYM"]
    MARGIN_X = core["MARGIN_X"]
    MARGIN_Y = core["MARGIN_Y"]
    BUS_A_Y, BUS_B_Y = core["BUS_A_Y"], core["BUS_B_Y"]
    end_pts = core["end_pts"]

    # Determine vertical canvas extent from wire endpoints + symbol heights
    up_ys   = [ep["wire_end_y"] - SYM for ep in end_pts.values() if ep["goes_up"]]
    down_ys = [ep["wire_end_y"] + SYM  for ep in end_pts.values() if not ep["goes_up"]]
    top_y   = min([BUS_A_Y] + up_ys)   - MARGIN_Y
    bot_y   = max([BUS_B_Y] + down_ys) + MARGIN_Y

    y_offset = -top_y
    BUS_X1, BUS_X2 = core["BUS_X1"], core["BUS_X2"]
    CANVAS_X1      = core["CANVAS_X1"]
    W = BUS_X2 - CANVAS_X1 + MARGIN_X
    H = bot_y - top_y

    boundary_pts = _cap(core, mode, y_offset, H)

    # Transform raw bbox coords → final SVG coords, then normalise to YOLO format
    # SVG transform is: translate(tx, ty) where tx = -CANVAS_X1, ty = y_offset
    tx = -CANVAS_X1
    ty = y_offset
    yolo_lines = []
    for (cls, x_left, y_top, w, h) in core["bbox_list"]:
        # Apply canvas transform
        fx = x_left + tx
        fy = y_top  + ty
        # YOLO: normalised centre
        xc = (fx + w / 2) / W
        yc = (fy + h / 2) / H
        nw = w / W
        nh = h / H
        yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    # Write .txt alongside the SVG

    full_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" '
        f'style="background:#fff">\n'
        f'<g transform="translate({-CANVAS_X1:.1f},{y_offset:.1f})">\n'
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
        print("Usage: python double_bus_renderer.py <topology.json> <output.svg> [standalone|hybrid]")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        topo = json.load(fh)
    m = sys.argv[3] if len(sys.argv) > 3 else "standalone"
    render_double_bus_svg(topo, sys.argv[2], mode=m)
    print(f"Written: {sys.argv[2]} ({m})")