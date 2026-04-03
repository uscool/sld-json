"""
single_bus_renderer.py
======================
Three-function renderer for SINGLE_BUS topology.

  _render_core(topology_json)
                        → parses topology, computes geometry, draws the bus
                          spine and all bay stacks (isolators, breakers, wires,
                          junction dots).  Returns a geometry/state dict
                          consumed by _cap.

  _cap(core, mode, y_offset, H)
                        → stamps terminal symbols + labels at every exposed
                          wire end.
                          standalone: all terminals get symbol + label.
                          hybrid:     Feeder/GridSource → symbol + label.
                                      Transformer       → wire to SVG boundary,
                                                          no symbol.

  render_single_bus_svg(topology_json, output_path, mode="standalone")
                        → orchestrates the above, writes SVG + YOLO .txt file.
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

def _sym_rotated_ccw(svg, name, cx, cy, size=32.0):
    """Draw a symbol rotated 90° CCW around its own centre."""
    scale = size / 32.0
    svg.append(
        f'<g transform="translate({cx:.1f},{cy:.1f}) scale({scale:.4f})">'
        f'<g transform="rotate(-90,0,0) translate({-16:.1f},{-16:.1f})">'
        + IEC_SYMBOLS[name] + '</g></g>'
    )

def _lbl(svg, text, x, y, anchor="start", size=7):
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="monospace" font-size="{size}" fill="#000">{text}</text>'
    )

def _lbl_rotated(svg, text, x, y, anchor="middle", size=7):
    """Label rotated +90° around (x, y) — runs top-to-bottom."""
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
    Parse topology, compute geometry, draw bus spine + bay stacks.
    Returns a geometry/state dict consumed by _cap.
    """
    # ── Constants ───────────────────────────────────────────────────────────
    MARGIN_X       = 50.0   # left/right canvas margin
    MARGIN_Y       = 0.0    # top/bottom margin (canvas trimmed by entry point)
    COL_W          = 80.0   # horizontal pitch per logical column
    SYM            = 32.0   # symbol size px
    BUS_SPINE_W    = 2.0
    NODE_R         = 0.0
    ISO_Y_OFF      = 0.0    # wire stub from bus tap to nearest iso terminal
    ISO_JCT_GAP    = 0.0    # gap between iso far edge and breaker near edge
    BRK_GAP        = 0.0    # gap between breaker far edge and terminal wire end
    COL_X_OFF      = 0.0    # x offset of infeed (UP) tap from column centre
    STAGGER        = 10.0  # additional x offset of outfeed (DOWN) tap vs infeed tap
    LBL_X_PAD      = -8.0   # horizontal gap: symbol edge → label (negative = tighter)
    LBL_Y_PAD      = 0.0    # baseline drop from symbol vertical midpoint
    BUS_OVERHANG_L = 40.0   # bus spine extends left  of leftmost tap
    BUS_OVERHANG_R = 30.0   # bus spine extends right of rightmost tap

    # ── Parse ────────────────────────────────────────────────────────────────
    devices = {d["id"]: d for d in topology_json["devices"]}

    def dev_label(dev_id):
        return devices[dev_id].get("label") or dev_id

    adj = defaultdict(set)
    for c in topology_json["connections"]:
        adj[c["from"]].add(c["to"])
        adj[c["to"]].add(c["from"])

    exposed_set = set(topology_json.get("layout", {}).get("terminals", []))

    # ── Identify main bus ─────────────────────────────────────────────────────
    bus_devs = [d for d in devices.values() if d["type"] == "Bus"]
    bus_id = next(
        (d["id"] for d in bus_devs if (d.get("bus_role") or "").upper() == "MAIN"),
        bus_devs[0]["id"] if bus_devs else None,
    )
    if bus_id is None:
        raise ValueError("No Bus device found in topology")

    # ── Trace each exposed terminal to its iso and brk ────────────────────────
    # Chain: terminal → Isolator → Breaker (or terminal → Breaker → Isolator).
    # Returns dict(iso=..., brk=..., terminal=...) or None.
    def trace_to_bus(terminal_id: str) -> dict | None:
        visited = {terminal_id}
        step1 = next(
            (n for n in adj[terminal_id]
             if n not in visited and devices[n]["type"] in ("Isolator", "Breaker")),
            None,
        )
        if step1 is None:
            return None
        visited.add(step1)
        step2 = next(
            (n for n in adj[step1]
             if n not in visited and devices[n]["type"] in ("Isolator", "Breaker")),
            None,
        )
        if step2 is None:
            return None
        iso = step1 if devices[step1]["type"] == "Isolator" else step2
        brk = step1 if devices[step1]["type"] == "Breaker"  else step2
        return dict(iso=iso, brk=brk, terminal=terminal_id)

    # ── Classify terminals into infeeds (up) and outfeeds (down) ──────────────
    infeeds:  list[dict] = []
    outfeeds: list[dict] = []
    for et in sorted(exposed_set):
        chain = trace_to_bus(et)
        if chain is None:
            continue
        if _goes_up(et, devices[et]["type"]):
            infeeds.append(chain)
        else:
            outfeeds.append(chain)

    infeeds.sort(key=lambda c: c["terminal"])

    # Outfeeds: transformers first (sorted by id), then all others (sorted by id)
    outfeeds_tr   = sorted(
        (c for c in outfeeds if devices[c["terminal"]]["type"] == "Transformer"),
        key=lambda c: c["terminal"],
    )
    outfeeds_rest = sorted(
        (c for c in outfeeds if devices[c["terminal"]]["type"] != "Transformer"),
        key=lambda c: c["terminal"],
    )
    outfeeds = outfeeds_tr + outfeeds_rest

    # ── Column pairing ────────────────────────────────────────────────────────
    # Outfeeds fill columns left-to-right: transformers first, then feeders.
    # Infeeds use a zigzag: col 0, col n-1, col 1, col n-2, col 2, …
    # so that grid sources bookend and then fill inward across the bus span.
    n_cols = max(len(infeeds), len(outfeeds), 1)

    slots_up: list = [None] * n_cols
    slots_dn: list = [None] * n_cols

    for i, chain in enumerate(infeeds):
        slots_up[i] = chain

    for i, chain in enumerate(outfeeds):
        slots_dn[i] = chain

    columns: list[tuple] = list(zip(slots_up, slots_dn))

    # ── Horizontal geometry ────────────────────────────────────────────────────
    def cx_of(col_idx: int) -> float:
        return MARGIN_X + col_idx * COL_W + COL_W / 2.0

    # Gather all tap x-coords to size the bus spine precisely
    tap_xs: list[float] = []
    for col_idx in range(n_cols):
        cx = cx_of(col_idx)
        if slots_up[col_idx] is not None:
            tap_xs.append(cx + COL_X_OFF)
        if slots_dn[col_idx] is not None:
            tap_xs.append(cx + COL_X_OFF + STAGGER)
    if not tap_xs:
        tap_xs = [cx_of(0)]

    BUS_X1   = min(tap_xs) - BUS_OVERHANG_L
    BUS_X2   = max(tap_xs) + BUS_OVERHANG_R
    CANVAS_W = BUS_X2 + MARGIN_X

    # ── Vertical geometry ──────────────────────────────────────────────────────
    # BUS_Y is the raw reference; canvas is trimmed by the entry point.
    BUS_Y = ISO_Y_OFF + SYM + ISO_JCT_GAP + SYM + BRK_GAP + SYM + MARGIN_Y

    # UP legs — from bus surface going upward:
    #   bus tap → wire(ISO_Y_OFF) → iso(flipped, SYM) → wire(ISO_JCT_GAP)
    #          → brk(SYM) → stub(BRK_GAP) → [terminal stamped by _cap]
    up_iso_bot  = BUS_Y - ISO_Y_OFF
    up_iso_top  = up_iso_bot - SYM
    up_brk_bot  = up_iso_top - ISO_JCT_GAP
    up_brk_top  = up_brk_bot - SYM
    up_wire_end = up_brk_top - BRK_GAP     # _cap: sym_top_y = wire_end_y - SYM

    # DOWN legs — from bus surface going downward:
    #   bus tap → wire(ISO_Y_OFF) → iso(SYM) → wire(ISO_JCT_GAP)
    #          → brk(SYM) → stub(BRK_GAP) → [terminal stamped by _cap]
    dn_iso_top  = BUS_Y + ISO_Y_OFF
    dn_iso_bot  = dn_iso_top + SYM
    dn_brk_top  = dn_iso_bot + ISO_JCT_GAP
    dn_brk_bot  = dn_brk_top + SYM
    dn_wire_end = dn_brk_bot + BRK_GAP     # _cap: sym_top_y = wire_end_y

    # ── Build end_pts ─────────────────────────────────────────────────────────
    # lbl_left=True  → device labels go to the LEFT  of the symbol (infeeds)
    # lbl_left=False → device labels go to the RIGHT of the symbol (outfeeds)
    end_pts: dict = {}
    for col_idx, (up_chain, dn_chain) in enumerate(columns):
        cx   = cx_of(col_idx)
        x_up = cx + COL_X_OFF
        x_dn = cx + COL_X_OFF + STAGGER

        if up_chain is not None:
            term = up_chain["terminal"]
            end_pts[term] = dict(
                mid_x=x_up, wire_end_y=up_wire_end, goes_up=True,
                sk=_sym_key(devices[term]["type"]), dev_type=devices[term]["type"],
                chain=up_chain, lbl_left=False,
            )

        if dn_chain is not None:
            term = dn_chain["terminal"]
            end_pts[term] = dict(
                mid_x=x_dn, wire_end_y=dn_wire_end, goes_up=False,
                sk=_sym_key(devices[term]["type"]), dev_type=devices[term]["type"],
                chain=dn_chain, lbl_left=False,
            )

    # ── Draw core SVG elements ────────────────────────────────────────────────
    svg = []

    # bbox_list: raw (pre-transform) symbol bounding boxes for YOLO export
    # class_id: 0=gridsource 1=transformer 2=breaker 3=feeder 5=isolator 6=bus
    bbox_list = []

    # Bus spine + label
    _bus_line(svg, BUS_X1, BUS_Y, BUS_X2, BUS_SPINE_W)
    bbox_list.append((6, BUS_X1, BUS_Y - BUS_SPINE_W / 2,
                      BUS_X2 - BUS_X1, BUS_SPINE_W))
    _lbl(svg, dev_label(bus_id), BUS_X1, BUS_Y - 4, anchor="middle")

    # ── Infeed stacks (going UP) ───────────────────────────────────────────────
    for term, ep in end_pts.items():
        if not ep["goes_up"]:
            continue
        cx       = ep["mid_x"]
        chain    = ep["chain"]
        lbl_left = ep["lbl_left"]

        def _lbl_bay(svg, text, cx, sym_top, lbl_left=lbl_left):
            if lbl_left:
                _lbl(svg, text, cx - SYM / 2 - LBL_X_PAD,
                     sym_top + SYM / 2 + LBL_Y_PAD, anchor="end")
            else:
                _lbl(svg, text, cx + SYM / 2 + LBL_X_PAD,
                     sym_top + SYM / 2 + LBL_Y_PAD, anchor="start")

        # Wire: iso bottom → bus tap
        _ln(svg, cx, up_iso_bot, cx, BUS_Y)
        _dot(svg, cx, BUS_Y, NODE_R)

        # Isolator (flipped: entry=bottom nearest bus, exit=top)
        _sym(svg, "isolator", cx, up_iso_top, SYM, rotate180=True)
        bbox_list.append((5, cx - SYM / 2, up_iso_top, SYM, SYM))
        _dot(svg, cx, up_iso_bot, NODE_R)
        _dot(svg, cx, up_iso_top, NODE_R)
        if chain["iso"]:
            _lbl_bay(svg, dev_label(chain["iso"]), cx, up_iso_top)

        # Wire: iso top → brk bottom
        _ln(svg, cx, up_iso_top, cx, up_brk_bot)

        # Breaker
        _sym(svg, "breaker", cx, up_brk_top, SYM)
        bbox_list.append((2, cx - SYM / 2, up_brk_top, SYM, SYM))
        _dot(svg, cx, up_brk_bot, NODE_R)
        _dot(svg, cx, up_brk_top, NODE_R)
        if chain["brk"]:
            _lbl_bay(svg, dev_label(chain["brk"]), cx, up_brk_top)

        # Stub wire: brk top → wire_end (_cap stamps terminal above)
        _ln(svg, cx, up_brk_top, cx, up_wire_end)

    # ── Outfeed stacks (going DOWN) ────────────────────────────────────────────
    for term, ep in end_pts.items():
        if ep["goes_up"]:
            continue
        cx       = ep["mid_x"]
        chain    = ep["chain"]
        lbl_left = ep["lbl_left"]

        def _lbl_bay(svg, text, cx, sym_top, lbl_left=lbl_left):
            if lbl_left:
                _lbl(svg, text, cx - SYM / 2 - LBL_X_PAD,
                     sym_top + SYM / 2 + LBL_Y_PAD, anchor="end")
            else:
                _lbl(svg, text, cx + SYM / 2 + LBL_X_PAD,
                     sym_top + SYM / 2 + LBL_Y_PAD, anchor="start")

        # Wire: bus tap → iso top
        _dot(svg, cx, BUS_Y, NODE_R)
        _ln(svg, cx, BUS_Y, cx, dn_iso_top)

        # Isolator (upright: entry=top, exit=bottom)
        _sym(svg, "isolator", cx, dn_iso_top, SYM)
        bbox_list.append((5, cx - SYM / 2, dn_iso_top, SYM, SYM))
        _dot(svg, cx, dn_iso_top, NODE_R)
        _dot(svg, cx, dn_iso_bot, NODE_R)
        if chain["iso"]:
            _lbl_bay(svg, dev_label(chain["iso"]), cx, dn_iso_top)

        # Wire: iso bottom → brk top
        _ln(svg, cx, dn_iso_bot, cx, dn_brk_top)

        # Breaker
        _sym(svg, "breaker", cx, dn_brk_top, SYM)
        bbox_list.append((2, cx - SYM / 2, dn_brk_top, SYM, SYM))
        _dot(svg, cx, dn_brk_top, NODE_R)
        _dot(svg, cx, dn_brk_bot, NODE_R)
        if chain["brk"]:
            _lbl_bay(svg, dev_label(chain["brk"]), cx, dn_brk_top)

        # Stub wire: brk bottom → wire_end (_cap stamps terminal below)
        _ln(svg, cx, dn_brk_bot, cx, dn_wire_end)

    return dict(
        svg=svg,
        BUS_Y=BUS_Y,
        BUS_X1=BUS_X1, BUS_X2=BUS_X2,
        CANVAS_W=CANVAS_W,
        end_pts=end_pts,
        devices=devices, dev_label=dev_label,
        SYM=SYM, MARGIN_X=MARGIN_X, MARGIN_Y=MARGIN_Y,
        LBL_X_PAD=LBL_X_PAD, LBL_Y_PAD=LBL_Y_PAD,
        bbox_list=bbox_list,
    )


# ── LABELLER ──────────────────────────────────────────────────────────────────

def _place_labels(svg, label_reqs):
    """Resolve vertical collisions among labels sharing the same x column."""
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
                col_items[j]  = (new_y, cur_i)

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
    MARGIN_Y  = core["MARGIN_Y"]
    end_pts   = core["end_pts"]
    dev_label = core["dev_label"]
    LBL_X_PAD = core["LBL_X_PAD"]
    LBL_Y_PAD = core["LBL_Y_PAD"]
    bbox_list  = core["bbox_list"]

    # YOLO class ids — must match render_core comment
    TERM_CLASS = {"GridSource": 0, "Transformer": 1, "Feeder": 3}

    label_reqs   = []
    boundary_pts = {}

    for et, ep in end_pts.items():
        mid_x      = ep["mid_x"]
        wire_end_y = ep["wire_end_y"]
        sk         = ep["sk"]
        goes_up    = ep["goes_up"]
        dev_type   = ep["dev_type"]
        lbl_left   = ep.get("lbl_left", False)

        lx_fn = (lambda cx: cx - SYM / 2 - LBL_X_PAD) if lbl_left \
                else (lambda cx: cx + SYM / 2 + LBL_X_PAD)
        anc   = "end" if lbl_left else "start"

        if mode == "hybrid" and dev_type == "Transformer":
            # Extend stub wire to the actual SVG boundary edge
            boundary_y = -y_offset if goes_up else -y_offset + H
            _ln(svg, mid_x, wire_end_y, mid_x, boundary_y)
            boundary_pts[et] = {
                "label": dev_label(et),
                "x": round(mid_x, 1),
                "y": round(boundary_y + y_offset, 1),
            }
        else:
            cls = TERM_CLASS.get(dev_type, 3)
            if goes_up:
                sym_top_y = wire_end_y - SYM
                _sym(svg, sk, mid_x, sym_top_y)
                bbox_list.append((cls, mid_x - SYM / 2, sym_top_y, SYM, SYM))
                label_reqs.append((
                    lx_fn(mid_x), sym_top_y + SYM / 2 + LBL_Y_PAD,
                    dev_label(et), anc,
                ))
            else:
                sym_top_y = wire_end_y
                _sym(svg, sk, mid_x, sym_top_y)
                bbox_list.append((cls, mid_x - SYM / 2, sym_top_y, SYM, SYM))
                label_reqs.append((
                    lx_fn(mid_x), sym_top_y + SYM / 2 + LBL_Y_PAD,
                    dev_label(et), anc,
                ))

    _place_labels(svg, label_reqs)
    return boundary_pts


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def render_single_bus_svg(
    topology_path: str,
    output_path: str,
    mode: str = "standalone",   # "standalone" | "hybrid"
) -> dict | None:

    import json

    with open(topology_path) as f:
        topology_json = json.load(f)

    layout = topology_json.get("layout", {})
    topo_type = layout.get("topology_type") or layout.get("mode", "")

    if topo_type != "SINGLE_BUS":
        raise ValueError(f"Expected SINGLE_BUS, got '{topo_type}'")

    core = _render_core(topology_json)

    svg      = core["svg"]
    SYM      = core["SYM"]
    MARGIN_Y = core["MARGIN_Y"]
    BUS_Y    = core["BUS_Y"]
    end_pts  = core["end_pts"]
    W        = core["CANVAS_W"]

    # Determine vertical canvas extent from wire endpoints + symbol heights
    up_ys   = [ep["wire_end_y"] - SYM for ep in end_pts.values() if     ep["goes_up"]]
    down_ys = [ep["wire_end_y"] + SYM  for ep in end_pts.values() if not ep["goes_up"]]
    top_y   = min([BUS_Y] + up_ys)   - MARGIN_Y
    bot_y   = max([BUS_Y] + down_ys) + MARGIN_Y

    y_offset = -top_y
    H = bot_y - top_y

    boundary_pts = _cap(core, mode, y_offset, H)

    # Transform raw bbox coords → final SVG coords, then emit YOLO .txt
    yolo_lines = []
    for (cls, x_left, y_top, w, h) in core["bbox_list"]:
        fy = y_top + y_offset
        xc = (x_left + w / 2) / W
        yc = (fy     + h / 2) / H
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

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 3:
        print("Usage: python render_sngl_bs.py <topology.json> <output.svg> [standalone|hybrid]")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        topo = json.load(fh)
    m = sys.argv[3] if len(sys.argv) > 3 else "standalone"
    render_single_bus_svg(topo, sys.argv[2], mode=m)
    print(f"Written: {sys.argv[2]} ({m})")