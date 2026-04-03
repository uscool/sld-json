"""
main_transfer_renderer.py
=========================
Three-function renderer for MAIN_AND_TRANSFER topology.

  _render_core(...)     → draws buses, bay stacks (isolators, breakers, wires,
                          junction dots). Returns layout geometry + svg list.

  _cap(core, mode)      → stamps terminal symbols + labels at every exposed
                          wire end.
                          standalone: all terminals get symbol + label.
                          hybrid:     Feeder/GridSource → symbol + label.
                                      Transformer       → wire to SVG boundary,
                                                          no symbol.

  render_main_and_transfer_svg(topology_json, output_path, mode="standalone")
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

def _sym_rotated_ccw(svg, name, cx, cy, size=32.0):
    """Draw a symbol rotated 90° CCW around its own centre — used for the bus-tie breaker."""
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
    """Label rotated +90° around (x,y) — runs top-to-bottom, placed to the right of symbol."""
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
    MARGIN_X           = 30.0   # left/right canvas margin
    MARGIN_Y           = 0.0    # top/bottom canvas margin
    COL_W              = 80.0   # horizontal pitch per bay column
    SYM                = 32.0   # symbol size px
    BUS_SPINE_W        = 2.0
    NODE_R             = 0.0
    WIRE_GAP           = 0.0    # short stub between bus tap and nearest iso terminal
    ISO_BRK_GAP        = 0.0    # isolator and breaker touch — no gap wire
    BRK_ISO_GAP        = 0.0    # breaker and far isolator touch — no gap wire
    ISO_TERM_GAP       = 0.0    # gap/stub between far iso edge and terminal symbol
    TIE_GAP            = 70.0   # horizontal gap between the two bus right and left ends
    FONT_W             = 0      # px per char at font-size 7 monospace (unused, kept for compat)
    LBL_X_PAD         = -8.0   # horizontal gap: symbol edge → label (start or end)
    LBL_Y_PAD         = 0.0    # baseline drop from symbol vertical midpoint
    MAIN_OVERHANG_L    = 45.0  # main bus: extends left  of leftmost tap
    MAIN_OVERHANG_R    = 15.0  # main bus: extends right of rightmost tap
    XFER_OVERHANG_L    = -20.0 # transfer bus: extends left  of leftmost tap
    XFER_OVERHANG_R    = 45.0  # transfer bus: extends right of rightmost tap
    BAY_STAGGER_X      = 4.0   # small nudge so incomer doesn't land on middle outgoer x
    TIE_Y_OFFSET       = -10.0   # vertical offset of the coupler label below BUS_Y + SYM/2

    # ── Parse ────────────────────────────────────────────────────────────────
    devices = {d["id"]: d for d in topology_json["devices"]}

    def dev_label(dev_id):
        return devices[dev_id].get("label") or dev_id

    adj = defaultdict(set)
    for c in topology_json["connections"]:
        adj[c["from"]].add(c["to"])
        adj[c["to"]].add(c["from"])

    exposed_set = set(topology_json.get("layout", {}).get("terminals", []))

    # ── Identify buses ────────────────────────────────────────────────────────
    bus_devs = [d for d in devices.values() if d["type"] == "Bus"]
    main_bus_id = transfer_bus_id = None
    for bd in bus_devs:
        role = (bd.get("bus_role") or "").upper()
        if role == "MAIN":
            main_bus_id = bd["id"]
        elif role == "TRANSFER":
            transfer_bus_id = bd["id"]
    if main_bus_id is None or transfer_bus_id is None:
        ids = sorted(bd["id"] for bd in bus_devs)
        main_bus_id, transfer_bus_id = ids[0], ids[1]

    # ── Trace chains from each bus ────────────────────────────────────────────
    # Chain: Bus → ISO_A → BRK → ISO_B → terminal
    def trace_chain(bus_id: str, start_iso: str):
        iso_a = start_iso
        visited = {bus_id, iso_a}
        brk = next((n for n in adj[iso_a] if n not in visited and
                     devices[n]["type"] == "Breaker"), None)
        if brk is None:
            return None
        visited.add(brk)
        iso_b = next((n for n in adj[brk] if n not in visited and
                       devices[n]["type"] == "Isolator"), None)
        if iso_b is None:
            return None
        visited.add(iso_b)
        terminal = next((n for n in adj[iso_b] if n not in visited and n in exposed_set), None)
        return dict(iso_a=iso_a, brk=brk, iso_b=iso_b, terminal=terminal)

    main_incomers: list[dict] = []
    main_outgoers: list[dict] = []
    xfer_incomers: list[dict] = []
    xfer_outgoers: list[dict] = []

    for bus_id, inc_list, out_list in [
        (main_bus_id,     main_incomers,  main_outgoers),
        (transfer_bus_id, xfer_incomers,  xfer_outgoers),
    ]:
        for nb in sorted(adj[bus_id]):
            if devices[nb]["type"] != "Isolator":
                continue
            chain = trace_chain(bus_id, nb)
            if chain is None:
                continue
            term = chain["terminal"]
            if term is None:
                continue
            if _goes_up(term, devices[term]["type"]):
                inc_list.append(chain)
            else:
                out_list.append(chain)

    # ── Detect bus-tie breaker ────────────────────────────────────────────────
    tie_breaker_id: str | None = None
    for dev in devices.values():
        if dev["type"] != "Breaker":
            continue
        brk_id = dev["id"]
        reached_buses = set()
        for nb in adj[brk_id]:
            if devices[nb]["type"] == "Bus":
                reached_buses.add(nb)
            elif devices[nb]["type"] == "Isolator":
                for nb2 in adj[nb]:
                    if devices[nb2]["type"] == "Bus":
                        reached_buses.add(nb2)
        if main_bus_id in reached_buses and transfer_bus_id in reached_buses:
            tie_breaker_id = brk_id
            break

    main_outgoers.sort(key=lambda c: c["terminal"])
    xfer_outgoers.sort(key=lambda c: c["terminal"])

    # ── Column positions ──────────────────────────────────────────────────────
    # Outgoers: evenly spaced at COL_W pitch.
    # Incomers: centred over the outgoers span, then nudged by BAY_STAGGER_X
    #           so they don't land dead-on the middle outgoer column.

    def _out_xs(base: float, n: int) -> list:
        return [base + i * COL_W + COL_W / 2 for i in range(n)]

    def _inc_xs_centred(out_xs: list, n_inc: int) -> list:
        if n_inc == 0:
            return []
        if not out_xs:
            return [0.0] * n_inc
        span_mid = (out_xs[0] + out_xs[-1]) / 2.0
        total_w  = (n_inc - 1) * COL_W if n_inc > 1 else 0.0
        start    = span_mid - total_w / 2.0 + BAY_STAGGER_X
        return [start + i * COL_W for i in range(n_inc)]

    # Main bus
    main_out_xs = _out_xs(MARGIN_X, len(main_outgoers))
    main_inc_xs = _inc_xs_centred(main_out_xs, len(main_incomers))
    all_main_xs = main_inc_xs + main_out_xs

    MAIN_BUS_X1 = (min(all_main_xs) if all_main_xs else MARGIN_X)         - MAIN_OVERHANG_L
    MAIN_BUS_X2 = (max(all_main_xs) if all_main_xs else MARGIN_X + COL_W) + MAIN_OVERHANG_R

    # Transfer bus
    XFER_BUS_X1   = MAIN_BUS_X2 + TIE_GAP
    xfer_out_base = XFER_BUS_X1 + XFER_OVERHANG_L
    xfer_out_xs   = _out_xs(xfer_out_base, len(xfer_outgoers))
    xfer_inc_xs   = _inc_xs_centred(xfer_out_xs, len(xfer_incomers))
    all_xfer_xs   = xfer_inc_xs + xfer_out_xs

    XFER_BUS_X2 = (max(all_xfer_xs) if all_xfer_xs else XFER_BUS_X1 + COL_W) + XFER_OVERHANG_R

    TIE_CX = (MAIN_BUS_X2 + XFER_BUS_X1) / 2.0

    # ── Vertical geometry ─────────────────────────────────────────────────────
    # BUS_Y is the reference; canvas is trimmed by the public entry point.
    BUS_Y = WIRE_GAP + 3 * SYM + ISO_TERM_GAP + SYM + MARGIN_Y

    # Outgoer positions (going DOWN from bus)
    dn_iso_a_top = BUS_Y + WIRE_GAP
    dn_brk_top   = dn_iso_a_top + SYM + ISO_BRK_GAP
    dn_iso_b_top = dn_brk_top   + SYM + BRK_ISO_GAP
    dn_wire_end  = dn_iso_b_top + SYM + ISO_TERM_GAP

    # Incomer positions (going UP from bus)
    up_iso_b_top = BUS_Y - WIRE_GAP - SYM
    up_brk_top   = up_iso_b_top - ISO_BRK_GAP - SYM
    up_iso_a_top = up_brk_top   - BRK_ISO_GAP - SYM
    up_wire_end  = up_iso_a_top - ISO_TERM_GAP

    # ── Build end_pts ─────────────────────────────────────────────────────────
    # lbl_left=True  → labels go to the LEFT  of the symbol (main bus bays)
    # lbl_left=False → labels go to the RIGHT of the symbol (transfer bus bays)
    end_pts: dict = {}

    for i, chain in enumerate(main_incomers):
        term = chain["terminal"]
        if term is None:
            continue
        end_pts[term] = dict(
            mid_x=main_inc_xs[i], wire_end_y=up_wire_end, goes_up=True,
            sk=_sym_key(devices[term]["type"]), dev_type=devices[term]["type"],
            chain=chain, lbl_left=True,
        )

    for i, chain in enumerate(xfer_incomers):
        term = chain["terminal"]
        if term is None:
            continue
        end_pts[term] = dict(
            mid_x=xfer_inc_xs[i], wire_end_y=up_wire_end, goes_up=True,
            sk=_sym_key(devices[term]["type"]), dev_type=devices[term]["type"],
            chain=chain, lbl_left=False,
        )

    for i, chain in enumerate(main_outgoers):
        term = chain["terminal"]
        if term is None:
            continue
        end_pts[term] = dict(
            mid_x=main_out_xs[i], wire_end_y=dn_wire_end, goes_up=False,
            sk=_sym_key(devices[term]["type"]), dev_type=devices[term]["type"],
            chain=chain, lbl_left=True,
        )

    for i, chain in enumerate(xfer_outgoers):
        term = chain["terminal"]
        if term is None:
            continue
        end_pts[term] = dict(
            mid_x=xfer_out_xs[i], wire_end_y=dn_wire_end, goes_up=False,
            sk=_sym_key(devices[term]["type"]), dev_type=devices[term]["type"],
            chain=chain, lbl_left=False,
        )

    # ── Draw core SVG elements ────────────────────────────────────────────────
    svg = []

    # bbox_list: raw (pre-transform) symbol bounding boxes for YOLO export
    # class_id: 0=gridsource 1=transformer 2=breaker 3=feeder 5=isolator 6=bus  (4=capacitor unused)
    bbox_list = []

    # Bus spines
    _bus_line(svg, MAIN_BUS_X1, BUS_Y, MAIN_BUS_X2, BUS_SPINE_W)
    _bus_line(svg, XFER_BUS_X1, BUS_Y, XFER_BUS_X2, BUS_SPINE_W)
    bbox_list.append((6, MAIN_BUS_X1, BUS_Y - BUS_SPINE_W / 2,
                       MAIN_BUS_X2 - MAIN_BUS_X1, BUS_SPINE_W))
    bbox_list.append((6, XFER_BUS_X1, BUS_Y - BUS_SPINE_W / 2,
                       XFER_BUS_X2 - XFER_BUS_X1, BUS_SPINE_W))

    # Bus labels: above the spine, left-aligned
    _lbl(svg, dev_label(main_bus_id),     MAIN_BUS_X1, BUS_Y - 4, anchor="start")
    _lbl(svg, dev_label(transfer_bus_id), XFER_BUS_X1, BUS_Y - 4, anchor="start")

    # Bus-tie breaker — symbol centred on BUS_Y (rotated CCW, sits across the wire).
    # TIE_Y_OFFSET only shifts the label downward from the symbol bottom edge.
    if tie_breaker_id:
        _ln(svg, MAIN_BUS_X2, BUS_Y, TIE_CX - SYM / 2.0, BUS_Y)
        _ln(svg, TIE_CX + SYM / 2.0, BUS_Y, XFER_BUS_X1, BUS_Y)
        _dot(svg, MAIN_BUS_X2, BUS_Y, NODE_R)
        _dot(svg, XFER_BUS_X1, BUS_Y, NODE_R)
        _sym_rotated_ccw(svg, "breaker", TIE_CX, BUS_Y)
        bbox_list.append((2, TIE_CX - SYM / 2, BUS_Y - SYM / 2, SYM, SYM))
        # Label is isolated from bay labelling — placed below symbol + offset
        _lbl(svg, dev_label(tie_breaker_id),
             TIE_CX, BUS_Y + SYM / 2.0 + TIE_Y_OFFSET + 6.0, anchor="middle")
    else:
        _ln(svg, MAIN_BUS_X2, BUS_Y, XFER_BUS_X1, BUS_Y)

    # ── Incomer stacks (going UP) ─────────────────────────────────────────────
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

        # Wire: iso_b bottom → bus tap
        _ln(svg, cx, up_iso_b_top + SYM, cx, BUS_Y)
        _dot(svg, cx, BUS_Y, NODE_R)

        # ISO_B (nearest to bus)
        _sym(svg, "isolator", cx, up_iso_b_top, SYM)
        bbox_list.append((5, cx - SYM / 2, up_iso_b_top, SYM, SYM))
        _dot(svg, cx, up_iso_b_top,       NODE_R)
        _dot(svg, cx, up_iso_b_top + SYM, NODE_R)
        if chain["iso_b"]:
            _lbl_bay(svg, dev_label(chain["iso_b"]), cx, up_iso_b_top)

        # Breaker
        _sym(svg, "breaker", cx, up_brk_top, SYM)
        bbox_list.append((2, cx - SYM / 2, up_brk_top, SYM, SYM))
        _dot(svg, cx, up_brk_top,       NODE_R)
        _dot(svg, cx, up_brk_top + SYM, NODE_R)
        if chain["brk"]:
            _lbl_bay(svg, dev_label(chain["brk"]), cx, up_brk_top)

        # ISO_A (far from bus)
        _sym(svg, "isolator", cx, up_iso_a_top, SYM)
        bbox_list.append((5, cx - SYM / 2, up_iso_a_top, SYM, SYM))
        _dot(svg, cx, up_iso_a_top,       NODE_R)
        _dot(svg, cx, up_iso_a_top + SYM, NODE_R)
        if chain["iso_a"]:
            _lbl_bay(svg, dev_label(chain["iso_a"]), cx, up_iso_a_top)

        # Stub wire: iso_a top → wire_end (capper stamps terminal above)
        _ln(svg, cx, up_iso_a_top, cx, up_wire_end)

    # ── Outgoer stacks (going DOWN) ───────────────────────────────────────────
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

        # Wire: bus tap → iso_a top
        _dot(svg, cx, BUS_Y, NODE_R)
        _ln(svg, cx, BUS_Y, cx, dn_iso_a_top)

        # ISO_A (nearest to bus)
        _sym(svg, "isolator", cx, dn_iso_a_top, SYM)
        bbox_list.append((5, cx - SYM / 2, dn_iso_a_top, SYM, SYM))
        _dot(svg, cx, dn_iso_a_top,       NODE_R)
        _dot(svg, cx, dn_iso_a_top + SYM, NODE_R)
        if chain["iso_a"]:
            _lbl_bay(svg, dev_label(chain["iso_a"]), cx, dn_iso_a_top)

        # Breaker
        _sym(svg, "breaker", cx, dn_brk_top, SYM)
        bbox_list.append((2, cx - SYM / 2, dn_brk_top, SYM, SYM))
        _dot(svg, cx, dn_brk_top,       NODE_R)
        _dot(svg, cx, dn_brk_top + SYM, NODE_R)
        if chain["brk"]:
            _lbl_bay(svg, dev_label(chain["brk"]), cx, dn_brk_top)

        # ISO_B (far from bus)
        _sym(svg, "isolator", cx, dn_iso_b_top, SYM)
        bbox_list.append((5, cx - SYM / 2, dn_iso_b_top, SYM, SYM))
        _dot(svg, cx, dn_iso_b_top,       NODE_R)
        _dot(svg, cx, dn_iso_b_top + SYM, NODE_R)
        if chain["iso_b"]:
            _lbl_bay(svg, dev_label(chain["iso_b"]), cx, dn_iso_b_top)

        # Stub wire: iso_b bottom → wire_end (capper stamps terminal below)
        _ln(svg, cx, dn_iso_b_top + SYM, cx, dn_wire_end)

    return dict(
        svg=svg,
        BUS_Y=BUS_Y,
        MAIN_BUS_X1=MAIN_BUS_X1, MAIN_BUS_X2=MAIN_BUS_X2,
        XFER_BUS_X1=XFER_BUS_X1, XFER_BUS_X2=XFER_BUS_X2,
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
    MARGIN_Y  = core["MARGIN_Y"]
    end_pts   = core["end_pts"]
    dev_label = core["dev_label"]
    LBL_X_PAD = core["LBL_X_PAD"]
    LBL_Y_PAD = core["LBL_Y_PAD"]
    bbox_list  = core["bbox_list"]

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
                "x": round(mid_x, 1),
                "y": round(boundary_y + y_offset, 1),
            }
        else:
            cls = TERM_CLASS.get(dev_type, 3)
            lbl_left = ep.get("lbl_left", False)
            lx_fn = (lambda cx: cx - SYM / 2 - LBL_X_PAD) if lbl_left else (lambda cx: cx + SYM / 2 + LBL_X_PAD)
            anc = "end" if lbl_left else "start"
            if goes_up:
                sym_top_y = wire_end_y - SYM
                _sym(svg, sk, mid_x, sym_top_y)
                bbox_list.append((cls, mid_x - SYM / 2, sym_top_y, SYM, SYM))
                label_reqs.append((lx_fn(mid_x), sym_top_y + SYM / 2 + LBL_Y_PAD, dev_label(et), anc))
            else:
                sym_top_y = wire_end_y
                _sym(svg, sk, mid_x, sym_top_y)
                bbox_list.append((cls, mid_x - SYM / 2, sym_top_y, SYM, SYM))
                label_reqs.append((lx_fn(mid_x), sym_top_y + SYM / 2 + LBL_Y_PAD, dev_label(et), anc))

    _place_labels(svg, label_reqs)
    return boundary_pts


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def render_main_and_transfer_svg(
    topology_path: str,
    output_path: str,
    mode: str = "standalone",   # "standalone" | "hybrid"
) -> dict | None:

    import json

    with open(topology_path) as f:
        topology_json = json.load(f)

    layout = topology_json.get("layout", {})
    topo_type = layout.get("topology_type") or layout.get("mode", "")

    if topo_type != "MAIN_AND_TRANSFER":
        raise ValueError(f"Expected MAIN_AND_TRANSFER, got '{topo_type}'")

    core = _render_core(topology_json)

    svg      = core["svg"]
    SYM      = core["SYM"]
    MARGIN_X = core["MARGIN_X"]
    MARGIN_Y = core["MARGIN_Y"]
    BUS_Y    = core["BUS_Y"]
    end_pts  = core["end_pts"]

    # Determine vertical canvas extent from wire endpoints + symbol heights
    up_ys   = [ep["wire_end_y"] - SYM for ep in end_pts.values() if ep["goes_up"]]
    down_ys = [ep["wire_end_y"] + SYM  for ep in end_pts.values() if not ep["goes_up"]]
    top_y   = min([BUS_Y] + up_ys)   - MARGIN_Y
    bot_y   = max([BUS_Y] + down_ys) + MARGIN_Y

    y_offset = -top_y
    W = core["XFER_BUS_X2"] + MARGIN_X
    H = bot_y - top_y

    boundary_pts = _cap(core, mode, y_offset, H)

    # Transform raw bbox coords → final SVG coords, then normalise to YOLO format
    # SVG transform is: translate(0, y_offset)
    ty = y_offset
    yolo_lines = []
    for (cls, x_left, y_top, w, h) in core["bbox_list"]:
        fy = y_top + ty
        xc = (x_left + w / 2) / W
        yc = (fy    + h / 2) / H
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
        print("Usage: python render_mn_trfr.py <topology.json> <output.svg> [standalone|hybrid]")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        topo = json.load(fh)
    m = sys.argv[3] if len(sys.argv) > 3 else "standalone"
    render_main_and_transfer_svg(topo, sys.argv[2], mode=m)
    print(f"Written: {sys.argv[2]} ({m})")