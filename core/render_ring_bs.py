"""
ring_bus_renderer.py
====================
Three-function renderer for RING_BUS topology.

  _render_core(...)     → draws ring structure (buses, ring breakers, tap legs).
                          Returns layout geometry + svg list + bbox list.

  _cap(core, mode)      → stamps terminal symbols + labels at every exposed
                          wire end.
                          standalone: all terminals get symbol + label.
                          hybrid:     Feeder/GridSource → symbol + label.
                                      Transformer       → wire to SVG boundary,
                                                          no symbol.

  render_ring_bus_svg(topology_json, output_path, mode="standalone")
                        → orchestrates the above, writes SVG file + YOLO labels.
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
    """Draw a symbol rotated 90 deg CCW around its centre - used for horizontal breakers."""
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
    """Label rotated +90 deg around (x,y) - used exclusively for vertical arm breakers."""
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="monospace" font-size="{size}" fill="#000" '
        f'transform="rotate(90,{x:.1f},{y:.1f})">{text}</text>'
    )

def _bus_wire(svg, x1, y1, x2, y2, w=2.0):
    svg.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="#000" stroke-width="{w}" stroke-linecap="square"/>'
    )


# ── CORE ───────────────────────────────────────────────────────────────────────

def _render_core(topology_json: dict) -> dict:
    """
    Parse topology, compute geometry, draw ring structure + tap legs.
    Returns a geometry/state dict consumed by _cap.
    """
    # ── Constants ───────────────────────────────────────────────────────────
    MARGIN_X            = 30.0   # left/right margin
    MARGIN_Y            = 0.0   # top/bottom margin
    SYM                 = 32.0   # symbol bounding box (square)
    BUS_SPINE_W         = 2.0    # bus-bar / arm stroke width
    NODE_R              = 0.0    # junction dot radius
    MIN_RING_WIDTH      = 300.0  # minimum span between the two vertical arm positions
    RING_HEIGHT         = 100.0  # vertical distance between top and bottom rails
    MIN_BUS_WIDTH       = 60.0  # minimum width for each bus segment
    RING_BRK_GAP        = 20.0   # gap between horizontal ring breaker and bus wire endpoints
    LEG_GAP             = 0.0    # wire gap from bus node to first tap leg symbol
    LEG_SYM_GAP         = 0.0    # gap between consecutive symbols on a tap leg
    LEG_COL_PITCH       = 0.0    # x-pitch for multiple legs on one bus node (0 = same column)
    BAY_OFFSET_RATIO    = 0.1   # bay attaches at this fraction from left edge of bus (0.0=left, 0.5=center, 1.0=right)
    LBL_FONT_SIZE       = 4      # label font size (pt)
    LBL_BASE_OFF        = 0.0    # text baseline drop below symbol vertical midpoint
    H_LBL_PAD          = -9.0   # gap: symbol right-edge -> horizontal label start (negative = overlap)
    LBL_X_PAD          = -8.0   # gap: symbol right-edge -> rotated label anchor (vertical arms only)
    LBL_Y_PAD          = 0.0    # vertical offset from symbol centre for rotated label (vertical arms only)
    HORIZ_BRK_BOX_H     = 5.0   # half-height of the rotated breaker rect body in y-direction
    BRK_LBL_GAP         = 2.0   # clearance between breaker body edge and label baseline
    BUS_LBL_INSIDE_TOP  = 9.0   # baseline offset below top rail  (positive = inside ring)
    BUS_LBL_INSIDE_BOT  = -4.0  # baseline offset above bot rail  (negative = inside ring)
    LEG_STACK = SYM * 3         # iso + brk + terminal, no gaps

    # ── Parse ────────────────────────────────────────────────────────────────
    devices = {d["id"]: d for d in topology_json["devices"]}

    def dev_label(dev_id):
        return devices[dev_id].get("label") or dev_id

    adj = defaultdict(set)
    for c in topology_json["connections"]:
        adj[c["from"]].add(c["to"])
        adj[c["to"]].add(c["from"])

    exposed_set = set(topology_json.get("layout", {}).get("terminals", []))

    # ── Read geometry hints ──────────────────────────────────────────────────
    geom       = topology_json["layout"]["geometry"]
    brk_orient = geom.get("breaker_orientation", {})
    brk_side   = geom.get("breaker_side", {})
    bus_side   = geom.get("bus_side", {})

    # ── Classify ring buses and breakers ─────────────────────────────────────
    ring_bus_set = {
        d["id"] for d in devices.values()
        if d["type"] == "Bus" and (d.get("bus_role") or "").upper() == "RING"
    }

    tap_brk_ids = {
        d["id"] for d in devices.values()
        if d["type"] == "Breaker" and (d.get("bay_id") or "") != "RING"
    }

    ring_brk_ids = sorted(
        d["id"] for d in devices.values()
        if d["type"] == "Breaker" and (d.get("bay_id") or "") == "RING"
    )

    # ── Build ring segments ───────────────────────────────────────────────────
    ring_segs = []
    seen_pairs = set()
    for brk_id in ring_brk_ids:
        nbs = sorted(nb for nb in adj[brk_id] if nb in ring_bus_set)
        if len(nbs) == 2:
            key = frozenset(nbs)
            if key not in seen_pairs:
                seen_pairs.add(key)
                ring_segs.append((nbs[0], brk_id, nbs[1]))

    # ── Partition segments by breaker side ────────────────────────────────────
    left_seg  = next((s for s in ring_segs if brk_side.get(s[1]) == "left"),  None)
    right_seg = next((s for s in ring_segs if brk_side.get(s[1]) == "right"), None)
    top_segs  = [s for s in ring_segs if brk_side.get(s[1]) == "top"]
    bot_segs  = [s for s in ring_segs if brk_side.get(s[1]) == "bottom"]

    # ── Identify corner buses ─────────────────────────────────────────────────
    def arm_buses(seg):
        a, _brk, b = seg
        return (a, b) if bus_side.get(a) == "top" else (b, a)

    tl_bus, bl_bus = arm_buses(left_seg)
    tr_bus, br_bus = arm_buses(right_seg)

    # ── Walk chains ───────────────────────────────────────────────────────────
    def walk_chain(start_bus, end_bus, segs):
        chain = [start_bus]
        visited = {start_bus}
        current = start_bus
        while current != end_bus:
            nxt = None
            for ba, _brk, bb in segs:
                if ba == current and bb not in visited:
                    nxt = bb; break
                if bb == current and ba not in visited:
                    nxt = ba; break
            if nxt is None:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        if end_bus not in chain:
            chain.append(end_bus)
        return chain

    top_chain = walk_chain(tl_bus, tr_bus, top_segs)
    bot_chain = walk_chain(bl_bus, br_bus, bot_segs)

    # ── Collect tap legs per bus ──────────────────────────────────────────────
    tap_legs_by_bus = defaultdict(list)
    for bus_id in ring_bus_set:
        side = bus_side.get(bus_id, "bottom")
        for iso_id in sorted(adj[bus_id]):
            if devices[iso_id]["type"] != "Isolator":
                continue
            for brk_id in sorted(adj[iso_id]):
                if brk_id == bus_id or brk_id not in tap_brk_ids:
                    continue
                for term_id in sorted(adj[brk_id]):
                    if term_id == iso_id:
                        continue
                    if term_id in exposed_set:
                        tap_legs_by_bus[bus_id].append(dict(
                            iso=iso_id, brk=brk_id,
                            terminal=term_id, goes_up=(side == "top"),
                        ))

    # ── Compute minimum RING_WIDTH from MIN_BUS_WIDTH ────────────────────────
    # For a chain of N buses with N-1 breakers, all evenly spaced:
    # Each breaker takes up: SYM + 2*RING_BRK_GAP (symbol plus gaps on both sides)
    #   end-bus wire length   = RING_WIDTH / (2*(N-1)) - (SYM + 2*RING_BRK_GAP)/2  >= MIN_BUS_WIDTH
    #   interior-bus wire len = RING_WIDTH / (N-1)    - (SYM + 2*RING_BRK_GAP)     >= MIN_BUS_WIDTH
    # Both resolve to: RING_WIDTH >= (N-1) * (2*MIN_BUS_WIDTH + SYM + 2*RING_BRK_GAP)
    def _chain_min_width(chain):
        n = len(chain)
        if n == 1:
            return 2.0 * MIN_BUS_WIDTH           # single bus: each half of rail >= MIN_BUS_WIDTH
        return (n - 1) * (2.0 * MIN_BUS_WIDTH + SYM + 2.0 * RING_BRK_GAP)

    RING_WIDTH = max(
        MIN_RING_WIDTH,
        _chain_min_width(top_chain),
        _chain_min_width(bot_chain),
    )

    # ── Assign x coordinates ──────────────────────────────────────────────────
    left_x  = MARGIN_X
    right_x = MARGIN_X + RING_WIDTH

    def even_x(chain):
        n = len(chain)
        if n == 1:
            return {chain[0]: (left_x + right_x) / 2}
        return {bus: left_x + i * (RING_WIDTH / (n - 1)) for i, bus in enumerate(chain)}

    top_bus_x = even_x(top_chain)
    bot_bus_x = even_x(bot_chain)
    all_bus_x = {**top_bus_x, **bot_bus_x}

    # ── Bus wire span helper ──────────────────────────────────────────────────
    def bus_wire_span(idx, chain, bus_x):
        """
        Returns (wire_left_x, wire_right_x) for the bus at chain[idx].
        This is the actual bus wire extent, i.e. from the arm (or right edge
        of the left breaker) to the arm (or left edge of the right breaker).
        The label for this bus should be centred at the midpoint of this span.
        Now accounts for RING_BRK_GAP on both sides of breakers.
        """
        n = len(chain)
        if n == 1:
            return left_x, right_x

        bx = bus_x[chain[idx]]

        # Left edge of this bus's wire
        if idx == 0:
            wx_left = left_x
        else:
            prev_bx = bus_x[chain[idx - 1]]
            # Right edge of the breaker gap between chain[idx-1] and chain[idx]
            wx_left = (prev_bx + bx) / 2 + SYM / 2 + RING_BRK_GAP

        # Right edge of this bus's wire
        if idx == n - 1:
            wx_right = right_x
        else:
            next_bx = bus_x[chain[idx + 1]]
            # Left edge of the breaker gap between chain[idx] and chain[idx+1]
            wx_right = (bx + next_bx) / 2 - SYM / 2 - RING_BRK_GAP

        return wx_left, wx_right

    # ── Assign y coordinates ──────────────────────────────────────────────────
    top_y = MARGIN_Y + LEG_STACK + LEG_GAP
    bot_y = top_y + RING_HEIGHT

    # ── Canvas ───────────────────────────────────────────────────────────────
    svg = []
    bbox_list = []
    end_pts = {}

    # ── Draw horizontal breaker chains ────────────────────────────────────────
    def draw_horiz_brk_chain(segs, chain, bus_x, rail_y, lbl_y_offset):
        """
        Draw horizontal breakers and bus nodes for one rail.
        lbl_y_offset added to rail_y positions the breaker label:
          positive  =>  below rail (top rail, label inside ring)
          negative  =>  above rail (bot rail, label inside ring)
        The breaker label is placed at the breaker's mid_x, i.e. centred on
        the breaker symbol, which is itself centred between the two bus nodes.
        """
        for bus_id in chain:
            _dot(svg, bus_x[bus_id], rail_y, NODE_R)
        for ba, brk_id, bb in segs:
            xa = bus_x.get(ba)
            xb = bus_x.get(bb)
            if xa is None or xb is None:
                continue
            x_lo, x_hi = min(xa, xb), max(xa, xb)
            mid_x     = (x_lo + x_hi) / 2
            brk_left  = mid_x - SYM / 2
            brk_right = mid_x + SYM / 2
            
            # Draw bus wires with RING_BRK_GAP between bus endpoint and breaker
            _bus_wire(svg, x_lo, rail_y, brk_left - RING_BRK_GAP, rail_y, BUS_SPINE_W)
            _ln(svg, brk_left - RING_BRK_GAP, rail_y, brk_left, rail_y, BUS_SPINE_W)
            _sym_rotated_ccw(svg, "breaker", mid_x, rail_y, SYM)
            bbox_list.append((2, mid_x - SYM / 2, rail_y - SYM / 2, SYM, SYM))
            _ln(svg, brk_right, rail_y, brk_right + RING_BRK_GAP, rail_y, BUS_SPINE_W)
            _bus_wire(svg, brk_right + RING_BRK_GAP, rail_y, x_hi, rail_y, BUS_SPINE_W)
            
            _dot(svg, brk_left,  rail_y, NODE_R)
            _dot(svg, brk_right, rail_y, NODE_R)
            # Breaker label centred on the breaker symbol midpoint
            _lbl(svg, dev_label(brk_id),
                 mid_x, rail_y + lbl_y_offset,
                 anchor="middle")

    # ── 1. TOP RAIL ────────────────────────────────────────────────────────────
    if len(top_chain) == 1:
        _bus_wire(svg, left_x, top_y, right_x, top_y, BUS_SPINE_W)
    else:
        top_brk_lbl_y = HORIZ_BRK_BOX_H + BRK_LBL_GAP + LBL_FONT_SIZE
        draw_horiz_brk_chain(top_segs, top_chain, top_bus_x, top_y,
                              lbl_y_offset=+top_brk_lbl_y)
    # Bus labels: centred on each bus's wire segment, inside ring below top rail
    for i, bus_id in enumerate(top_chain):
        wx_l, wx_r = bus_wire_span(i, top_chain, top_bus_x)
        bus_center_x = (wx_l + wx_r) / 2
        bus_width = wx_r - wx_l
        # Add bus bounding box (class 6, positioned at the bus wire segment)
        bbox_list.append((6, wx_l, top_y - BUS_SPINE_W/2, bus_width, BUS_SPINE_W))
        _lbl(svg, dev_label(bus_id),
             bus_center_x, top_y + BUS_LBL_INSIDE_TOP,
             anchor="middle")

    # ── 2. BOTTOM RAIL ─────────────────────────────────────────────────────────
    if len(bot_chain) == 1:
        _bus_wire(svg, left_x, bot_y, right_x, bot_y, BUS_SPINE_W)
    else:
        bot_brk_lbl_y = -(HORIZ_BRK_BOX_H + BRK_LBL_GAP)
        draw_horiz_brk_chain(bot_segs, bot_chain, bot_bus_x, bot_y,
                              lbl_y_offset=bot_brk_lbl_y)
    # Bus labels: centred on each bus's wire segment, inside ring above bottom rail
    for i, bus_id in enumerate(bot_chain):
        wx_l, wx_r = bus_wire_span(i, bot_chain, bot_bus_x)
        bus_center_x = (wx_l + wx_r) / 2
        bus_width = wx_r - wx_l
        # Add bus bounding box (class 6, positioned at the bus wire segment)
        bbox_list.append((6, wx_l, bot_y - BUS_SPINE_W/2, bus_width, BUS_SPINE_W))
        _lbl(svg, dev_label(bus_id),
             bus_center_x, bot_y + BUS_LBL_INSIDE_BOT,
             anchor="middle")

    # ── 3. VERTICAL ARMS ───────────────────────────────────────────────────────
    for seg, _top_bus, _bot_bus in [
        (left_seg,  tl_bus, bl_bus),
        (right_seg, tr_bus, br_bus),
    ]:
        brk_id  = seg[1]
        vx      = left_x if brk_side.get(brk_id) == "left" else right_x
        mid_y   = (top_y + bot_y) / 2
        brk_top = mid_y - SYM / 2
        brk_bot = mid_y + SYM / 2

        _bus_wire(svg, vx, top_y,   vx, brk_top, BUS_SPINE_W)
        _sym(svg, "breaker", vx, brk_top, SYM)
        bbox_list.append((2, vx - SYM / 2, brk_top, SYM, SYM))
        _bus_wire(svg, vx, brk_bot, vx, bot_y, BUS_SPINE_W)
        _dot(svg, vx, top_y,   NODE_R)
        _dot(svg, vx, brk_top, NODE_R)
        _dot(svg, vx, brk_bot, NODE_R)
        _dot(svg, vx, bot_y,   NODE_R)
        # Vertical arm breaker: ONLY element using a rotated label
        _lbl_rotated(svg, dev_label(brk_id),
                     vx + SYM / 2 + LBL_X_PAD,
                     brk_top + SYM / 2 + LBL_Y_PAD,
                     anchor="middle")

    # ── 4. TAP LEGS ────────────────────────────────────────────────────────────
    for bus_id in sorted(tap_legs_by_bus.keys()):
        legs     = tap_legs_by_bus[bus_id]
        n        = len(legs)
        side     = bus_side.get(bus_id, "bottom")
        
        # Find which chain this bus belongs to
        if bus_id in top_bus_x:
            chain = top_chain
            bus_x = top_bus_x
        else:
            chain = bot_chain
            bus_x = bot_bus_x
        
        # Get the bus wire span
        idx = chain.index(bus_id) if bus_id in chain else 0
        wx_left, wx_right = bus_wire_span(idx, chain, bus_x)
        bus_width = wx_right - wx_left
        
        # Attach tap legs at BAY_OFFSET_RATIO from left edge of bus segment
        tap_attach_x = wx_left + bus_width * BAY_OFFSET_RATIO
        tap_y = top_y if side == "top" else bot_y

        offsets = [(i - (n - 1) / 2.0) * LEG_COL_PITCH for i in range(n)]

        for leg, x_off in zip(sorted(legs, key=lambda l: l["terminal"]), offsets):
            term_id = leg["terminal"]
            iso_id  = leg["iso"]
            brk_id  = leg["brk"]
            goes_up = leg["goes_up"]
            lx      = tap_attach_x + x_off

            # Draw horizontal connection from bus center to tap point if offset
            centre_x = all_bus_x[bus_id]
            if abs(lx - centre_x) > 0.5:
                _ln(svg, centre_x, tap_y, lx, tap_y)
                _dot(svg, centre_x, tap_y, NODE_R)
                _dot(svg, lx, tap_y, NODE_R)

            if goes_up:
                iso_bot  = tap_y - LEG_GAP
                iso_top  = iso_bot - SYM
                brk_bot  = iso_top - LEG_SYM_GAP
                brk_top  = brk_bot - SYM
                term_bot = brk_top - LEG_SYM_GAP
                term_top = term_bot - SYM

                _ln(svg, lx, tap_y,   lx, iso_bot)
                _sym(svg, "isolator", lx, iso_top, SYM, rotate180=True)
                bbox_list.append((5, lx - SYM / 2, iso_top, SYM, SYM))
                _dot(svg, lx, iso_bot, NODE_R)
                _dot(svg, lx, iso_top, NODE_R)
                _lbl(svg, dev_label(iso_id),
                     lx + SYM / 2 + H_LBL_PAD,
                     iso_top + SYM / 2 + LBL_BASE_OFF)

                _ln(svg, lx, iso_top, lx, brk_bot)
                _sym(svg, "breaker", lx, brk_top, SYM)
                bbox_list.append((2, lx - SYM / 2, brk_top, SYM, SYM))
                _dot(svg, lx, brk_bot, NODE_R)
                _dot(svg, lx, brk_top, NODE_R)
                _lbl(svg, dev_label(brk_id),
                     lx + SYM / 2 + H_LBL_PAD,
                     brk_top + SYM / 2 + LBL_BASE_OFF)

                _ln(svg, lx, brk_top, lx, term_bot)

                end_pts[term_id] = {
                    "mid_x": lx,
                    "wire_end_y": term_bot,
                    "sk": _sym_key(devices[term_id]["type"]),
                    "goes_up": True,
                    "dev_type": devices[term_id]["type"],
                }

            else:
                iso_top  = tap_y + LEG_GAP
                iso_bot  = iso_top + SYM
                brk_top  = iso_bot + LEG_SYM_GAP
                brk_bot  = brk_top + SYM
                term_top = brk_bot + LEG_SYM_GAP
                term_bot = term_top + SYM

                _ln(svg, lx, tap_y,   lx, iso_top)
                _sym(svg, "isolator", lx, iso_top, SYM)
                bbox_list.append((5, lx - SYM / 2, iso_top, SYM, SYM))
                _dot(svg, lx, iso_top, NODE_R)
                _dot(svg, lx, iso_bot, NODE_R)
                _lbl(svg, dev_label(iso_id),
                     lx + SYM / 2 + H_LBL_PAD,
                     iso_top + SYM / 2 + LBL_BASE_OFF)

                _ln(svg, lx, iso_bot, lx, brk_top)
                _sym(svg, "breaker", lx, brk_top, SYM)
                bbox_list.append((2, lx - SYM / 2, brk_top, SYM, SYM))
                _dot(svg, lx, brk_top, NODE_R)
                _dot(svg, lx, brk_bot, NODE_R)
                _lbl(svg, dev_label(brk_id),
                     lx + SYM / 2 + H_LBL_PAD,
                     brk_top + SYM / 2 + LBL_BASE_OFF)

                _ln(svg, lx, brk_bot, lx, term_top)

                end_pts[term_id] = {
                    "mid_x": lx,
                    "wire_end_y": term_top,
                    "sk": _sym_key(devices[term_id]["type"]),
                    "goes_up": False,
                    "dev_type": devices[term_id]["type"],
                }

    return dict(
        svg=svg,
        end_pts=end_pts,
        devices=devices,
        dev_label=dev_label,
        SYM=SYM,
        MARGIN_X=MARGIN_X,
        MARGIN_Y=MARGIN_Y,
        H_LBL_PAD=H_LBL_PAD,
        LBL_BASE_OFF=LBL_BASE_OFF,
        bbox_list=bbox_list,
        CANVAS_W=left_x + RING_WIDTH + MARGIN_X,   # symmetric margins
        top_y=top_y,
        bot_y=bot_y,
        LEG_STACK=LEG_STACK,
        LEG_GAP=LEG_GAP,
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
    hybrid:     Feeder/GridSource -> symbol + label.
                Transformer       -> wire extended to SVG boundary, no symbol.

    Returns transformer boundary points (hybrid mode only):
      { device_id: {"label": str, "x": float, "y": float} }
    """
    svg          = core["svg"]
    SYM          = core["SYM"]
    H_LBL_PAD   = core["H_LBL_PAD"]
    LBL_BASE_OFF = core["LBL_BASE_OFF"]
    end_pts      = core["end_pts"]
    dev_label    = core["dev_label"]
    bbox_list    = core["bbox_list"]

    TERM_CLASS = {
        "GridSource": 0,
        "Transformer": 1,
        "Feeder": 3,
        "Capacitor": 4,
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
            # Draw wire from current endpoint to boundary
            _ln(svg, mid_x, wire_end_y, mid_x, boundary_y)
            boundary_pts[et] = {
                "label": dev_label(et),
                "x": round(mid_x, 1),
                "y": round(wire_end_y + y_offset, 1),
            }
        else:
            cls = TERM_CLASS.get(dev_type, 3)
            lx  = mid_x + SYM / 2 + H_LBL_PAD
            if goes_up:
                sym_top_y = wire_end_y - SYM
                _sym(svg, sk, mid_x, sym_top_y, SYM)
                bbox_list.append((cls, mid_x - SYM / 2, sym_top_y, SYM, SYM))
                label_reqs.append((lx, sym_top_y + SYM / 2 + LBL_BASE_OFF, dev_label(et), "start"))
            else:
                sym_top_y = wire_end_y
                _sym(svg, sk, mid_x, sym_top_y, SYM)
                bbox_list.append((cls, mid_x - SYM / 2, sym_top_y, SYM, SYM))
                label_reqs.append((lx, sym_top_y + SYM / 2 + LBL_BASE_OFF, dev_label(et), "start"))

    _place_labels(svg, label_reqs)
    return boundary_pts


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def render_ring_bus_svg(
    topology_path: str,
    output_path: str,
    mode: str = "standalone",
) -> dict | None:

    import json

    with open(topology_path) as f:
        topology_json = json.load(f)

    layout = topology_json.get("layout", {})
    topo_type = layout.get("topology_type") or layout.get("mode", "")

    if topo_type != "RING_BUS":
        raise ValueError(f"Expected RING_BUS, got '{topo_type}'")

    core = _render_core(topology_json)

    svg       = core["svg"]
    SYM       = core["SYM"]
    MARGIN_Y  = core["MARGIN_Y"]
    end_pts   = core["end_pts"]
    top_y     = core["top_y"]
    bot_y     = core["bot_y"]
    LEG_STACK = core["LEG_STACK"]
    LEG_GAP   = core["LEG_GAP"]

    up_ys      = [ep["wire_end_y"] - SYM for ep in end_pts.values() if ep["goes_up"]]
    down_ys    = [ep["wire_end_y"] + SYM  for ep in end_pts.values() if not ep["goes_up"]]
    top_extent = min([top_y] + up_ys)   - MARGIN_Y
    bot_extent = max([bot_y] + down_ys) + MARGIN_Y

    y_offset = -top_extent
    W = core["CANVAS_W"]
    H = bot_extent - top_extent

    boundary_pts = _cap(core, mode, y_offset, H)

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
        print("Usage: python render_ring_bs.py <topology.json> <output.svg> [standalone|hybrid]")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        topo = json.load(fh)
    m = sys.argv[3] if len(sys.argv) > 3 else "standalone"
    render_ring_bus_svg(topo, sys.argv[2], mode=m)
    print(f"Written: {sys.argv[2]} ({m})")