"""
topology.py
===========
Factory functions for generating electrical substation topology graphs.

Each function constructs a set of Devices and directed edges representing
a specific bus arrangement (single bus, double bus, ring bus, mesh, etc.).
Results are returned as plain Python sets, or serialised to JSON when a
filepath is supplied.

Public API
----------
breaker_and_half(...)       - Breaker-and-a-half topology
double_bus(...)             - Double-bus (single or double breaker per bay)
main_and_transfer(...)      - Main-and-transfer bus topology
mesh(...)                   - Mesh topology (configurable bus count)
ring_bus(...)               - Ring-bus topology
single_bus(...)             - Single-bus topology
generate_display_labels(...) - Human-readable label map for a device set
topology_to_json(...)       - Serialise a topology graph to JSON
"""

import json
import random
from collections import defaultdict

from core.grammar import Device, DeviceType, BusRole


# ---------------------------------------------------------------------------
# Breaker-and-a-half
# ---------------------------------------------------------------------------

def breaker_and_half(
    voltage_level,
    bayCount=3,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None,
):
    """
    Generate a breaker-and-a-half topology graph.

    Each bay contains three breakers (BRK_x_1, BRK_x_2, BRK_x_3) and two
    isolators (ISO_TOP_x, ISO_BOT_x) connecting the two main buses.

    Infeeds (upstream transformers then generators) are tapped onto the first
    breaker (BRK_x_1) of the first and last bays.  Outfeeds (downstream
    transformers then feeders) are tapped onto the middle breakers first,
    then onto any unused first breakers.

    Parameters
    ----------
    voltage_level : str | int
        Nominal voltage level shared by all devices in this topology.
    bayCount : int
        Number of breaker-and-a-half bays to generate.  Defaults to 2.
    num_upstream_transformers : int
        Number of upstream (HV-side) transformer infeeds.  Combined with
        ``num_generators``, the total may not exceed 2.
    num_generators : int
        Number of grid-source generator infeeds.
    num_downstream_transformers : int
        Number of downstream (LV-side) transformer outfeeds.  Capped at 6
        and must not exceed ``bayCount``.
    num_feeders : int
        Number of feeder outfeeds.
    filepath : str or None
        If supplied, the topology is serialised to this JSON file and the
        function returns ``None``.  If ``None``, the raw data structures are
        returned instead.

    Returns
    -------
    tuple or None
        ``(devices, edges, terminals, exposed_terminals)`` when ``filepath``
        is ``None``; otherwise ``None`` (result is written to disk).

    Raises
    ------
    ValueError
        If any capacity constraint is violated.
    """

    devices = set()
    edges = set()
    terminals = set()
    exposed_terminals = {"upstream": [], "downstream": []}

    TOPOLOGY_NAME = "BREAKER_AND_HALF"

    # ---------------- VALIDATION ----------------
    if bayCount < 3:
        raise ValueError("Minimum 3 bays required")
    
    if (num_upstream_transformers + num_generators) > 2:
        raise ValueError("Maximum 2 incomers allowed")

    if num_downstream_transformers > 6:
        raise ValueError("Number of downstream transformers is capped at 6")

    if num_downstream_transformers > bayCount:
        raise ValueError("Downstream transformers exceed bay count")

    # ---------------- BUSES ----------------
    bus1 = Device("BUS_001", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    bus2 = Device("BUS_002", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    devices.update([bus1, bus2])

    # ---------------- BAY CORE ----------------
    # Each bay: bus1 → ISO_TOP → BRK_1 → BRK_2 → BRK_3 ← ISO_BOT ← bus2
    bay_breakers = []

    for i in range(bayCount):
        bay_id = f"BAY_{i + 1}"

        brk1 = Device(f"BRK_{i+1}_1", DeviceType.BREAKER,  voltage_level, None, bay_id)
        brk2 = Device(f"BRK_{i+1}_2", DeviceType.BREAKER,  voltage_level, None, bay_id)
        brk3 = Device(f"BRK_{i+1}_3", DeviceType.BREAKER,  voltage_level, None, bay_id)
        devices.update([brk1, brk2, brk3])

        iso_top = Device(f"ISO_TOP_{i+1}", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        iso_bot = Device(f"ISO_BOT_{i+1}", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        devices.update([iso_top, iso_bot])

        edges.add((bus1, iso_top))
        edges.add((iso_top, brk1))

        edges.add((bus2, iso_bot))
        edges.add((iso_bot, brk3))

        edges.add((brk1, brk2))
        edges.add((brk2, brk3))

        bay_breakers.append((brk1, brk2, brk3))

    # ---------------- INFEED TARGET SELECTION ----------------
    # Infeeds tap onto BRK_1 of the first bay, then BRK_1 of the last bay.
    infeed_targets = []
    used_breaker1 = set()

    if bayCount >= 1:
        infeed_targets.append(bay_breakers[0][0])   # Bay 1  - first breaker

    if bayCount >= 2:
        infeed_targets.append(bay_breakers[-1][0])  # Last bay - first breaker

    # ---------------- INFEEDS ----------------
    uid = 1
    idx = 0

    for _ in range(num_upstream_transformers):
        if idx >= len(infeed_targets):
            break

        target = infeed_targets[idx]
        tr = Device(f"TR_UP_{uid}", DeviceType.TRANSFORMER, voltage_level, None, None)

        devices.add(tr)
        edges.add((tr, target))
        used_breaker1.add(target)
        exposed_terminals["upstream"].append(tr.device_id)
        terminals.add(tr.device_id)

        idx += 1
        uid += 1

    for _ in range(num_generators):
        if idx >= len(infeed_targets):
            break

        target = infeed_targets[idx]
        gs = Device(f"GS_{uid}", DeviceType.GRID_SOURCE, voltage_level, None, None)

        devices.add(gs)
        terminals.add(gs.device_id)
        edges.add((gs, target))
        used_breaker1.add(target)

        idx += 1
        uid += 1

    # ---------------- OUTFEED TARGET ORDERING ----------------
    # Priority 1: middle breakers (BRK_2) of every bay, left to right.
    # Priority 2: remaining first breakers (BRK_1) not claimed by an infeed.
    outfeed_targets = []

    for i in range(bayCount):
        _, brk2, _ = bay_breakers[i]
        outfeed_targets.append(brk2)

    for i in range(bayCount):
        brk1, _, _ = bay_breakers[i]
        if brk1 not in used_breaker1:
            outfeed_targets.append(brk1)

    # ---------------- OUTFEEDS ----------------
    did = 1
    idx = 0

    for _ in range(num_downstream_transformers):
        if idx >= len(outfeed_targets):
            break

        tr = Device(f"TR_DN_{did}", DeviceType.TRANSFORMER, voltage_level, None, None)

        devices.add(tr)
        edges.add((outfeed_targets[idx], tr))
        exposed_terminals["downstream"].append(tr.device_id)
        terminals.add(tr.device_id)

        idx += 1
        did += 1

    for _ in range(num_feeders):
        if idx >= len(outfeed_targets):
            break

        fd = Device(f"FDR_{did}", DeviceType.FEEDER, voltage_level, None, None)

        devices.add(fd)
        terminals.add(fd.device_id)
        edges.add((outfeed_targets[idx], fd))

        idx += 1
        did += 1

    # ---------------- OUTPUT ----------------
    if filepath is not None:
        topology_to_json(
            devices=devices,
            edges=edges,
            terminals=terminals,
            exposed_terminals=exposed_terminals,
            filepath=filepath,
            topology=TOPOLOGY_NAME,
        )
    else:
        return devices, edges, terminals, exposed_terminals


# ---------------------------------------------------------------------------
# Double-bus
# ---------------------------------------------------------------------------

def double_bus(
    voltage_level,
    breaker_per_bay,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None,
):
    """
    Generate a double-bus topology graph.

    Two main buses run in parallel.  Each bay has an isolator on each bus
    and either one shared breaker (``breaker_per_bay == 1``) or two
    individual breakers (``breaker_per_bay == 2``).  Infeeds are placed in
    the leftmost bays; outfeeds fill the remaining bays to the right.

    Parameters
    ----------
    voltage_level : str | int
        Nominal voltage level shared by all devices.
    breaker_per_bay : int
        Number of breakers per bay: ``1`` (single breaker) or ``2``
        (double breaker, one per bus).
    num_upstream_transformers : int
        Number of upstream transformer infeeds.  Capped at 6.
    num_generators : int
        Number of grid-source generator infeeds.
    num_downstream_transformers : int
        Number of downstream transformer outfeeds.  Capped at 6.
    num_feeders : int
        Number of feeder outfeeds.
    filepath : str or None
        If supplied, the result is written to this JSON file and ``None`` is
        returned.  Otherwise the raw data structures are returned.

    Returns
    -------
    tuple or None
        ``(devices, edges, terminals, exposed_terminals)`` when ``filepath``
        is ``None``; otherwise ``None``.

    Raises
    ------
    ValueError
        If any capacity constraint is violated.
    """

    devices = set()
    edges = set()
    terminals = set()
    exposed_terminals = {"upstream": [], "downstream": []}

    TOPOLOGY_NAME = "DOUBLE_BUS"

    # ---------------- VALIDATION ----------------
    if num_upstream_transformers > 6:
        raise ValueError("Number of upstream transformers is capped at 6")

    if num_downstream_transformers > 6:
        raise ValueError("Number of downstream transformers is capped at 6")

    if breaker_per_bay not in [1, 2]:
        raise ValueError("Either 1 or 2 breakers per bay required")
    # ---------------- BUSES ----------------
    bus1 = Device("BUS_001", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    bus2 = Device("BUS_002", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    devices.update([bus1, bus2])

    total_in = num_upstream_transformers + num_generators
    total_out = num_downstream_transformers + num_feeders
    total_bays = total_in + total_out

    # Incoming bays occupy the leftmost positions.
    incoming_positions = list(range(1, total_in + 1))

    upstream_remaining  = num_upstream_transformers
    generator_remaining = num_generators
    dn_tr_remaining     = num_downstream_transformers
    feeder_remaining    = num_feeders

    # ---------------- BAY GENERATION ----------------
    for bay_index in range(1, total_bays + 1):
        bay_id = f"BAY_{bay_index}"

        iso_a = Device(f"ISO_A_{bay_index}", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        iso_b = Device(f"ISO_B_{bay_index}", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        devices.update([iso_a, iso_b])

        edges.add((bus1, iso_a))
        edges.add((bus2, iso_b))

        # Build the breaker arrangement and record the attachment point(s).
        if breaker_per_bay == 1:
            brk = Device(f"BRK_{bay_index}_1", DeviceType.BREAKER, voltage_level, None, bay_id)
            devices.add(brk)
            edges.add((iso_a, brk))
            edges.add((iso_b, brk))
            terminal_attach = brk
        else:
            brk_a = Device(f"BRK_{bay_index}_A", DeviceType.BREAKER, voltage_level, None, bay_id)
            brk_b = Device(f"BRK_{bay_index}_B", DeviceType.BREAKER, voltage_level, None, bay_id)
            devices.update([brk_a, brk_b])
            edges.add((iso_a, brk_a))
            edges.add((iso_b, brk_b))
            terminal_attach = (brk_a, brk_b)

        # ---------------- INCOMING DEVICES ----------------
        if bay_index in incoming_positions:

            if upstream_remaining > 0:
                tr = Device(f"TR_UP_{upstream_remaining}", DeviceType.TRANSFORMER, voltage_level, None, None)
                devices.add(tr)

                if breaker_per_bay == 1:
                    edges.add((tr, terminal_attach))
                else:
                    edges.add((terminal_attach[0], tr))
                    edges.add((terminal_attach[1], tr))

                exposed_terminals["upstream"].append(tr.device_id)
                terminals.add(tr.device_id)
                upstream_remaining -= 1

            elif generator_remaining > 0:
                gs = Device(f"GS_{generator_remaining}", DeviceType.GRID_SOURCE, voltage_level, None, None)
                devices.add(gs)
                terminals.add(gs.device_id)

                if breaker_per_bay == 1:
                    edges.add((gs, terminal_attach))
                else:
                    edges.add((terminal_attach[0], gs))
                    edges.add((terminal_attach[1], gs))

                generator_remaining -= 1

        # ---------------- DOWNSTREAM TRANSFORMERS ----------------
        elif dn_tr_remaining > 0:
            tr = Device(f"TR_DN_{dn_tr_remaining}", DeviceType.TRANSFORMER, voltage_level, None, None)
            devices.add(tr)

            if breaker_per_bay == 1:
                edges.add((terminal_attach, tr))
            else:
                edges.add((terminal_attach[0], tr))
                edges.add((terminal_attach[1], tr))

            exposed_terminals["downstream"].append(tr.device_id)
            terminals.add(tr.device_id)
            dn_tr_remaining -= 1

        # ---------------- FEEDERS ----------------
        elif feeder_remaining > 0:
            fd = Device(f"FDR_{feeder_remaining}", DeviceType.FEEDER, voltage_level, None, None)
            devices.add(fd)
            terminals.add(fd.device_id)

            if breaker_per_bay == 1:
                edges.add((terminal_attach, fd))
            else:
                edges.add((terminal_attach[0], fd))
                edges.add((terminal_attach[1], fd))

            feeder_remaining -= 1

    # ---------------- OUTPUT ----------------
    if filepath is not None:
        topology_to_json(
            devices=devices,
            edges=edges,
            terminals=terminals,
            exposed_terminals=exposed_terminals,
            filepath=filepath,
            topology=TOPOLOGY_NAME,
            breakers_per_bay=breaker_per_bay,
        )
    else:
        return devices, edges, terminals, exposed_terminals


# ---------------------------------------------------------------------------
# Main-and-transfer bus
# ---------------------------------------------------------------------------

def main_and_transfer(
    voltage_level,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None,
):
    """
    Generate a main-and-transfer bus topology graph.

    Exactly two incomers (upstream transformers and/or generators) feed a
    MAIN bus and a TRANSFER bus that are coupled through a single bus-coupler
    breaker.  Each incomer/outgoer chain is modelled as its own bay
    (ISO → BRK → ISO).

    Load is distributed roughly evenly between the two buses: if the total
    outgoer count is odd the extra bay is assigned to the transfer bus.

    Parameters
    ----------
    voltage_level : str | int
        Nominal voltage level shared by all devices.
    num_upstream_transformers : int
        Number of upstream transformer infeeds.  Together with
        ``num_generators``, the total must equal exactly 2.
    num_generators : int
        Number of grid-source generator infeeds.
    num_downstream_transformers : int
        Number of downstream transformer outfeeds.  Capped at 6.
    num_feeders : int
        Number of feeder outfeeds.
    filepath : str or None
        If supplied, the result is serialised to this JSON path.

    Returns
    -------
    tuple or None
        ``(devices, edges, terminals, exposed_terminals)`` when ``filepath``
        is ``None``; otherwise ``None``.

    Raises
    ------
    ValueError
        If the incomer count is not exactly 2, or any cap is exceeded.
    """

    devices = set()
    edges = set()
    terminals = set()
    exposed_terminals = {"upstream": [], "downstream": []}

    TOPOLOGY_NAME = "MAIN_AND_TRANSFER"

    # ---------------- VALIDATION ----------------
    if (num_upstream_transformers + num_generators) != 2:
        raise ValueError("Exactly two incomers allowed")

    if num_downstream_transformers > 6:
        raise ValueError("Number of downstream transformers is capped at 6")

    # ---------------- BUSES ----------------
    bus_main     = Device("BUS_MAIN",     DeviceType.BUS,     voltage_level, BusRole.MAIN,     None)
    bus_transfer = Device("BUS_TRANSFER", DeviceType.BUS,     voltage_level, BusRole.TRANSFER, None)
    bus_coupler  = Device("BRK_BUS_COUPLER", DeviceType.BREAKER, voltage_level, None, "BAY_COUPLER")

    devices.update([bus_main, bus_transfer, bus_coupler])
    edges.add((bus_main, bus_coupler))
    edges.add((bus_coupler, bus_transfer))

    # ---------------- INCOMERS ----------------
    # First incomer connects to MAIN bus; second incomer to TRANSFER bus.
    incomer_targets = [bus_main, bus_transfer]

    uid = 1
    idx = 0

    def add_incomer(source_device, bus, tag):
        """Wire source_device into the given bus through ISO → BRK → ISO."""
        bay_id = f"BAY_{tag}"

        iso1 = Device(f"ISO_IN_{tag}_A", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        brk  = Device(f"BRK_IN_{tag}",   DeviceType.BREAKER,  voltage_level, None, bay_id)
        iso2 = Device(f"ISO_IN_{tag}_B", DeviceType.ISOLATOR, voltage_level, None, bay_id)

        devices.update([iso1, brk, iso2])
        edges.add((source_device, iso1))
        edges.add((iso1, brk))
        edges.add((brk, iso2))
        edges.add((iso2, bus))

    for _ in range(num_upstream_transformers):
        if idx >= len(incomer_targets):
            break

        tr = Device(f"TR_UP_{uid}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(tr)
        add_incomer(tr, incomer_targets[idx], uid)
        exposed_terminals["upstream"].append(tr.device_id)
        terminals.add(tr.device_id)

        idx += 1
        uid += 1

    for _ in range(num_generators):
        if idx >= len(incomer_targets):
            break

        gs = Device(f"GS_{uid}", DeviceType.GRID_SOURCE, voltage_level, None, None)
        devices.add(gs)
        terminals.add(gs.device_id)
        add_incomer(gs, incomer_targets[idx], uid)

        idx += 1
        uid += 1

    # ---------------- LOAD DISTRIBUTION ----------------
    # Split outgoers evenly; any odd remainder goes to the right (transfer) side.
    total_outgoing = num_downstream_transformers + num_feeders

    lhs_count = total_outgoing // 2
    rhs_count = total_outgoing - lhs_count

    if total_outgoing % 2 == 1:
        lhs_count -= 1
        rhs_count += 1

    out_targets = [bus_main] * lhs_count + [bus_transfer] * rhs_count

    # ---------------- OUTGOERS ----------------
    did = 1
    idx = 0

    def add_outgoer(load_device, bus, tag):
        """Wire load_device out of bus through ISO → BRK → ISO.

        Bay IDs are offset by 100 to avoid collisions with incomer bay IDs.
        """
        bay_id = f"BAY_{tag + 100}"

        iso1 = Device(f"ISO_OUT_{tag}_A", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        brk  = Device(f"BRK_OUT_{tag}",   DeviceType.BREAKER,  voltage_level, None, bay_id)
        iso2 = Device(f"ISO_OUT_{tag}_B", DeviceType.ISOLATOR, voltage_level, None, bay_id)

        devices.update([iso1, brk, iso2])
        edges.add((bus, iso1))
        edges.add((iso1, brk))
        edges.add((brk, iso2))
        edges.add((iso2, load_device))

    for _ in range(num_downstream_transformers):
        if idx >= len(out_targets):
            break

        tr = Device(f"TR_DN_{did}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(tr)
        add_outgoer(tr, out_targets[idx], did)
        exposed_terminals["downstream"].append(tr.device_id)
        terminals.add(tr.device_id)

        idx += 1
        did += 1

    for _ in range(num_feeders):
        if idx >= len(out_targets):
            break

        fd = Device(f"FDR_{did}", DeviceType.FEEDER, voltage_level, None, None)
        devices.add(fd)
        terminals.add(fd.device_id)
        add_outgoer(fd, out_targets[idx], did)

        idx += 1
        did += 1

    # ---------------- OUTPUT ----------------
    if filepath is not None:
        topology_to_json(
            devices=devices,
            edges=edges,
            terminals=terminals,
            exposed_terminals=exposed_terminals,
            filepath=filepath,
            topology=TOPOLOGY_NAME,
        )
    else:
        return devices, edges, terminals, exposed_terminals


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def mesh(
    voltage_level,
    num_buses,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None,
):
    """
    Generate a mesh topology graph.

    Buses are arranged in a rectangular grid (rows × cols) and connected
    horizontally and vertically.  Additional inter-bus breakers are added
    at random until the per-configuration breaker target is reached.

    Infeeds attach to the top row of buses (left to right); outfeeds attach
    to the bottom row (left to right, wrapping if needed).

    Supported bus counts and their constraints
    ------------------------------------------
    ======  ========  ===========  ===========
    Buses   Breakers  Max infeeds  Max outfeeds
    ======  ========  ===========  ===========
      6        7          6            6
      8       10          8            8
      9       12          6            6
     10       13         10           10
     12       16         12           12
     15       22         10           10
    ======  ========  ===========  ===========

    Parameters
    ----------
    voltage_level : str | int
        Nominal voltage level shared by all devices.
    num_buses : int
        Total number of buses.  Must be one of {6, 8, 9, 10, 12, 15}.
    num_upstream_transformers : int
        Number of upstream transformer infeeds.  Capped at 6 and at the
        per-configuration ``max_infeeds`` value.
    num_generators : int
        Number of grid-source generator infeeds.
    num_downstream_transformers : int
        Number of downstream transformer outfeeds.  Capped at 6 and at the
        per-configuration ``max_outfeeds`` value.
    num_feeders : int
        Number of feeder outfeeds.
    filepath : str or None
        If supplied, the result is serialised to this JSON path.

    Returns
    -------
    tuple or None
        ``(devices, edges, terminals, exposed_terminals, BUS_POSITIONS)``
        when ``filepath`` is ``None``; otherwise ``None``.

    Raises
    ------
    ValueError
        If ``num_buses`` is unsupported or any capacity constraint is exceeded.
    """

    # ---------------- CONFIGURATION TABLE ----------------
    BUS_CONFIG = {
        6:  {"breakers": 7,  "max_infeeds": 6,  "max_outfeeds": 6},
        8:  {"breakers": 10, "max_infeeds": 8,  "max_outfeeds": 8},
        9:  {"breakers": 12, "max_infeeds": 6,  "max_outfeeds": 6},
        10: {"breakers": 13, "max_infeeds": 10, "max_outfeeds": 10},
        12: {"breakers": 16, "max_infeeds": 12, "max_outfeeds": 12},
        15: {"breakers": 22, "max_infeeds": 10, "max_outfeeds": 10},
    }

    # ---------------- VALIDATION ----------------
    if num_buses not in BUS_CONFIG:
        raise ValueError(f"num_buses must be one of {set(BUS_CONFIG.keys())}")

    config = BUS_CONFIG[num_buses]

    if (num_upstream_transformers + num_generators) > config["max_infeeds"]:
        raise ValueError(
            f"For {num_buses} buses, maximum infeeds allowed is {config['max_infeeds']}"
        )

    if (num_downstream_transformers + num_feeders) > config["max_outfeeds"]:
        raise ValueError(
            f"For {num_buses} buses, maximum outfeeds allowed is {config['max_outfeeds']}"
        )

    if num_upstream_transformers > 6:
        raise ValueError("Number of upstream transformers is capped at 6")

    if num_downstream_transformers > 6:
        raise ValueError("Number of downstream transformers is capped at 6")

    # ---------------- BUSES ----------------
    devices = set()
    edges = set()
    terminals = set()
    exposed_terminals = {"upstream": [], "downstream": []}

    TOPOLOGY_NAME = "MESH"

    target_breakers = config["breakers"]

    buses = []
    for i in range(num_buses):
        bus = Device(f"BUS_{i+1:03d}", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
        buses.append(bus)
        devices.add(bus)

    # ---------------- GRID FACTORING ----------------
    # Prefer a wider grid (more columns) for layout aesthetics.
    def factor_grid(n):
        """Return (rows, cols) that factor n, favouring more columns."""
        candidates = []
        for r in (2, 3):
            if n % r == 0:
                candidates.append((r, n // r))
        return max(candidates, key=lambda x: x[1])

    rows, cols = factor_grid(num_buses)

    def idx(r, c):
        """Flatten a (row, col) position to a linear bus index."""
        return r * cols + c

    # ---------------- BUS POSITIONS ----------------
    # Canvas coordinates used downstream for rendering.
    BUS_POSITIONS = {}
    spacing_x, spacing_y = 220, 100

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            if i >= len(buses):
                break
            BUS_POSITIONS[buses[i].device_id] = (c * spacing_x, r * spacing_y)

    # ---------------- INTERNAL GRID CONNECTIONS ----------------
    brk_id = 1
    connected_pairs = set()

    def add_bus_link(b1, b2):
        """Insert a mesh breaker between two buses (no duplicates)."""
        nonlocal brk_id

        key = tuple(sorted((b1.device_id, b2.device_id)))
        if key in connected_pairs:
            return False

        brk = Device(f"BRK_M_{brk_id}", DeviceType.BREAKER, voltage_level, None, "MESH")
        devices.add(brk)
        edges.add((b1, brk))
        edges.add((brk, b2))
        connected_pairs.add(key)
        brk_id += 1
        return True

    # Horizontal links (within each row)
    for r in range(rows):
        for c in range(cols - 1):
            add_bus_link(buses[idx(r, c)], buses[idx(r, c + 1)])

    # Vertical links (within each column)
    for c in range(cols):
        for r in range(rows - 1):
            add_bus_link(buses[idx(r, c)], buses[idx(r + 1, c)])

    # Additional random links to meet the target breaker count
    all_pairs = [
        (buses[i], buses[j])
        for i in range(len(buses))
        for j in range(i + 1, len(buses))
    ]
    random.shuffle(all_pairs)

    for b1, b2 in all_pairs:
        if brk_id > target_breakers:
            break
        add_bus_link(b1, b2)

    # ---------------- ROW CLASSIFICATION ----------------
    top_row_buses    = buses[0:cols]
    bottom_row_buses = buses[(rows - 1) * cols: rows * cols]

    # ---------------- ATTACH ORDER (LEFT → RIGHT) ----------------
    infeed_attach_points  = list(top_row_buses)
    outfeed_attach_points = list(bottom_row_buses)

    # ---------------- INFEEDS ----------------
    uid = 1
    aidx_in = 0

    def add_infeed(dev):
        """Attach an infeed device to the next available top-row bus."""
        nonlocal uid, aidx_in

        bus    = infeed_attach_points[aidx_in % len(infeed_attach_points)]
        bay_id = f"BAY_{uid}"

        iso = Device(f"ISO_IN_{uid}", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        brk = Device(f"BRK_IN_{uid}", DeviceType.BREAKER,  voltage_level, None, bay_id)

        devices.update([iso, brk])
        edges.add((dev, iso))
        edges.add((iso, brk))
        edges.add((brk, bus))

        uid += 1
        aidx_in += 1

    for _ in range(num_upstream_transformers):
        tr = Device(f"TR_UP_{uid}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(tr)
        add_infeed(tr)
        exposed_terminals["upstream"].append(tr.device_id)
        terminals.add(tr.device_id)

    for _ in range(num_generators):
        gs = Device(f"GS_{uid}", DeviceType.GRID_SOURCE, voltage_level, None, None)
        devices.add(gs)
        terminals.add(gs.device_id)
        add_infeed(gs)

    # ---------------- OUTFEEDS ----------------
    did = 1
    aidx_out = 0

    def add_outfeed(dev):
        """Attach an outfeed device to the next available bottom-row bus."""
        nonlocal did, aidx_out

        bus    = outfeed_attach_points[aidx_out % len(outfeed_attach_points)]
        bay_id = f"BAY_{100 + did}"

        brk = Device(f"BRK_OUT_{did}", DeviceType.BREAKER,  voltage_level, None, bay_id)
        iso = Device(f"ISO_OUT_{did}", DeviceType.ISOLATOR, voltage_level, None, bay_id)

        devices.update([iso, brk])
        edges.add((bus, brk))
        edges.add((brk, iso))
        edges.add((iso, dev))

        did += 1
        aidx_out += 1

    for _ in range(num_downstream_transformers):
        tr = Device(f"TR_DN_{did}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(tr)
        add_outfeed(tr)
        exposed_terminals["downstream"].append(tr.device_id)
        terminals.add(tr.device_id)

    for _ in range(num_feeders):
        fd = Device(f"FDR_{did}", DeviceType.FEEDER, voltage_level, None, None)
        devices.add(fd)
        terminals.add(fd.device_id)
        add_outfeed(fd)

    # ---------------- OUTPUT ----------------
    if filepath is not None:
        topology_to_json(
            devices=devices,
            edges=edges,
            terminals=terminals,
            exposed_terminals=exposed_terminals,
            filepath=filepath,
            topology=TOPOLOGY_NAME,
            bus_positions=BUS_POSITIONS,
        )
    else:
        return devices, edges, terminals, exposed_terminals, BUS_POSITIONS


# ---------------------------------------------------------------------------
# Ring-bus
# ---------------------------------------------------------------------------

def ring_bus(
    voltage_level,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None,
):
    """
    Generate a ring-bus topology graph.

    Buses are arranged in two rows - infeeds attach to top-row buses (left
    to right: transformers first, then generators) and outfeeds attach to
    bottom-row buses (left to right: feeders first, then downstream
    transformers).  Section breakers link adjacent buses around the ring,
    with left and right vertical breakers closing the loop.

    A ``ring_geometry`` metadata dict is returned (or embedded in the JSON)
    to assist rendering: it records orientation and side for each breaker,
    and side for each bus.

    Parameters
    ----------
    voltage_level : str | int
        Nominal voltage level shared by all devices.
    num_upstream_transformers : int
        Number of upstream transformer infeeds.  Capped at 6.
    num_generators : int
        Number of grid-source generator infeeds.
    num_downstream_transformers : int
        Number of downstream transformer outfeeds.  Capped at 6.
    num_feeders : int
        Number of feeder outfeeds.
    filepath : str or None
        If supplied, the result is serialised to this JSON path.

    Returns
    -------
    tuple or None
        ``(devices, edges, terminals, exposed_terminals, ring_geometry)``
        when ``filepath`` is ``None``; otherwise ``None``.

    Raises
    ------
    ValueError
        If any capacity constraint is violated, or if the total circuit
        count is less than 3.
    """

    # ---------------- VALIDATION ----------------
    if num_upstream_transformers > 6:
        raise ValueError("Number of upstream transformers is capped at 6")

    if num_downstream_transformers > 6:
        raise ValueError("Number of downstream transformers is capped at 6")

    devices = set()
    edges = set()
    terminals = set()
    exposed_terminals = {"upstream": [], "downstream": []}

    TOPOLOGY_NAME = "RING_BUS"

    # ---------------- CIRCUIT COUNTS ----------------
    num_infeeds  = num_generators + num_upstream_transformers
    num_outfeeds = num_downstream_transformers + num_feeders
    total_buses  = num_infeeds + num_outfeeds

    if total_buses < 3:
        raise ValueError("Ring bus requires at least 3 circuits")

    # ---------------- HELPERS ----------------
    bus_counter = 0
    brk_counter = 0

    buses               = []
    breaker_orientation = {}   # device_id → "horizontal" | "vertical"
    breaker_side        = {}   # device_id → "top" | "bottom" | "left" | "right"
    bus_side            = {}   # device_id → "top" | "bottom"

    def new_bus(side):
        """Create a RING-role bus and register it."""
        nonlocal bus_counter
        bus_counter += 1
        b = Device(
            f"BUS_R_{bus_counter:03d}",
            DeviceType.BUS,
            voltage_level,
            BusRole.RING,
            None,
        )
        devices.add(b)
        buses.append(b)
        bus_side[b.device_id] = side
        return b

    def new_ring_breaker(orientation, side):
        """Create a ring section breaker (not part of any bay)."""
        nonlocal brk_counter
        brk_counter += 1
        brk = Device(
            f"BRK_R_{brk_counter:03d}",
            DeviceType.BREAKER,
            voltage_level,
            None,
            "RING",
        )
        devices.add(brk)
        breaker_orientation[brk.device_id] = orientation
        breaker_side[brk.device_id]        = side
        return brk

    def connect(b1, brk, b2):
        """Add both edges for a bus-breaker-bus section."""
        edges.add((b1, brk))
        edges.add((brk, b2))

    # ---------------- BUS GROUPS ----------------
    top_buses    = [new_bus("top")    for _ in range(num_infeeds)]
    bottom_buses = [new_bus("bottom") for _ in range(num_outfeeds)]

    # ---------------- RING WIRING ----------------
    # Left vertical breaker → top row (left to right) → right vertical
    # breaker → bottom row (right to left) → back to left vertical breaker.
    brk_left = new_ring_breaker("vertical", "left")
    edges.add((brk_left, top_buses[0]))

    prev_bus = top_buses[0]
    for i in range(1, len(top_buses)):
        brk = new_ring_breaker("horizontal", "top")
        connect(prev_bus, brk, top_buses[i])
        prev_bus = top_buses[i]

    brk_right = new_ring_breaker("vertical", "right")
    edges.add((prev_bus, brk_right))
    edges.add((brk_right, bottom_buses[0]))

    prev_bus = bottom_buses[0]
    for i in range(1, len(bottom_buses)):
        brk = new_ring_breaker("horizontal", "bottom")
        connect(prev_bus, brk, bottom_buses[i])
        prev_bus = bottom_buses[i]

    edges.add((prev_bus, brk_left))

    # ---------------- DEVICE ATTACHMENT ----------------
    # Each attached device forms its own bay (bus → ISO → tap BRK → device).
    bay_counter = 1

    def attach_device(dev, bus):
        """Attach dev to bus via an isolator and tap breaker."""
        nonlocal bay_counter

        bay_id = f"BAY_{bay_counter}"

        iso = Device(
            f"ISO_R_{dev.device_id}",
            DeviceType.ISOLATOR,
            voltage_level,
            None,
            bay_id,
        )
        tap = Device(
            f"BRK_TAP_{dev.device_id}",
            DeviceType.BREAKER,
            voltage_level,
            None,
            bay_id,
        )

        devices.update([iso, tap])
        edges.add((bus, iso))
        edges.add((iso, tap))
        edges.add((tap, dev))

        bay_counter += 1

    # ---------------- INFEEDS (left → right: transformers first) ----------------
    idx = 0

    for i in range(num_upstream_transformers):
        d = Device(f"TR_UP_{i+1}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(d)
        attach_device(d, top_buses[idx])
        exposed_terminals["upstream"].append(d.device_id)
        terminals.add(d.device_id)
        idx += 1

    for i in range(num_generators):
        d = Device(f"GS_{i+1}", DeviceType.GRID_SOURCE, voltage_level, None, None)
        devices.add(d)
        terminals.add(d.device_id)
        attach_device(d, top_buses[idx])
        idx += 1

    # ---------------- OUTFEEDS (left → right: feeders first) ----------------
    idx = 0

    for i in range(num_feeders):
        d = Device(f"FDR_{i+1}", DeviceType.FEEDER, voltage_level, None, None)
        devices.add(d)
        terminals.add(d.device_id)
        attach_device(d, bottom_buses[idx])
        idx += 1

    for i in range(num_downstream_transformers):
        d = Device(f"TR_DN_{i+1}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(d)
        attach_device(d, bottom_buses[idx])
        exposed_terminals["downstream"].append(d.device_id)
        terminals.add(d.device_id)
        idx += 1

    # ---------------- GEOMETRY METADATA ----------------
    ring_geometry = {
        "breaker_orientation": breaker_orientation,
        "breaker_side":        breaker_side,
        "bus_side":            bus_side,
    }

    # ---------------- OUTPUT ----------------
    if filepath is not None:
        topology_to_json(
            devices=devices,
            edges=edges,
            terminals=terminals,
            exposed_terminals=exposed_terminals,
            filepath=filepath,
            topology=TOPOLOGY_NAME,
            ring_geometry=ring_geometry,
        )
    else:
        return devices, edges, terminals, exposed_terminals, ring_geometry


# ---------------------------------------------------------------------------
# Single-bus
# ---------------------------------------------------------------------------

def single_bus(
    voltage_level,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None,
):
    """
    Generate a single-bus topology graph.

    All infeeds and outfeeds connect to a single main bus.  Each connection
    is protected by an isolator and a breaker.  Infeed bays use IDs starting
    at BAY_1; outfeed bays use IDs starting at BAY_101 to avoid collisions.

    Parameters
    ----------
    voltage_level : str | int
        Nominal voltage level shared by all devices.
    num_upstream_transformers : int
        Number of upstream transformer infeeds.  Capped at 6.
    num_generators : int
        Number of grid-source generator infeeds.
    num_downstream_transformers : int
        Number of downstream transformer outfeeds.  Capped at 6.
    num_feeders : int
        Number of feeder outfeeds.
    filepath : str or None
        If supplied, the result is serialised to this JSON path.

    Returns
    -------
    tuple or None
        ``(devices, edges, terminals, exposed_terminals)`` when ``filepath``
        is ``None``; otherwise ``None``.

    Raises
    ------
    ValueError
        If any capacity constraint is violated.
    """

    # ---------------- VALIDATION ----------------
    if num_upstream_transformers > 6:
        raise ValueError("Number of upstream transformers is capped at 6")

    if num_downstream_transformers > 6:
        raise ValueError("Number of downstream transformers is capped at 6")

    devices = set()
    edges = set()
    terminals = set()
    exposed_terminals = {"upstream": [], "downstream": []}

    TOPOLOGY_NAME = "SINGLE_BUS"

    # ---------------- BUS ----------------
    bus = Device("BUS_001", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    devices.add(bus)

    # ---------------- INFEEDS ----------------
    uid = 1

    def add_infeed(dev):
        """Wire dev into bus through ISO → BRK."""
        bay_id = f"BAY_{uid}"

        iso = Device(f"ISO_IN_{uid}", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        brk = Device(f"BRK_IN_{uid}", DeviceType.BREAKER,  voltage_level, None, bay_id)

        devices.update([iso, brk])
        edges.add((dev, iso))
        edges.add((iso, brk))
        edges.add((brk, bus))

    for _ in range(num_upstream_transformers):
        tr = Device(f"TR_UP_{uid}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(tr)
        add_infeed(tr)
        exposed_terminals["upstream"].append(tr.device_id)
        terminals.add(tr.device_id)
        uid += 1

    for _ in range(num_generators):
        gs = Device(f"GS_{uid}", DeviceType.GRID_SOURCE, voltage_level, None, None)
        devices.add(gs)
        terminals.add(gs.device_id)
        add_infeed(gs)
        uid += 1

    # ---------------- OUTFEEDS ----------------
    did = 1

    def add_outfeed(dev):
        """Wire bus out to dev through ISO → BRK.

        Bay IDs are offset by 100 to avoid collisions with infeed bay IDs.
        """
        bay_id = f"BAY_{100 + did}"

        iso = Device(f"ISO_OUT_{did}", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        brk = Device(f"BRK_OUT_{did}", DeviceType.BREAKER,  voltage_level, None, bay_id)

        devices.update([iso, brk])
        edges.add((bus, iso))
        edges.add((iso, brk))
        edges.add((brk, dev))

    for _ in range(num_downstream_transformers):
        tr = Device(f"TR_DN_{did}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(tr)
        add_outfeed(tr)
        exposed_terminals["downstream"].append(tr.device_id)
        terminals.add(tr.device_id)
        did += 1

    for _ in range(num_feeders):
        fd = Device(f"FDR_{did}", DeviceType.FEEDER, voltage_level, None, None)
        devices.add(fd)
        terminals.add(fd.device_id)
        add_outfeed(fd)
        did += 1

    # ---------------- OUTPUT ----------------
    if filepath is not None:
        topology_to_json(
            devices=devices,
            edges=edges,
            terminals=terminals,
            exposed_terminals=exposed_terminals,
            filepath=filepath,
            topology=TOPOLOGY_NAME,
        )
    else:
        return devices, edges, terminals, exposed_terminals


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def generate_display_labels(devices):
    """
    Build a human-readable label map for every device in a topology.

    Labels follow substation naming conventions:

    * Buses:       ``<voltage>_BUS_<A|B|C...>``
    * Breakers:    ``<voltage>_<bay>_CB<n>``
    * Isolators:   ``<voltage>_<bay>_DS<A|B|C...>``
    * Transformers: ``<voltage>-TR-<nn>``
    * Generators:  ``<voltage>-GEN-<nn>``
    * Feeders:     ``<voltage>-FDR-<nn>``

    Parameters
    ----------
    devices : iterable of Device
        The complete device set from a topology generator.

    Returns
    -------
    dict
        Mapping from ``device_id`` to display label string.
    """

    label_map = {}

    buses        = []
    bays         = defaultdict(list)
    transformers = []
    generators   = []
    feeders      = []

    for d in devices:
        if d.device_type == DeviceType.BUS:
            buses.append(d)
        elif d.bay_id:
            bays[d.bay_id].append(d)
        elif d.device_type == DeviceType.TRANSFORMER:
            transformers.append(d)
        elif d.device_type == DeviceType.GRID_SOURCE:
            generators.append(d)
        elif d.device_type == DeviceType.FEEDER:
            feeders.append(d)

    voltage_prefix = str(next(iter(devices)).voltage)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # ---------------- BUS LABELS ----------------
    buses = sorted(buses, key=lambda x: x.device_id)

    for i, bus in enumerate(buses):
        label_map[bus.device_id] = f"{voltage_prefix}_BUS_{alphabet[i]}"

    # ---------------- BAY DEVICES ----------------
    for bay_id, devs in bays.items():
        try:
            bay_index = int(bay_id.split("_")[1])
        except (IndexError, ValueError):
            bay_index = 0

        bay_label = f"{bay_index:02d}"

        breakers  = sorted(
            [d for d in devs if d.device_type == DeviceType.BREAKER],
            key=lambda x: x.device_id,
        )
        isolators = sorted(
            [d for d in devs if d.device_type == DeviceType.ISOLATOR],
            key=lambda x: x.device_id,
        )

        for i, brk in enumerate(breakers, start=1):
            label_map[brk.device_id] = f"{voltage_prefix}_{bay_label}_CB{i}"

        for i, iso in enumerate(isolators):
            label_map[iso.device_id] = f"{voltage_prefix}_{bay_label}_DS{alphabet[i]}"

    # ---------------- TRANSFORMERS ----------------
    for i, tr in enumerate(sorted(transformers, key=lambda x: x.device_id), start=1):
        label_map[tr.device_id] = f"{voltage_prefix}-TR-{i:02d}"

    # ---------------- GENERATORS ----------------
    for i, g in enumerate(sorted(generators, key=lambda x: x.device_id), start=1):
        label_map[g.device_id] = f"{voltage_prefix}-GEN-{i:02d}"

    # ---------------- FEEDERS ----------------
    for i, f in enumerate(sorted(feeders, key=lambda x: x.device_id), start=1):
        label_map[f.device_id] = f"{voltage_prefix}-FDR-{i:02d}"

    return label_map


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def topology_to_json(
    devices,
    edges,
    terminals,
    exposed_terminals,
    filepath,
    topology,
    breakers_per_bay=None,
    bus_positions=None,
    ring_geometry=None,
    topology_mode="STANDARD",
):
    """
    Serialise a topology graph to a JSON file.

    The output schema contains three top-level keys:

    * ``devices``     - sorted list of device objects
    * ``connections`` - list of directed edge objects
    * ``layout``      - topology metadata including terminal lists and any
                        topology-specific geometry

    Topology-specific additions:

    * ``DOUBLE_BUS`` - ``layout.breakersPerBay``
    * ``MESH``       - ``layout.bus_positions``
    * ``RING_BUS``   - ``layout.geometry``

    Parameters
    ----------
    devices : set of Device
        All devices in the topology.
    edges : set of tuple[Device, Device]
        Directed connections between devices.
    terminals : set of str
        Device IDs that serve as external connection points.
    exposed_terminals : dict
        ``{"upstream": [...], "downstream": [...]}`` terminal ID lists.
    filepath : str
        Destination file path for the JSON output.
    topology : str
        Topology name constant (e.g. ``"SINGLE_BUS"``, ``"MESH"``).
    breakers_per_bay : int or None
        Number of breakers per bay; required for ``DOUBLE_BUS`` topologies.
    bus_positions : dict or None
        ``{device_id: (x, y)}`` canvas coordinates; required for ``MESH``.
    ring_geometry : dict or None
        Rendering metadata; required for ``RING_BUS``.
    topology_mode : str
        Rendering mode hint.  Defaults to ``"STANDARD"``.
    """

    label_map = generate_display_labels(devices)

    def device_to_dict(d):
        return {
            "id":       d.device_id,
            "label":    label_map.get(d.device_id),
            "type":     d.device_type.value,
            "voltage":  str(d.voltage),
            "bus_role": d.bus_role.name if d.bus_role else None,
            "bay_id":   d.bay_id,
        }

    def edge_to_dict(a, b):
        return {"from": a.device_id, "to": b.device_id}

    data = {
        "devices": [
            device_to_dict(d)
            for d in sorted(devices, key=lambda x: x.device_id)
        ],
        "connections": [
            edge_to_dict(a, b)
            for a, b in edges
        ],
        "layout": {
            "topology_mode": topology_mode,
            "topology_type": topology,
            "terminals":     sorted(list(terminals)),
            "exposed_terminals": {
                "upstream":   sorted(exposed_terminals.get("upstream",   [])),
                "downstream": sorted(exposed_terminals.get("downstream", [])),
            },
        },
    }

    # Topology-specific layout extensions
    if topology == "MESH" and bus_positions:
        data["layout"]["bus_positions"] = bus_positions

    if topology == "DOUBLE_BUS":
        data["layout"]["breakersPerBay"] = breakers_per_bay

    if topology == "RING_BUS" and ring_geometry:
        data["layout"]["geometry"] = ring_geometry

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)