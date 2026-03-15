from core.grammar import Device, DeviceType, BusRole
import json, random
from collections import defaultdict

def breaker_and_half(
    voltage_level,
    bayCount=2,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None
):
    """
    Generate a breaker-and-a-half topology graph.
    """

    devices = set()
    edges = set()
    terminals = set()
    TOPOLOGY_NAME = "BREAKER_AND_HALF"
    exposed_terminals = {"upstream": [], "downstream": []}

    # ---------------- BUSES ----------------
    bus1 = Device("BUS_001", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    bus2 = Device("BUS_002", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    devices.update([bus1, bus2])

    # ---------------- VALIDATION ----------------
    if (num_upstream_transformers + num_generators) > 2:
        raise ValueError("Maximum 2 incomers allowed")

    # ---------------- BAY CORE ----------------
    bay_breakers = []

    for i in range(bayCount):

        bay_id = f"BAY_{i+1}"

        brk1 = Device(f"BRK_{i+1}_1", DeviceType.BREAKER, voltage_level, None, bay_id)
        brk2 = Device(f"BRK_{i+1}_2", DeviceType.BREAKER, voltage_level, None, bay_id)
        brk3 = Device(f"BRK_{i+1}_3", DeviceType.BREAKER, voltage_level, None, bay_id)

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

    # ---------------- TAP TARGET ORDERING ----------------

    used_breaker1 = set()

    # ---- Infeed targets ----
    infeed_targets = []

    if bayCount >= 1:
        infeed_targets.append(bay_breakers[0][0])   # Bay1 breaker1

    if bayCount >= 2:
        infeed_targets.append(bay_breakers[-1][0])  # Last bay breaker1

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

    # ---------------- OUTFEED TARGETS ----------------

    outfeed_targets = []

    # Step 1: breaker2 in all bays
    for i in range(bayCount):
        _, brk2, _ = bay_breakers[i]
        outfeed_targets.append(brk2)

    # Step 2: remaining breaker1s not used by infeeds
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

    if filepath is not None:
        topology_to_json(
            devices=devices, edges=edges,
            terminals=terminals, exposed_terminals=exposed_terminals,
            filepath=filepath, topology=TOPOLOGY_NAME,
        )
    else:
        return devices, edges, terminals, exposed_terminals

def double_bus(
    voltage_level,
    breaker_per_bay,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None
):
    devices = set()
    edges = set()
    terminals = set()
    TOPOLOGY_NAME = "DOUBLE_BUS"
    exposed_terminals = {"upstream": [], "downstream": []}

    # ---------------- BUSES ----------------
    bus1 = Device("BUS_001", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    bus2 = Device("BUS_002", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    devices.update([bus1, bus2])

    total_in = num_upstream_transformers + num_generators
    total_out = num_downstream_transformers + num_feeders
    total_bays = total_in + total_out

    # ---------------- INCOMING BAY ORDER ----------------
    incoming_total = total_in

    left = 1
    right = total_bays
    incoming_positions = []

    while len(incoming_positions) < incoming_total and left <= right:
        incoming_positions.append(left)
        if len(incoming_positions) >= incoming_total:
            break
        incoming_positions.append(right)
        left += 1
        right -= 1

    upstream_remaining = num_upstream_transformers
    generator_remaining = num_generators
    dn_tr_remaining = num_downstream_transformers
    feeder_remaining = num_feeders

    # ---------------- BAY GENERATION ----------------
    for bay_index in range(1, total_bays + 1):

        bay_id = f"BAY_{bay_index}"

        iso_a = Device(f"ISO_A_{bay_index}", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        iso_b = Device(f"ISO_B_{bay_index}", DeviceType.ISOLATOR, voltage_level, None, bay_id)
        devices.update([iso_a, iso_b])

        edges.add((bus1, iso_a))
        edges.add((bus2, iso_b))

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

    if filepath is not None:
        topology_to_json(
            devices=devices, edges=edges,
            terminals=terminals, exposed_terminals=exposed_terminals,
            filepath=filepath, topology=TOPOLOGY_NAME,
            breakers_per_bay=breaker_per_bay,
        )
    else:
        return devices, edges, terminals, exposed_terminals

def main_and_transfer(
    voltage_level,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None
):
    """
    Generate a main-and-transfer bus topology graph with explicit bays.
    Each incomer/outgoer chain is treated as one bay.
    """

    devices = set()
    edges = set()
    terminals = set()
    TOPOLOGY_NAME = "MAIN_AND_TRANSFER"
    exposed_terminals = {"upstream": [], "downstream": []}

    # ---------------- BUSES ----------------
    bus_main = Device("BUS_MAIN", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    bus_transfer = Device("BUS_TRANSFER", DeviceType.BUS, voltage_level, BusRole.TRANSFER, None)

    bus_coupler = Device(
        "BRK_BUS_COUPLER",
        DeviceType.BREAKER,
        voltage_level,
        None,
        "BAY_COUPLER"
    )

    devices.update([bus_main, bus_transfer, bus_coupler])

    edges.add((bus_main, bus_coupler))
    edges.add((bus_coupler, bus_transfer))

    # ---------------- VALIDATION ----------------
    total_incoming = num_upstream_transformers + num_generators
    if total_incoming > 2:
        raise ValueError("Maximum two incomers allowed")

    # ---------------- INCOMERS ----------------
    incomer_targets = [bus_main, bus_transfer]

    uid = 1
    idx = 0

    def add_incomer(source_device, bus, tag):

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
    total_outgoing = num_downstream_transformers + num_feeders

    rhs_count = total_outgoing // 2
    lhs_count = total_outgoing - rhs_count

    if total_outgoing % 2 == 1:
        lhs_count -= 1
        rhs_count += 1

    lhs_targets = [bus_main] * lhs_count
    rhs_targets = [bus_transfer] * rhs_count
    out_targets = lhs_targets + rhs_targets

    # ---------------- OUTGOERS ----------------
    did = 1
    idx = 0

    def add_outgoer(load_device, bus, tag):

        bay_id = f"BAY_{tag + 100}"   # offset so incomer/outgoer bays never collide

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

    if filepath is not None:
        topology_to_json(
            devices=devices, edges=edges,
            terminals=terminals, exposed_terminals=exposed_terminals,
            filepath=filepath, topology=TOPOLOGY_NAME,
        )
    else:
        return devices, edges, terminals, exposed_terminals

def mesh(
    voltage_level,
    num_buses,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None
):

    BUS_CONFIG = {
        6:  {"breakers": 7,  "max_infeeds": 6,  "max_outfeeds": 6},
        8:  {"breakers": 10, "max_infeeds": 8,  "max_outfeeds": 8},
        9:  {"breakers": 12, "max_infeeds": 6,  "max_outfeeds": 6},
        10: {"breakers": 13, "max_infeeds": 10, "max_outfeeds": 10},
        12: {"breakers": 16, "max_infeeds": 12, "max_outfeeds": 12},
        15: {"breakers": 22, "max_infeeds": 10, "max_outfeeds": 10},
    }

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

    target_breakers = config["breakers"]

    devices = set()
    edges = set()
    terminals = set()
    TOPOLOGY_NAME = "MESH"
    exposed_terminals = {"upstream": [], "downstream": []}
    buses = []
    for i in range(num_buses):
        bus = Device(f"BUS_{i+1:03d}", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
        buses.append(bus)
        devices.add(bus)

    # ---------------- GRID FACTORING ----------------
    def factor_grid(n):
        candidates = []
        for r in (2, 3):
            if n % r == 0:
                c = n // r
                candidates.append((r, c))
        return max(candidates, key=lambda x: x[1])

    rows, cols = factor_grid(num_buses)

    def idx(r, c): 
        return r * cols + c

    # ---------------- BUS POSITIONS ----------------
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
        nonlocal brk_id

        key = tuple(sorted((b1.device_id, b2.device_id)))
        if key in connected_pairs:
            return False

        brk = Device(
            f"BRK_M_{brk_id}",
            DeviceType.BREAKER,
            voltage_level,
            None,
            "MESH"
        )

        devices.add(brk)
        edges.add((b1, brk))
        edges.add((brk, b2))

        connected_pairs.add(key)
        brk_id += 1
        return True

    # Horizontal
    for r in range(rows):
        for c in range(cols - 1):
            add_bus_link(buses[idx(r, c)], buses[idx(r, c+1)])

    # Vertical
    for c in range(cols):
        for r in range(rows - 1):
            add_bus_link(buses[idx(r, c)], buses[idx(r+1, c)])

    # Extra links to reach breaker target
    all_pairs = []
    for i in range(len(buses)):
        for j in range(i+1, len(buses)):
            all_pairs.append((buses[i], buses[j]))

    random.shuffle(all_pairs)

    for b1, b2 in all_pairs:
        if brk_id > target_breakers:
            break
        add_bus_link(b1, b2)

    # ---------------- ROW CLASSIFICATION ----------------
    top_row_buses = buses[0:cols]
    bottom_row_buses = buses[(rows-1)*cols : rows*cols]

    def alternating_lr_order(row):
        order = []
        l, r = 0, len(row) - 1

        while l <= r:
            if l == r:
                order.append(row[l])
            else:
                order.append(row[l])
                order.append(row[r])
            l += 1
            r -= 1

        return order

    infeed_attach_points = alternating_lr_order(top_row_buses)
    outfeed_attach_points = list(bottom_row_buses)

    # ---------------- INFEEDS ----------------
    uid = 1
    aidx_in = 0

    def add_infeed(dev):
        nonlocal uid, aidx_in

        bus = infeed_attach_points[aidx_in % len(infeed_attach_points)]
        bay_id = f"BAY_{uid}"

        iso = Device(
            f"ISO_IN_{uid}",
            DeviceType.ISOLATOR,
            voltage_level,
            None,
            bay_id
        )

        brk = Device(
            f"BRK_IN_{uid}",
            DeviceType.BREAKER,
            voltage_level,
            None,
            bay_id
        )

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
        nonlocal did, aidx_out

        bus = outfeed_attach_points[aidx_out % len(outfeed_attach_points)]
        bay_id = f"BAY_{100 + did}"

        brk = Device(
            f"BRK_OUT_{did}",
            DeviceType.BREAKER,
            voltage_level,
            None,
            bay_id
        )

        iso = Device(
            f"ISO_OUT_{did}",
            DeviceType.ISOLATOR,
            voltage_level,
            None,
            bay_id
        )

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

    if filepath is not None:
        topology_to_json(
            devices=devices, edges=edges,
            terminals=terminals, exposed_terminals=exposed_terminals,
            filepath=filepath, topology=TOPOLOGY_NAME,
            bus_positions=BUS_POSITIONS,
        )
    else:
        return devices, edges, terminals, exposed_terminals, BUS_POSITIONS

def ring_bus(
    voltage_level,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None
):
    from core.grammar import Device, DeviceType, BusRole

    devices = set()
    edges = set()
    terminals = set()
    TOPOLOGY_NAME = "RING_BUS"
    exposed_terminals = {"upstream": [], "downstream": []}

    # -------- COUNTS --------
    num_infeeds = num_generators + num_upstream_transformers
    num_outfeeds = num_downstream_transformers + num_feeders
    total_buses = num_infeeds + num_outfeeds

    if total_buses < 3:
        raise ValueError("Ring bus requires at least 3 circuits")

    # -------- HELPERS --------
    bus_counter = 0
    brk_counter = 0

    buses = []
    breaker_orientation = {}
    breaker_side = {}
    bus_side = {}

    def new_bus(side):
        nonlocal bus_counter
        bus_counter += 1
        b = Device(
            f"BUS_R_{bus_counter:03d}",
            DeviceType.BUS,
            voltage_level,
            BusRole.RING,
            None
        )
        devices.add(b)
        buses.append(b)
        bus_side[b.device_id] = side
        return b

    # ring section breakers (not bays)
    def new_ring_breaker(orientation, side):
        nonlocal brk_counter
        brk_counter += 1
        brk = Device(
            f"BRK_R_{brk_counter:03d}",
            DeviceType.BREAKER,
            voltage_level,
            None,
            "RING"
        )
        devices.add(brk)
        breaker_orientation[brk.device_id] = orientation
        breaker_side[brk.device_id] = side
        return brk

    def connect(b1, brk, b2):
        edges.add((b1, brk))
        edges.add((brk, b2))

    # -------- CREATE BUS GROUPS --------
    top_buses = [new_bus("top") for _ in range(num_infeeds)]
    bottom_buses = [new_bus("bottom") for _ in range(num_outfeeds)]

    # -------- BUILD RING --------
    brk_left = new_ring_breaker("vertical", "left")

    first_top_bus = top_buses[0]
    edges.add((brk_left, first_top_bus))

    prev_bus = first_top_bus
    for i in range(1, len(top_buses)):
        brk = new_ring_breaker("horizontal", "top")
        connect(prev_bus, brk, top_buses[i])
        prev_bus = top_buses[i]

    brk_right = new_ring_breaker("vertical", "right")
    edges.add((prev_bus, brk_right))

    first_bottom_bus = bottom_buses[0]
    edges.add((brk_right, first_bottom_bus))
    prev_bus = first_bottom_bus

    for i in range(1, len(bottom_buses)):
        brk = new_ring_breaker("horizontal", "bottom")
        connect(prev_bus, brk, bottom_buses[i])
        prev_bus = bottom_buses[i]

    edges.add((prev_bus, brk_left))

    # -------- ATTACH DEVICES (each is a bay) --------
    bay_counter = 1

    def attach_device(dev, bus):
        nonlocal bay_counter

        bay_id = f"BAY_{bay_counter}"

        iso = Device(
            f"ISO_R_{dev.device_id}",
            DeviceType.ISOLATOR,
            voltage_level,
            None,
            bay_id
        )

        tap = Device(
            f"BRK_TAP_{dev.device_id}",
            DeviceType.BREAKER,
            voltage_level,
            None,
            bay_id
        )

        devices.update([iso, tap])

        edges.add((bus, iso))
        edges.add((iso, tap))
        edges.add((tap, dev))

        bay_counter += 1

    # -------- INFEEDS --------
    idx = 0

    for i in range(num_generators):
        d = Device(f"GS_{i+1}", DeviceType.GRID_SOURCE, voltage_level, None, None)
        devices.add(d)
        terminals.add(d.device_id)
        attach_device(d, top_buses[idx])
        idx += 1

    for i in range(num_upstream_transformers):
        d = Device(f"TR_UP_{i+1}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(d)
        attach_device(d, top_buses[idx])
        exposed_terminals["upstream"].append(d.device_id)
        terminals.add(d.device_id)
        idx += 1

    # -------- OUTFEEDS --------
    idx = 0

    for i in range(num_downstream_transformers):
        d = Device(f"TR_DN_{i+1}", DeviceType.TRANSFORMER, voltage_level, None, None)
        devices.add(d)
        attach_device(d, bottom_buses[idx])
        exposed_terminals["downstream"].append(d.device_id)
        terminals.add(d.device_id)
        idx += 1

    for i in range(num_feeders):
        d = Device(f"FDR_{i+1}", DeviceType.FEEDER, voltage_level, None, None)
        devices.add(d)
        terminals.add(d.device_id)
        attach_device(d, bottom_buses[idx])
        idx += 1

    # -------- GEOMETRY METADATA --------
    ring_geometry = {
        "breaker_orientation": breaker_orientation,
        "breaker_side": breaker_side,
        "bus_side": bus_side
    }

    if filepath is not None:
        topology_to_json(
            devices=devices, edges=edges,
            terminals=terminals, exposed_terminals=exposed_terminals,
            filepath=filepath, topology=TOPOLOGY_NAME,
            ring_geometry=ring_geometry,
        )
    else:
        return devices, edges, terminals, exposed_terminals, ring_geometry

def single_bus(
    voltage_level,
    num_upstream_transformers=0,
    num_generators=0,
    num_downstream_transformers=0,
    num_feeders=0,
    filepath=None
):
    from core.grammar import Device, DeviceType, BusRole

    devices = set()
    edges = set()
    terminals = set()
    TOPOLOGY_NAME = "SINGLE_BUS"
    exposed_terminals = {"upstream": [], "downstream": []}

    # ---------------- BUS ----------------
    bus = Device("BUS_001", DeviceType.BUS, voltage_level, BusRole.MAIN, None)
    devices.add(bus)
    
    # ---------------- INFEEDS ----------------
    uid = 1

    def add_infeed(dev):
        bay_id = f"BAY_{uid}"

        iso = Device(
            f"ISO_IN_{uid}",
            DeviceType.ISOLATOR,
            voltage_level,
            None,
            bay_id
        )

        brk = Device(
            f"BRK_IN_{uid}",
            DeviceType.BREAKER,
            voltage_level,
            None,
            bay_id
        )

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
        bay_id = f"BAY_{100 + did}"

        iso = Device(
            f"ISO_OUT_{did}",
            DeviceType.ISOLATOR,
            voltage_level,
            None,
            bay_id
        )

        brk = Device(
            f"BRK_OUT_{did}",
            DeviceType.BREAKER,
            voltage_level,
            None,
            bay_id
        )

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

    if filepath is not None:
        topology_to_json(
            devices=devices, edges=edges,
            terminals=terminals, exposed_terminals=exposed_terminals,
            filepath=filepath, topology=TOPOLOGY_NAME,
        )
    else:
        return devices, edges, terminals, exposed_terminals

def generate_display_labels(devices):
    label_map = {}

    buses = []
    bays = defaultdict(list)
    transformers = []
    generators = []
    feeders = []

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

    # ---------------- BUS LABELS ----------------
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    buses = sorted(buses, key=lambda x: x.device_id)

    for i, bus in enumerate(buses):
        label_map[bus.device_id] = f"{voltage_prefix}_BUS_{alphabet[i]}"

    # ---------------- BAY DEVICES ----------------
    for bay_id, devs in bays.items():

        try:
            bay_index = int(bay_id.split("_")[1])
        except:
            bay_index = 0

        bay_label = f"{bay_index:02d}"

        breakers = []
        isolators = []

        for d in devs:
            if d.device_type == DeviceType.BREAKER:
                breakers.append(d)
            elif d.device_type == DeviceType.ISOLATOR:
                isolators.append(d)

        breakers = sorted(breakers, key=lambda x: x.device_id)
        isolators = sorted(isolators, key=lambda x: x.device_id)

        for i, brk in enumerate(breakers, start=1):
            label_map[brk.device_id] = f"{voltage_prefix}_{bay_label}_CB{i}"

        iso_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, iso in enumerate(isolators):
            label_map[iso.device_id] = f"{voltage_prefix}_{bay_label}_DS{iso_letters[i]}"

    # ---------------- TRANSFORMERS ----------------
    transformers = sorted(transformers, key=lambda x: x.device_id)

    for i, tr in enumerate(transformers, start=1):
        label_map[tr.device_id] = f"{voltage_prefix}-TR-{i:02d}"

    # ---------------- GENERATORS ----------------
    generators = sorted(generators, key=lambda x: x.device_id)

    for i, g in enumerate(generators, start=1):
        label_map[g.device_id] = f"{voltage_prefix}-GEN-{i:02d}"

    # ---------------- FEEDERS ----------------
    feeders = sorted(feeders, key=lambda x: x.device_id)

    for i, f in enumerate(feeders, start=1):
        label_map[f.device_id] = f"{voltage_prefix}-FDR-{i:02d}"

    return label_map
      
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
    topology_mode="STANDARD"
):

    label_map = generate_display_labels(devices)

    def device_to_dict(d):
        return {
            "id": d.device_id,
            "label": label_map.get(d.device_id),
            "type": d.device_type.value,
            "voltage": str(d.voltage),
            "bus_role": d.bus_role.name if d.bus_role else None,
            "bay_id": d.bay_id
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
            "terminals": sorted(list(terminals)),
            "exposed_terminals": {
                "upstream":   sorted(exposed_terminals.get("upstream",   [])),
                "downstream": sorted(exposed_terminals.get("downstream", [])),
            },
        }
    }

    if topology == "MESH" and bus_positions:
        data["layout"]["bus_positions"] = bus_positions

    if topology == "DOUBLE_BUS":
        data["layout"]["breakersPerBay"] = breakers_per_bay

    if topology == "RING_BUS" and ring_geometry:
        data["layout"]["geometry"] = ring_geometry

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)