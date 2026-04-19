"""
domain_conversion.py — robust version using label-based matching
"""

import re
import json
from itertools import combinations


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def build_component_index_to_device_id(components, devices, max_dist_px=120):
    device_lookup = []
    for dev in devices:
        dev_id = dev.get("id")
        if not dev_id:
            continue
        sc = dev.get("symbol_center")
        if sc:
            cx, cy = sc[0], sc[1]
        elif dev.get("bbox"):
            cx, cy = _bbox_center(dev["bbox"])
        else:
            continue
        device_lookup.append({
            "id":    dev_id,
            "type":  dev.get("type", ""),
            "label": dev.get("label", ""),
            "cx":    cx,
            "cy":    cy,
        })

    if not device_lookup:
        print("  WARNING: No devices with spatial data — index_map will be empty")
        return {}

    YOLO_CLASS_TO_DEV_TYPE = {
        "Generator":       "GridSource",
        "Transformer":     "Transformer",
        "Circuit_Breaker": "Breaker",
        "Load":            "Feeder",
        "Capacitor":       "Feeder",
        "Switch":          "Isolator",
    }

    index_map = {}

    for comp_idx, comp in enumerate(components):
        if "center" in comp:
            cx, cy = comp["center"][0], comp["center"][1]
        elif "bbox" in comp:
            cx, cy = _bbox_center(comp["bbox"])
        else:
            print(f"  WARNING: comp[{comp_idx}] has no spatial data — skipping")
            continue

        comp_class    = comp.get("class_name", comp.get("class", ""))
        expected_type = YOLO_CLASS_TO_DEV_TYPE.get(comp_class, "")

        best_id   = None
        best_dist = float("inf")

        # Pass 1: type-filtered proximity
        candidates = [d for d in device_lookup if d["type"] == expected_type] \
                     if expected_type else device_lookup

        for dev in candidates:
            d = _dist((cx, cy), (dev["cx"], dev["cy"]))
            if d < best_dist:
                best_dist = d
                best_id   = dev["id"]

        # Pass 2: fall back to all devices if no typed match or too far
        if best_id is None or best_dist > max_dist_px:
            for dev in device_lookup:
                d = _dist((cx, cy), (dev["cx"], dev["cy"]))
                if d < best_dist:
                    best_dist = d
                    best_id   = dev["id"]

        if best_dist > max_dist_px * 3:
            print(f"  WARNING: comp[{comp_idx}] ({comp_class}) @ ({cx:.0f},{cy:.0f}) "
                  f"— nearest device '{best_id}' is {best_dist:.0f}px away.")

        if best_id is not None:
            index_map[comp_idx] = best_id

    return index_map


def extract_connections_from_line_results(connections, index_map):
    seen   = set()
    result = []

    for conn in connections:
        connected_boxes = conn.get("connected_boxes", [])

        device_ids = []
        for entry in connected_boxes:
            comp_idx = entry[0] if isinstance(entry, (list, tuple)) else entry
            dev_id   = index_map.get(comp_idx)
            if dev_id:
                device_ids.append(dev_id)

        for a, b in combinations(device_ids, 2):
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)
            result.append({"from": a, "to": b})

    return result


def convert_devices_to_domain(devices):
    """
    Rename DEV_N ids to meaningful domain ids: BRK, ISO_TOP, ISO_BOT, TR_DN, GS.
    Returns (final_devices list, id_mapping dict {old_id: new_id}).
    """
    final_devices = []
    id_mapping    = {}
    seen_ids      = {}

    for d in devices:
        old_id = d.get("id")
        label  = d.get("label", "")

        bay_match = re.search(r'_(\d+)_', label)
        bay       = bay_match.group(1) if bay_match else None

        if "CB" in label:
            num    = re.search(r'CB(\d+)', label)
            new_id = f"BRK_{bay}_{num.group(1)}" if num else f"BRK_{bay}_X"

        elif "DSA" in label:
            new_id = f"ISO_BOT_{bay}"

        elif "DSB" in label:
            new_id = f"ISO_TOP_{bay}"

        elif "TR" in label:
            num    = re.search(r'TR-(\d+)', label)
            new_id = f"TR_DN_{num.group(1)}" if num else "TR_DN_X"

        elif "GEN" in label:
            num    = re.search(r'GEN-(\d+)', label)
            new_id = f"GS_{num.group(1)}" if num else "GS_X"

        else:
            new_id = old_id  # keep DEV_N if label matches no pattern

        # Deduplicate in case two devices resolve to the same new_id
        if new_id in seen_ids:
            seen_ids[new_id] += 1
            new_id = f"{new_id}_{seen_ids[new_id]}"
        else:
            seen_ids[new_id] = 0

        id_mapping[old_id] = new_id

        final_devices.append({
            "id":           new_id,
            "label":        label,
            "type":         d.get("type"),
            "voltage":      d.get("voltage"),
            "bay_id":       f"BAY_{bay}" if bay else None,
            "bay_number":   d.get("bay_number"),
            "bus_role":     None
        })

    return final_devices, id_mapping


def build_final_json(sld_nodes_path, line_results_path):
    sld  = load_json(sld_nodes_path)
    line = load_json(line_results_path)

    devices         = sld.get("devices", [])
    components      = line.get("components", [])
    connections_raw = line.get("connections", [])

    print(f"  Devices in sld_nodes      : {len(devices)}")
    print(f"  Components in line_results: {len(components)}")
    print(f"  Raw connections           : {len(connections_raw)}")

    if not devices:
        raise ValueError(f"No devices found in {sld_nodes_path}")
    if not components:
        raise ValueError(f"No components found in {line_results_path}")

    # Coordinate alignment check
    dev_centers  = [d["symbol_center"] for d in devices    if d.get("symbol_center")]
    comp_centers = [c["center"]         for c in components if "center" in c]
    if dev_centers and comp_centers:
        dev_xs  = [c[0] for c in dev_centers]
        comp_xs = [c[0] for c in comp_centers]
        print(f"  Device x range   : [{min(dev_xs):.0f} .. {max(dev_xs):.0f}]")
        print(f"  Component x range: [{min(comp_xs):.0f} .. {max(comp_xs):.0f}]")
        ratio = max(dev_xs) / max(comp_xs) if max(comp_xs) > 0 else 0
        if ratio < 0.5 or ratio > 2.0:
            print(f"  !! SCALE MISMATCH (ratio={ratio:.2f})")

    # Step 1: component index → DEV_N  (uses raw DEV_N ids for stable proximity match)
    index_map = build_component_index_to_device_id(components, devices)

    # Step 2: resolve connections as DEV_N pairs
    raw_connections = extract_connections_from_line_results(connections_raw, index_map)

    # Step 3: rename DEV_N → domain ids (BRK, ISO, TR, GS)
    final_devices, id_mapping = convert_devices_to_domain(devices)

    # Step 4: remap connection ids to domain ids
    seen = set()
    final_connections = []
    for conn in raw_connections:
        f = id_mapping.get(conn["from"], conn["from"])
        t = id_mapping.get(conn["to"],   conn["to"])
        if f == t:
            continue
        key = tuple(sorted([f, t]))
        if key in seen:
            continue
        seen.add(key)
        final_connections.append({"from": f, "to": t})

    print(f"  Mapped {len(index_map)}/{len(components)} components → devices")
    print(f"  Extracted {len(final_connections)} unique connections")

    if not index_map:
        print("  !! index_map is empty — check symbol_center values and coordinate spaces")

    return {
        "devices":     final_devices,
        "connections": final_connections,
    }