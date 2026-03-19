"""
HYBRID TRAINING DATA GENERATOR
================================
Generates random hybrid substation layouts for YOLO training data.

Each sample is a random stack of 2–5 voltage levels, each with an independently
chosen topology type (all six types are eligible at every level, including all
levels being the same type).  Adjacent levels are wired together via a randomly
chosen number of inter-level transformers that respects every topology's hard
constraints.

Rendering constraint: max(infeeds, outfeeds) ≤ MAX_INFEEDS_OUTFEEDS per level,
so no single level becomes visually cluttered.

Output layout (one flat file-set per sample):
    output_hybrid_train/
        hybrid_0000.json   ← merged JSON (kept for inspection / debugging)
        hybrid_0000.svg
        hybrid_0000.txt    ← YOLO annotations
        hybrid_0000.png    ← if cairosvg is available
"""

import math
import os
import random
import shutil

from core.grammar import Voltage
from core.hybrid_json_builder import HybridJSONBuilder
from core.hybrid_svg_renderer import HybridSVGRenderer

try:
    import cairosvg
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False
    print("[WARN] cairosvg not found — PNG export will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DIR    = "output_hybrid"
NUM_SAMPLES  = 10
MIN_LEVELS   = 2
MAX_LEVELS   = 5

# Rendering constraint applied to every level independently
MAX_INFEEDS_OUTFEEDS = 15

# Voltage levels available (top → bottom).  A random contiguous slice of
# length num_levels is chosen from this list for each sample.
VOLTAGE_SEQUENCE = [
    Voltage.V220KV,
    Voltage.V132KV,
    Voltage.V66KV,
    Voltage.V33KV,
    Voltage.V11KV,
]
VOLTAGE_LABELS = ["220kV", "132kV", "66kV", "33kV", "11kV"]

ALL_TOPOLOGIES = [
    "SINGLE_BUS",
    "DOUBLE_BUS",
    "BREAKER_AND_HALF",
    "MAIN_AND_TRANSFER",
    "RING_BUS",
    "MESH",
]

# Valid bus counts for MESH and their (max_infeeds, max_outfeeds)
MESH_BUS_COUNTS = [6, 8, 9, 10, 12, 15]
MESH_LIMITS = {
    6:  (6,  6),
    8:  (8,  8),
    9:  (6,  6),
    10: (10, 10),
    12: (12, 12),
    15: (10, 10),
}


# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY CAPACITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _max_upstream(topo: str) -> int:
    """
    Maximum num_upstream_transformers this topology can absorb as its sole
    incomers (zero generators assumed — the safe lower bound).
    """
    if topo in ("BREAKER_AND_HALF", "MAIN_AND_TRANSFER"):
        return 2                              # hard incomer cap
    if topo == "MESH":
        return min(v[0] for v in MESH_LIMITS.values())   # 6 — conservative
    return MAX_INFEEDS_OUTFEEDS               # SINGLE_BUS, DOUBLE_BUS, RING_BUS


def _max_downstream(topo: str) -> int:
    """
    Conservative maximum num_downstream_transformers before any bay-count or
    bus-count choices are made.
    """
    if topo == "MESH":
        return min(v[1] for v in MESH_LIMITS.values())   # 6 — conservative
    if topo == "BREAKER_AND_HALF":
        # With bayCount=6 and 0 incomers: max_outfeeds = 2*6 - 0 = 12
        return 12
    return MAX_INFEEDS_OUTFEEDS


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


# ─────────────────────────────────────────────────────────────────────────────
# PER-TOPOLOGY PARAMETER GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def rand_params(
    topo: str,
    up_tr: int,
    dn_tr: int,
    is_top: bool = False,
) -> dict | None:
    """
    Generate random parameters for `topo` with exactly `up_tr` upstream
    transformers and `dn_tr` downstream transformers already fixed by the
    inter-level connection plan.

    `is_top=True` forces at least one generator when up_tr == 0, so the top
    level always has an incomer.

    Returns a valid params dict or None if the combination is infeasible.
    """

    # ── SINGLE_BUS ────────────────────────────────────────────────────────────
    if topo == "SINGLE_BUS":
        spare_in  = MAX_INFEEDS_OUTFEEDS - up_tr
        gens      = random.randint(1 if (is_top and up_tr == 0) else 0,
                                   _clamp(spare_in, 0, 3))
        infeeds   = up_tr + gens
        spare_out   = MAX_INFEEDS_OUTFEEDS - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        feeders     = random.randint(min_feeders, _clamp(spare_out, min_feeders, 10))
        outfeeds    = dn_tr + feeders
        if max(infeeds, outfeeds) > MAX_INFEEDS_OUTFEEDS:
            return None
        return dict(
            num_upstream_transformers   = up_tr,
            num_generators              = gens,
            num_downstream_transformers = dn_tr,
            num_feeders                 = feeders,
        )

    # ── DOUBLE_BUS ────────────────────────────────────────────────────────────
    elif topo == "DOUBLE_BUS":
        spare_in  = MAX_INFEEDS_OUTFEEDS - up_tr
        gens      = random.randint(1 if (is_top and up_tr == 0) else 0,
                                   _clamp(spare_in, 0, 3))
        infeeds   = up_tr + gens
        spare_out   = MAX_INFEEDS_OUTFEEDS - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        feeders     = random.randint(min_feeders, _clamp(spare_out, min_feeders, 8))
        outfeeds    = dn_tr + feeders
        if max(infeeds, outfeeds) > MAX_INFEEDS_OUTFEEDS:
            return None
        return dict(
            breaker_per_bay             = random.choice([1, 2]),
            num_upstream_transformers   = up_tr,
            num_generators              = gens,
            num_downstream_transformers = dn_tr,
            num_feeders                 = feeders,
        )

    # ── BREAKER_AND_HALF ──────────────────────────────────────────────────────
    elif topo == "BREAKER_AND_HALF":
        # Hard incomer cap: up_tr + gens <= 2
        if up_tr > 2:
            return None
        min_gens = 1 if (is_top and up_tr == 0) else 0
        gens     = random.randint(min_gens, 2 - up_tr)
        infeeds  = up_tr + gens

        # bayCount must satisfy: 2*bayCount - infeeds >= dn_tr
        min_bays = math.ceil((dn_tr + infeeds) / 2)
        min_bays = max(min_bays, max(1, infeeds))  # also need bays for infeeds
        if min_bays > 6:
            return None
        bayCount = random.randint(min_bays, 6)
        max_out  = 2 * bayCount - infeeds

        spare_out   = max_out - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        feeders     = random.randint(min_feeders, _clamp(spare_out, min_feeders, 8))
        outfeeds  = dn_tr + feeders
        if max(infeeds, outfeeds) > MAX_INFEEDS_OUTFEEDS:
            return None
        return dict(
            bayCount                    = bayCount,
            num_upstream_transformers   = up_tr,
            num_generators              = gens,
            num_downstream_transformers = dn_tr,
            num_feeders                 = feeders,
        )

    # ── MAIN_AND_TRANSFER ─────────────────────────────────────────────────────
    elif topo == "MAIN_AND_TRANSFER":
        # Hard incomer cap: up_tr + gens <= 2
        if up_tr > 2:
            return None
        min_gens = 1 if (is_top and up_tr == 0) else 0
        gens     = random.randint(min_gens, 2 - up_tr)
        infeeds  = up_tr + gens
        spare_out   = MAX_INFEEDS_OUTFEEDS - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        feeders     = random.randint(min_feeders, _clamp(spare_out, min_feeders, 10))
        outfeeds    = dn_tr + feeders
        if max(infeeds, outfeeds) > MAX_INFEEDS_OUTFEEDS:
            return None
        return dict(
            num_upstream_transformers   = up_tr,
            num_generators              = gens,
            num_downstream_transformers = dn_tr,
            num_feeders                 = feeders,
        )

    # ── RING_BUS ──────────────────────────────────────────────────────────────
    elif topo == "RING_BUS":
        spare_in  = MAX_INFEEDS_OUTFEEDS - up_tr
        min_gens  = 1 if (is_top and up_tr == 0) else 0
        gens      = random.randint(min_gens, _clamp(spare_in, min_gens, 3))
        infeeds   = up_tr + gens
        spare_out = MAX_INFEEDS_OUTFEEDS - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        feeders     = random.randint(min_feeders, _clamp(spare_out, min_feeders, 5))
        outfeeds  = dn_tr + feeders

        # Ring requires total circuits >= 3
        total = infeeds + outfeeds
        if total < 3:
            # Prefer adding a feeder; fall back to a generator
            if outfeeds < MAX_INFEEDS_OUTFEEDS:
                feeders  += 3 - total
                outfeeds  = dn_tr + feeders
            elif infeeds < MAX_INFEEDS_OUTFEEDS:
                gens     += 3 - total
                infeeds   = up_tr + gens
            else:
                return None

        if max(infeeds, outfeeds) > MAX_INFEEDS_OUTFEEDS:
            return None
        return dict(
            num_upstream_transformers   = up_tr,
            num_generators              = gens,
            num_downstream_transformers = dn_tr,
            num_feeders                 = feeders,
        )

    # ── MESH ──────────────────────────────────────────────────────────────────
    elif topo == "MESH":
        # Find bus counts that can accommodate the fixed up_tr and dn_tr
        valid_buses = [
            nb for nb in MESH_BUS_COUNTS
            if MESH_LIMITS[nb][0] >= up_tr and MESH_LIMITS[nb][1] >= dn_tr
        ]
        if not valid_buses:
            return None
        num_buses        = random.choice(valid_buses)
        max_in, max_out  = MESH_LIMITS[num_buses]

        spare_in  = max_in - up_tr
        min_gens  = 1 if (is_top and up_tr == 0) else 0
        gens      = random.randint(min_gens, _clamp(spare_in, min_gens, 2))
        infeeds   = up_tr + gens

        spare_out = max_out - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        feeders     = random.randint(min_feeders, _clamp(spare_out, min_feeders, 4))
        outfeeds  = dn_tr + feeders

        if max(infeeds, outfeeds) > MAX_INFEEDS_OUTFEEDS:
            return None
        return dict(
            num_buses                   = num_buses,
            num_upstream_transformers   = up_tr,
            num_generators              = gens,
            num_downstream_transformers = dn_tr,
            num_feeders                 = feeders,
        )

    return None  # unknown topology


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID LAYOUT SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def sample_hybrid_layout(num_levels: int) -> list | None:
    """
    Sample a random feasible hybrid layout with `num_levels` voltage levels.

    Returns a list of dicts:
        [{"topology": str, "params": dict, "voltage": Voltage, "label": str}, ...]
    or None if a feasible combination cannot be found for the chosen topologies.
    """
    # Random topology for each level (all combinations are valid as a starting point)
    topologies = [random.choice(ALL_TOPOLOGIES) for _ in range(num_levels)]

    # Decide inter-level connection counts (transformers bridging adjacent levels).
    # The count is bounded by both the upstream level's max downstream capacity
    # and the downstream level's max upstream capacity.
    connection_counts = []
    for i in range(num_levels - 1):
        max_dn = _max_downstream(topologies[i])
        max_up = _max_upstream(topologies[i + 1])
        hi     = _clamp(min(max_dn, max_up), 1, MAX_INFEEDS_OUTFEEDS)
        connection_counts.append(random.randint(1, hi))

    # Choose a contiguous voltage slice
    start = random.randint(0, len(VOLTAGE_SEQUENCE) - num_levels)
    voltages = VOLTAGE_SEQUENCE[start : start + num_levels]
    labels   = VOLTAGE_LABELS[start : start + num_levels]

    # Generate params for every level
    levels = []
    for i, (topo, voltage, label) in enumerate(zip(topologies, voltages, labels)):
        up_tr  = connection_counts[i - 1] if i > 0          else 0
        dn_tr  = connection_counts[i]     if i < num_levels - 1 else 0
        is_top = (i == 0)

        params = rand_params(topo, up_tr, dn_tr, is_top=is_top)
        if params is None:
            return None   # infeasible — caller will retry

        levels.append({
            "topology": topo,
            "params":   params,
            "voltage":  voltage,
            "label":    label,
        })

    return levels


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)

    total      = 0
    errors     = 0
    sample_idx = 0
    attempts   = 0

    while total < NUM_SAMPLES:
        attempts += 1
        if attempts > NUM_SAMPLES * 30:
            print(f"[WARN] Reached attempt limit — stopping at {total}/{NUM_SAMPLES} samples.")
            break

        # ── Sample a feasible layout ──────────────────────────────────────────
        num_levels = random.randint(MIN_LEVELS, MAX_LEVELS)
        layout = sample_hybrid_layout(num_levels)
        if layout is None:
            continue   # infeasible combination — retry silently

        sample_id  = f"hybrid_{sample_idx:04d}"
        temp_dir   = os.path.join(TRAIN_DIR, f"_temp_{sample_id}")
        merged_path = os.path.join(TRAIN_DIR, f"{sample_id}.json")
        svg_path    = os.path.join(TRAIN_DIR, f"{sample_id}.svg")
        yolo_path   = os.path.join(TRAIN_DIR, f"{sample_id}.txt")
        png_path    = os.path.join(TRAIN_DIR, f"{sample_id}.png")

        # ── Build merged JSON ─────────────────────────────────────────────────
        try:
            builder = HybridJSONBuilder()
            for lvl in layout:
                builder.add_level(
                    voltage       = lvl["voltage"],
                    topology_type = lvl["topology"],
                    params        = lvl["params"],
                    voltage_label = lvl["label"],
                )
            builder.generate(output_path=merged_path, temp_dir=temp_dir)
        except Exception as e:
            errors += 1
            shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(merged_path):
                os.remove(merged_path)
            print(f"  [ERR] JSON build (attempt {attempts}): {e}")
            continue

        # ── Render SVG + YOLO ─────────────────────────────────────────────────
        try:
            individual_paths = [
                os.path.join(temp_dir, f"{lvl['label']}_{i}.json")
                for i, lvl in enumerate(layout)
            ]
            topo_types = [lvl["topology"] for lvl in layout]

            renderer = HybridSVGRenderer()
            renderer.render(
                merged_json_path      = merged_path,
                individual_json_paths = individual_paths,
                topology_types        = topo_types,
                svg_output            = svg_path,
                yolo_output           = yolo_path,
            )
        except Exception as e:
            errors += 1
            print(f"  [ERR] SVG render (attempt {attempts}): {e}")
            for p in (svg_path, yolo_path, merged_path):
                if os.path.exists(p):
                    os.remove(p)
            continue
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # ── Render PNG ────────────────────────────────────────────────────────
        if CAIRO_AVAILABLE:
            try:
                cairosvg.svg2png(
                    url              = svg_path,
                    write_to         = png_path,
                    background_color = "white",
                    scale            = 8,
                )
            except Exception as e:
                print(f"  [WARN] PNG failed for {sample_id}: {e}")

        # ── Log ───────────────────────────────────────────────────────────────
        level_desc = "  |  ".join(
            f"{lvl['label']} {lvl['topology']}" for lvl in layout
        )
        print(f"  [{total + 1:>{len(str(NUM_SAMPLES))}}/{NUM_SAMPLES}]  "
              f"{sample_id}  —  {level_desc}")

        total      += 1
        sample_idx += 1

    print(f"\nDone. {total} samples written to '{TRAIN_DIR}/'  ({errors} errors skipped).")


if __name__ == "__main__":
    main()