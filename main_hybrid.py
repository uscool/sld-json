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
NUM_SAMPLES  = 100
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
    Hard maximum num_downstream_transformers this topology can emit,
    matching the caps enforced in all_params.
    """
    if topo in ("BREAKER_AND_HALF", "MAIN_AND_TRANSFER", "MESH"):
        return 6   # global transformer cap
    return 6       # all others also capped at 6 (MAX soft cap)


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


# ─────────────────────────────────────────────────────────────────────────────
# PER-TOPOLOGY PARAMETER GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def all_params(
    topo: str,
    up_tr: int,
    dn_tr: int,
    is_top: bool = False,
) -> list[dict]:
    """
    Enumerate ALL valid deterministic parameter combinations for `topo` with
    exactly `up_tr` upstream transformers and `dn_tr` downstream transformers
    fixed by the inter-level connection plan.

    `is_top=True` requires at least one generator when up_tr == 0 (top level
    must always have an incomer).

    Returns a (possibly empty) list of valid params dicts.  The caller iterates
    over this list in order — no randomness.
    """
    results = []
    min_gens_global = 1 if (is_top and up_tr == 0) else 0

    # ── SINGLE_BUS ────────────────────────────────────────────────────────────
    if topo == "SINGLE_BUS":
        spare_in    = MAX_INFEEDS_OUTFEEDS - up_tr
        max_gens    = _clamp(spare_in, 0, 6)
        spare_out   = MAX_INFEEDS_OUTFEEDS - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        max_feeders = _clamp(spare_out, min_feeders, 10)
        for gens in range(min_gens_global, max_gens + 1):
            infeeds = up_tr + gens
            for feeders in range(min_feeders, max_feeders + 1):
                outfeeds = dn_tr + feeders
                if max(infeeds, outfeeds) <= MAX_INFEEDS_OUTFEEDS:
                    results.append(dict(
                        num_upstream_transformers   = up_tr,
                        num_generators              = gens,
                        num_downstream_transformers = dn_tr,
                        num_feeders                 = feeders,
                    ))

    # ── DOUBLE_BUS ────────────────────────────────────────────────────────────
    elif topo == "DOUBLE_BUS":
        spare_in    = MAX_INFEEDS_OUTFEEDS - up_tr
        max_gens    = _clamp(spare_in, 0, 6)
        spare_out   = MAX_INFEEDS_OUTFEEDS - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        max_feeders = _clamp(spare_out, min_feeders, 10)
        for bpb in (1, 2):
            for gens in range(min_gens_global, max_gens + 1):
                infeeds = up_tr + gens
                for feeders in range(min_feeders, max_feeders + 1):
                    outfeeds = dn_tr + feeders
                    if max(infeeds, outfeeds) <= MAX_INFEEDS_OUTFEEDS:
                        results.append(dict(
                            breaker_per_bay             = bpb,
                            num_upstream_transformers   = up_tr,
                            num_generators              = gens,
                            num_downstream_transformers = dn_tr,
                            num_feeders                 = feeders,
                        ))

    # ── BREAKER_AND_HALF ──────────────────────────────────────────────────────
    elif topo == "BREAKER_AND_HALF":
        # Hard incomer cap: up_tr + gens <= 2
        if up_tr > 2:
            return results
        max_gens = 2 - up_tr
        for gens in range(min_gens_global, max_gens + 1):
            infeeds  = up_tr + gens
            # bayCount must satisfy: 2*bayCount - infeeds >= dn_tr
            min_bays = max(math.ceil((dn_tr + infeeds) / 2), max(1, infeeds))
            if min_bays > 6:
                continue
            for bayCount in range(min_bays, 7):
                max_out     = 2 * bayCount - infeeds
                spare_out   = max_out - dn_tr
                min_feeders = 0 if dn_tr > 0 else 1
                max_feeders = _clamp(spare_out, min_feeders, 10)
                for feeders in range(min_feeders, max_feeders + 1):
                    outfeeds = dn_tr + feeders
                    if max(infeeds, outfeeds) <= MAX_INFEEDS_OUTFEEDS:
                        results.append(dict(
                            bayCount                    = bayCount,
                            num_upstream_transformers   = up_tr,
                            num_generators              = gens,
                            num_downstream_transformers = dn_tr,
                            num_feeders                 = feeders,
                        ))

    # ── MAIN_AND_TRANSFER ─────────────────────────────────────────────────────
    elif topo == "MAIN_AND_TRANSFER":
        # Exactly 2 incomers: up_tr + gens == 2
        if up_tr > 2:
            return results
        # gens is fully determined by the equality constraint
        gens_required = 2 - up_tr
        if gens_required < min_gens_global:
            return results
        infeeds     = 2
        spare_out   = MAX_INFEEDS_OUTFEEDS - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        max_feeders = _clamp(spare_out, min_feeders, 10)
        for feeders in range(min_feeders, max_feeders + 1):
            outfeeds = dn_tr + feeders
            if max(infeeds, outfeeds) <= MAX_INFEEDS_OUTFEEDS:
                results.append(dict(
                    num_upstream_transformers   = up_tr,
                    num_generators              = gens_required,
                    num_downstream_transformers = dn_tr,
                    num_feeders                 = feeders,
                ))

    # ── RING_BUS ──────────────────────────────────────────────────────────────
    elif topo == "RING_BUS":
        spare_in    = MAX_INFEEDS_OUTFEEDS - up_tr
        max_gens    = _clamp(spare_in, 0, 6)
        spare_out   = MAX_INFEEDS_OUTFEEDS - dn_tr
        min_feeders = 0 if dn_tr > 0 else 1
        max_feeders = _clamp(spare_out, min_feeders, 10)
        for gens in range(min_gens_global, max_gens + 1):
            infeeds = up_tr + gens
            for feeders in range(min_feeders, max_feeders + 1):
                outfeeds = dn_tr + feeders
                # Ring constraint: total circuits >= 3
                if (infeeds + outfeeds) >= 3 and max(infeeds, outfeeds) <= MAX_INFEEDS_OUTFEEDS:
                    results.append(dict(
                        num_upstream_transformers   = up_tr,
                        num_generators              = gens,
                        num_downstream_transformers = dn_tr,
                        num_feeders                 = feeders,
                    ))

    # ── MESH ──────────────────────────────────────────────────────────────────
    elif topo == "MESH":
        valid_buses = [
            nb for nb in MESH_BUS_COUNTS
            if MESH_LIMITS[nb][0] >= up_tr and MESH_LIMITS[nb][1] >= dn_tr
        ]
        for num_buses in valid_buses:
            max_in, max_out = MESH_LIMITS[num_buses]
            spare_in    = max_in - up_tr
            max_gens    = _clamp(spare_in, 0, 6)
            spare_out   = max_out - dn_tr
            min_feeders = 0 if dn_tr > 0 else 1
            max_feeders = _clamp(spare_out, min_feeders, 10)
            for gens in range(min_gens_global, max_gens + 1):
                infeeds = up_tr + gens
                if infeeds > max_in:
                    continue
                for feeders in range(min_feeders, max_feeders + 1):
                    outfeeds = dn_tr + feeders
                    if outfeeds <= max_out and max(infeeds, outfeeds) <= MAX_INFEEDS_OUTFEEDS:
                        results.append(dict(
                            num_buses                   = num_buses,
                            num_upstream_transformers   = up_tr,
                            num_generators              = gens,
                            num_downstream_transformers = dn_tr,
                            num_feeders                 = feeders,
                        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID LAYOUT SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def sample_layout() -> list[dict] | None:
    """
    Randomly sample one valid hybrid layout.

    At every choice point, picks uniformly from the valid options derived
    from the deterministic bounds — so every generated layout is guaranteed
    feasible with no retry needed.

    Returns a layout list or None if a degenerate edge case makes a level
    infeasible (extremely rare; caller may retry once).
    """
    num_levels = random.randint(MIN_LEVELS, MAX_LEVELS)

    # Random contiguous voltage slice
    v_start  = random.randint(0, len(VOLTAGE_SEQUENCE) - num_levels)
    voltages = VOLTAGE_SEQUENCE[v_start : v_start + num_levels]
    labels   = VOLTAGE_LABELS[v_start : v_start + num_levels]

    # Random topology per level
    topologies = [random.choice(ALL_TOPOLOGIES) for _ in range(num_levels)]

    # Random connection counts within valid bounds
    connection_counts = []
    for i in range(num_levels - 1):
        max_dn = _max_downstream(topologies[i])
        max_up = _max_upstream(topologies[i + 1])
        hi     = _clamp(min(max_dn, max_up), 1, MAX_INFEEDS_OUTFEEDS)
        connection_counts.append(random.randint(1, hi))

    # Random params per level — pick from the full valid set for that level
    levels = []
    for i, (topo, voltage, label) in enumerate(zip(topologies, voltages, labels)):
        up_tr  = connection_counts[i - 1] if i > 0              else 0
        dn_tr  = connection_counts[i]     if i < num_levels - 1 else 0
        is_top = (i == 0)

        options = all_params(topo, up_tr, dn_tr, is_top=is_top)
        if not options:
            return None   # infeasible edge case — caller retries
        params = random.choice(options)
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

    while total < NUM_SAMPLES:
        layout = sample_layout()
        if layout is None:
            continue   # rare infeasible edge case — retry immediately
        sample_id   = f"hybrid_{sample_idx:04d}"
        temp_dir    = os.path.join(TRAIN_DIR, f"_temp_{sample_id}")
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
            print(f"  [ERR] JSON build ({sample_id}): {e}")
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
            print(f"  [ERR] SVG render ({sample_id}): {e}")
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
        chain = " -> ".join(
            f"{lvl['topology']}_{lvl['label']}" for lvl in layout
        )
        print(f"[{total + 1:>{len(str(NUM_SAMPLES))}}/{NUM_SAMPLES}] [{chain}]")

        total      += 1
        sample_idx += 1

    print(f"\nDone. {total} samples written to '{TRAIN_DIR}/'  ({errors} errors skipped).")


if __name__ == "__main__":
    main()