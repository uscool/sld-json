import os
import random

import core.topology as t
from core.grammar import Voltage

from core.render_brk_hlf import render_breaker_and_half_svg
from core.render_dbl_bs import render_double_bus_svg
from core.render_mn_trfr import render_main_and_transfer_svg
from core.render_mesh import render_mesh_svg
from core.render_ring_bs import render_ring_bus_svg
from core.render_sngl_bs import render_single_bus_svg

try:
    import cairosvg
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False
    print("[WARN] cairosvg not found — PNG export will be skipped.")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

TRAIN_DIR   = "output_standalone"
NUM_SAMPLES = 50

# Fixed voltage — visual variety comes from topology structure, not voltage label
VOLTAGE = Voltage.V220KV

# Valid values enforced by the mesh topology builder
MESH_BUS_COUNTS = [6, 8, 9, 10, 12, 15]

# Max infeeds / outfeeds per mesh bus count
# infeeds  = num_upstream_transformers + num_generators
# outfeeds = num_downstream_transformers + num_feeders
MESH_LIMITS = {
    6:  (6,  6),
    8:  (8,  8),
    9:  (6,  6),
    10: (10, 10),
    12: (12, 12),
    15: (10, 10),
}

RENDER_MAP = {
    "SINGLE_BUS":        render_single_bus_svg,
    "DOUBLE_BUS":        render_double_bus_svg,
    "BREAKER_AND_HALF":  render_breaker_and_half_svg,
    "MAIN_AND_TRANSFER": render_main_and_transfer_svg,
    "RING_BUS":          render_ring_bus_svg,
    "MESH":              render_mesh_svg,
}

# ─────────────────────────────────────────────
# Per-topology random parameter generators
# ─────────────────────────────────────────────

def rand_params_single_bus():
    up_tr   = random.randint(0, 6)
    gens    = random.randint(0, 6)
    dn_tr   = random.randint(0, 6)
    feeders = random.randint(0, 10)
    return dict(
        num_upstream_transformers   = up_tr,
        num_generators              = gens,
        num_downstream_transformers = dn_tr,
        num_feeders                 = feeders,
    )


def rand_params_double_bus():
    up_tr   = random.randint(0, 6)
    gens    = random.randint(0, 6)
    dn_tr   = random.randint(0, 6)
    feeders = random.randint(0, 10)
    return dict(
        breaker_per_bay             = random.choice([1, 2]),
        num_upstream_transformers   = up_tr,
        num_generators              = gens,
        num_downstream_transformers = dn_tr,
        num_feeders                 = feeders,
    )


def rand_params_breaker_and_half():
    # Hard incomer cap: up_tr + gens <= 2
    up_tr = random.randint(0, 2)
    gens  = random.randint(0, 2 - up_tr)
    infeeds = up_tr + gens

    # bayCount: min 3, must satisfy 2*bayCount - infeeds >= dn_tr
    bayCount = random.randint(3, 6)
    max_out  = 2 * bayCount - infeeds

    dn_tr   = random.randint(0, min(6, bayCount, max(0, max_out)))
    spare   = max_out - dn_tr
    feeders = random.randint(0, min(10, max(0, spare)))
    return dict(
        bayCount                    = bayCount,
        num_upstream_transformers   = up_tr,
        num_generators              = gens,
        num_downstream_transformers = dn_tr,
        num_feeders                 = feeders,
    )


def rand_params_main_and_transfer():
    # Exactly 2 incomers: up_tr + gens == 2
    up_tr = random.randint(0, 2)
    gens  = 2 - up_tr
    dn_tr   = random.randint(0, 6)
    feeders = random.randint(0, 10)
    return dict(
        num_upstream_transformers   = up_tr,
        num_generators              = gens,
        num_downstream_transformers = dn_tr,
        num_feeders                 = feeders,
    )


def rand_params_ring_bus():
    # total circuits >= 3
    up_tr   = random.randint(0, 6)
    gens    = random.randint(0, 6)
    dn_tr   = random.randint(0, 6)
    feeders = random.randint(0, 10)
    infeeds  = up_tr + gens
    outfeeds = dn_tr + feeders
    # Bump feeders if total < 3
    if infeeds + outfeeds < 3:
        feeders += 3 - (infeeds + outfeeds)
    return dict(
        num_upstream_transformers   = up_tr,
        num_generators              = gens,
        num_downstream_transformers = dn_tr,
        num_feeders                 = feeders,
    )


def rand_params_mesh():
    num_buses        = random.choice(MESH_BUS_COUNTS)
    max_in, max_out  = MESH_LIMITS[num_buses]

    up_tr   = random.randint(0, min(6, max_in))
    gens    = random.randint(0, min(6, max_in - up_tr))
    dn_tr   = random.randint(0, min(6, max_out))
    feeders = random.randint(0, min(10, max_out - dn_tr))
    return dict(
        num_buses                   = num_buses,
        num_upstream_transformers   = up_tr,
        num_generators              = gens,
        num_downstream_transformers = dn_tr,
        num_feeders                 = feeders,
    )


PARAM_GEN = {
    "SINGLE_BUS":        rand_params_single_bus,
    "DOUBLE_BUS":        rand_params_double_bus,
    "BREAKER_AND_HALF":  rand_params_breaker_and_half,
    "MAIN_AND_TRANSFER": rand_params_main_and_transfer,
    "RING_BUS":          rand_params_ring_bus,
    "MESH":              rand_params_mesh,
}

# ─────────────────────────────────────────────
# Topology builder
# Mirrors build.py: each function writes JSON via filepath= automatically.
# ─────────────────────────────────────────────

def build_topology(topology: str, params: dict, filepath: str):
    """Call the appropriate topology builder with filepath so JSON is written automatically."""

    if topology == "SINGLE_BUS":
        t.single_bus(voltage_level=VOLTAGE, filepath=filepath, **params)

    elif topology == "DOUBLE_BUS":
        t.double_bus(voltage_level=VOLTAGE, filepath=filepath, **params)

    elif topology == "BREAKER_AND_HALF":
        t.breaker_and_half(voltage_level=VOLTAGE, filepath=filepath, **params)

    elif topology == "MAIN_AND_TRANSFER":
        t.main_and_transfer(voltage_level=VOLTAGE, filepath=filepath, **params)

    elif topology == "RING_BUS":
        t.ring_bus(voltage_level=VOLTAGE, filepath=filepath, **params)

    elif topology == "MESH":
        t.mesh(voltage_level=VOLTAGE, filepath=filepath, **params)

    else:
        raise ValueError(f"Unknown topology: {topology}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)

    total  = 0
    errors = 0

    for topology in RENDER_MAP:

        generated = 0
        attempts  = 0

        while generated < NUM_SAMPLES:
            attempts += 1
            if attempts > NUM_SAMPLES * 10:
                print(f"  [WARN] Could not reach {NUM_SAMPLES} samples for {topology} — stopping at {generated}.")
                break

            params    = PARAM_GEN[topology]()
            sample_id = f"{topology.lower()}_{generated:03d}"
            json_path = os.path.join(TRAIN_DIR, f"{sample_id}.json")
            svg_path  = os.path.join(TRAIN_DIR, f"{sample_id}.svg")
            png_path  = os.path.join(TRAIN_DIR, f"{sample_id}.png")

            # ── build topology + write JSON via filepath ──
            try:
                build_topology(topology, params, json_path)
            except Exception as e:
                errors += 1
                print(f"  [ERR] {topology} build attempt {attempts}: {e}")
                continue

            # ── render SVG — render function loads JSON from json_path directly ──
            try:
                RENDER_MAP[topology](json_path, svg_path, "standalone")
            except Exception as e:
                errors += 1
                print(f"  [ERR] {topology} SVG render attempt {attempts}: {e}")
                for p in (json_path, svg_path):
                    if os.path.exists(p):
                        os.remove(p)
                continue

            # ── render PNG ──
            if CAIRO_AVAILABLE:
                try:
                    cairosvg.svg2png(
                        url=svg_path,
                        write_to=png_path,
                        background_color="white",
                        scale=8,
                    )
                except Exception as e:
                    print(f"  [WARN] PNG conversion failed for {sample_id}: {e}")

            generated += 1
            total     += 1
            print(f"  [{topology}] {generated}/{NUM_SAMPLES}  →  {sample_id}")

    print(f"\nDone. {total} samples written to '{TRAIN_DIR}/'  ({errors} errors skipped).")


if __name__ == "__main__":
    main()