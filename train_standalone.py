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
NUM_SAMPLES = 10

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

def _rand_incomers():
    """Return (upstream_tx, generators) with sum <= 2, at least one incomer."""
    combos = [(1, 0), (2, 0), (1, 1), (0, 1), (0, 2)]
    return random.choice(combos)


def rand_params_single_bus():
    # No constraints
    return dict(
        num_upstream_transformers   = random.randint(1, 4),
        num_generators              = random.randint(0, 3),
        num_downstream_transformers = random.randint(1, 4),
        num_feeders                 = random.randint(2, 10),
    )


def rand_params_double_bus():
    # breaker_per_bay must be 1 or 2
    return dict(
        breaker_per_bay             = random.choice([1, 2]),
        num_upstream_transformers   = random.randint(1, 4),
        num_generators              = random.randint(0, 3),
        num_downstream_transformers = random.randint(1, 4),
        num_feeders                 = random.randint(1, 8),
    )


def rand_params_breaker_and_half():
    # num_upstream_transformers + num_generators <= 2
    upstream, gens = _rand_incomers()
    return dict(
        bayCount                    = random.randint(2, 6),
        num_upstream_transformers   = upstream,
        num_generators              = gens,
        num_downstream_transformers = random.randint(1, 4),
        num_feeders                 = random.randint(2, 8),
    )


def rand_params_main_and_transfer():
    # num_upstream_transformers + num_generators <= 2
    upstream, gens = _rand_incomers()
    return dict(
        num_upstream_transformers   = upstream,
        num_generators              = gens,
        num_downstream_transformers = random.randint(1, 4),
        num_feeders                 = random.randint(2, 10),
    )


def rand_params_ring_bus():
    # total connections >= 3; guaranteed here since upstream >= 1,
    # downstream >= 1, feeders >= 1, so min total = 3.
    return dict(
        num_upstream_transformers   = random.randint(1, 4),
        num_generators              = random.randint(0, 3),
        num_downstream_transformers = random.randint(1, 4),
        num_feeders                 = random.randint(1, 5),
    )


def rand_params_mesh():
    # num_buses in MESH_BUS_COUNTS
    # infeeds  (upstream + generators)  <= max_infeeds for that bus count
    # outfeeds (downstream + feeders)   <= max_outfeeds for that bus count
    num_buses = random.choice(MESH_BUS_COUNTS)
    max_in, max_out = MESH_LIMITS[num_buses]

    # --- infeeds ---
    max_upstream = min(3, max_in - 0)           # leave room for at least 0 generators
    upstream     = random.randint(1, max_upstream)
    max_gens     = min(2, max_in - upstream)
    gens         = random.randint(0, max_gens)

    # --- outfeeds ---
    max_downstream = min(3, max_out - 1)        # leave room for at least 1 feeder
    downstream     = random.randint(1, max_downstream)
    max_feeders    = min(4, max_out - downstream)
    feeders        = random.randint(1, max(1, max_feeders))

    return dict(
        num_buses                   = num_buses,
        num_upstream_transformers   = upstream,
        num_generators              = gens,
        num_downstream_transformers = downstream,
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
        topo_dir = os.path.join(TRAIN_DIR, topology.lower())
        os.makedirs(topo_dir, exist_ok=True)

        generated = 0
        attempts  = 0

        while generated < NUM_SAMPLES:
            attempts += 1
            if attempts > NUM_SAMPLES * 10:
                print(f"  [WARN] Could not reach {NUM_SAMPLES} samples for {topology} — stopping at {generated}.")
                break

            params   = PARAM_GEN[topology]()
            sample_id = f"{topology.lower()}_{generated:03d}"
            txt_path  = os.path.join(topo_dir, f"{sample_id}.txt")
            svg_path  = os.path.join(topo_dir, f"{sample_id}.svg")
            png_path  = os.path.join(topo_dir, f"{sample_id}.png")

            # ── build topology + write JSON via filepath ──
            try:
                build_topology(topology, params, txt_path)
            except Exception as e:
                errors += 1
                print(f"  [ERR] {topology} build attempt {attempts}: {e}")
                continue

            # ── render SVG — render function loads JSON from txt_path directly ──
            try:
                RENDER_MAP[topology](txt_path, svg_path, "standalone")
            except Exception as e:
                errors += 1
                print(f"  [ERR] {topology} SVG render attempt {attempts}: {e}")
                for p in (txt_path, svg_path):
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