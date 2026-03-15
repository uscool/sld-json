import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import core.topology as t
from core.grammar import Voltage
import os

from core.render_brk_hlf import render_breaker_and_half_svg
from core.render_dbl_bs import render_double_bus_svg
from core.render_mn_trfr import render_main_and_transfer_svg
from core.render_mesh import render_mesh_svg
from core.render_ring_bs import render_ring_bus_svg
from core.render_sngl_bs import render_single_bus_svg

def main():

    TOPOLOGY = "MAIN_AND_TRANSFER"
    
    BASE = os.path.dirname(__file__)
    JSON_PATH = os.path.join(BASE, "json", f"{TOPOLOGY.lower()}.json")
    SVG_PATH  = os.path.join(BASE, "svg",  f"{TOPOLOGY.lower()}.svg")

    os.makedirs(os.path.join(BASE, "json"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "svg"),  exist_ok=True)

    voltage = Voltage.V220KV

    if TOPOLOGY == "SINGLE_BUS":

        t.single_bus(
            voltage_level=voltage,
            num_upstream_transformers=2,
            num_generators=0,
            num_downstream_transformers=1,
            num_feeders=5,
            filepath=JSON_PATH,
        )

    elif TOPOLOGY == "DOUBLE_BUS":

        t.double_bus(
            voltage_level=voltage,
            breaker_per_bay=1,
            num_upstream_transformers=2,
            num_generators=0,
            num_downstream_transformers=1,
            num_feeders=1,
            filepath=JSON_PATH,
        )

    elif TOPOLOGY == "BREAKER_AND_HALF":

        t.breaker_and_half(
            voltage_level=voltage,
            bayCount=3,
            num_upstream_transformers=2,
            num_generators=0,
            num_downstream_transformers=1,
            num_feeders=3,
            filepath=JSON_PATH,
        )

    elif TOPOLOGY == "MAIN_AND_TRANSFER":

        t.main_and_transfer(
            voltage_level=voltage,
            num_upstream_transformers=2,
            num_generators=0,
            num_downstream_transformers=1,
            num_feeders=5,
            filepath=JSON_PATH,
        )

    elif TOPOLOGY == "RING_BUS":

        t.ring_bus(
            voltage_level=voltage,
            num_upstream_transformers=3,
            num_generators=0,
            num_downstream_transformers=2,
            num_feeders=2,
            filepath=JSON_PATH,
        )

    elif TOPOLOGY == "MESH":

        t.mesh(
            voltage_level=voltage,
            num_buses=12,
            num_upstream_transformers=2,
            num_generators=0,
            num_downstream_transformers=1,
            num_feeders=1,
            filepath=JSON_PATH,
        )

    else:
        raise ValueError(f"Unknown topology: {TOPOLOGY}")

    # ---------- load JSON + render ----------

    render_map = {
        "SINGLE_BUS":        render_single_bus_svg,
        "DOUBLE_BUS":        render_double_bus_svg,
        "BREAKER_AND_HALF":  render_breaker_and_half_svg,
        "MAIN_AND_TRANSFER": render_main_and_transfer_svg,
        "RING_BUS":          render_ring_bus_svg,
        "MESH":              render_mesh_svg,
    }

    result = render_map[TOPOLOGY](JSON_PATH, SVG_PATH, "standalone")
    if result:
        print(result)

if __name__ == "__main__":
    main()