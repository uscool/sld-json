import shutil
import os

from core.grammar import Voltage
from hybrid_json_builder import HybridJSONBuilder
from hybrid_svg_renderer import HybridSVGRenderer


def main():
    # ---------------- PATHS ----------------
    BASE_DIR = "output_hybrid"
    TEMP_JSON_DIR = os.path.join(BASE_DIR, "temp_json")
    MERGED_JSON_PATH = os.path.join(BASE_DIR, "json", "hybrid.json")
    SVG_OUTPUT_PATH = os.path.join(BASE_DIR, "svg", "hybrid.svg")
    YOLO_OUTPUT_PATH = os.path.join(BASE_DIR, "yolo", "hybrid.txt")

    # ---------------- CLEAN TEMP ----------------
    if os.path.exists(TEMP_JSON_DIR):
        shutil.rmtree(TEMP_JSON_DIR)

    # ---------------- BUILD JSON ----------------
    builder = HybridJSONBuilder()

    builder.add_level(
        voltage=Voltage.V220KV,
        topology_type="MESH",
        params={
            "num_buses": 6,
            "num_upstream_transformers": 0,
            "num_generators": 3,
            "num_downstream_transformers": 3,
            "num_feeders": 3
        },
        voltage_label="220kV"
    )

    builder.add_level(
        voltage=Voltage.V132KV,
        topology_type="RING_BUS",
        params={
            "num_upstream_transformers": 3,
            "num_generators": 0,
            "num_downstream_transformers": 2,
            "num_feeders": 2
        },
        voltage_label="132kV"
    )

    builder.add_level(
        voltage=Voltage.V66KV,
        topology_type="BREAKER_AND_HALF",
        params={
            "bayCount": 3,
            "num_upstream_transformers": 2,
            "num_generators": 0,
            "num_downstream_transformers": 1,
            "num_feeders": 2
        },
        voltage_label="66kV"
    )

    builder.add_level(
        voltage=Voltage.V33KV,
        topology_type="MAIN_AND_TRANSFER",
        params={
            "num_upstream_transformers": 1,
            "num_generators": 1,
            "num_downstream_transformers": 2,
            "num_feeders": 3
        },
        voltage_label="33kV"
    )

    builder.add_level(
        voltage=Voltage.V11KV,
        topology_type="DOUBLE_BUS",
        params={
            "breaker_per_bay": 2,
            "num_upstream_transformers": 2,
            "num_generators": 1,
            "num_downstream_transformers": 0,
            "num_feeders": 4
        },
        voltage_label="11kV"
    )

    merged_path = builder.generate(
        output_path=MERGED_JSON_PATH,
        temp_dir=TEMP_JSON_DIR
    )

    # ---------------- RENDER SVG ----------------
    renderer = HybridSVGRenderer()

    individual_json_paths = [
        os.path.join(TEMP_JSON_DIR, "220kV_0.json"),
        os.path.join(TEMP_JSON_DIR, "132kV_1.json"),
        os.path.join(TEMP_JSON_DIR, "66kV_2.json"),
        os.path.join(TEMP_JSON_DIR, "33kV_3.json"),
        os.path.join(TEMP_JSON_DIR, "11kV_4.json"),
    ]

    topology_types = [
        "MESH",
        "RING_BUS",
        "BREAKER_AND_HALF",
        "MAIN_AND_TRANSFER",
        "DOUBLE_BUS"
    ]

    renderer.render(
        merged_json_path=merged_path,
        individual_json_paths=individual_json_paths,
        topology_types=topology_types,
        svg_output=SVG_OUTPUT_PATH,
        yolo_output=YOLO_OUTPUT_PATH
    )

    # ---------------- CLEAN TEMP AGAIN ----------------
    if os.path.exists(TEMP_JSON_DIR):
        shutil.rmtree(TEMP_JSON_DIR)


if __name__ == "__main__":
    main()