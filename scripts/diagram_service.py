"""
Main service for generating single line diagrams.
This service generates randomized diagrams based on set rules.
"""

import random
import os
import argparse
from scripts.legacy.components import ComponentGenerator
from layout_engine import LayoutEngine


class DiagramService:
    """Service for generating single line diagrams."""

    def __init__(self, output_dir="output", symbol_dir="symbols"):
        """Initialize the diagram service."""
        self.output_dir = output_dir
        self.symbol_dir = symbol_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(symbol_dir, exist_ok=True)

        self.component_gen = ComponentGenerator(symbol_dir)
        self.component_gen.generate_all_symbols()

        # Load all component symbols
        self.component_symbols = {}
        for comp_type in self.component_gen.get_component_types():
            self.component_symbols[comp_type] = self.component_gen.generate_symbol(
                comp_type
            )

    def generate_random_components(self, min_components=5, max_components=12):
        """Generate a random list of components with high randomization."""
        num_components = random.randint(min_components, max_components)
        component_types = self.component_gen.get_component_types()

        # More randomized weights - can vary per generation
        base_weights = {
            "generator": random.uniform(0.10, 0.20),
            "transformer": random.uniform(0.10, 0.20),
            "load": random.uniform(0.20, 0.30),
            "breaker": random.uniform(0.10, 0.20),
            "bus": random.uniform(0.05, 0.15),
            "capacitor": random.uniform(0.05, 0.15),
            "switch": random.uniform(0.05, 0.15),
        }

        # Normalize weights
        total = sum(base_weights.values())
        weights = {k: v / total for k, v in base_weights.items()}

        components = []
        for _ in range(num_components):
            # Sometimes use completely random selection instead of weighted
            if random.random() < 0.3:  # 30% chance for pure random
                comp_type = random.choice(component_types)
            else:
                comp_type = random.choices(
                    list(weights.keys()), weights=list(weights.values())
                )[0]
            components.append((comp_type, self.component_symbols[comp_type]))

        # Randomly shuffle component order for more variety
        random.shuffle(components)

        return components

    def generate_linear_diagram(self, num_diagrams=3):
        """Generate linear layout diagrams."""
        diagrams = []
        for i in range(num_diagrams):
            # Much larger component counts and canvas sizes
            num_comps = random.randint(15, 35)
            components = self.generate_random_components(num_comps, num_comps)
            # Huge canvas sizes
            width = random.randint(8000, 12000)
            height = random.randint(2000, 4000)
            engine = LayoutEngine(width=width, height=height)
            # Random starting positions
            start_x = random.randint(200, 500)
            start_y = random.randint(height // 4, height // 2)
            engine.generate_linear_layout(components, start_x=start_x, start_y=start_y)

            output_path = os.path.join(self.output_dir, f"linear_diagram_{i+1}.png")
            json_path = os.path.join(self.output_dir, f"linear_diagram_{i+1}.json")
            engine.render(output_path)
            engine.export_to_json(json_path)
            diagrams.append(output_path)
            print(f"Generated linear diagram: {output_path} ({width}x{height})")
            print(f"  JSON data: {json_path}")

        return diagrams

    def generate_bus_diagram(self, num_diagrams=3):
        """Generate bus layout diagrams."""
        diagrams = []
        for i in range(num_diagrams):
            # Much larger component counts
            num_comps = random.randint(20, 50)
            components = self.generate_random_components(num_comps, num_comps)
            # Huge canvas sizes
            width = random.randint(10000, 15000)
            height = random.randint(4000, 6000)
            engine = LayoutEngine(width=width, height=height)
            bus_symbol = self.component_symbols["bus"]
            # Random starting positions
            start_x = random.randint(200, 500)
            start_y = random.randint(300, 800)
            engine.generate_bus_layout(
                components, bus_symbol, start_x=start_x, start_y=start_y
            )

            output_path = os.path.join(self.output_dir, f"bus_diagram_{i+1}.png")
            json_path = os.path.join(self.output_dir, f"bus_diagram_{i+1}.json")
            engine.render(output_path)
            engine.export_to_json(json_path)
            diagrams.append(output_path)
            print(f"Generated bus diagram: {output_path} ({width}x{height})")
            print(f"  JSON data: {json_path}")

        return diagrams

    def generate_radial_diagram(self, num_diagrams=3):
        """Generate radial layout diagrams."""
        diagrams = []
        for i in range(num_diagrams):
            # Much larger component counts
            num_comps = random.randint(12, 30)
            components = self.generate_random_components(num_comps, num_comps)
            # Huge square canvas
            size = random.randint(6000, 10000)
            engine = LayoutEngine(width=size, height=size)
            # Random center position
            center_x = random.randint(size // 4, 3 * size // 4)
            center_y = random.randint(size // 4, 3 * size // 4)
            engine.generate_radial_layout(
                components, center_x=center_x, center_y=center_y
            )

            output_path = os.path.join(self.output_dir, f"radial_diagram_{i+1}.png")
            json_path = os.path.join(self.output_dir, f"radial_diagram_{i+1}.json")
            engine.render(output_path)
            engine.export_to_json(json_path)
            diagrams.append(output_path)
            print(f"Generated radial diagram: {output_path} ({size}x{size})")
            print(f"  JSON data: {json_path}")

        return diagrams

    def generate_mesh_diagram(self, num_diagrams=3):
        """Generate mesh/grid layout diagrams."""
        diagrams = []
        for i in range(num_diagrams):
            # Much larger component counts
            num_comps = random.randint(25, 60)
            components = self.generate_random_components(num_comps, num_comps)
            # Random grid dimensions
            rows = random.randint(4, 8)
            cols = random.randint(5, 12)
            # Huge canvas sizes
            width = random.randint(10000, 15000)
            height = random.randint(8000, 12000)
            engine = LayoutEngine(width=width, height=height)
            # Random starting positions
            start_x = random.randint(200, 500)
            start_y = random.randint(200, 500)
            engine.generate_mesh_layout(
                components, rows=rows, cols=cols, start_x=start_x, start_y=start_y
            )

            output_path = os.path.join(self.output_dir, f"mesh_diagram_{i+1}.png")
            json_path = os.path.join(self.output_dir, f"mesh_diagram_{i+1}.json")
            engine.render(output_path)
            engine.export_to_json(json_path)
            diagrams.append(output_path)
            print(
                f"Generated mesh diagram: {output_path} ({width}x{height}, {rows}x{cols} grid)"
            )
            print(f"  JSON data: {json_path}")

        return diagrams

    def generate_mesh_radial_diagram(self, num_diagrams=3):
        """Generate mesh+radial combined layout diagrams."""
        diagrams = []
        for i in range(num_diagrams):
            num_comps = random.randint(40, 80)
            components = self.generate_random_components(num_comps, num_comps)
            width = random.randint(12000, 18000)
            height = random.randint(10000, 15000)
            engine = LayoutEngine(width=width, height=height)

            num_centers = random.randint(3, 6)
            rows = random.randint(5, 10)
            cols = random.randint(6, 12)
            start_x = random.randint(200, 500)
            start_y = random.randint(200, 500)

            engine.generate_mesh_radial_layout(
                components,
                num_centers=num_centers,
                rows=rows,
                cols=cols,
                start_x=start_x,
                start_y=start_y,
            )

            output_path = os.path.join(
                self.output_dir, f"mesh_radial_diagram_{i+1}.png"
            )
            json_path = os.path.join(self.output_dir, f"mesh_radial_diagram_{i+1}.json")
            engine.render(output_path)
            engine.export_to_json(json_path)
            diagrams.append(output_path)
            print(
                f"Generated mesh+radial diagram: {output_path} ({width}x{height}, {num_centers} centers)"
            )
            print(f"  JSON data: {json_path}")

        return diagrams

    def generate_multi_bus_diagram(self, num_diagrams=3):
        """Generate multi-bus layout diagrams."""
        diagrams = []
        for i in range(num_diagrams):
            num_comps = random.randint(30, 70)
            components = self.generate_random_components(num_comps, num_comps)
            width = random.randint(12000, 18000)
            height = random.randint(8000, 12000)
            engine = LayoutEngine(width=width, height=height)

            num_buses = random.randint(3, 6)
            start_x = random.randint(200, 500)
            start_y = random.randint(200, 500)

            engine.generate_multi_bus_layout(
                components, num_buses=num_buses, start_x=start_x, start_y=start_y
            )

            output_path = os.path.join(self.output_dir, f"multi_bus_diagram_{i+1}.png")
            json_path = os.path.join(self.output_dir, f"multi_bus_diagram_{i+1}.json")
            engine.render(output_path)
            engine.export_to_json(json_path)
            diagrams.append(output_path)
            print(
                f"Generated multi-bus diagram: {output_path} ({width}x{height}, {num_buses} buses)"
            )
            print(f"  JSON data: {json_path}")

        return diagrams

    def generate_hierarchical_diagram(self, num_diagrams=3):
        """Generate hierarchical layout diagrams."""
        diagrams = []
        for i in range(num_diagrams):
            num_comps = random.randint(35, 75)
            components = self.generate_random_components(num_comps, num_comps)
            width = random.randint(12000, 18000)
            height = random.randint(10000, 15000)
            engine = LayoutEngine(width=width, height=height)

            levels = random.randint(3, 6)
            start_x = random.randint(200, 500)
            start_y = random.randint(200, 500)

            engine.generate_hierarchical_layout(
                components, levels=levels, start_x=start_x, start_y=start_y
            )

            output_path = os.path.join(
                self.output_dir, f"hierarchical_diagram_{i+1}.png"
            )
            json_path = os.path.join(
                self.output_dir, f"hierarchical_diagram_{i+1}.json"
            )
            engine.render(output_path)
            engine.export_to_json(json_path)
            diagrams.append(output_path)
            print(
                f"Generated hierarchical diagram: {output_path} ({width}x{height}, {levels} levels)"
            )
            print(f"  JSON data: {json_path}")

        return diagrams

    def generate_all_diagrams(self):
        """Generate all types of diagrams."""
        print("Generating single line diagrams...")
        print("=" * 50)

        all_diagrams = []

        # Generate basic layout types
        all_diagrams.extend(self.generate_linear_diagram(num_diagrams=2))
        all_diagrams.extend(self.generate_bus_diagram(num_diagrams=2))
        all_diagrams.extend(self.generate_radial_diagram(num_diagrams=2))
        all_diagrams.extend(self.generate_mesh_diagram(num_diagrams=2))

        # Generate complex combined layouts
        all_diagrams.extend(self.generate_mesh_radial_diagram(num_diagrams=3))
        all_diagrams.extend(self.generate_multi_bus_diagram(num_diagrams=3))
        all_diagrams.extend(self.generate_hierarchical_diagram(num_diagrams=3))

        print("=" * 50)
        print(f"Generated {len(all_diagrams)} diagrams in {self.output_dir}/")
        return all_diagrams

    def generate_bulk_diagrams(self, total_diagrams=500):
        """
        Generate a bulk set of diagrams distributed across all layout types.

        Args:
            total_diagrams: Total number of diagrams to generate.
        """
        print(f"Generating {total_diagrams} diagrams in bulk mode...")
        print("=" * 50)

        generators = [
            ("linear", self.generate_linear_diagram),
            ("bus", self.generate_bus_diagram),
            ("radial", self.generate_radial_diagram),
            ("mesh", self.generate_mesh_diagram),
            ("mesh_radial", self.generate_mesh_radial_diagram),
            ("multi_bus", self.generate_multi_bus_diagram),
            ("hierarchical", self.generate_hierarchical_diagram),
        ]

        num_types = len(generators)
        base_count = total_diagrams // num_types
        remainder = total_diagrams % num_types

        all_diagrams = []
        for idx, (name, generator) in enumerate(generators):
            count = base_count + (1 if idx < remainder else 0)
            if count <= 0:
                continue
            print(f"\n-> Generating {count} {name} diagrams...")
            all_diagrams.extend(generator(num_diagrams=count))

        print("=" * 50)
        print(
            f"Bulk generation complete. Generated {len(all_diagrams)} diagrams in {self.output_dir}/"
        )
        return all_diagrams

    def generate_custom_diagram(self, layout_type="linear", num_components=None):
        """Generate a single custom diagram."""
        if num_components is None:
            num_components = random.randint(6, 12)

        components = self.generate_random_components(num_components, num_components)
        engine = LayoutEngine(width=2500, height=1500)

        if layout_type == "linear":
            engine.generate_linear_layout(components, start_x=100, start_y=500)
        elif layout_type == "bus":
            bus_symbol = self.component_symbols["bus"]
            engine.generate_bus_layout(components, bus_symbol, start_x=100, start_y=100)
        elif layout_type == "radial":
            engine.generate_radial_layout(components)
        elif layout_type == "mesh":
            rows = random.randint(2, 4)
            cols = random.randint(3, 5)
            engine.generate_mesh_layout(components, rows=rows, cols=cols)
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")

        output_path = os.path.join(self.output_dir, f"custom_{layout_type}_diagram.png")
        json_path = os.path.join(self.output_dir, f"custom_{layout_type}_diagram.json")
        engine.render(output_path)
        engine.export_to_json(json_path)
        print(f"Generated custom diagram: {output_path}")
        print(f"  JSON data: {json_path}")
        return output_path


def main():
    """Main entry point for the diagram service."""
    parser = argparse.ArgumentParser(
        description="Single Line Diagram Generator Service"
    )
    parser.add_argument(
        "--bulk-count",
        type=int,
        default=None,
        help="Generate this many diagrams in bulk (distributed across layout types).",
    )
    args = parser.parse_args()

    service = DiagramService()

    if args.bulk_count is not None:
        service.generate_bulk_diagrams(total_diagrams=args.bulk_count)
    else:
        # Generate the default set of all diagram types
        service.generate_all_diagrams()
        # Optionally generate a custom diagram
        # service.generate_custom_diagram(layout_type='mesh', num_components=12)


if __name__ == "__main__":
    main()
