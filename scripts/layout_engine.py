"""
Layout engine for single line diagrams.
Handles component placement and connection routing (left/right only).
"""

from PIL import Image, ImageDraw
import random
import math
import json
from typing import List, Dict, Any, Tuple


class Component:
    """Represents a component in the diagram."""

    def __init__(
        self,
        comp_type,
        symbol_image,
        x,
        y,
        width=None,
        height=None,
        component_id=None,
        voltage=None,
    ):
        self.comp_type = comp_type
        self.symbol_image = symbol_image
        self.x = x
        self.y = y
        self.component_id = component_id  # Unique identifier for JSON export
        self.voltage = voltage  # Voltage value in volts
        if symbol_image is not None:
            self.width = symbol_image.width
            self.height = symbol_image.height
        else:
            # Allow manual width/height for components without images
            self.width = width if width is not None else 50
            self.height = height if height is not None else 50
        self.left_connection = (x, y + self.height // 2)
        self.right_connection = (x + self.width, y + self.height // 2)
        self.connected_left = []
        self.connected_right = []

    def to_dict(self):
        """Convert component to dictionary for JSON export."""
        return {
            "id": self.component_id,
            "type": self.comp_type,
            "voltage": self.voltage,
            "position": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "connection_points": {
                "left": {"x": self.left_connection[0], "y": self.left_connection[1]},
                "right": {"x": self.right_connection[0], "y": self.right_connection[1]},
            },
            "connections": {
                "left_connected_to": [
                    comp.component_id
                    for comp in self.connected_left
                    if comp.component_id
                ],
                "right_connected_to": [
                    comp.component_id
                    for comp in self.connected_right
                    if comp.component_id
                ],
            },
        }


class LayoutEngine:
    """Engine for generating diagram layouts with left/right connections."""

    COMPONENT_SPACING = 150  # Base horizontal spacing between components
    ROW_SPACING = 200  # Base vertical spacing between rows
    LINE_WIDTH = 4
    LABEL_MARGIN = 4  # Minimum gap between labels and components/other labels

    def __init__(self, width=2000, height=1500, randomize_spacing=True):
        """Initialize layout engine with optional randomized spacing."""
        self.width = width
        self.height = height
        self.components = []
        self.connections = []
        self.randomize_spacing = randomize_spacing
        self._component_counter = 0  # For generating unique component IDs
        # Randomize spacing based on canvas size
        if randomize_spacing:
            scale_factor = max(width, height) / 2000.0
            self.component_spacing = int(
                self.COMPONENT_SPACING * scale_factor * random.uniform(0.8, 1.5)
            )
            self.row_spacing = int(
                self.ROW_SPACING * scale_factor * random.uniform(0.8, 1.5)
            )
        else:
            self.component_spacing = self.COMPONENT_SPACING
            self.row_spacing = self.ROW_SPACING

    def _generate_voltage(self, comp_type):
        """Generate a random voltage value based on component type."""
        # Different voltage ranges for different component types
        voltage_ranges = {
            "generator": (11000, 23000),  # 11kV to 23kV
            "transformer": (400, 33000),  # 400V to 33kV
            "load": (230, 415),  # 230V to 415V
            "breaker": (400, 33000),  # 400V to 33kV
            "bus": (400, 33000),  # 400V to 33kV
            "capacitor": (400, 11000),  # 400V to 11kV
            "switch": (400, 33000),  # 400V to 33kV
        }

        if comp_type in voltage_ranges:
            min_v, max_v = voltage_ranges[comp_type]
            # Generate voltage in steps (e.g., 230, 400, 415, 11000, 22000, etc.)
            voltage = random.randint(min_v, max_v)
            # Round to common voltage levels
            if voltage < 1000:
                voltage = random.choice([230, 400, 415, 480])
            elif voltage < 5000:
                voltage = random.choice([1100, 2300, 3300, 4150])
            elif voltage < 15000:
                voltage = random.choice([6600, 11000, 13200])
            else:
                voltage = random.choice([22000, 33000, 66000])
        else:
            voltage = random.choice([230, 400, 11000, 22000])

        return voltage

    def add_component(
        self, comp_type, symbol_image, x, y, width=None, height=None, voltage=None
    ):
        """Add a component to the layout."""
        self._component_counter += 1
        component_id = f"{comp_type}_{self._component_counter}"
        if voltage is None:
            voltage = self._generate_voltage(comp_type)
        component = Component(
            comp_type,
            symbol_image,
            x,
            y,
            width=width,
            height=height,
            component_id=component_id,
            voltage=voltage,
        )
        self.components.append(component)
        return component

    def connect_components(self, comp1, comp2, use_left_right=True):
        """
        Connect two components.
        If use_left_right is True, connects left-right only.
        """
        if use_left_right:
            # Determine which side to connect based on positions
            if comp1.x < comp2.x:
                # comp1 is left, comp2 is right
                comp1.connected_right.append(comp2)
                comp2.connected_left.append(comp1)
                self.connections.append((comp1.right_connection, comp2.left_connection))
            else:
                # comp2 is left, comp1 is right
                comp2.connected_right.append(comp1)
                comp1.connected_left.append(comp2)
                self.connections.append((comp2.right_connection, comp1.left_connection))
        else:
            # Direct connection (shouldn't be used in this implementation)
            comp1.connected_right.append(comp2)
            comp2.connected_left.append(comp1)
            self.connections.append((comp1.right_connection, comp2.left_connection))

    def route_connection(self, point1, point2):
        """
        Route a connection between two points using only horizontal and vertical lines.
        Returns list of points for the path.
        """
        x1, y1 = point1
        x2, y2 = point2

        # Simple L-shaped routing (horizontal then vertical, or vertical then horizontal)
        # Choose based on which is shorter
        path1 = [point1, (x2, y1), point2]  # Horizontal then vertical
        path2 = [point1, (x1, y2), point2]  # Vertical then horizontal

        # Calculate total length for each path
        len1 = abs(x2 - x1) + abs(y2 - y1)
        len2 = abs(x1 - x1) + abs(y2 - y1)  # Same as len1, but let's use path1

        # Use path1 (horizontal then vertical) for consistency
        return path1

    def generate_linear_layout(self, components_data, start_x=100, start_y=100):
        """Generate a linear horizontal layout with randomization."""
        x = start_x
        y = start_y

        # Random vertical variation for more interesting layouts
        vertical_variation = random.randint(0, 100) if self.randomize_spacing else 0

        for comp_type, symbol_image in components_data:
            # Add random vertical offset for some components
            y_offset = (
                random.randint(-vertical_variation, vertical_variation)
                if random.random() < 0.3
                else 0
            )
            comp = self.add_component(comp_type, symbol_image, x, y + y_offset)
            # Random spacing variation
            spacing = (
                self.component_spacing + random.randint(-30, 50)
                if self.randomize_spacing
                else self.component_spacing
            )
            x += spacing

        # Connect components sequentially
        for i in range(len(self.components) - 1):
            self.connect_components(self.components[i], self.components[i + 1])

    def generate_bus_layout(
        self, components_data, bus_component, start_x=100, start_y=100
    ):
        """Generate a bus layout with components connected to a bus."""
        # Place bus in the middle - create as a long horizontal line
        bus_y = start_y + 300
        bus_width = self.width - 200
        self._component_counter += 1
        bus = Component(
            "bus",
            None,
            start_x,
            bus_y - 10,
            width=bus_width,
            height=20,
            component_id=f"bus_{self._component_counter}",
            voltage=self._generate_voltage("bus"),
        )
        bus.left_connection = (start_x, bus_y)
        bus.right_connection = (start_x + bus_width, bus_y)
        self.components.append(bus)

        # Place components above and below the bus
        x = start_x + 150
        above_y = start_y
        below_y = bus_y + 150

        for i, (comp_type, symbol_image) in enumerate(components_data):
            if i % 2 == 0:
                y = above_y
            else:
                y = below_y

            comp = self.add_component(comp_type, symbol_image, x, y)

            # Connect to bus using proper left/right connections
            # Components connect horizontally to a vertical drop, then to bus
            bus_connection_x = x
            bus_connection_y = bus_y

            if y < bus_y:
                # Component above bus - connect from right side down
                comp_y_center = comp.y + comp.height // 2
                # Route: comp right -> horizontal -> vertical down -> bus
                mid_x = comp.right_connection[0] + 30
                self.connections.append((comp.right_connection, (mid_x, comp_y_center)))
                self.connections.append(
                    ((mid_x, comp_y_center), (mid_x, bus_connection_y))
                )
                self.connections.append(
                    ((mid_x, bus_connection_y), (bus_connection_x, bus_connection_y))
                )
            else:
                # Component below bus - connect from left side up
                comp_y_center = comp.y + comp.height // 2
                # Route: comp left -> horizontal -> vertical up -> bus
                mid_x = comp.left_connection[0] - 30
                self.connections.append((comp.left_connection, (mid_x, comp_y_center)))
                self.connections.append(
                    ((mid_x, comp_y_center), (mid_x, bus_connection_y))
                )
                self.connections.append(
                    ((mid_x, bus_connection_y), (bus_connection_x, bus_connection_y))
                )

            # Random spacing variation
            spacing = (
                self.component_spacing + random.randint(-30, 50)
                if self.randomize_spacing
                else self.component_spacing
            )
            x += spacing

    def generate_radial_layout(self, components_data, center_x=None, center_y=None):
        """Generate a radial layout with components arranged around a center."""
        if center_x is None:
            center_x = self.width // 2
        if center_y is None:
            center_y = self.height // 2

        num_components = len(components_data)
        # Randomize radius for more variation
        base_radius = min(self.width, self.height) // 3
        radius = (
            int(base_radius * random.uniform(0.6, 0.9))
            if self.randomize_spacing
            else base_radius
        )

        # Store component positions with angles for proper connection routing
        component_data = []

        # Place components in a circle with random angle offset for variation
        angle_offset = random.uniform(0, 2 * math.pi) if self.randomize_spacing else 0
        for i, (comp_type, symbol_image) in enumerate(components_data):
            angle = 2 * math.pi * i / num_components + angle_offset
            x = center_x + radius * math.cos(angle) - symbol_image.width // 2
            y = center_y + radius * math.sin(angle) - symbol_image.height // 2

            comp = self.add_component(comp_type, symbol_image, int(x), int(y))
            component_data.append((comp, angle))

        # Connect to center (create a central bus)
        self._component_counter += 1
        center_comp = Component(
            "bus",
            None,
            center_x - 25,
            center_y - 25,
            width=50,
            height=50,
            component_id=f"bus_{self._component_counter}",
            voltage=self._generate_voltage("bus"),
        )
        center_comp.left_connection = (center_x - 25, center_y)
        center_comp.right_connection = (center_x + 25, center_y)
        self.components.append(center_comp)

        # Connect all components to center using proper left/right connections
        for comp, angle in component_data:
            comp_center_x = comp.x + comp.width // 2
            comp_center_y = comp.y + comp.height // 2

            # Determine which side to use based on angle
            # Components on left side (90 to 270 degrees) use right connection
            # Components on right side (270 to 90 degrees) use left connection
            angle_deg = math.degrees(angle) % 360

            if 90 <= angle_deg <= 270:
                # Component is on left side - use right connection
                comp_conn = comp.right_connection
                # Route: comp right -> horizontal -> vertical -> center left
                mid_x = (comp.right_connection[0] + center_comp.left_connection[0]) // 2
                self.connections.append((comp_conn, (mid_x, comp_center_y)))
                self.connections.append(((mid_x, comp_center_y), (mid_x, center_y)))
                self.connections.append(
                    ((mid_x, center_y), center_comp.left_connection)
                )
            else:
                # Component is on right side - use left connection
                comp_conn = comp.left_connection
                # Route: comp left -> horizontal -> vertical -> center right
                mid_x = (comp.left_connection[0] + center_comp.right_connection[0]) // 2
                self.connections.append((comp_conn, (mid_x, comp_center_y)))
                self.connections.append(((mid_x, comp_center_y), (mid_x, center_y)))
                self.connections.append(
                    ((mid_x, center_y), center_comp.right_connection)
                )

    def generate_mesh_layout(
        self, components_data, rows=3, cols=4, start_x=100, start_y=100
    ):
        """Generate a mesh/grid layout with components in rows and columns."""
        components_2d = []
        idx = 0

        for row in range(rows):
            components_2d.append([])
            for col in range(cols):
                if idx < len(components_data):
                    comp_type, symbol_image = components_data[idx]
                    # Use randomized spacing
                    spacing_x = self.component_spacing + (
                        random.randint(-20, 30)
                        if self.randomize_spacing and random.random() < 0.3
                        else 0
                    )
                    spacing_y = self.row_spacing + (
                        random.randint(-20, 30)
                        if self.randomize_spacing and random.random() < 0.3
                        else 0
                    )
                    x = start_x + col * spacing_x
                    y = start_y + row * spacing_y
                    comp = self.add_component(comp_type, symbol_image, x, y)
                    components_2d[row].append(comp)
                    idx += 1

        # Connect horizontally within rows
        for row in components_2d:
            for i in range(len(row) - 1):
                self.connect_components(row[i], row[i + 1])

        # Connect vertically between rows (using horizontal routing)
        for row_idx in range(len(components_2d) - 1):
            current_row = components_2d[row_idx]
            next_row = components_2d[row_idx + 1]

            # Connect each component to corresponding component in next row
            for i in range(min(len(current_row), len(next_row))):
                comp1 = current_row[i]
                comp2 = next_row[i]

                # Route: comp1 right -> horizontal -> vertical -> horizontal -> comp2 left
                mid_x = (comp1.x + comp1.width + comp2.x) // 2
                mid_y1 = comp1.y + comp1.height // 2
                mid_y2 = comp2.y + comp2.height // 2

                self.connections.append((comp1.right_connection, (mid_x, mid_y1)))
                self.connections.append(((mid_x, mid_y1), (mid_x, mid_y2)))
                self.connections.append(((mid_x, mid_y2), comp2.left_connection))

    def generate_mesh_radial_layout(
        self, components_data, num_centers=3, rows=4, cols=5, start_x=100, start_y=100
    ):
        """Generate an integrated mesh+radial layout where components are part of both structures."""
        # Create mesh grid - these components will also connect radially
        components_2d = []
        idx = 0
        mesh_start_x = start_x
        mesh_start_y = start_y

        for row in range(rows):
            components_2d.append([])
            for col in range(cols):
                if idx < len(components_data):
                    comp_type, symbol_image = components_data[idx]
                    spacing_x = self.component_spacing + (
                        random.randint(-20, 30)
                        if self.randomize_spacing and random.random() < 0.3
                        else 0
                    )
                    spacing_y = self.row_spacing + (
                        random.randint(-20, 30)
                        if self.randomize_spacing and random.random() < 0.3
                        else 0
                    )
                    x = mesh_start_x + col * spacing_x
                    y = mesh_start_y + row * spacing_y
                    comp = self.add_component(comp_type, symbol_image, x, y)
                    components_2d[row].append(comp)
                    idx += 1

        # Connect mesh horizontally (mesh structure)
        for row in components_2d:
            for i in range(len(row) - 1):
                self.connect_components(row[i], row[i + 1])

        # Connect mesh vertically (mesh structure)
        for row_idx in range(len(components_2d) - 1):
            current_row = components_2d[row_idx]
            next_row = components_2d[row_idx + 1]
            for i in range(min(len(current_row), len(next_row))):
                comp1 = current_row[i]
                comp2 = next_row[i]
                mid_x = (comp1.x + comp1.width + comp2.x) // 2
                mid_y1 = comp1.y + comp1.height // 2
                mid_y2 = comp2.y + comp2.height // 2
                self.connections.append((comp1.right_connection, (mid_x, mid_y1)))
                self.connections.append(((mid_x, mid_y1), (mid_x, mid_y2)))
                self.connections.append(((mid_x, mid_y2), comp2.left_connection))

        # Create radial centers from mesh components - select some mesh components as centers
        radial_centers = []
        center_spacing_x = (self.width - 200) // (num_centers + 1)

        # Choose mesh components as radial centers (integrated approach)
        center_cols = [
            int((i + 1) * cols / (num_centers + 1)) for i in range(num_centers)
        ]
        center_row = rows // 2

        for i, center_col in enumerate(center_cols):
            if (
                center_col < cols
                and center_row < rows
                and len(components_2d[center_row]) > center_col
            ):
                # Use existing mesh component as center
                center_comp = components_2d[center_row][center_col]
                center_x = center_comp.x + center_comp.width // 2
                center_y = center_comp.y + center_comp.height // 2
                radial_centers.append((center_comp, center_x, center_y))
            else:
                # Create a bus center if mesh component not available
                center_x = center_spacing_x * (i + 1)
                center_y = mesh_start_y + center_row * self.row_spacing
                self._component_counter += 1
                center_comp = Component(
                    "bus",
                    None,
                    center_x - 25,
                    center_y - 25,
                    width=50,
                    height=50,
                    component_id=f"bus_{self._component_counter}",
                    voltage=self._generate_voltage("bus"),
                )
                center_comp.left_connection = (center_x - 25, center_y)
                center_comp.right_connection = (center_x + 25, center_y)
                self.components.append(center_comp)
                radial_centers.append((center_comp, center_x, center_y))

        # Connect ALL mesh components to their nearest radial center (radial structure)
        for row in components_2d:
            for comp in row:
                comp_center_x = comp.x + comp.width // 2
                comp_center_y = comp.y + comp.height // 2

                # Find nearest radial center
                nearest_center = min(
                    radial_centers,
                    key=lambda c: (
                        (comp_center_x - c[1]) ** 2 + (comp_center_y - c[2]) ** 2
                    )
                    ** 0.5,
                )
                center_comp, center_x, center_y = nearest_center

                # Skip if this component IS the center
                if comp == center_comp:
                    continue

                # Connect to center using proper left/right routing
                if comp_center_x < center_x:
                    # Component left of center - use right connection
                    mid_x = (
                        comp.right_connection[0] + center_comp.left_connection[0]
                    ) // 2
                    comp_y = comp.right_connection[1]
                    self.connections.append((comp.right_connection, (mid_x, comp_y)))
                    self.connections.append(((mid_x, comp_y), (mid_x, center_y)))
                    self.connections.append(
                        ((mid_x, center_y), center_comp.left_connection)
                    )
                else:
                    # Component right of center - use left connection
                    mid_x = (
                        comp.left_connection[0] + center_comp.right_connection[0]
                    ) // 2
                    comp_y = comp.left_connection[1]
                    self.connections.append((comp.left_connection, (mid_x, comp_y)))
                    self.connections.append(((mid_x, comp_y), (mid_x, center_y)))
                    self.connections.append(
                        ((mid_x, center_y), center_comp.right_connection)
                    )

        # Connect radial centers to each other (inter-center connections)
        for i in range(len(radial_centers) - 1):
            center1, x1, y1 = radial_centers[i]
            center2, x2, y2 = radial_centers[i + 1]
            # Connect centers horizontally
            mid_y = (y1 + y2) // 2
            self.connections.append((center1.right_connection, (x1 + 50, mid_y)))
            self.connections.append(((x1 + 50, mid_y), (x2 - 50, mid_y)))
            self.connections.append(((x2 - 50, mid_y), center2.left_connection))

    def generate_multi_bus_layout(
        self, components_data, num_buses=3, start_x=100, start_y=100
    ):
        """Generate an integrated multi-bus layout where buses connect to each other and components span buses."""
        # Create multiple buses at different levels
        buses = []
        bus_spacing_y = (self.height - 400) // (num_buses + 1)

        for i in range(num_buses):
            bus_y = start_y + bus_spacing_y * (i + 1)
            bus_width = self.width - 200
            self._component_counter += 1
            bus = Component(
                "bus",
                None,
                start_x,
                bus_y - 10,
                width=bus_width,
                height=20,
                component_id=f"bus_{self._component_counter}",
                voltage=self._generate_voltage("bus"),
            )
            bus.left_connection = (start_x, bus_y)
            bus.right_connection = (start_x + bus_width, bus_y)
            self.components.append(bus)
            buses.append((bus, bus_y))

        # Connect buses to each other vertically (integrated bus structure)
        for i in range(len(buses) - 1):
            bus1, y1 = buses[i]
            bus2, y2 = buses[i + 1]
            # Connect buses at multiple points
            for conn_x in [
                start_x + bus_width // 4,
                start_x + bus_width // 2,
                start_x + 3 * bus_width // 4,
            ]:
                mid_y = (y1 + y2) // 2
                self.connections.append(((conn_x, y1), (conn_x, mid_y)))
                self.connections.append(((conn_x, mid_y), (conn_x, y2)))

        # Distribute components - some connect to single bus, some span multiple buses
        x = start_x + 200
        comp_idx = 0

        for i, (bus, bus_y) in enumerate(buses):
            # Calculate how many components for this bus
            remaining_buses = num_buses - i
            remaining_comps = len(components_data) - comp_idx
            comps_for_this_bus = (
                remaining_comps // remaining_buses if remaining_buses > 0 else 0
            )

            for j in range(comps_for_this_bus):
                if comp_idx >= len(components_data):
                    break

                comp_type, symbol_image = components_data[comp_idx]

                # Some components connect to single bus, some span to adjacent buses
                if random.random() < 0.3 and i < len(buses) - 1:
                    # Component spans to next bus - place between buses
                    next_bus, next_y = buses[i + 1]
                    y = (bus_y + next_y) // 2

                    comp = self.add_component(comp_type, symbol_image, x, y)

                    # Connect to both buses
                    bus_conn_x = x
                    # Connect to current bus
                    mid_x1 = comp.left_connection[0] - 30
                    self.connections.append(
                        (comp.left_connection, (mid_x1, comp.y + comp.height // 2))
                    )
                    self.connections.append(
                        ((mid_x1, comp.y + comp.height // 2), (mid_x1, bus_y))
                    )
                    self.connections.append(((mid_x1, bus_y), (bus_conn_x, bus_y)))
                    # Connect to next bus
                    mid_x2 = comp.right_connection[0] + 30
                    self.connections.append(
                        (comp.right_connection, (mid_x2, comp.y + comp.height // 2))
                    )
                    self.connections.append(
                        ((mid_x2, comp.y + comp.height // 2), (mid_x2, next_y))
                    )
                    self.connections.append(((mid_x2, next_y), (bus_conn_x, next_y)))
                else:
                    # Component connects to single bus
                    if j % 2 == 0:
                        y = bus_y - 200
                    else:
                        y = bus_y + 150

                    comp = self.add_component(comp_type, symbol_image, x, y)

                    # Connect to bus
                    bus_connection_x = x
                    bus_connection_y = bus_y

                    if y < bus_y:
                        comp_y_center = comp.y + comp.height // 2
                        mid_x = comp.right_connection[0] + 30
                        self.connections.append(
                            (comp.right_connection, (mid_x, comp_y_center))
                        )
                        self.connections.append(
                            ((mid_x, comp_y_center), (mid_x, bus_connection_y))
                        )
                        self.connections.append(
                            (
                                (mid_x, bus_connection_y),
                                (bus_connection_x, bus_connection_y),
                            )
                        )
                    else:
                        comp_y_center = comp.y + comp.height // 2
                        mid_x = comp.left_connection[0] - 30
                        self.connections.append(
                            (comp.left_connection, (mid_x, comp_y_center))
                        )
                        self.connections.append(
                            ((mid_x, comp_y_center), (mid_x, bus_connection_y))
                        )
                        self.connections.append(
                            (
                                (mid_x, bus_connection_y),
                                (bus_connection_x, bus_connection_y),
                            )
                        )

                spacing = (
                    self.component_spacing + random.randint(-30, 50)
                    if self.randomize_spacing
                    else self.component_spacing
                )
                x += spacing
                comp_idx += 1

            # Reset x for next bus with some variation
            x = start_x + 200 + random.randint(-50, 50)

    def generate_hierarchical_layout(
        self, components_data, levels=3, start_x=100, start_y=100
    ):
        """Generate an integrated hierarchical layout where levels connect to each other organically."""
        # Split components into levels
        comps_per_level = len(components_data) // levels
        level_height = (self.height - 200) // levels
        level_components_list = []

        for level in range(levels):
            start_idx = level * comps_per_level
            end_idx = (
                start_idx + comps_per_level
                if level < levels - 1
                else len(components_data)
            )
            level_components = components_data[start_idx:end_idx]
            level_y = start_y + level * level_height + 100
            level_components_list.append((level_components, level_y, level))

        # Place components at each level with different patterns
        level_comps_dict = {}  # level -> list of components

        for level_components, level_y, level in level_components_list:
            level_comps = []

            if level % 3 == 0:
                # Linear layout for this level
                x = start_x
                for comp_type, symbol_image in level_components:
                    comp = self.add_component(comp_type, symbol_image, x, level_y)
                    spacing = (
                        self.component_spacing + random.randint(-30, 50)
                        if self.randomize_spacing
                        else self.component_spacing
                    )
                    x += spacing
                    level_comps.append(comp)

                # Connect sequentially
                for i in range(len(level_comps) - 1):
                    self.connect_components(level_comps[i], level_comps[i + 1])

            elif level % 3 == 1:
                # Radial sub-layout for this level
                center_x = self.width // 2
                center_y = level_y
                radius = min(self.width, level_height) // 4

                self._component_counter += 1
                center_comp = Component(
                    "bus",
                    None,
                    center_x - 25,
                    center_y - 25,
                    width=50,
                    height=50,
                    component_id=f"bus_{self._component_counter}",
                    voltage=self._generate_voltage("bus"),
                )
                center_comp.left_connection = (center_x - 25, center_y)
                center_comp.right_connection = (center_x + 25, center_y)
                self.components.append(center_comp)
                level_comps.append(center_comp)

                angle_offset = (
                    random.uniform(0, 2 * math.pi) if self.randomize_spacing else 0
                )
                for i, (comp_type, symbol_image) in enumerate(level_components):
                    angle = 2 * math.pi * i / len(level_components) + angle_offset
                    x = center_x + radius * math.cos(angle) - symbol_image.width // 2
                    y = center_y + radius * math.sin(angle) - symbol_image.height // 2
                    comp = self.add_component(comp_type, symbol_image, int(x), int(y))
                    level_comps.append(comp)

                    # Connect to center
                    angle_deg = math.degrees(angle) % 360
                    if 90 <= angle_deg <= 270:
                        mid_x = (
                            comp.right_connection[0] + center_comp.left_connection[0]
                        ) // 2
                        self.connections.append(
                            (comp.right_connection, (mid_x, comp.y + comp.height // 2))
                        )
                        self.connections.append(
                            ((mid_x, comp.y + comp.height // 2), (mid_x, center_y))
                        )
                        self.connections.append(
                            ((mid_x, center_y), center_comp.left_connection)
                        )
                    else:
                        mid_x = (
                            comp.left_connection[0] + center_comp.right_connection[0]
                        ) // 2
                        self.connections.append(
                            (comp.left_connection, (mid_x, comp.y + comp.height // 2))
                        )
                        self.connections.append(
                            ((mid_x, comp.y + comp.height // 2), (mid_x, center_y))
                        )
                        self.connections.append(
                            ((mid_x, center_y), center_comp.right_connection)
                        )

            else:
                # Bus layout for this level
                bus_y = level_y
                bus_width = self.width - 200
                self._component_counter += 1
                bus = Component(
                    "bus",
                    None,
                    start_x,
                    bus_y - 10,
                    width=bus_width,
                    height=20,
                    component_id=f"bus_{self._component_counter}",
                    voltage=self._generate_voltage("bus"),
                )
                bus.left_connection = (start_x, bus_y)
                bus.right_connection = (start_x + bus_width, bus_y)
                self.components.append(bus)
                level_comps.append(bus)

                x = start_x + 200
                for j, (comp_type, symbol_image) in enumerate(level_components):
                    if j % 2 == 0:
                        y = bus_y - 150
                    else:
                        y = bus_y + 150

                    comp = self.add_component(comp_type, symbol_image, x, y)
                    level_comps.append(comp)

                    bus_connection_x = x
                    if y < bus_y:
                        mid_x = comp.right_connection[0] + 30
                        self.connections.append(
                            (comp.right_connection, (mid_x, comp.y + comp.height // 2))
                        )
                        self.connections.append(
                            ((mid_x, comp.y + comp.height // 2), (mid_x, bus_y))
                        )
                        self.connections.append(
                            ((mid_x, bus_y), (bus_connection_x, bus_y))
                        )
                    else:
                        mid_x = comp.left_connection[0] - 30
                        self.connections.append(
                            (comp.left_connection, (mid_x, comp.y + comp.height // 2))
                        )
                        self.connections.append(
                            ((mid_x, comp.y + comp.height // 2), (mid_x, bus_y))
                        )
                        self.connections.append(
                            ((mid_x, bus_y), (bus_connection_x, bus_y))
                        )

                    spacing = (
                        self.component_spacing + random.randint(-30, 50)
                        if self.randomize_spacing
                        else self.component_spacing
                    )
                    x += spacing

            level_comps_dict[level] = level_comps

        # Connect levels to each other organically
        for level in range(levels - 1):
            current_comps = level_comps_dict[level]
            next_comps = level_comps_dict[level + 1]

            # Connect components between levels - select key components to connect
            num_connections = min(len(current_comps), len(next_comps), 5)
            current_selected = random.sample(
                [c for c in current_comps if c.comp_type != "bus"],
                min(
                    num_connections,
                    len([c for c in current_comps if c.comp_type != "bus"]),
                ),
            )
            next_selected = random.sample(
                [c for c in next_comps if c.comp_type != "bus"],
                min(
                    num_connections,
                    len([c for c in next_comps if c.comp_type != "bus"]),
                ),
            )

            for i in range(min(len(current_selected), len(next_selected))):
                comp1 = current_selected[i]
                comp2 = next_selected[i]

                # Route connection between levels
                mid_x = (comp1.x + comp1.width + comp2.x) // 2
                mid_y1 = comp1.y + comp1.height // 2
                mid_y2 = comp2.y + comp2.height // 2

                self.connections.append((comp1.right_connection, (mid_x, mid_y1)))
                self.connections.append(((mid_x, mid_y1), (mid_x, mid_y2)))
                self.connections.append(((mid_x, mid_y2), comp2.left_connection))

    def _rects_overlap(self, a, b, margin=0):
        """Check if two rectangles (x1, y1, x2, y2) overlap with optional margin."""
        ax1, ay1, ax2, ay2 = a[0] - margin, a[1] - margin, a[2] + margin, a[3] + margin
        bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def _find_label_position(self, comp, label_w, label_h, occupied_rects, padding):
        """Find a non-overlapping position for a label near the component.
        Returns (x, y) or None, keeping labels as close as possible while
        still avoiding components, lines, and other labels.

        Strategy:
        - Prefer a tight ring around the component (just outside its bounds).
        - Search in multiple angles around the component (not just 4 directions),
          gradually growing the radius but capped so labels never go far away.
        """
        margin = self.LABEL_MARGIN
        gap = padding + margin  # base gap between label edge and component

        base_spacing = max(self.component_spacing, self.row_spacing)
        # Keep labels visually \"adjacent\": don't let radius grow beyond a small fraction
        # of the grid spacing, or more than a few label heights.
        max_radius = min(base_spacing * 0.4, max(label_h * 3, 40))
        radius_step = max(label_h // 2, 10)
        angle_steps = 12  # sample 12 directions around the component

        center_x = comp.x + comp.width // 2
        center_y = comp.y + comp.height // 2

        radius = gap
        while radius <= max_radius:
            for i in range(angle_steps):
                angle = 2 * math.pi * i / angle_steps
                # Center of the label at (cx, cy) on the ring around the component
                cx = center_x + radius * math.cos(angle)
                cy = center_y + radius * math.sin(angle)
                lx = int(cx - label_w / 2)
                ly = int(cy - label_h / 2)

                box = (
                    lx - padding,
                    ly - padding,
                    lx + label_w + padding,
                    ly + label_h + padding,
                )

                # Must be inside canvas
                if (
                    box[0] < 0
                    or box[1] < 0
                    or box[2] > self.width
                    or box[3] > self.height
                ):
                    continue

                # No overlap with this component (label must be at least margin away)
                comp_box = (comp.x, comp.y, comp.x + comp.width, comp.y + comp.height)
                if self._rects_overlap(box, comp_box, margin):
                    continue

                # No overlap with any other occupied rect (other components, lines, labels)
                if any(self._rects_overlap(box, r) for r in occupied_rects):
                    continue

                return (lx, ly)

            radius += radius_step

        # If we reach here, we couldn't find a valid non-overlapping position
        return None

    def render(self, output_path="diagram_output.png"):
        """Render the diagram to an image."""
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img)
        line_rects = []

        # Draw connections first (so components appear on top)
        for connection in self.connections:
            if isinstance(connection, tuple) and len(connection) == 2:
                point1, point2 = connection
                # Check if points are tuples (coordinates)
                if isinstance(point1, tuple) and isinstance(point2, tuple):
                    path = self.route_connection(point1, point2)
                    # Draw the path
                    for i in range(len(path) - 1):
                        p1 = path[i]
                        p2 = path[i + 1]
                        draw.line([p1, p2], fill="black", width=self.LINE_WIDTH)
                        # Record occupied area around this line segment (thin buffer so labels can be nearby)
                        margin = self.LINE_WIDTH + self.LABEL_MARGIN // 2
                        x1, y1 = p1
                        x2, y2 = p2
                        line_rects.append(
                            (
                                min(x1, x2) - margin,
                                min(y1, y2) - margin,
                                max(x1, x2) + margin,
                                max(y1, y2) + margin,
                            )
                        )
                else:
                    # Direct line if points are already coordinates
                    draw.line([point1, point2], fill="black", width=self.LINE_WIDTH)
                    margin = self.LINE_WIDTH + self.LABEL_MARGIN // 2
                    x1, y1 = point1
                    x2, y2 = point2
                    line_rects.append(
                        (
                            min(x1, x2) - margin,
                            min(y1, y2) - margin,
                            max(x1, x2) + margin,
                            max(y1, y2) + margin,
                        )
                    )
            elif isinstance(connection, (list, tuple)) and len(connection) > 2:
                # Handle multi-point connections
                for i in range(len(connection) - 1):
                    p1 = connection[i]
                    p2 = connection[i + 1]
                    draw.line([p1, p2], fill="black", width=self.LINE_WIDTH)
                    margin = self.LINE_WIDTH + self.LABEL_MARGIN // 2
                    x1, y1 = p1
                    x2, y2 = p2
                    line_rects.append(
                        (
                            min(x1, x2) - margin,
                            min(y1, y2) - margin,
                            max(x1, x2) + margin,
                            max(y1, y2) + margin,
                        )
                    )

        # Draw components
        for comp in self.components:
            if comp.symbol_image:
                # Convert to RGBA if needed for transparency
                if comp.symbol_image.mode != "RGBA":
                    comp_img = comp.symbol_image.convert("RGBA")
                else:
                    comp_img = comp.symbol_image
                img.paste(comp_img, (comp.x, comp.y), comp_img)
            elif comp.comp_type == "bus":
                # Draw bus as a thick horizontal line
                bus_y = comp.y + comp.height // 2
                draw.line(
                    [(comp.x, bus_y), (comp.x + comp.width, bus_y)],
                    fill="black",
                    width=8,
                )

        # Draw labels for components (component name and voltage) in empty space only
        try:
            from PIL import ImageFont

            font_size = max(12, min(self.width, self.height) // 100)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        padding = 2
        margin = self.LABEL_MARGIN
        # Occupied rectangles: components + lines + labels (labels must not overlap these)
        occupied_rects = []
        for c in self.components:
            occupied_rects.append(
                (
                    c.x - margin,
                    c.y - margin,
                    c.x + c.width + margin,
                    c.y + c.height + margin,
                )
            )
        # Add line rectangles so labels don't encroach on drawn lines
        occupied_rects.extend(line_rects)

        for comp in self.components:
            comp_name = comp.comp_type.capitalize()
            if comp.voltage:
                voltage_str = (
                    f"{comp.voltage // 1000}kV"
                    if comp.voltage >= 1000
                    else f"{comp.voltage}V"
                )
            else:
                voltage_str = "N/A"
            label_text = f"{comp_name}\n{voltage_str}"
            lines = label_text.split("\n")
            line_spacing = 4
            max_width = 0
            total_height = 0
            try:
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_width = bbox[2] - bbox[0]
                    line_height = bbox[3] - bbox[1]
                    max_width = max(max_width, line_width)
                    total_height += line_height
                if len(lines) > 1:
                    total_height += line_spacing * (len(lines) - 1)
            except Exception:
                max_width = 40
                total_height = 20

            pos = self._find_label_position(
                comp, max_width, total_height, occupied_rects, padding
            )
            if pos is None:
                continue
            label_x, label_y = pos

            # Draw background and text
            try:
                draw.rectangle(
                    [
                        (label_x - padding, label_y - padding),
                        (
                            label_x + max_width + padding,
                            label_y + total_height + padding,
                        ),
                    ],
                    fill="white",
                    outline="white",
                )
                current_y = label_y
                for idx, line in enumerate(lines):
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_height = bbox[3] - bbox[1]
                    draw.text((label_x, current_y), line, fill="black", font=font)
                    current_y += line_height + (
                        line_spacing if idx < len(lines) - 1 else 0
                    )
            except Exception:
                draw.text((label_x, label_y), label_text, fill="black", font=font)

            # Reserve this label area so future labels don't overlap
            occupied_rects.append(
                (
                    label_x - padding - margin,
                    label_y - padding - margin,
                    label_x + max_width + padding + margin,
                    label_y + total_height + padding + margin,
                )
            )

        img.save(output_path)
        return output_path

    def export_to_json(self, output_path):
        """
        Export diagram data to JSON file.

        Args:
            output_path: Path to save JSON file

        Returns:
            Path to saved JSON file
        """
        # Build connections list with component IDs
        connections_data = []
        for connection in self.connections:
            if isinstance(connection, tuple) and len(connection) == 2:
                point1, point2 = connection
                # Find components that these points belong to
                comp1_id = self._find_component_by_point(point1)
                comp2_id = self._find_component_by_point(point2)

                connections_data.append(
                    {
                        "from_point": {"x": point1[0], "y": point1[1]},
                        "to_point": {"x": point2[0], "y": point2[1]},
                        "from_component": comp1_id,
                        "to_component": comp2_id,
                        "type": "direct",
                    }
                )
            elif isinstance(connection, (list, tuple)) and len(connection) > 2:
                # Multi-point connection
                points = [{"x": p[0], "y": p[1]} for p in connection]
                comp_ids = [self._find_component_by_point(p) for p in connection]
                connections_data.append(
                    {"points": points, "components": comp_ids, "type": "multi_point"}
                )

        # Build complete diagram data
        diagram_data = {
            "metadata": {
                "canvas_size": {"width": self.width, "height": self.height},
                "total_components": len(self.components),
                "total_connections": len(self.connections),
            },
            "components": [comp.to_dict() for comp in self.components],
            "connections": connections_data,
        }

        # Save to JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(diagram_data, f, indent=2, ensure_ascii=False)

        return output_path

    def _find_component_by_point(self, point):
        """
        Find component ID that contains or is closest to the given point.

        Args:
            point: (x, y) coordinate

        Returns:
            Component ID or None
        """
        x, y = point
        for comp in self.components:
            # Check if point is on left or right connection point
            if (
                abs(comp.left_connection[0] - x) < 5
                and abs(comp.left_connection[1] - y) < 5
            ):
                return comp.component_id
            if (
                abs(comp.right_connection[0] - x) < 5
                and abs(comp.right_connection[1] - y) < 5
            ):
                return comp.component_id
            # Check if point is within component bounds
            if (
                comp.x <= x <= comp.x + comp.width
                and comp.y <= y <= comp.y + comp.height
            ):
                return comp.component_id

        # If no exact match, find closest component
        if self.components:
            closest = min(
                self.components,
                key=lambda c: (
                    (c.x + c.width // 2 - x) ** 2 + (c.y + c.height // 2 - y) ** 2
                )
                ** 0.5,
            )
            return closest.component_id

        return None
