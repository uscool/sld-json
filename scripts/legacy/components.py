from PIL import Image, ImageDraw
import os


class ComponentGenerator:
    SYMBOL_SIZE = (100, 100)  # Standard symbol size
    CONNECTION_POINT_OFFSET = 10  # Distance from edge for connection points

    def __init__(self, symbol_dir="symbols"):
        self.symbol_dir = symbol_dir
        os.makedirs(symbol_dir, exist_ok=True)

    def _get_connection_points(self):
        width, height = self.SYMBOL_SIZE
        left_point = (0, height // 2)
        right_point = (width, height // 2)
        return left_point, right_point

    def generate_generator(self):
        img = Image.new("RGB", self.SYMBOL_SIZE, "white")
        draw = ImageDraw.Draw(img)

        # Draw circle
        margin = 20
        draw.ellipse(
            [
                margin,
                margin,
                self.SYMBOL_SIZE[0] - margin,
                self.SYMBOL_SIZE[1] - margin,
            ],
            outline="black",
            width=3,
        )

        # Draw 'G' text
        try:
            from PIL import ImageFont

            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()

        text = "G"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (
                (self.SYMBOL_SIZE[0] - text_width) // 2,
                (self.SYMBOL_SIZE[1] - text_height) // 2,
            ),
            text,
            fill="black",
            font=font,
        )

        # Connection points (left and right)
        left, right = self._get_connection_points()
        draw.ellipse(
            [left[0] - 5, left[1] - 5, left[0] + 5, left[1] + 5],
            fill="red",
            outline="red",
        )
        draw.ellipse(
            [right[0] - 5, right[1] - 5, right[0] + 5, right[1] + 5],
            fill="red",
            outline="red",
        )

        return img

    def generate_transformer(self):
        img = Image.new("RGB", self.SYMBOL_SIZE, "white")
        draw = ImageDraw.Draw(img)

        # Draw two circles side by side
        circle_size = 30
        y_center = self.SYMBOL_SIZE[1] // 2

        # Left circle
        draw.ellipse(
            [
                20,
                y_center - circle_size // 2,
                20 + circle_size,
                y_center + circle_size // 2,
            ],
            outline="black",
            width=3,
        )

        # Right circle
        draw.ellipse(
            [
                self.SYMBOL_SIZE[0] - 20 - circle_size,
                y_center - circle_size // 2,
                self.SYMBOL_SIZE[0] - 20,
                y_center + circle_size // 2,
            ],
            outline="black",
            width=3,
        )

        # Connection points
        left, right = self._get_connection_points()
        draw.ellipse(
            [left[0] - 5, left[1] - 5, left[0] + 5, left[1] + 5],
            fill="red",
            outline="red",
        )
        draw.ellipse(
            [right[0] - 5, right[1] - 5, right[0] + 5, right[1] + 5],
            fill="red",
            outline="red",
        )

        return img

    def generate_load(self):
        img = Image.new("RGB", self.SYMBOL_SIZE, "white")
        draw = ImageDraw.Draw(img)

        # Draw zigzag pattern
        width, height = self.SYMBOL_SIZE
        x_start = 20
        x_end = width - 20
        y_center = height // 2
        segment_width = (x_end - x_start) // 4

        points = [(x_start, y_center)]
        for i in range(1, 5):
            x = x_start + i * segment_width
            y = y_center + (20 if i % 2 == 1 else -20)
            points.append((x, y))
        points.append((x_end, y_center))

        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill="black", width=3)

        # Connection points
        left, right = self._get_connection_points()
        draw.ellipse(
            [left[0] - 5, left[1] - 5, left[0] + 5, left[1] + 5],
            fill="red",
            outline="red",
        )
        draw.ellipse(
            [right[0] - 5, right[1] - 5, right[0] + 5, right[1] + 5],
            fill="red",
            outline="red",
        )

        return img

    def generate_breaker(self):
        """Generate a circuit breaker symbol (rectangle with gap)."""
        img = Image.new("RGB", self.SYMBOL_SIZE, "white")
        draw = ImageDraw.Draw(img)

        # Draw rectangle with gap
        width, height = self.SYMBOL_SIZE
        rect_width = 60
        rect_height = 30
        x_start = (width - rect_width) // 2
        y_start = (height - rect_height) // 2

        # Top and bottom lines
        draw.line(
            [(x_start, y_start), (x_start + rect_width, y_start)], fill="black", width=3
        )
        draw.line(
            [
                (x_start, y_start + rect_height),
                (x_start + rect_width, y_start + rect_height),
            ],
            fill="black",
            width=3,
        )

        # Left and right vertical lines
        draw.line(
            [(x_start, y_start), (x_start, y_start + rect_height)],
            fill="black",
            width=3,
        )
        draw.line(
            [
                (x_start + rect_width, y_start),
                (x_start + rect_width, y_start + rect_height),
            ],
            fill="black",
            width=3,
        )

        # Gap in the middle (diagonal line)
        gap_x = x_start + rect_width // 2
        draw.line(
            [(gap_x, y_start), (gap_x + 10, y_start + rect_height)],
            fill="black",
            width=3,
        )

        # Connection points
        left, right = self._get_connection_points()
        draw.ellipse(
            [left[0] - 5, left[1] - 5, left[0] + 5, left[1] + 5],
            fill="red",
            outline="red",
        )
        draw.ellipse(
            [right[0] - 5, right[1] - 5, right[0] + 5, right[1] + 5],
            fill="red",
            outline="red",
        )

        return img

    def generate_bus(self):
        img = Image.new("RGB", self.SYMBOL_SIZE, "white")
        draw = ImageDraw.Draw(img)

        # Draw thick horizontal line
        width, height = self.SYMBOL_SIZE
        y_center = height // 2
        draw.line([(0, y_center), (width, y_center)], fill="black", width=8)

        # Connection points
        left, right = self._get_connection_points()
        draw.ellipse(
            [left[0] - 5, left[1] - 5, left[0] + 5, left[1] + 5],
            fill="red",
            outline="red",
        )
        draw.ellipse(
            [right[0] - 5, right[1] - 5, right[0] + 5, right[1] + 5],
            fill="red",
            outline="red",
        )

        return img

    def generate_capacitor(self):
        img = Image.new("RGB", self.SYMBOL_SIZE, "white")
        draw = ImageDraw.Draw(img)

        width, height = self.SYMBOL_SIZE
        x_center = width // 2
        line_length = 30

        # Two parallel vertical lines
        draw.line(
            [
                (x_center, height // 2 - line_length // 2),
                (x_center, height // 2 + line_length // 2),
            ],
            fill="black",
            width=3,
        )
        draw.line(
            [
                (x_center + 15, height // 2 - line_length // 2),
                (x_center + 15, height // 2 + line_length // 2),
            ],
            fill="black",
            width=3,
        )

        # Connection points
        left, right = self._get_connection_points()
        draw.ellipse(
            [left[0] - 5, left[1] - 5, left[0] + 5, left[1] + 5],
            fill="red",
            outline="red",
        )
        draw.ellipse(
            [right[0] - 5, right[1] - 5, right[0] + 5, right[1] + 5],
            fill="red",
            outline="red",
        )

        return img

    def generate_switch(self):
        """Generate a switch symbol (line with gap and diagonal)."""
        img = Image.new("RGB", self.SYMBOL_SIZE, "white")
        draw = ImageDraw.Draw(img)

        width, height = self.SYMBOL_SIZE
        y_center = height // 2
        x_center = width // 2

        # Horizontal line with gap
        draw.line([(20, y_center), (x_center - 10, y_center)], fill="black", width=3)
        draw.line(
            [(x_center + 10, y_center), (width - 20, y_center)], fill="black", width=3
        )

        # Diagonal line (switch blade)
        draw.line(
            [(x_center - 10, y_center), (x_center + 5, y_center - 15)],
            fill="black",
            width=3,
        )

        # Connection points
        left, right = self._get_connection_points()
        draw.ellipse(
            [left[0] - 5, left[1] - 5, left[0] + 5, left[1] + 5],
            fill="red",
            outline="red",
        )
        draw.ellipse(
            [right[0] - 5, right[1] - 5, right[0] + 5, right[1] + 5],
            fill="red",
            outline="red",
        )

        return img

    def get_component_types(self):
        return [
            "generator",
            "transformer",
            "load",
            "breaker",
            "bus",
            "capacitor",
            "switch",
        ]

    def generate_symbol(self, component_type):
        generators = {
            "generator": self.generate_generator,
            "transformer": self.generate_transformer,
            "load": self.generate_load,
            "breaker": self.generate_breaker,
            "bus": self.generate_bus,
            "capacitor": self.generate_capacitor,
            "switch": self.generate_switch,
        }

        if component_type not in generators:
            raise ValueError(f"Unknown component type: {component_type}")

        return generators[component_type]()

    def save_symbol(self, component_type, filename=None):
        if filename is None:
            filename = os.path.join(self.symbol_dir, f"{component_type}.png")

        symbol = self.generate_symbol(component_type)
        symbol.save(filename)
        return filename

    def generate_all_symbols(self):
        for comp_type in self.get_component_types():
            self.save_symbol(comp_type)
        print(f"Generated all symbols in {self.symbol_dir}/")
