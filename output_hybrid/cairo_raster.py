import os
import cairosvg

input_dir = "output_hybrid/svg"
output_dir = "output_hybrid/image"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".svg"):
        svg_path = os.path.join(input_dir, filename)
        png_name = os.path.splitext(filename)[0] + ".png"
        png_path = os.path.join(output_dir, png_name)

        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            background_color="white",
            scale=8
        )

        print(f"Converted {svg_path} -> {png_path}")