import os
import cv2

from preprocessing_symbols import preprocess_image_symbols
from preprocessing_text import preprocess_image_text
from symbol_text_matching import (
    detect_symbols_from_yolo,
    match_symbols_to_text_ray,
    export_to_json,
    draw_matches
)

import traceback


def get_valid_file(prompt):
    while True:
        path = input(prompt).strip()
        if os.path.isfile(path):
            return path
        print("Invalid file path. Try again.")


def main():
    image_path = get_valid_file("Enter image file location: ")

    os.makedirs("output", exist_ok=True)

    model_path = os.path.join(
        "models",
        "yolo_model",
        "runs",
        "detect",
        "train",
        "weights",
        "best.pt"
    )

    try:
        # Step 1: preprocess image for symbol detection
        symbol_ready_image = preprocess_image_symbols(image_path)
        symbol_output_path = os.path.join("output", "preprocessed_symbols.png")
        cv2.imwrite(symbol_output_path, symbol_ready_image)
        print(f"Symbol preprocessing complete: {symbol_output_path}")

        # Step 2: detect symbols
        symbols = detect_symbols_from_yolo(
            image_path=symbol_output_path,
            model_path=model_path,
            conf=0.2
        )

        if not symbols:
            print("No symbols detected. Pipeline stopped.")
            return

        # Step 3: preprocess image for OCR and save to disk.
        text_ready_image, text_scale = preprocess_image_text(image_path)
        text_output_path = os.path.join("output", "preprocessed_text.png")
        cv2.imwrite(text_output_path, text_ready_image)
        print(f"Text preprocessing complete: {text_output_path}")

        # Step 4: match symbols to OCR text.
        matched_results = match_symbols_to_text_ray(
            base_image_path=image_path,
            ocr_image_path=text_output_path,
            ocr_scale=text_scale,
            symbols=symbols,
            max_distance=300,
            beam_width=80
        )
        print("Symbol-text matching complete.")

        # Step 5: save outputs
        json_output_path = os.path.join("output", "sld_nodes.json")
        vis_output_path = os.path.join("output", "ray_match_results.png")

        export_to_json(matched_results, json_output_path)
        draw_matches(image_path, matched_results, vis_output_path)

        print("Pipeline completed successfully.")
        print(f"JSON saved to: {json_output_path}")
        print(f"Visualization saved to: {vis_output_path}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()