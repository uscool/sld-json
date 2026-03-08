import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import cv2
from noise_component_classifier import ComponentClassifier, NoiseGenerator
from components import ComponentGenerator


class SlidingWindowDetector:
    def __init__(
        self, classifier, window_size=100, step_size=10, confidence_threshold=0.6
    ):
        self.classifier = classifier
        self.window_size = window_size
        self.step_size = step_size
        self.confidence_threshold = confidence_threshold
        self.detections = []

    def detect(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            image = Image.open(image_path_or_array).convert("RGB")
        else:
            image = Image.fromarray(image_path_or_array).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        self.detections = []

        print(f"Scanning image ({width}x{height}) with sliding window...")

        total_windows = ((width - self.window_size) // self.step_size + 1) * (
            (height - self.window_size) // self.step_size + 1
        )
        window_count = 0

        for y in range(0, height - self.window_size, self.step_size):
            for x in range(0, width - self.window_size, self.step_size):
                window_count += 1
                window = image_array[y : y + self.window_size, x : x + self.window_size]
                window_normalized = window.astype(np.float32) / 255.0
                label, confidence, _ = self.classifier.predict(window_normalized)
                if confidence >= self.confidence_threshold:
                    self.detections.append(
                        {
                            "bbox": (x, y, self.window_size, self.window_size),
                            "label": label,
                            "confidence": float(confidence),
                            "center": (
                                x + self.window_size // 2,
                                y + self.window_size // 2,
                            ),
                        }
                    )

                if window_count % 500 == 0:
                    print(f"  Processed {window_count}/{total_windows} windows...")

        print(f"Found {len(self.detections)} detections before NMS")
        self.detections = self._non_maximum_suppression(self.detections)
        print(f"Found {len(self.detections)} detections after NMS")
        return self.detections

    def _non_maximum_suppression(self, detections, iou_threshold=0.3):
        if not detections:
            return []
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)

        keep = []
        while sorted_dets:
            current = sorted_dets.pop(0)
            keep.append(current)
            remaining = []
            for det in sorted_dets:
                iou = self._compute_iou(current["bbox"], det["bbox"])
                if iou < iou_threshold:
                    remaining.append(det)

            sorted_dets = remaining

        return keep

    @staticmethod
    def _compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        return intersection / (union + 1e-6)


class ComponentAnnotator:
    def __init__(self, line_width=2, font_size=12):
        self.line_width = line_width
        self.font_size = font_size
        self.colors = {
            "generator": "#FF6B6B",
            "transformer": "#4ECDC4",
            "load": "#45B7D1",
            "breaker": "#FFA07A",
            "bus": "#98D8C8",
            "capacitor": "#F7DC6F",
            "switch": "#BB8FCE",
        }

    def annotate(
        self,
        image_path_or_array,
        detections,
        output_path=None,
        show_confidence=True,
        thickness=2,
    ):
        if isinstance(image_path_or_array, str):
            image = Image.open(image_path_or_array).convert("RGB")
        else:
            image = Image.fromarray(image_path_or_array).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", self.font_size)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", self.font_size)
            except:
                font = ImageFont.load_default()

        for i, det in enumerate(detections):
            x, y, w, h = det["bbox"]
            label = det["label"]
            confidence = det["confidence"]
            color = self.colors.get(label, "#FFFFFF")
            color_rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
            draw.rectangle([(x, y), (x + w, y + h)], outline=color_rgb, width=thickness)
            text = f"{label}"
            if show_confidence:
                text += f" ({confidence:.2f})"
            bbox_text = draw.textbbox((x, y - 20), text, font=font)
            draw.rectangle(
                [
                    (bbox_text[0] - 2, bbox_text[1] - 2),
                    (bbox_text[2] + 2, bbox_text[3] + 2),
                ],
                fill=color_rgb,
            )
            draw.text((x, y - 20), text, fill="white", font=font)

        if output_path:
            image.save(output_path)
            print(f"Annotated image saved to {output_path}")

        return image


class ComponentDetectionPipeline:
    def __init__(self, model_path="component_classifier.h5", symbol_dir="symbols"):
        self.model_path = model_path
        self.symbol_dir = symbol_dir
        self.classifier = None
        self.detector = None
        self.annotator = ComponentAnnotator()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        gen = ComponentGenerator()
        component_types = gen.get_component_types()
        self.classifier = ComponentClassifier(num_classes=len(component_types))
        self.classifier.load_model(self.model_path)
        self.detector = SlidingWindowDetector(
            self.classifier,
            window_size=100,
            step_size=20,  # Increase for speed, decrease for accuracy
            confidence_threshold=0.5,
        )

        print("Model loaded successfully")

    def detect_in_image(self, image_path, output_path=None, annotate=True):
        print(f"\nDetecting components in {image_path}...")
        detections = self.detector.detect(image_path)
        print(f"\nDetections found:")
        for i, det in enumerate(detections, 1):
            x, y, w, h = det["bbox"]
            print(
                f"  {i}. {det['label']:12s} - Box({x}, {y}, {w}x{h}) - Confidence: {det['confidence']:.3f}"
            )
        annotated_image = None
        if annotate:
            annotated_image = self.annotator.annotate(
                image_path,
                detections,
                output_path=output_path,
                show_confidence=True,
                thickness=2,
            )
        return detections, annotated_image

    def detect_batch(
        self,
        image_directory,
        output_directory=None,
        recursive=False,
        file_pattern="*.png",
    ):
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        results = {}
        if recursive:
            image_paths = list(Path(image_directory).rglob(file_pattern))
        else:
            image_paths = list(Path(image_directory).glob(file_pattern))

        print(f"Found {len(image_paths)} images to process")

        for idx, image_path in enumerate(image_paths, 1):
            print(f"\n[{idx}/{len(image_paths)}] Processing {image_path.name}...")

            try:
                out_path = None
                if output_directory:
                    out_path = os.path.join(
                        output_directory, f"annotated_{image_path.stem}.png"
                    )
                detections, _ = self.detect_in_image(
                    str(image_path), out_path, annotate=True
                )
                results[image_path.name] = {
                    "image_path": str(image_path),
                    "num_detections": len(detections),
                    "detections": detections,
                    "output_path": out_path,
                }

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[image_path.name] = {"error": str(e)}

        return results

    def export_results_json(self, detections, output_path):
        data = {"detections": detections, "count": len(detections)}

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {output_path}")

    def export_results_csv(self, detections, output_path):
        """Export detection results to CSV."""
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "label",
                    "x",
                    "y",
                    "width",
                    "height",
                    "confidence",
                    "center_x",
                    "center_y",
                ],
            )
            writer.writeheader()

            for det in detections:
                x, y, w, h = det["bbox"]
                cx, cy = det["center"]
                writer.writerow(
                    {
                        "label": det["label"],
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": f"{det['confidence']:.4f}",
                        "center_x": cx,
                        "center_y": cy,
                    }
                )

        print(f"Results saved to {output_path}")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect components in diagrams with bounding boxes"
    )
    parser.add_argument("--image", type=str, help="Image file to process")
    parser.add_argument("--directory", type=str, help="Directory of images to process")
    parser.add_argument(
        "--output", type=str, help="Output directory for annotated images"
    )
    parser.add_argument(
        "--model", type=str, default="component_classifier.h5", help="Model path"
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=20,
        help="Sliding window step size (smaller=slower but more accurate)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold"
    )
    parser.add_argument("--save-json", type=str, help="Save detection results as JSON")
    parser.add_argument("--save-csv", type=str, help="Save detection results as CSV")

    args = parser.parse_args()
    pipeline = ComponentDetectionPipeline(model_path=args.model)
    pipeline.load_model()

    pipeline.detector.step_size = args.step_size
    pipeline.detector.confidence_threshold = args.confidence

    if args.image:
        detections, annotated = pipeline.detect_in_image(
            args.image, output_path=args.output, annotate=True
        )

        if args.save_json:
            pipeline.export_results_json(detections, args.save_json)
        if args.save_csv:
            pipeline.export_results_csv(detections, args.save_csv)

    elif args.directory:
        results = pipeline.detect_batch(
            args.directory, output_directory=args.output, recursive=True
        )

        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)

        total_images = len(results)
        total_detections = sum(r.get("num_detections", 0) for r in results.values())
        failed = sum(1 for r in results.values() if "error" in r)

        print(f"Total images: {total_images}")
        print(f"Failed: {failed}")
        print(f"Total detections: {total_detections}")
        print(
            f"Average detections per image: {total_detections / (total_images - failed) if (total_images - failed) > 0 else 0:.1f}"
        )

    else:
        print("Specify --image or --directory")
        parser.print_help()


if __name__ == "__main__":
    main()
