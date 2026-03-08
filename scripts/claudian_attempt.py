import os
import csv
import json
import argparse
from itertools import groupby
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from noise_component_classifier import ComponentClassifier, NoiseGenerator
from components import ComponentGenerator


# ──────────────────────────────────────────────────────────────────────────────
# Multi-scale CNN detector with red dot marker detection
# ──────────────────────────────────────────────────────────────────────────────


class MultiScaleCNNDetector:
    def __init__(
        self,
        classifier,
        window_size=100,
        step_size=30,
        confidence_threshold=0.88,
        batch_size=256,
        num_workers=4,
    ):
        self.classifier = classifier
        self.window_size = window_size
        self.step_size = step_size
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.detections = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            image_array = cv2.imread(image_path_or_array)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_array = np.asarray(image_path_or_array)
            if image_array.ndim == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        height, width = image_array.shape[:2]
        print(f"Scanning image ({width}x{height}) using red dot markers...")

        # ── 1. Isolate red pixels ──────────────────────────────────────────
        # Work in HSV for reliable hue-based red detection (red wraps around 0°/180°)
        image_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

        # Red occupies two hue ranges in HSV
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask_r1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_r1, mask_r2)

        # ── 2. Find individual red dot blobs ──────────────────────────────
        # Light morphological close to join fragmented dots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            red_mask, connectivity=8
        )

        # Filter blobs by area — genuine marker dots are small but not noise
        MIN_DOT_AREA = 10  # pixels  (ignore single stray pixels)
        MAX_DOT_AREA = 2000  # pixels  (ignore large red regions that aren't dots)

        dots = []  # list of (cx, cy)
        for i in range(1, num_labels):  # skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if MIN_DOT_AREA <= area <= MAX_DOT_AREA:
                cx, cy = centroids[i]
                dots.append((float(cx), float(cy)))

        print(f"  Found {len(dots)} red marker dots")
        if len(dots) < 2:
            print(
                "  Not enough dots to form a bounding box — falling back to empty result"
            )
            return []

        # ── 3. Pair dots → component bounding boxes ───────────────────────
        # Strategy: cluster nearby dot pairs.  Each component is bounded by the
        # dots on its left and right sides (or top/bottom), so we pair dots that
        # are closest to each other and form a tight bounding box around each pair.
        #
        # Simple greedy nearest-neighbour pairing (works well when dots come in
        # pairs on opposite sides of a component).
        used = [False] * len(dots)
        pairs = []

        # Sort by x so left-dot always comes first in each pair
        dots_sorted = sorted(enumerate(dots), key=lambda t: t[1][0])

        for rank, (i, (cx, cy)) in enumerate(dots_sorted):
            if used[i]:
                continue
            best_j, best_dist = None, float("inf")
            for j, (cx2, cy2) in dots_sorted[rank + 1 :]:
                if used[j]:
                    continue
                dist = ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is not None:
                used[i] = used[best_j] = True
                pairs.append((dots[i], dots[best_j]))

        print(f"  Formed {len(pairs)} dot pairs → component candidates")

        # ── 4. Crop each pair region and classify ─────────────────────────
        ws = self.window_size
        image_float = image_array.astype(np.float32) / 255.0
        self.detections = []

        batch_windows = []
        batch_meta = []  # (x1, y1, w, h) for each window

        for (ax, ay), (bx, by) in pairs:
            # Bounding box of the two dots with a small margin
            x1 = max(0, int(min(ax, bx)) - 10)
            y1 = max(0, int(min(ay, by)) - 10)
            x2 = min(width, int(max(ax, bx)) + 10)
            y2 = min(height, int(max(ay, by)) + 10)

            # Ensure the crop is at least ws × ws so the classifier sees a
            # full-resolution patch; expand symmetrically if needed.
            bw, bh = x2 - x1, y2 - y1
            if bw < ws:
                pad = (ws - bw) // 2
                x1 = max(0, x1 - pad)
                x2 = min(width, x2 + pad)
            if bh < ws:
                pad = (ws - bh) // 2
                y1 = max(0, y1 - pad)
                y2 = min(height, y2 + pad)

            crop = image_float[y1:y2, x1:x2]

            # Resize to the fixed window the classifier expects
            crop_resized = cv2.resize(crop, (ws, ws), interpolation=cv2.INTER_AREA)

            batch_windows.append(crop_resized)
            batch_meta.append((x1, y1, x2 - x1, y2 - y1))

        # ── 5. Batch inference ────────────────────────────────────────────
        for batch_start in range(0, len(batch_windows), self.batch_size):
            b_wins = batch_windows[batch_start : batch_start + self.batch_size]
            b_meta = batch_meta[batch_start : batch_start + self.batch_size]
            b_arr = np.stack(b_wins, axis=0)

            if hasattr(self.classifier, "predict_batch"):
                result = self.classifier.predict_batch(b_arr)
                labels, confidences = result[0], result[1]
            else:
                pairs_out = [self.classifier.predict(w) for w in b_wins]
                labels = [p[0] for p in pairs_out]
                confidences = [p[1] for p in pairs_out]

            for (x, y, w, h), label, conf in zip(b_meta, labels, confidences):
                if float(conf) >= self.confidence_threshold:
                    self.detections.append(
                        {
                            "bbox": (x, y, w, h),
                            "label": label,
                            "confidence": float(conf),
                            "center": (x + w // 2, y + h // 2),
                        }
                    )

        print(f"Found {len(self.detections)} detections before NMS")
        self.detections = self._non_maximum_suppression(self.detections)
        print(f"Found {len(self.detections)} detections after NMS")
        self.detections = self._merge_collinear_detections(self.detections)
        print(f"Found {len(self.detections)} detections after merging collinear boxes")
        return self.detections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _non_maximum_suppression(self, detections, iou_threshold=0.45):
        """Vectorised NMS — keeps the highest-confidence box in each overlapping group."""
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        order = scores.argsort()[::-1]
        keep_indices = []

        while order.size > 0:
            i = order[0]
            keep_indices.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < iou_threshold]

        return [detections[i] for i in keep_indices]

    @staticmethod
    def _merge_collinear_detections(
        detections, y_tolerance=30, gap_tolerance=150, min_run=5
    ):
        """
        Collapse detections of the same class that form a horizontal line
        (e.g. a bus bar misclassified as many adjacent switches) into a single
        wide bounding box.

        A "run" is a sequence of same-label boxes whose y-centres are within
        *y_tolerance* of each other and whose x-gaps are within *gap_tolerance*.
        Runs of at least *min_run* boxes are merged; shorter ones are kept as-is.
        """
        if not detections:
            return []

        merged = []
        sorted_dets = sorted(detections, key=lambda d: (d["label"], d["bbox"][1]))

        for label, group_iter in groupby(sorted_dets, key=lambda d: d["label"]):
            group = list(group_iter)

            # Cluster detections whose y-coordinates are close together
            clusters = []
            for det in group:
                _, y, _, _ = det["bbox"]
                placed = False
                for cluster in clusters:
                    if abs(y - cluster[0]["bbox"][1]) <= y_tolerance:
                        cluster.append(det)
                        placed = True
                        break
                if not placed:
                    clusters.append([det])

            for cluster in clusters:
                # Sort cluster left-to-right
                cluster.sort(key=lambda d: d["bbox"][0])

                # Split into contiguous runs (large x-gaps break a run)
                runs = [[cluster[0]]]
                for det in cluster[1:]:
                    last_box = runs[-1][-1]["bbox"]
                    gap = det["bbox"][0] - (last_box[0] + last_box[2])
                    if gap <= gap_tolerance:
                        runs[-1].append(det)
                    else:
                        runs.append([det])

                for run in runs:
                    if len(run) >= min_run:
                        # Merge the whole run into one encompassing box
                        all_x1 = [d["bbox"][0] for d in run]
                        all_y1 = [d["bbox"][1] for d in run]
                        all_x2 = [d["bbox"][0] + d["bbox"][2] for d in run]
                        all_y2 = [d["bbox"][1] + d["bbox"][3] for d in run]
                        mx, my = min(all_x1), min(all_y1)
                        mw, mh = max(all_x2) - mx, max(all_y2) - my
                        best_conf = max(d["confidence"] for d in run)
                        merged.append(
                            {
                                "bbox": (mx, my, mw, mh),
                                "label": label,
                                "confidence": best_conf,
                                "center": (mx + mw // 2, my + mh // 2),
                                "merged_count": len(run),
                            }
                        )
                        print(
                            f"  Merged {len(run)} collinear '{label}' boxes "
                            f"-> single box ({mx},{my},{mw}x{mh})"
                        )
                    else:
                        merged.extend(run)

        return merged


# ──────────────────────────────────────────────────────────────────────────────
# Annotator
# ──────────────────────────────────────────────────────────────────────────────


class ComponentAnnotator:
    COLORS = {
        "generator": "#FF6B6B",
        "transformer": "#4ECDC4",
        "load": "#45B7D1",
        "breaker": "#FFA07A",
        "bus": "#98D8C8",
        "capacitor": "#F7DC6F",
        "switch": "#BB8FCE",
    }

    def __init__(self, font_size=12):
        self.font_size = font_size

    def _load_font(self):
        for path in ("arial.ttf", "C:/Windows/Fonts/arial.ttf"):
            try:
                return ImageFont.truetype(path, self.font_size)
            except Exception:
                pass
        return ImageFont.load_default()

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
            image = Image.fromarray(np.asarray(image_path_or_array)).convert("RGB")

        draw = ImageDraw.Draw(image)
        font = self._load_font()

        for det in detections:
            x, y, w, h = det["bbox"]
            label = det["label"]
            confidence = det["confidence"]
            hex_color = self.COLORS.get(label, "#FFFFFF")
            color_rgb = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))

            draw.rectangle([(x, y), (x + w, y + h)], outline=color_rgb, width=thickness)

            count = det.get("merged_count", 1)
            text = label
            if count > 1:
                text += f" x{count}"
            if show_confidence:
                text += f" ({confidence:.2f})"

            tb = draw.textbbox((x, y - 20), text, font=font)
            draw.rectangle(
                [(tb[0] - 2, tb[1] - 2), (tb[2] + 2, tb[3] + 2)], fill=color_rgb
            )
            draw.text((x, y - 20), text, fill="white", font=font)

        if output_path:
            # Auto-create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            image.save(output_path)
            print(f"Annotated image saved to {output_path}")

        return image


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────


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

        self.detector = MultiScaleCNNDetector(
            self.classifier,
            window_size=100,
            step_size=50,
            confidence_threshold=0.95,
            batch_size=256,
            num_workers=4,
        )
        print("Model loaded successfully")

    def detect_in_image(self, image_path, output_path=None, annotate=True):
        print(f"\nDetecting components in {image_path}...")
        detections = self.detector.detect(image_path)

        print(f"\nDetections found: {len(detections)}")
        for i, det in enumerate(detections, 1):
            x, y, w, h = det["bbox"]
            count = det.get("merged_count", 1)
            count_str = f" (merged {count})" if count > 1 else ""
            print(
                f"  {i:3}. {det['label']:12s}{count_str} "
                f"- Box({x},{y},{w}x{h}) "
                f"- Confidence: {det['confidence']:.3f}"
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

        image_paths = (
            list(Path(image_directory).rglob(file_pattern))
            if recursive
            else list(Path(image_directory).glob(file_pattern))
        )
        print(f"Found {len(image_paths)} images to process")

        results = {}

        def process_one(args):
            idx, image_path = args
            print(f"\n[{idx}/{len(image_paths)}] Processing {image_path.name}...")
            try:
                out_path = (
                    os.path.join(output_directory, f"annotated_{image_path.stem}.png")
                    if output_directory
                    else None
                )
                detections, _ = self.detect_in_image(
                    str(image_path), out_path, annotate=True
                )
                return image_path.name, {
                    "image_path": str(image_path),
                    "num_detections": len(detections),
                    "detections": detections,
                    "output_path": out_path,
                }
            except Exception as exc:
                print(f"Error processing {image_path}: {exc}")
                return image_path.name, {"error": str(exc)}

        with ThreadPoolExecutor(max_workers=self.detector.num_workers) as ex:
            for name, result in ex.map(process_one, enumerate(image_paths, 1)):
                results[name] = result

        return results

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_results_json(self, detections, output_path):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"count": len(detections), "detections": detections}, f, indent=2)
        print(f"Results saved to {output_path}")

    def export_results_csv(self, detections, output_path):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fields = [
            "label",
            "x",
            "y",
            "width",
            "height",
            "confidence",
            "center_x",
            "center_y",
            "merged_count",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
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
                        "merged_count": det.get("merged_count", 1),
                    }
                )
        print(f"Results saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Detect electrical components in diagrams using red dot markers"
    )
    parser.add_argument("--image", type=str, help="Single image file to process")
    parser.add_argument("--directory", type=str, help="Directory of images to process")
    parser.add_argument(
        "--output", type=str, help="Output path / directory for annotated images"
    )
    parser.add_argument("--model", type=str, default="component_classifier.h5")
    parser.add_argument(
        "--step-size", type=int, default=50, help="Unused (kept for CLI compatibility)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Detection confidence threshold (0-1)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel workers for batch processing"
    )
    parser.add_argument("--save-json", type=str, help="Save detections as JSON")
    parser.add_argument("--save-csv", type=str, help="Save detections as CSV")
    parser.add_argument(
        "--min-dot-area",
        type=int,
        default=10,
        help="Minimum red dot blob area in pixels (default: 10)",
    )
    parser.add_argument(
        "--max-dot-area",
        type=int,
        default=2000,
        help="Maximum red dot blob area in pixels (default: 2000)",
    )
    args = parser.parse_args()

    pipeline = ComponentDetectionPipeline(model_path=args.model)
    pipeline.load_model()

    pipeline.detector.confidence_threshold = args.confidence
    pipeline.detector.batch_size = args.batch_size
    pipeline.detector.num_workers = args.workers

    if args.image:
        detections, _ = pipeline.detect_in_image(
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
        total = len(results)
        failed = sum(1 for r in results.values() if "error" in r)
        successful = total - failed
        total_dets = sum(r.get("num_detections", 0) for r in results.values())

        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total images     : {total}")
        print(f"Successful       : {successful}")
        print(f"Failed           : {failed}")
        print(f"Total detections : {total_dets}")
        print(
            f"Avg per image    : " f"{total_dets / successful:.1f}"
            if successful
            else "N/A"
        )

    else:
        print("Please specify --image or --directory")
        parser.print_help()


if __name__ == "__main__":
    main()
