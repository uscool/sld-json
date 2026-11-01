# SLD-JSON

This program is meant to convert Single Line Diagrams (electrical diagrams) into a standardised JSON format. This program is undergoing integrations, but has approached runnable capabilities. This is currently limited, and open for developer assistance, mainly because it currently works for clean diagrams, performance on messy diagrams is limited. It is meant to aid exports to AM, GIS and SCADA systems 

## diagram_service.py

The main orchestrator. Calls `main_standalone.py` and `main_hybrid.py` with the right arguments depending on how you invoke it. Outputs go into two separate folders — one for standalone diagrams, one for hybrid.

---

```bash
python diagram_service.py
```
Runs both generators with their built-in default counts.

- **Standalone** — 50 samples per topology × 6 topologies = **300 diagrams** → `output_standalone/`
- **Hybrid** — **100 diagrams** → `output_hybrid/`
- **Total: 400 diagrams**

---

```bash
python diagram_service.py --bulk-count 500
```
Splits 500 evenly — 250 to standalone, 250 to hybrid.

- **Standalone** — 250 ÷ 6 = 41 samples per topology = **246 diagrams** → `output_standalone/`
- **Hybrid** — **250 diagrams** → `output_hybrid/`
- **Total: 496 diagrams** (remainder from the 250 ÷ 6 division is dropped)

---

```bash
python diagram_service.py --standalone-samples 100 --hybrid-samples 200
```
Explicit control over both counts independently. Does not do any splitting — you specify each directly.

- **Standalone** — 100 samples per topology × 6 topologies = **600 diagrams** → `output_standalone/`
- **Hybrid** — **200 diagrams** → `output_hybrid/`
- **Total: 800 diagrams**

---

```bash
python diagram_service.py --standalone-only
```
Runs only the standalone generator with default counts (50 per topology).

- **Standalone** — 50 × 6 = **300 diagrams** → `output_standalone/`
- Hybrid is skipped entirely.

---

```bash
python diagram_service.py --hybrid-only
```
Runs only the hybrid generator with default counts.

- **Hybrid** — **100 diagrams** → `output_hybrid/`
- Standalone is skipped entirely.

---

```bash
python diagram_service.py --standalone-dir data/standalone --hybrid-dir data/hybrid
```
Same as the default run but outputs go to custom directories instead.

- **Standalone** — 50 × 6 = **300 diagrams** → `data/standalone/`
- **Hybrid** — **100 diagrams** → `data/hybrid/`

---

You can combine this with any other flag, e.g.:
```bash
python diagram_service.py --bulk-count 500 --standalone-dir data/standalone --hybrid-dir data/hybrid
```

---

### Output folder structure

#### `output_standalone/`
Flat folder. One set of files per diagram, named by topology and index.
```
output_standalone/
    single_bus_000.json
    single_bus_000.svg
    single_bus_000.png
    single_bus_001.json
    ...
    double_bus_000.json
    ...
    breaker_and_half_000.json
    ...
    main_and_transfer_000.json
    ...
    ring_bus_000.json
    ...
    mesh_000.json
    ...
    mesh_049.json
    mesh_049.svg
    mesh_049.png
```

#### `output_hybrid/`
Flat folder. One set of files per diagram, named by index. Each diagram is a multi-voltage-level stack so it also gets a YOLO annotations file.
```
output_hybrid/
    hybrid_0000.json    ← merged topology JSON (all levels combined)
    hybrid_0000.svg
    hybrid_0000.png
    hybrid_0000.txt     ← YOLO bounding box annotations
    hybrid_0001.json
    ...
    hybrid_0099.json
    hybrid_0099.svg
    hybrid_0099.png
    hybrid_0099.txt
```

> Temporary folders named `_temp_hybrid_XXXX/` are created during hybrid generation but are deleted automatically once each diagram finishes.
# Other Files
*Note: Please keep updating as and when you add files. Update status as well. Those never meant to be part of structure due to being plain wrong, mark as gray.*

1. components.py: Creates components to use to generate diagrams.
2. layout_engine.py: Structures the image to be generated based on layouts. Some with AI aided code, some solo (core ones: generally us, except where stuck for long, others: AI aided merges).
3. diagram_service.py: Service automating generation of database.
4. preprocessing.py: To be integrated, needs to be connected to rest of code.
5. noise_component_classifier.py: Produces model by creating noisy components and training on them. Produces h5 (legacy keras) file. 
6. component_detector: Detects components in image using sliding window. Discarded now, owing to long processing time and hyper false positives owing to small scaled images and similar features in texts and lines.
7. claudian_attempt.py: Enhanced version of component_detector for speed and hacky red dot implementation of placing kernel.
   
## Path notes

1. /output: Stores sample SLD diagrams and JSONs (complete diagrams)
2. /symbols: Stores original symbols used to create SLDs
3. /dataset: Stores noisy components used for detection
4. /scripts: Stores code
5. /results: Marked symbol diagrams and JSONs stored here

## Important import functions

1. preprocessing_image_symbols() (from preprocessing_symbols)
2. preprocessing_image_text() (from preprocessing_text)
