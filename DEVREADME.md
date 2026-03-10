# Status and functionality

@@ File repo README for a quick refresher dev guide @@
*Courtesy of Ujjwal Sharma TM and Ishaan Sarode*
*Courtesy of Pranav Iyer*

```diff
+ components.py
+ layout_engine.py
+ diagram_service.py
! preprocessing.py
! noise_component_classifier.py
# component_detector.py
! claudian_attempt.py
- Add as you go
```

## File info

*Note: Please keep updating as and when you add files. Update status as well. Those never meant to be part of structure due to being plain wrong, mark as gray.*

1. components.py: Creates components to use to generate diagrams. Yours truly.
2. layout_engine.py: Structures the image to be generated based on layouts. Some with AI aided code, some solo (core ones: generally us, except where stuck for long, others: AI aided merges). Yours Truly+Pranav Iyer
3. diagram_service.py: Service automating generation of database. Courtesy of Claude.
4. preprocessing.py: To be integrated, needs to be connected to rest of code. Courtesy of Ishaan Sarode
5. noise_component_classifier.py: Produces model by creating noisy components and training on them. Produces h5 (legacy keras) file. Yours truly.
6. component_detector: Detects components in image using sliding window. Written by yours truly and discarded now, owing to long processing time and hyper false positives owing to small scaled images and similar features in texts and lines.
7. claudian_attempt.py: Enhanced version of component_detector for speed and hacky red dot implementation of placing kernel. Yours truly and Claude (To be enhanced by Pranav Iyer)

## Path notes

1. /output: Stores sample SLD diagrams and JSONs (complete diagrams)
2. /symbols: Stores original symbols used to create SLDs
3. /dataset: Stores noisy components used for detection
4. /scripts: Stores code
5. /results: Marked symbol diagrams and JSONs stored here

## Important import functions

1. preprocessing_image_symbols() (from preprocessing_symbols)
2. preprocessing_image_text() (from preprocessing_text)
