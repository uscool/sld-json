import cv2
import numpy as np

MAX_SAFE_DIM = 32000

def preprocess_image_text(image_path):
    """
    Preprocess an image for OCR.

    Returns:
        bw    (np.ndarray) : upscaled binary image ready for PaddleOCR
        scale (float)      : factor applied; divide OCR bbox coords by this
                             to convert back to original image coordinates
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    
    desired_scale = 1.8
    max_allowed_scale = MAX_SAFE_DIM / max(orig_w, orig_h)
    scale = min(desired_scale, max_allowed_scale)

    if scale < desired_scale:
        print(f"[preprocess_image_text] Scale clamped {desired_scale} -> {scale:.3f} "
              f"(image {orig_w}x{orig_h} would exceed {MAX_SAFE_DIM}px at full scale)")

    img_up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    sharp = cv2.addWeighted(g, 1.15, blur, -0.15, 0)

    bw = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=12
    )

    return bw, scale
