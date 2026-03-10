import cv2
import numpy as np

def preprocess_image_text(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    orig_h, orig_w = img.shape[:2]

    scale = 2.0
    img_up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2) Gray
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)

    # 3) Mild CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # General Denoise
    g = cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4) Mild Unsharp mask
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    sharp = cv2.addWeighted(g, 1.15, blur, -0.15, 0)

    # 5) Binary threshold
    _, bw = cv2.threshold(sharp, 200, 255, cv2.THRESH_BINARY)

    # 6)Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw_dilated = cv2.erode(bw, kernel, iterations=1)

    bw_final = cv2.resize(bw_dilated, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

    return bw_final
