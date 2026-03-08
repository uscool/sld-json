import cv2
import numpy as np


img = cv2.imread('images/sld_2.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
g = clahe.apply(gray)

# Unsharp mask
blur = cv2.GaussianBlur(g, (3, 3), 0)
sharp = cv2.addWeighted(g, 1.5, blur, -0.4, 0)

# Grayscale dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
sharp_erode = cv2.erode(sharp, kernel, iterations=1)
sharp_dilated = cv2.dilate(sharp_erode, kernel, iterations=1)

# Unsharp mask
blur2 = cv2.GaussianBlur(sharp_dilated, (3, 3), 0)
sharp_2 = cv2.addWeighted(sharp_dilated, 1.3, blur2, -0.3, 0)

# Binary
_, bw = cv2.threshold(sharp_2, 190, 255, cv2.THRESH_BINARY)

# Display
cv2.imshow("Final - Symbol and Lines", bw)

cv2.waitKey(0)
cv2.destroyAllWindows()