import os
import cv2
import numpy as np
from PIL import Image

def restore_lab_color(color_folder, gray_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    color_files = [f for f in os.listdir(color_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
    for fname in color_files:
        color_path = os.path.join(color_folder, fname)
        gray_path = os.path.join(gray_folder, fname)
        out_path = os.path.join(output_folder, fname)
        if not os.path.exists(gray_path):
            print(f"Missing grayscale for {fname}, skipping.")
            continue
        # Load color image and grayscale image
        color_img = cv2.imread(color_path)
        gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        # Convert color image to Lab
        lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Replace L channel with grayscale
        lab_restored = cv2.merge([gray_img, a, b])
        # Convert back to BGR
        restored = cv2.cvtColor(lab_restored, cv2.COLOR_LAB2BGR)
        cv2.imwrite(out_path, restored)
        print(f"Restored: {out_path}")

if __name__ == "__main__":
    color_folder = "coloured images/input_images/human"
    gray_folder = "coloured images/output_grayscale/human"
    output_folder = "coloured images/output_colorized/human"
    restore_lab_color(color_folder, gray_folder, output_folder)
