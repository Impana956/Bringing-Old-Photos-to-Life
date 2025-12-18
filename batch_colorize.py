import os
import streamlit as st
import numpy as np
from PIL import Image
from colorizer import colorize_image

gray_folder = "coloured images/output_grayscale/human"
output_folder = "coloured images/output_colorized/human"
os.makedirs(output_folder, exist_ok=True)

def get_image_list(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def save_image(np_img, path):
    Image.fromarray(np_img).save(path)

def run_batch_colorize():
    st.header("Single Image Lab Restoration")
    uploaded_file = st.file_uploader("Upload a grayscale image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if uploaded_file is not None:
        fname = uploaded_file.name
        gray_img = np.array(Image.open(uploaded_file).convert("RGB"))
        # Try to find the original color image
        color_folder = "coloured images/input_images/human"
        color_path = os.path.join(color_folder, fname)
        if not os.path.exists(color_path):
            st.error(f"Original color image not found: {color_path}")
            return
        color_img = np.array(Image.open(color_path).convert("RGB"))
        # Convert both to LAB
        import cv2
        color_lab = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB)
        gray_lab = cv2.cvtColor(gray_img, cv2.COLOR_RGB2LAB)
        l_gray = gray_lab[:, :, 0]
        a = color_lab[:, :, 1]
        b = color_lab[:, :, 2]
        lab_restored = cv2.merge([l_gray, a, b])
        lab_restored_rgb = cv2.cvtColor(lab_restored, cv2.COLOR_LAB2RGB)

        # Pre-trained model colorization
        from colorizer import colorize_image
        model_colorized = colorize_image(gray_img)

        # Blend both results (50% Lab, 50% model)
        blend_ratio = 0.5
        blended = cv2.addWeighted(lab_restored_rgb, blend_ratio, model_colorized, 1-blend_ratio, 0)

        out_path = os.path.join(output_folder, fname)
        Image.fromarray(blended).save(out_path)
        st.success(f"Saved blended image: {out_path}")
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_img, caption="Grayscale Input", use_column_width=True)
        with col2:
            st.image(blended, caption="Blended Output (50% Lab, 50% Model)", use_column_width=True)

run_batch_colorize()
