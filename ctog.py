import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Color to Grayscale Image Converter")

# Upload button
uploaded_file = st.file_uploader("Upload a Color Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Display original image
    st.subheader("Original Image")
    st.image(image_np, channels="RGB")

    # Convert to grayscale (no background, same size)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)   # convert back to 3 channels

    st.subheader("Grayscale Image (Same size, No background)")
    st.image(gray_rgb, channels="RGB")

    # Download button
    gray_pil = Image.fromarray(gray_rgb)
    st.download_button(
        label="Download Grayscale Image",
        data=cv2.imencode('.png', gray_rgb)[1].tobytes(),
        file_name="grayscale.png",
        mime="image/png"
    )