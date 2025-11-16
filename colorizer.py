#!/usr/bin/env python
# coding: utf-8

"""
Colorizer Module
Handles B&W image colorization functionality
"""

import io
import time
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
from PIL import Image


@st.cache_resource(show_spinner=False)
def load_colorization_net():
    """Load the pre-trained colorization neural network"""
    prototxt = r"models\model_colorization.prototxt"
    model = r"models\colorization_model.model"
    points = r"models\pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net


@st.cache_data(show_spinner=False)
def colorize_image(image_np: np.ndarray) -> np.ndarray:
    """Colorize a grayscale image using deep learning"""
    try:
        net = load_colorization_net()
        # Convert to grayscale → RGB to stabilize tones (model expects L channel)
        img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        scaled = img.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
        L_full = cv2.split(lab)[0]
        colorized = np.concatenate((L_full[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
        return colorized
    except Exception as e:
        # If colorization fails, return a properly colored version using a different approach
        if st.session_state.get("debug_color_detection", False):
            st.warning(f"Colorization model issue: {str(e)}")
        else:
            st.warning(f"Colorization model issue: {str(e)[:50]}...")
        
        # Fallback: enhance the image with a more sophisticated approach
        try:
            # Try to boost colors more effectively
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype("float32")
            # Boost saturation more aggressively but with limits
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
            # Slightly boost value (brightness) as well
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
            result = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)
            
            # If the result is still very similar to original, try a different approach
            orig_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            if np.allclose(orig_gray, result_gray, atol=10):
                # Add some artificial color variation
                result = add_pseudo_colors(image_np)
            
            return result
        except Exception as fallback_e:
            if st.session_state.get("debug_color_detection", False):
                st.error(f"Fallback colorization also failed: {str(fallback_e)}")
            # Ultimate fallback: return original image with minimal enhancement
            return image_np

def add_pseudo_colors(image_np: np.ndarray) -> np.ndarray:
    """Add pseudo colors to a grayscale image as a last resort"""
    try:
        # Convert to LAB
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Create artificial color channels based on luminance variations
        # This creates a blue/white/red effect based on brightness
        a_channel = (l.astype(float) - 128) * 0.5  # Blue-red channel
        b_channel = (l.astype(float) - 128) * 0.3  # Green-magenta channel
        
        # Combine channels
        colored_lab = cv2.merge([l, a_channel.astype(np.uint8), b_channel.astype(np.uint8)])
        colored_rgb = cv2.cvtColor(colored_lab, cv2.COLOR_LAB2RGB)
        
        # Blend with original to keep it subtle
        blended = cv2.addWeighted(image_np, 0.7, colored_rgb, 0.3, 0)
        return blended
    except:
        # If all else fails, return the original image
        return image_np


def is_color_image(image_np: np.ndarray) -> bool:
    """
    Detect if an image is already colored.
    Returns True if the image has significant color information.
    Uses multiple methods for robust detection.
    """
    # Method 1: Check if image is truly grayscale (R=G=B everywhere)
    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    if np.allclose(r, g, atol=5) and np.allclose(g, b, atol=5):
        return False  # Pure grayscale (with small tolerance)
    
    # Method 2: Check saturation in HSV space
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype("float32") / 255.0
    sat_mean = float(sat.mean())
    sat_std = float(sat.std())
    
    # Method 3: Count pixels with significant saturation
    color_pixels = np.sum(sat > 0.15)  # Pixels with >15% saturation (increased threshold)
    total_pixels = sat.size
    color_ratio = color_pixels / total_pixels
    
    # Image is colored if:
    # - More than 3% of pixels have noticeable saturation, OR
    # - Average saturation is above 2%, OR
    # - High saturation variance (std > 15)
    is_colored = bool(color_ratio > 0.03 or sat_mean > 0.02 or sat_std > 0.15)
    
    # Debug information
    if st.session_state.get("debug_color_detection", False):
        st.sidebar.write(f"Color Detection Debug:")
        st.sidebar.write(f"  Color Ratio: {color_ratio:.4f}")
        st.sidebar.write(f"  Sat Mean: {sat_mean:.4f}")
        st.sidebar.write(f"  Sat Std: {sat_std:.4f}")
        st.sidebar.write(f"  Is Colored: {is_colored}")
    
    return is_colored


def adjust_colors(image_np: np.ndarray, hue_shift: int = 0, saturation: float = 1.0, brightness: float = 1.0) -> np.ndarray:
    """Apply color adjustments to an image"""
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    
    if hue_shift != 0:
        h = (h + hue_shift / 2.0) % 180.0
    
    if saturation != 1.0:
        s = np.clip(s * saturation, 0, 255)
    
    if brightness != 1.0:
        v = np.clip(v * brightness, 0, 255)
    
    hsv_adjusted = cv2.merge([h, s, v]).astype("uint8")
    rgb_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    
    return rgb_adjusted


def to_download_bytes(img_np: np.ndarray, fmt: str = "PNG", **save_kwargs) -> bytes:
    """Return image bytes for an ndarray image"""
    image = Image.fromarray(img_np)
    buf = io.BytesIO()
    image.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def render_colorizer_tab():
    """Render the colorizer tab UI"""
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1.5rem;">🎨 B&W → Colorizer</h1>', unsafe_allow_html=True)
    
    # File uploader - horizontal at top (like converter)
    uploaded_bw = st.file_uploader("Upload a B&W image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="colorizer_upload")
    
    # Helper function to format file size
    def format_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
    
    # Check if uploaded image is colored (needed for button state)
    is_color_detected = st.session_state.get("is_already_colored", False)
    
    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1_disp = k1.empty()
    k2_disp = k2.empty()
    k3_disp = k3.empty()
    k4_disp = k4.empty()
    k1_disp.metric("Image size", "—")
    k2_disp.metric("Processing time", "—")
    k3_disp.metric("Colorfulness", "—")
    k4_disp.metric("File size change", "—")
    
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown("**Original**")
        input_image_np = None
        is_already_colored = False
        
        if uploaded_bw is not None:
            original_bytes = uploaded_bw.getvalue()
            image = Image.open(io.BytesIO(original_bytes)).convert("RGB")
            input_image_np = np.array(image)
            
            # Re-check color detection if file changed
            current_file_hash = hash(uploaded_bw.name + str(len(original_bytes)))
            cached_hash = st.session_state.get("file_hash", None)
            
            if cached_hash != current_file_hash or "orig_size" not in st.session_state:
                # File changed - clear all previous results and recalculate
                st.session_state.original_colorized = None
                st.session_state.original_metrics = None
                st.session_state.hue_shift_value = 0
                st.session_state.saturation_value = 1.0
                st.session_state.brightness_value = 1.0
                
                is_already_colored = is_color_image(input_image_np)
                
                st.session_state.file_hash = current_file_hash
                st.session_state.orig_size = len(original_bytes)
                st.session_state.orig_format = getattr(image, "format", None) or None
                st.session_state.is_already_colored = is_already_colored
                try:
                    st.session_state.orig_norm_size = len(to_download_bytes(input_image_np, fmt="JPEG", quality=85, optimize=True))
                except Exception:
                    st.session_state.orig_norm_size = st.session_state.orig_size
            else:
                is_already_colored = st.session_state.get("is_already_colored", False)
            
            st.image(image, use_column_width=True)
            
            # Display original file size below the image
            if "orig_size" in st.session_state:
                orig_size = st.session_state.orig_size
                st.caption(f"📦 Original Size: {format_size(orig_size)}")
            
            if is_already_colored:
                st.error("⚠️ This image is already in colour! The colorizer is designed for black & white images only.")
                st.info("💡 Please upload a grayscale or black & white image to use this feature.")
        else:
            st.info("Upload an image to get started.")
    
    with col_right:
        st.markdown("**Colorized**")
        output_np = None
        is_reset = False
        
        # Check if reset to original is requested
        if st.session_state.get("reset_to_original", False) and st.session_state.get("original_colorized") is not None:
            # Reset slider values
            st.session_state.hue_shift_value = 0
            st.session_state.saturation_value = 1.0
            st.session_state.brightness_value = 1.0
            st.session_state.reset_to_original = False
            # Don't call st.rerun() here to avoid tab switching issue
        
        # Display colorized image if available
        original_colorized = st.session_state.get("original_colorized")
        if original_colorized is not None:
            # Apply current color adjustments
            output_np = adjust_colors(
                original_colorized,
                st.session_state.get("hue_shift_value", 0),
                st.session_state.get("saturation_value", 1.0),
                st.session_state.get("brightness_value", 1.0)
            )
            st.image(output_np, use_column_width=True)
            
            # Display colorized file size below the image
            # Calculate current colorized image size
            colorized_bytes = to_download_bytes(output_np, fmt="PNG")
            colorized_size = len(colorized_bytes)
            st.caption(f"📦 Colorized Size: {format_size(colorized_size)}")
            
            # Display cached metrics from original colorization
            if "original_metrics" in st.session_state:
                metrics = st.session_state.original_metrics
                if metrics is not None:
                    # Extract dimensions from the size string and calculate aspect ratio
                    if "size" in metrics and "×" in metrics["size"]:
                        dimensions = metrics["size"].split("×")
                        if len(dimensions) == 2:
                            try:
                                w = int(dimensions[0])
                                h = int(dimensions[1])
                                # Calculate aspect ratio
                                gcd_value = np.gcd(w, h)
                                ratio_w = w // gcd_value
                                ratio_h = h // gcd_value
                                aspect_ratio = f"{ratio_w}:{ratio_h}"
                                k1_disp.metric("Image size", aspect_ratio)
                            except:
                                # Fallback to original size if parsing fails
                                k1_disp.metric("Image size", metrics["size"])
                        else:
                            k1_disp.metric("Image size", metrics["size"])
                    else:
                        k1_disp.metric("Image size", metrics["size"])
                    k2_disp.metric("Processing time", metrics["time"])
                    k3_disp.metric("Colorfulness", metrics["colorfulness"])
                    k4_disp.metric("File size change", metrics["size_change"])
        else:
            st.caption("Click Colorize to process the image.")
        
        # Store reference for later use in controls
        st.session_state._input_image_np = input_image_np if 'input_image_np' in locals() else None
    
    # Colorize button and Color Adjustments - only show if image uploaded
    is_color_detected = st.session_state.get("is_already_colored", False)
    force_colorize = st.session_state.get("force_colorize", False)
    run_btn = False  # Default: button not clicked
    
    # Initialize slider values
    hue_shift = st.session_state.get("hue_shift_value", 0)
    saturation = st.session_state.get("saturation_value", 1.0)
    brightness = st.session_state.get("brightness_value", 1.0)
    
    # Initialize show_color_adjustments state if not present
    if "show_color_adjustments" not in st.session_state:
        st.session_state.show_color_adjustments = False
    
    # Initialize control values
    enhance = True
    force_colorize = st.session_state.get("force_colorize", False)
    
    # Check if reset to original is requested (before any other processing)
    if st.session_state.get("reset_to_original", False) and st.session_state.get("original_colorized") is not None:
        # Reset slider values to neutral
        st.session_state.hue_shift_value = 0
        st.session_state.saturation_value = 1.0
        st.session_state.brightness_value = 1.0
        st.session_state.reset_to_original = False
        # Update local variables to reflect the reset
        hue_shift = 0
        saturation = 1.0
        brightness = 1.0
        # Don't call st.rerun() here to avoid tab switching issue

    # Only show Colorize button and Color Adjustments if an image has been uploaded
    if uploaded_bw is not None and uploaded_bw.size > 0:
        st.markdown("---")
        
        # First row: Color Adjustments button (left) and Colorize button (center)
        col_adjust_btn, col_colorize_btn, col_download_btn = st.columns([1, 1, 1])
        
        with col_adjust_btn:
            # Color Adjustments button - looks like Colorize button
            if st.button("🎨 Color Adjustments", use_container_width=True):
                st.session_state.show_color_adjustments = not st.session_state.show_color_adjustments
                st.rerun()
        
        with col_colorize_btn:
            # Colorize button in the center
            run_btn = st.button("Colorize", disabled=(is_color_detected and not force_colorize), use_container_width=True)
        
        # Download button - only visible after colorizing the image
        with col_download_btn:
            if st.session_state.get("original_colorized") is not None:
                original_colorized = st.session_state.get("original_colorized")
                # Apply current adjustments for download
                if original_colorized is not None:
                    output_np = adjust_colors(
                        original_colorized,
                        st.session_state.get("hue_shift_value", 0),
                        st.session_state.get("saturation_value", 1.0),
                        st.session_state.get("brightness_value", 1.0)
                    )
                    dl_bytes = to_download_bytes(output_np, fmt="PNG")
                    st.download_button(
                        label="📥 Download",
                        data=dl_bytes,
                        file_name="colorized.png",
                        mime="image/png",
                        use_container_width=True,
                    )
        
        # Show adjustments only when Color Adjustments button is clicked
        if st.session_state.show_color_adjustments:
            st.markdown("#### Adjustments & Controls")
            
            # Use the complete horizontal width with spacing between sliders and controls
            col_hue, col_bright, col_sat, col_space, col_enhance, col_reset_col = st.columns([2, 2, 2, 1, 2, 2])
            
            # Sliders
            with col_hue:
                st.write("**Hue Shift:**")
                hue_shift = st.slider(
                    "Hue Shift", 
                    -180, 180, 
                    st.session_state.get("hue_shift_value", 0),
                    key="hue_shift_slider",
                    help="Shift colors (useful to fix green/yellow tints)",
                    label_visibility="collapsed"
                )
            
            with col_bright:
                st.write("**Brightness:**")
                brightness = st.slider(
                    "Brightness", 
                    0.5, 1.5, 
                    st.session_state.get("brightness_value", 1.0),
                    0.1,
                    key="brightness_slider",
                    help="Adjust overall brightness",
                    label_visibility="collapsed"
                )
            
            with col_sat:
                st.write("**Saturation:**")
                saturation = st.slider(
                    "Saturation", 
                    0.0, 2.0, 
                    st.session_state.get("saturation_value", 1.0),
                    0.1,
                    key="saturation_slider",
                    help="Adjust color intensity",
                    label_visibility="collapsed"
                )
            
            # Empty space column
            with col_space:
                st.write("")
            
            # Enhance control only
            with col_enhance:
                enhance = st.checkbox("Enhance", value=True)
            
            # Reset button in its own column
            with col_reset_col:
                if st.session_state.get("original_colorized") is not None:
                    if st.button("🔄 Reset to Original", use_container_width=True):
                        # Reset all adjustments to neutral values
                        st.session_state.hue_shift_value = 0
                        st.session_state.saturation_value = 1.0
                        st.session_state.brightness_value = 1.0
                        st.session_state.reset_to_original = True
                        st.rerun()
        
            # Store current slider values (but only if not resetting)
            if not st.session_state.get("reset_to_original", False):
                st.session_state.hue_shift_value = hue_shift
                st.session_state.saturation_value = saturation
                st.session_state.brightness_value = brightness
    
    else:
        # No image uploaded - don't show the adjustment button or options
        st.markdown("---")

    # Disable colorize button if color image detected
    if is_color_detected:
        st.error("⚠️ Color image detected!")
        st.caption("Colorizer only works with B&W images")
    
    # Process colorization if button clicked
    input_image_np = st.session_state.get("_input_image_np", None)
    is_reset = st.session_state.get("_is_reset", False)
    is_already_colored = st.session_state.get("is_already_colored", False)
    
    if run_btn and input_image_np is not None and not is_reset:
        # Check if we should proceed with colorization
        should_colorize = not is_already_colored or force_colorize
        
        if not should_colorize:
            st.error("❌ Cannot colorize: This image is already in color!")
            st.info("💡 The colorizer is designed for black & white images only. Please upload a grayscale image to use this feature.")
        else:
            # Show spinner next to the original image
            with st.spinner("Colorizing…"):
                start = time.perf_counter()
                enhanced_src = input_image_np.copy()
                if enhance:
                    lab = cv2.cvtColor(enhanced_src, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    lab = cv2.merge([l, a, b])
                    enhanced_src = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                output_np = colorize_image(enhanced_src)
                
                elapsed = (time.perf_counter() - start) * 1000.0
                
                # Store original colorization (first time only)
                if st.session_state.get("original_colorized") is None:
                    st.session_state.original_colorized = output_np.copy()
                    
                    h, w = output_np.shape[:2]
                    hsv_temp = cv2.cvtColor(output_np, cv2.COLOR_RGB2HSV)
                    sat_temp = hsv_temp[:, :, 1].astype("float32") / 255.0
                    colorfulness_temp = float(sat_temp.std() * 100.0)
                    
                    st.session_state.original_metrics = {
                        "size": f"{w}×{h}",
                        "time": f"{elapsed:.0f} ms",
                        "colorfulness": f"{colorfulness_temp:.1f}",
                        "size_change": "—",
                        "size_delta": ""
                    }
                
                # Apply color adjustments
                output_np = adjust_colors(output_np, hue_shift, saturation, brightness)
                st.session_state._output_np = output_np
                st.session_state._colorization_done = True
                
                # Metrics
                h, w = output_np.shape[:2]
                hsv = cv2.cvtColor(output_np, cv2.COLOR_RGB2HSV)
                sat = hsv[:, :, 1].astype("float32") / 255.0
                colorfulness = float(sat.std() * 100.0)

                # Calculate aspect ratio
                gcd_value = np.gcd(w, h)
                ratio_w = w // gcd_value
                ratio_h = h // gcd_value
                aspect_ratio = f"{ratio_w}:{ratio_h}"

                k1_disp.metric("Image size", aspect_ratio)
                k2_disp.metric("Processing time", f"{elapsed:.0f} ms")
                k3_disp.metric("Colorfulness", f"{colorfulness:.1f}")

                # File size comparison - compare actual original size to PNG colorized size
                color_bytes_png = to_download_bytes(output_np, fmt="PNG")
                color_size_png = len(color_bytes_png)
                orig_size = st.session_state.get("orig_size", 0)
                
                if orig_size:
                    pct_change = int(round(((color_size_png - orig_size) / orig_size) * 100.0))
                    pct_text = f"+{pct_change}%" if pct_change > 0 else f"{pct_change}%"
                    
                    k4_disp.metric(
                        "File size change", 
                        pct_text
                    )
                    
                    original_metrics = st.session_state.get("original_metrics")
                    if original_metrics is not None and "size_change" in original_metrics and original_metrics["size_change"] == "—":
                        original_metrics["size_change"] = pct_text
                else:
                    k4_disp.metric("File size change", "—")

                # Save to history
                thumb = cv2.resize(output_np, (min(320, w), int(h * min(320, w) / w)))
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, {
                    "type": "colorized",
                    "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "size": f"{w}×{h}",
                    "colorfulness": colorfulness,
                    "image": thumb,
                    "full": output_np,
                })
