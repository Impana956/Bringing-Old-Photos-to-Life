#!/usr/bin/env python
# coding: utf-8

# Merged Image Processing App - Colorizer + Format Converter

import os
import io
import base64
import time
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import img2pdf
import zipfile

# ---------- Page config & theming ----------
st.set_page_config(
    page_title="Colour & Convert",
    page_icon="🎨",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Base UI polish */
    .stApp {background: linear-gradient(180deg, #e8f0fe 0%, #f0e7ff 50%, #ffffff 100%) !important;}    
    .app-title {font-size: 2.2rem; font-weight: 700; letter-spacing: .3px;}
    .muted {color: #6b7280;}
    .foot {font-size: 0.85rem; color: #6b7280;}
    .stDownloadButton>button {border-radius: 12px; font-weight: 600;}
    .stButton>button {border-radius: 12px; font-weight: 600;}
    
    /* Center and enlarge tabs */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.5rem;
        font-weight: 700;
        padding: 1rem 2rem;
    }
    
    /* Override file uploader limit text */
    [data-testid="stFileUploader"] small {
        display: none;
    }
    [data-testid="stFileUploader"] section::after {
        content: "Limit 50MB per file";
        display: block;
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create necessary folders
CONVERTED_FOLDER = "converted"
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

# ---------- Model loading (cached) ----------
@st.cache_resource(show_spinner=False)
def load_colorization_net():
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


def _ensure_rgb_from_any(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    else:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) if image_np.shape[2] == 3 else image_np
    return image_np


@st.cache_data(show_spinner=False)
def colorize_image(image_np: np.ndarray) -> np.ndarray:
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


def to_download_bytes(img_np: np.ndarray, fmt: str = "PNG", **save_kwargs) -> bytes:
    """Return image bytes for an ndarray image."""
    image = Image.fromarray(img_np)
    buf = io.BytesIO()
    image.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def is_color_image(image_np: np.ndarray, threshold: float = 15.0) -> bool:
    """
    Detect if an image is already colored.
    Returns True if the image has significant color information.
    
    Method: Calculate standard deviation of saturation in HSV.
    If std is above threshold, image is considered colored.
    Threshold of 15.0 allows for mostly grayscale images while detecting truly colored ones.
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype("float32") / 255.0
    sat_std = float(sat.std() * 100.0)
    return sat_std > threshold


def adjust_colors(image_np: np.ndarray, hue_shift: int = 0, saturation: float = 1.0, brightness: float = 1.0) -> np.ndarray:
    """
    Apply color adjustments to an image.
    
    Args:
        image_np: Input image in RGB format
        hue_shift: Shift hue by degrees (-180 to 180). Negative values shift towards blue/purple, positive towards yellow/red
        saturation: Multiply saturation (0.0 to 2.0). < 1.0 reduces color intensity, > 1.0 increases it
        brightness: Multiply value/brightness (0.5 to 1.5)
    
    Returns:
        Adjusted image in RGB format
    """
    # Convert to HSV for color adjustments
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    
    # Apply hue shift (wrap around 0-180 range in OpenCV)
    if hue_shift != 0:
        h = (h + hue_shift / 2.0) % 180.0  # OpenCV uses 0-180 range for hue
    
    # Apply saturation adjustment
    if saturation != 1.0:
        s = np.clip(s * saturation, 0, 255)
    
    # Apply brightness adjustment
    if brightness != 1.0:
        v = np.clip(v * brightness, 0, 255)
    
    # Merge and convert back to RGB
    hsv_adjusted = cv2.merge([h, s, v]).astype("uint8")
    rgb_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    
    return rgb_adjusted


def compress_image_to_target(img: Image.Image, target_format: str, original_size_bytes: int) -> bytes:
    """
    Compress image with high quality while trying to reduce file size.
    Priority: Quality > File Size
    
    Args:
        img: PIL Image object
        target_format: Output format ("jpg", "png", "bmp")
        original_size_bytes: Original file size in bytes
    
    Returns:
        Compressed image bytes
    """
    if target_format == "bmp":
        buf = io.BytesIO()
        img.save(buf, "BMP")
        return buf.getvalue()
    
    elif target_format == "jpg":
        # Start with very high quality (minimal quality loss)
        quality = 95
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        
        # If already smaller, return high quality version
        if buf.tell() < original_size_bytes:
            return buf.getvalue()
        
        # Try slightly lower quality (still excellent)
        quality = 90
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        
        if buf.tell() < original_size_bytes:
            return buf.getvalue()
        
        # Use quality 85 (very good, widely used standard)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85, optimize=True)
        return buf.getvalue()
    
    elif target_format == "png":
        # PNG is lossless - quality never changes
        buf = io.BytesIO()
        img.save(buf, "PNG", compress_level=9, optimize=True)
        return buf.getvalue()
    
    return b""


# ---------- Initialize session state ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "target_format" not in st.session_state:
    st.session_state.target_format = "png"
if "uploaded_format" not in st.session_state:
    st.session_state.uploaded_format = None
if "original_colorized" not in st.session_state:
    st.session_state.original_colorized = None

# ---------- Header ----------
st.markdown('<div class="app-title" style="text-align: center;">🎨 Colour & Convert</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; margin-bottom: 2rem;">Colorize photos and convert image formats - all in one place.</p>', unsafe_allow_html=True)

# ---------- Main Tabs ----------
tab_colorizer, tab_converter, tab_history = st.tabs(["🎨 Colorizer", "🔄 Format Converter", "📂 History"])

# ==================== COLORIZER TAB ====================
with tab_colorizer:
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1.5rem;">🎨 B&W → Colorizer</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    col_sidebar, col_main = st.columns([1, 3])
    
    with col_sidebar:
        st.markdown("### Controls")
        theme = st.selectbox("Themes", ["Light", "Theme1", "Theme2", "Theme3"], index=0, key="colorizer_theme")
        uploaded_bw = st.file_uploader("Upload a B&W image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="colorizer_upload")
        enhance = st.checkbox("Enhance contrast", value=True)
        run_btn = st.button("Colorize")
        
        # Color adjustment controls
        st.markdown("---")
        st.markdown("### Color Adjustments")
        hue_shift = st.slider("Hue Shift", -180, 180, 0, help="Shift colors (useful to fix green/yellow tints)")
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1, help="Adjust color intensity")
        brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1, help="Adjust overall brightness")
        
        # Reset button
        if st.session_state.original_colorized is not None:
            if st.button("🔄 Reset to Original", help="Reset to first colorization before adjustments"):
                st.session_state.reset_to_original = True
    
    # Apply theme
    css = ""
    if theme == "Light":
        css = """
        <style>
            .stApp {background: linear-gradient(180deg, #e8f0fe 0%, #f0e7ff 50%, #ffffff 100%) !important;} 
            .app-title {background: linear-gradient(90deg,#5b6fd8,#7c4dff); -webkit-background-clip:text; background-clip:text; color: transparent;}
            .stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#5b6fd8,#7c4dff); color: white; border: 0;}
        </style>
        """
    elif theme == "Theme1":
        css = """
        <style>
            .stApp {background: linear-gradient(160deg, #ede9fe 0%, #f5f3ff 40%, #faf5ff 100%) !important;} 
            .app-title {background: linear-gradient(90deg,#6d28d9,#8b5cf6,#a78bfa); -webkit-background-clip:text; background-clip:text; color: transparent;}
            .stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#7c3aed,#8b5cf6); color: white; border: 0;}
        </style>
        """
    elif theme == "Theme2":
        css = """
        <style>
            .stApp {background: linear-gradient(145deg, #f0f9ff 0%, #fde68a 35%, #fbcfe8 100%) !important;} 
            .app-title {background: linear-gradient(90deg,#f97316,#ef4444,#a855f7,#06b6d4); -webkit-background-clip:text; background-clip:text; color: transparent;}
            .stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#f97316,#ef4444); color: white; border: 0;}
        </style>
        """
    elif theme == "Theme3":
        css = """
        <style>
            .stApp {background: linear-gradient(160deg, #ffedd5 0%, #fecaca 45%, #e9d5ff 100%) !important;} 
            .app-title {background: linear-gradient(90deg,#fb923c,#f43f5e,#8b5cf6); -webkit-background-clip:text; background-clip:text; color: transparent;}
            .stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#fb923c,#f43f5e); color: white; border: 0;}
        </style>
        """
    
    if css:
        st.markdown(css, unsafe_allow_html=True)
    
    with col_main:
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
                
                # Only check color and calculate metrics if not already done
                if "orig_size" not in st.session_state:
                    # Check if image is already colored
                    is_already_colored = is_color_image(input_image_np)
                    
                    # Store metrics once
                    st.session_state.orig_size = len(original_bytes)
                    st.session_state.orig_format = getattr(image, "format", None) or None
                    st.session_state.is_already_colored = is_already_colored
                    try:
                        st.session_state.orig_norm_size = len(to_download_bytes(input_image_np, fmt="JPEG", quality=85, optimize=True))
                    except Exception:
                        st.session_state.orig_norm_size = st.session_state.orig_size
                else:
                    # Use cached value
                    is_already_colored = st.session_state.get("is_already_colored", False)
                
                st.image(image, use_column_width=True)
                
                # Show warning if image is already colored
                if is_already_colored:
                    st.warning("⚠️ This image appears to be already colored! The colorizer is designed for black & white images. Processing will convert it to grayscale first, then re-colorize it, which may not preserve the original colors.")
            else:
                st.info("Upload an image from the sidebar.")
        
        with col_right:
            st.markdown("**Colorized**")
            output_np = None
            is_reset = False
            
            # Check if reset to original is requested
            if st.session_state.get("reset_to_original", False) and st.session_state.original_colorized is not None:
                output_np = st.session_state.original_colorized.copy()
                st.session_state.reset_to_original = False
                is_reset = True
                st.image(output_np, use_column_width=True)
                st.success("✅ Restored to original colorization (instant!)")
                
                # Display cached metrics from original colorization
                if "original_metrics" in st.session_state:
                    metrics = st.session_state.original_metrics
                    k1_disp.metric("Image size", metrics["size"])
                    k2_disp.metric("Processing time", metrics["time"])
                    k3_disp.metric("Colorfulness", metrics["colorfulness"])
                    k4_disp.metric("File size change", metrics["size_change"], delta=metrics.get("size_delta", ""))
            
            if run_btn and input_image_np is not None and not is_reset:
                # Check if image is colored and skip processing
                if is_already_colored:
                    st.error("❌ Cannot colorize: This image is already in color!")
                    st.info("💡 The colorizer is designed for black & white images only. Please upload a grayscale image to use this feature.")
                else:
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
                        if st.session_state.original_colorized is None:
                            st.session_state.original_colorized = output_np.copy()
                            
                            # Also store the metrics for this original colorization
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
                        st.image(output_np, use_column_width=True)
                        st.success(f"Done in {elapsed:.0f} ms")

                        # Metrics
                        h, w = output_np.shape[:2]
                        hsv = cv2.cvtColor(output_np, cv2.COLOR_RGB2HSV)
                        sat = hsv[:, :, 1].astype("float32") / 255.0
                        colorfulness = float(sat.std() * 100.0)

                        k1_disp.metric("Image size", f"{w}×{h}")
                        k2_disp.metric("Processing time", f"{elapsed:.0f} ms")
                        k3_disp.metric("Colorfulness", f"{colorfulness:.1f}")

                        # File size comparison
                        color_bytes_norm = to_download_bytes(output_np, fmt="JPEG", quality=85, optimize=True)
                        color_size_norm = len(color_bytes_norm)
                        orig_size_norm = st.session_state.get("orig_norm_size", 0)
                        
                        if orig_size_norm:
                            pct_change = int(round(((color_size_norm - orig_size_norm) / orig_size_norm) * 100.0))
                            pct_text = f"+{pct_change}%" if pct_change > 0 else f"{pct_change}%"
                            
                            # Format file sizes in KB or MB
                            def format_size(size_bytes):
                                if size_bytes < 1024 * 1024:  # Less than 1MB
                                    return f"{size_bytes / 1024:.1f} KB"
                                else:
                                    return f"{size_bytes / (1024 * 1024):.2f} MB"
                            
                            orig_size_text = format_size(orig_size_norm)
                            new_size_text = format_size(color_size_norm)
                            
                            k4_disp.metric(
                                "File size change", 
                                pct_text,
                                delta=f"{orig_size_text} → {new_size_text}"
                            )
                            
                            # Update original metrics with file size info if this is the first colorization
                            if "original_metrics" in st.session_state and st.session_state.original_metrics["size_change"] == "—":
                                st.session_state.original_metrics["size_change"] = pct_text
                                st.session_state.original_metrics["size_delta"] = f"{orig_size_text} → {new_size_text}"
                        else:
                            k4_disp.metric("File size change", "—")

                        # Save to history
                        thumb = cv2.resize(output_np, (min(320, w), int(h * min(320, w) / w)))
                        st.session_state.history.insert(0, {
                            "type": "colorized",
                            "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "size": f"{w}×{h}",
                            "colorfulness": colorfulness,
                            "image": thumb,
                            "full": output_np,
                        })
            elif input_image_np is not None:
                st.caption("Click Colorize in the sidebar to process.")
        
        # Download button
        if 'output_np' in locals() and output_np is not None:
            dl_bytes = to_download_bytes(output_np, fmt="PNG")
            st.download_button(
                label="📥 Download colorized image",
                data=dl_bytes,
                file_name="colorized.png",
                mime="image/png",
            )

# ==================== CONVERTER TAB ====================
with tab_converter:
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">🔄 Image ⇄ PDF Converter</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280; margin-bottom: 1.5rem;">Upload an image or PDF and choose the target format.</p>', unsafe_allow_html=True)
    
    # Conversion settings
    uploaded_file = st.file_uploader("Choose a file to convert", type=["pdf", "png", "jpg", "jpeg", "bmp"], key="file_upload")
    
    if uploaded_file:
        filename = uploaded_file.name
        base_name, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        current_format = ext.replace(".", "")
        if current_format == "jpeg":
            current_format = "jpg"
        
        # Create preview columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original File**")
            try:
                if ext == ".pdf":
                    uploaded_bytes = uploaded_file.getvalue()
                    doc = fitz.open(stream=uploaded_bytes)
                    page = doc.load_page(0)
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, caption=f"{filename} (page 1)", use_column_width=True)
                else:
                    st.image(uploaded_file, caption=filename, use_column_width=True)
            except Exception:
                st.warning("Couldn't generate preview.")
        
        # Format selection
        all_formats = ["jpg", "png", "bmp", "pdf"]
        available_formats = [f for f in all_formats if f != current_format]
        
        if st.session_state.target_format not in available_formats:
            st.session_state.target_format = available_formats[0]
        
        format_choice = st.selectbox("Convert to", available_formats, index=available_formats.index(st.session_state.target_format), key="format_select")
        st.session_state.target_format = format_choice
        
        convert_all_pages = False
        if ext == ".pdf" and format_choice in ["jpg", "png"]:
            convert_all_pages = st.checkbox("Convert all pages", value=False)
        
        # Convert button
        if st.button("🔄 Convert", key="convert_btn"):
            progress = st.progress(0)
            time.sleep(0.05)
            progress.progress(10)
            
            try:
                uploaded_bytes = uploaded_file.getvalue()
                
                if ext == ".pdf" and format_choice in ["jpg", "png"]:
                    doc = fitz.open(stream=uploaded_bytes)
                    pages = range(doc.page_count) if convert_all_pages else [0]
                    converted_images = []
                    
                    for i, pnum in enumerate(pages):
                        page = doc.load_page(pnum)
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes(format_choice)
                        converted_images.append((f"{base_name}_page{pnum + 1}.{format_choice}", img_bytes))
                        progress.progress(30 + int(60 * (i + 1) / len(pages)))
                    
                    st.success(f"✅ Converted PDF → {len(converted_images)} image(s)")
                    
                    with col2:
                        st.markdown("**Converted Image**")
                        st.image(converted_images[0][1], caption=converted_images[0][0], use_column_width=True)
                    
                    if len(converted_images) > 1:
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                            for fname, fbytes in converted_images:
                                zf.writestr(fname, fbytes)
                        zip_buf.seek(0)
                        st.download_button("📥 Download all pages (.zip)", zip_buf.read(), file_name=f"{base_name}_pages.zip")
                    else:
                        fname, fbytes = converted_images[0]
                        st.download_button("📥 Download file", fbytes, file_name=fname)
                
                elif format_choice == "pdf":
                    img = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
                    temp_buf = io.BytesIO()
                    img.save(temp_buf, format="JPEG", quality=85, optimize=True)
                    temp_buf.seek(0)
                    
                    pdf_bytes = img2pdf.convert(temp_buf.getvalue())
                    progress.progress(95)
                    st.success(f"✅ Converted image → {base_name}.pdf")
                    
                    with col2:
                        st.markdown("**Converted PDF (Preview)**")
                        doc = fitz.open(stream=pdf_bytes)
                        page = doc.load_page(0)
                        pix = page.get_pixmap()
                        pdf_preview_bytes = pix.tobytes("png")
                        st.image(pdf_preview_bytes, caption=f"{base_name}.pdf", use_column_width=True)
                    
                    st.download_button("📥 Download PDF", pdf_bytes, file_name=f"{base_name}.pdf")
                
                else:
                    img = Image.open(io.BytesIO(uploaded_bytes))
                    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                        alpha = img.convert("RGBA").split()[-1]
                        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                        bg.paste(img, mask=alpha)
                        img = bg.convert("RGB")
                    else:
                        img = img.convert("RGB")
                    
                    output_filename = f"{base_name}.{format_choice}"
                    
                    # Compress with high quality
                    original_size = len(uploaded_bytes)
                    compressed_bytes = compress_image_to_target(img, format_choice, original_size)
                    output_buf = io.BytesIO(compressed_bytes)
                    
                    original_size_kb = original_size / 1024
                    file_size_kb = len(compressed_bytes) / 1024
                    
                    if len(compressed_bytes) < original_size:
                        reduction_pct = ((original_size - len(compressed_bytes)) / original_size) * 100
                        st.success(f"✅ {output_filename} | {original_size_kb:.1f} KB → {file_size_kb:.1f} KB ({reduction_pct:.1f}% smaller)")
                    else:
                        increase_pct = ((len(compressed_bytes) - original_size) / original_size) * 100
                        st.info(f"ℹ️ {output_filename} | {original_size_kb:.1f} KB → {file_size_kb:.1f} KB (+{increase_pct:.1f}% larger - high quality preserved)")
                    
                    progress.progress(95)
                    
                    with col2:
                        st.markdown("**Converted Image**")
                        st.image(output_buf.getvalue(), caption=output_filename, use_column_width=True)
                    
                    output_buf.seek(0)
                    st.download_button("📥 Download converted file", output_buf.read(), file_name=output_filename)
                
                progress.progress(100)
            
            except Exception as e:
                st.error(f"Error during conversion: {e}")
                progress.progress(0)

# ==================== HISTORY TAB ====================
with tab_history:
    st.subheader("Processing History")
    if not st.session_state.history:
        st.info("No results yet. Process an image to see your gallery here.")
    else:
        cols = st.columns(3)
        for idx, item in enumerate(st.session_state.history):
            with cols[idx % 3]:
                caption_text = f"{item['when']} • {item['size']}"
                if item.get('colorfulness'):
                    caption_text += f" • {item['colorfulness']:.1f}"
                st.image(item["image"], caption=caption_text, use_column_width=True)
                b = to_download_bytes(item["full"], fmt="PNG")
                st.download_button(
                    label="📥 Download",
                    data=b,
                    file_name=f"processed_{idx+1}.png",
                    mime="image/png",
                    key=f"dl_hist_{idx}",
                )
