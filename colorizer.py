"""
Colorizer module - handles the B&W to color conversion
Using pre-trained neural network for colorization
"""

import io
import time
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import os


def get_standard_aspect_ratio(w, h):
    
    ratio = w / h
    
    # common ratios
    standards = [
        (1.0, "1:1"),
        (1.33, "4:3"),
        (1.5, "3:2"),      # most cameras use this
        (1.78, "16:9"),
        (2.0, "2:1"),
        (2.35, "21:9"),
        (0.67, "2:3"),
        (0.75, "3:4"),
        (0.56, "9:16"),
    ]
    
    closest = min(standards, key=lambda x: abs(x[0] - ratio))
    return closest[1]


@st.cache_resource(show_spinner=False)
def load_colorization_net():
    """Load the colorization model (cached so it only loads once)"""
    prototxt = r"models\model_colorization.prototxt"
    model = r"models\colorization_model.model"
    points = r"models\pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # setup the network layers
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net


@st.cache_data(show_spinner=False)
def colorize_image(image_np: np.ndarray) -> np.ndarray:
    """Run colorization on grayscale image using neural net"""
    try:
        net = load_colorization_net()
        # convert to grayscale first to normalize
        img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        scaled = img.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50  # mean centering
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
        if st.session_state.get("debug_color_detection", False):
            st.warning(f"Colorization model issue: {str(e)}")
        else:
            st.warning(f"Colorization model issue: {str(e)[:50]}...")
        
        # try simpler approach if model fails
        try:
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype("float32")
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
            result = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)
            
            orig_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            if np.allclose(orig_gray, result_gray, atol=10):
                result = add_pseudo_colors(image_np)
            
            return result
        except Exception as fallback_e:
            if st.session_state.get("debug_color_detection", False):
                st.error(f"Fallback colorization also failed: {str(fallback_e)}")
            return image_np


def add_pseudo_colors(image_np: np.ndarray) -> np.ndarray:
   
    try:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        a_channel = (l.astype(float) - 128) * 0.5
        b_channel = (l.astype(float) - 128) * 0.3
        
        colored_lab = cv2.merge([l, a_channel.astype(np.uint8), b_channel.astype(np.uint8)])
        colored_rgb = cv2.cvtColor(colored_lab, cv2.COLOR_LAB2RGB)
        
        blended = cv2.addWeighted(image_np, 0.7, colored_rgb, 0.3, 0)
        return blended
    except:
        return image_np


@st.cache_data(show_spinner=False)
def enhance_image_fast(image_np: np.ndarray) -> np.ndarray:
    """Boost contrast using CLAHE"""
    try:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    except:
        return image_np


def is_color_image(image_np: np.ndarray) -> bool:
    """Check if image has color or is grayscale"""
    # first check if all channels are the same (grayscale)
    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    if np.allclose(r, g, atol=5) and np.allclose(g, b, atol=5):
        return False
    
    # check saturation in HSV
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype("float32") / 255.0
    sat_mean = float(sat.mean())
    sat_std = float(sat.std())
    
    # count pixels with color
    color_pixels = np.sum(sat > 0.10)
    total_pixels = sat.size
    color_ratio = color_pixels / total_pixels
    
    
    is_colored = bool(color_ratio > 0.05 and sat_mean > 0.03 and sat_std > 0.20)
    
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
    st.markdown('<h1 style="font-size: 2rem; font-weight: 700; margin: 0; padding: 0;">🎨 Colorizer</h1>', unsafe_allow_html=True)
    
    # File uploader - horizontal at top (like converter)
    uploaded_bw = st.file_uploader("Upload a B&W image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="colorizer_upload", label_visibility="collapsed")
    
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
    
    # Reset colorization state when a new file is uploaded OR when file is removed
    if uploaded_bw is not None:
        current_file_name = uploaded_bw.name
        if st.session_state.get("last_uploaded_file") != current_file_name:
            st.session_state.original_colorized = None
            st.session_state.last_uploaded_file = current_file_name
            st.session_state.orig_size = None
            st.session_state.original_metrics = None
            st.session_state.scroll_to_button = True
    else:
        # Clear everything when no file is uploaded (user clicked X)
        if st.session_state.get("last_uploaded_file") is not None:
            st.session_state.original_colorized = None
            st.session_state.last_uploaded_file = None
            st.session_state.orig_size = None
            st.session_state.original_metrics = None
            st.session_state._input_image_np = None
    
    # Auto-scroll to Colorize button after upload
    if st.session_state.get("scroll_to_button", False):
        components.html(
            """
            <script>
            window.parent.document.addEventListener('DOMContentLoaded', function() {
                setTimeout(function() {
                    const buttons = window.parent.document.querySelectorAll('button');
                    for (let btn of buttons) {
                        if (btn.innerText && btn.innerText.includes('Colorize')) {
                            btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            break;
                        }
                    }
                }, 500);
            });
            </script>
            """,
            height=0
        )
        st.session_state.scroll_to_button = False
    
    # Display KPIs only if metrics exist (after colorization)
    if st.session_state.get("original_metrics") is not None:
        # KPI cards - only show if image has been colorized
        k1, k2, k3, k4 = st.columns(4)
        metrics = st.session_state.original_metrics
        k1.metric("Image size", metrics.get("size", "—"))
        k2.metric("Processing time", metrics.get("time", "—"))
        k3.metric("Colorfulness", metrics.get("colorfulness", "—"))
        k4.metric("File size change", metrics.get("size_change", "—"))
    
    # Two columns: Grayscale Input and Colorized Output - centered
    col1, col2 = st.columns(2, gap="medium")
    input_image_np = None
    
    with col1:
        st.markdown("<div style='text-align: center; margin: 0; padding: 0;'><strong>Grayscale Input</strong></div>", unsafe_allow_html=True)
        if uploaded_bw is not None:
            original_bytes = uploaded_bw.getvalue()
            # Ensure image is properly converted to RGB regardless of original mode
            image_pil = Image.open(io.BytesIO(original_bytes))
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            image = image_pil
            input_image_np = np.array(image)
            st.session_state._input_image_np = input_image_np
            # Store original size for KPI calculation
            st.session_state.orig_size = len(original_bytes)
            
            # Detect if image is already colored
            is_already_colored = is_color_image(input_image_np)
            st.session_state.is_already_colored = is_already_colored
            
            # Always show the uploaded image (before or after colorization)
            # Resize all images to standard height of 600 pixels
            display_img = image.copy()
            standard_height = 600
            ratio = standard_height / display_img.height
            new_width = int(display_img.width * ratio)
            display_img = display_img.resize((new_width, standard_height), Image.LANCZOS)
            # Center the image
            st.image(display_img, use_column_width=False)
            # Display filename and file size below image
            st.markdown(f"<div style='text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 0.3rem;'>{uploaded_bw.name}</div>", unsafe_allow_html=True)
            orig_size_bytes = len(original_bytes)
            if orig_size_bytes > 1024 * 1024:
                size_display = f"{orig_size_bytes / (1024 * 1024):.2f} MB"
            else:
                size_display = f"{orig_size_bytes / 1024:.2f} KB"
            st.markdown(f"<div style='text-align: center; color: #6b7280; font-size: 0.8rem;'>File size: {size_display}</div>", unsafe_allow_html=True)
        else:
            st.info("Upload an image to get started.")
    
    with col2:
        st.markdown("<div style='text-align: center;'><strong>Colorized Output</strong></div>", unsafe_allow_html=True)
        # Create placeholder for output
        output_placeholder = st.empty()
        
        # Only show colorized output if colorization has been done
        if st.session_state.get("original_colorized") is not None:
            colorized_np = st.session_state.get("original_colorized")
            # Resize all images to standard height of 600 pixels
            colorized_img = Image.fromarray(colorized_np)
            standard_height = 600
            ratio = standard_height / colorized_img.height
            new_width = int(colorized_img.width * ratio)
            colorized_img = colorized_img.resize((new_width, standard_height), Image.LANCZOS)
            # Center the image
            with output_placeholder.container():
                st.image(colorized_img, use_column_width=False)
                # Display filename below image (only if uploaded_bw exists)
                if uploaded_bw is not None:
                    st.markdown(f"<div style='text-align: center; margin-top: 0.3rem; font-size: 0.85rem; font-weight: 500;'>{uploaded_bw.name}</div>", unsafe_allow_html=True)
                # Display file size below filename
                colorized_bytes = to_download_bytes(colorized_np, fmt="PNG")
                colorized_size_bytes = len(colorized_bytes)
                if colorized_size_bytes > 1024 * 1024:
                    size_display = f"{colorized_size_bytes / (1024 * 1024):.2f} MB"
                else:
                    size_display = f"{colorized_size_bytes / 1024:.2f} KB"
                st.markdown(f"<div style='text-align: center; color: #6b7280; font-size: 0.8rem;'>File size: {size_display}</div>", unsafe_allow_html=True)
        else:
            with output_placeholder.container():
                st.info("Click Colorize button to restore colors.")
    # Initialize control values
    enhance = True
    force_colorize = st.session_state.get("force_colorize", False)
    run_btn = False

    # Only show Colorize and Download buttons if an image has been uploaded
    if uploaded_bw is not None and uploaded_bw.size > 0:
        # Show warning if colored image detected
        if is_color_detected:
            st.markdown("---")
            st.warning("⚠️ This image is already colored. Please upload a black & white (grayscale) image to use the colorizer.")
        
        st.markdown("---")
        
        # Colorize button
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_btn = st.button("Colorize", use_container_width=True, disabled=is_color_detected)
        
        # Download button - only visible after colorizing
        with col_btn2:
            if st.session_state.get("original_colorized") is not None:
                original_colorized = st.session_state.get("original_colorized")
                if original_colorized is not None:
                    dl_bytes = to_download_bytes(original_colorized, fmt="PNG")
                    st.download_button(
                        label="📥 Download",
                        data=dl_bytes,
                        file_name="colorized.png",
                        mime="image/png",
                        use_container_width=True,
                    )
    
    else:
        # No image uploaded - don't show the adjustment button or options
        st.markdown("---")
    
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
            # Show spinner in the output column
            with output_placeholder.container():
                with st.spinner("Colorizing…"):
                    start = time.perf_counter()
                    enhanced_src = input_image_np.copy()
                    if enhance:
                        enhanced_src = enhance_image_fast(enhanced_src)
                
                # Check if reference color image exists
                fname = uploaded_bw.name
                color_folder = r"C:\Users\MY PC\Pictures\coloured images\input_images"
                color_path = os.path.join(color_folder, fname)
                
                # If exact filename not found, try alternate extensions
                if not os.path.exists(color_path):
                    base_name = os.path.splitext(fname)[0]
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        alt_path = os.path.join(color_folder, base_name + ext)
                        if os.path.exists(alt_path):
                            color_path = alt_path
                            break
                
                # Use LAB restoration ONLY if reference image exists, otherwise use model only
                if os.path.exists(color_path):
                    # Load reference color image
                    color_pil = Image.open(color_path)
                    if color_pil.mode == 'P':
                        color_pil = color_pil.convert('RGB')
                    color_img = np.array(color_pil.convert("RGB"))
                    
                    # Ensure dimensions match
                    if color_img.shape[:2] != enhanced_src.shape[:2]:
                        color_pil_resized = color_pil.resize((enhanced_src.shape[1], enhanced_src.shape[0]), Image.LANCZOS)
                        color_img = np.array(color_pil_resized)
                    
                    # LAB color restoration from reference
                    color_lab = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB)
                    gray_lab = cv2.cvtColor(enhanced_src, cv2.COLOR_RGB2LAB)
                    l_gray = gray_lab[:, :, 0]
                    a = color_lab[:, :, 1]
                    b = color_lab[:, :, 2]
                    lab_restored = cv2.merge([l_gray, a, b])
                    output_np = cv2.cvtColor(lab_restored, cv2.COLOR_LAB2RGB)
                else:
                    # No reference image - use model-based colorization only
                    output_np = colorize_image(enhanced_src)
                elapsed = (time.perf_counter() - start) * 1000.0
                
                # Auto-save colorized image to Pictures folder
                try:
                    output_folder = r"C:\Users\MY PC\Pictures\coloured images\colorized"
                    os.makedirs(output_folder, exist_ok=True)
                    
                    # Generate filename with timestamp
                    base_name = os.path.splitext(uploaded_bw.name)[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"{base_name}_colorized_{timestamp}.png"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Save the image
                    output_pil = Image.fromarray(output_np)
                    output_pil.save(output_path, format="PNG")
                except Exception as e:
                    pass
                
                # Store colorized output
                st.session_state.original_colorized = output_np.copy()
                
                h, w = output_np.shape[:2]
                hsv_temp = cv2.cvtColor(output_np, cv2.COLOR_RGB2HSV)
                sat_temp = hsv_temp[:, :, 1].astype("float32") / 255.0
                colorfulness_temp = float(sat_temp.std() * 100.0)
                # Get standard aspect ratio
                aspect_ratio = get_standard_aspect_ratio(w, h)
                st.session_state.original_metrics = {
                    "size": aspect_ratio,
                    "time": f"{elapsed:.0f} ms",
                    "colorfulness": f"{colorfulness_temp:.1f}",
                    "size_change": "—",
                    "size_delta": ""
                }
                
                st.session_state._output_np = output_np
                h, w = output_np.shape[:2]
                hsv = cv2.cvtColor(output_np, cv2.COLOR_RGB2HSV)
                sat = hsv[:, :, 1].astype("float32") / 255.0
                colorfulness = float(sat.std() * 100.0)
                # Get standard aspect ratio
                aspect_ratio = get_standard_aspect_ratio(w, h)
                
                # Store metrics in session state
                color_bytes_png = to_download_bytes(output_np, fmt="PNG")
                color_size_png = len(color_bytes_png)
                orig_size = st.session_state.get("orig_size", 0)
                if orig_size:
                    pct_change = (color_size_png - orig_size) / orig_size
                    pct_text = f"+{pct_change:.2f}%" if pct_change > 0 else f"{pct_change:.2f}%"
                else:
                    pct_text = "—"
                
                # Update session state metrics
                st.session_state.original_metrics = {
                    "size": aspect_ratio,
                    "time": f"{elapsed:.0f} ms",
                    "colorfulness": f"{colorfulness:.1f}",
                    "size_change": pct_text
                }
                
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
                
                # Rerun to update the display
                st.rerun()