#!/usr/bin/env python
# coding: utf-8

"""
Colour & Convert - Image Processing Application
Main entry point
"""

import os
import pickle
import streamlit as st
from colorizer import render_colorizer_tab, to_download_bytes
from converter import render_converter_tab
import math

# History file path
HISTORY_FILE = os.path.join(os.path.dirname(__file__), ".app_history.pkl")
WELCOME_FILE = os.path.join(os.path.dirname(__file__), ".welcome_dismissed.pkl")
TAB_FILE = os.path.join(os.path.dirname(__file__), ".active_tab.pkl")

def load_history():
    """Load history from disk"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    return []

def save_history(history):
    """Save history to disk, keeping only last 10 items"""
    try:
        # Keep only the 10 most recent items
        history_to_save = history[:10]
        with open(HISTORY_FILE, 'wb') as f:
            pickle.dump(history_to_save, f)
    except Exception:
        pass

def load_welcome_status():
    """Load welcome dismissed status from disk"""
    try:
        if os.path.exists(WELCOME_FILE):
            with open(WELCOME_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    return False

def save_welcome_status(dismissed):
    """Save welcome dismissed status to disk"""
    try:
        with open(WELCOME_FILE, 'wb') as f:
            pickle.dump(dismissed, f)
    except Exception:
        pass

def load_active_tab():
    """Load active tab from disk"""
    try:
        if os.path.exists(TAB_FILE):
            with open(TAB_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    return "🎨 Colorizer"

def save_active_tab(tab_name):
    """Save active tab to disk"""
    try:
        with open(TAB_FILE, 'wb') as f:
            pickle.dump(tab_name, f)
    except Exception:
        pass

# ---------- Page config & theming ----------
st.set_page_config(
    page_title="Colour & Convert",
    page_icon="🎨",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Remove all top spacing */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: 0rem !important;
    }
    header {
        display: none !important;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Hide file uploader image preview */
    [data-testid="stFileUploader"] img {
        display: none !important;
    }
    
    /* Compress all vertical spacing */
    .element-container {
        margin-bottom: 0.2rem !important;
    }
    h1, h2, h3 {
        margin-top: 0 !important;
        margin-bottom: 0.3rem !important;
        padding: 0 !important;
    }
    p {
        margin-top: 0 !important;
        margin-bottom: 0.3rem !important;
    }
    .stMarkdown {
        margin-bottom: 0.2rem !important;
    }
    hr {
        margin: 0.5rem 0 !important;
    }
    
    /* Hide file uploader filename and reduce spacing */
    [data-testid="stFileUploader"] {
        margin-bottom: 0rem !important;
        padding: 0rem !important;
        margin-top: 0rem !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0rem !important;
        margin: 0rem !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    [data-testid="stFileUploader"] section > div {
        padding-bottom: 0rem !important;
        padding-top: 0rem !important;
        margin: 0rem !important;
    }
    [data-testid="stFileUploader"] section + div {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
    [data-testid="stFileUploader"] small {
        display: none !important;
    }
    [data-testid="stFileUploader"] button {
        margin: 0rem !important;
        margin-right: 0.1rem !important;
        padding: 0.25rem 0.5rem !important;
        order: 2 !important;
    }
    [data-testid="stFileUploader"] > div {
        padding: 0rem !important;
        gap: 0rem !important;
        display: flex !important;
        justify-content: space-between !important;
    }
    
    /* Base UI polish */
    .stApp {background: linear-gradient(180deg, #d0c5ff 0%, #e8d5ff 50%, #f5f0ff 100%) !important;}    
    .app-title {
        font-size: 3rem; 
        font-weight: 800; 
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .muted {color: #6b7280;}
    .foot {font-size: 0.85rem; color: #6b7280;}
    
    /* Enhanced buttons with hover effects */
    .stDownloadButton>button {
        border-radius: 12px; 
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stButton>button {
        border-radius: 12px; 
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stButton>button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    /* Style radio buttons as tabs */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:has(div[role="radiogroup"]) {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    /* Hide Navigation label */
    div[role="radiogroup"] > label.row-widget.stRadio > div[data-testid="stMarkdownContainer"] {
        display: none !important;
    }
    div[data-testid="stRadio"] > label {
        display: none !important;
    }
    div[role="radiogroup"] {
        display: flex !important;
        gap: 2rem !important;
        justify-content: center !important;
    }
    div[role="radiogroup"] label {
        display: flex !important;
        align-items: center !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        padding: 1.2rem 2.5rem !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        position: relative !important;
        pointer-events: auto !important;
        user-select: none !important;
    }
    /* Hide radio button circles */
    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
        margin-left: 0 !important;
        pointer-events: none !important;
    }
    div[role="radiogroup"] label input[type="radio"] {
        position: absolute !important;
        opacity: 0 !important;
        width: 100% !important;
        height: 100% !important;
        top: 0 !important;
        left: 0 !important;
        cursor: pointer !important;
        z-index: 10 !important;
    }
    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    div[role="radiogroup"] label span[data-baseweb="radio"] {
        display: none !important;
    }
    
    /* Center images and constrain size */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    [data-testid="stImage"] img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }
    
    /* Override file uploader limit text */
    [data-testid="stFileUploader"] small {
        display: none;
    }
    
    /* Enhanced metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(250,250,255,0.95) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        border: 2px solid rgba(124, 58, 237, 0.3);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.3);
        border-color: rgba(124, 58, 237, 0.6);
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        font-weight: 700;
        color: #5b21b6 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4a148c !important;
    }
    
    /* Enhanced file uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.5);
        border-radius: 15px;
        padding: 1rem !important;
        border: 2px dashed #a855f7;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        background: rgba(255,255,255,0.8);
        border-color: #7c3aed;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
    }
    
    /* Radio buttons enhancement - Full clickable area */
    .stRadio > label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #5b21b6;
    }
    [data-testid="stRadio"] > div {
        gap: 2rem;
        justify-content: center;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 3rem;
    }
    [data-testid="stRadio"] label {
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1rem 3rem;
        transition: all 0.3s ease;
        background: transparent;
        border-radius: 15px;
        border: 2px solid transparent;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 1rem;
        min-width: 150px;
        justify-content: center;
        color: #5b21b6;
    }
    [data-testid="stRadio"] label:hover {
        color: #7c3aed;
        background: rgba(124, 58, 237, 0.05);
    }
    /* Make radio circle larger and styled */
    [data-testid="stRadio"] input[type="radio"] {
        width: 20px;
        height: 20px;
        cursor: pointer;
        accent-color: #7c3aed;
    }
    /* Entire label area is clickable */
    [data-testid="stRadio"] label > div {
        pointer-events: none;
    }
    
    /* Enhanced info/error boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Initialize session state ----------
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "target_format" not in st.session_state:
    st.session_state.target_format = "png"
if "uploaded_format" not in st.session_state:
    st.session_state.uploaded_format = None
if "original_colorized" not in st.session_state:
    st.session_state.original_colorized = None
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = not load_welcome_status()

# Auto-save history when it changes
if "last_history_len" not in st.session_state:
    st.session_state.last_history_len = len(st.session_state.history)
elif st.session_state.last_history_len != len(st.session_state.history):
    save_history(st.session_state.history)
    st.session_state.last_history_len = len(st.session_state.history)

# Initialize theme
if "colorizer_theme" not in st.session_state:
    st.session_state.colorizer_theme = "Light"

# Initialize active tab - persist across refreshes by loading from disk
if "active_tab" not in st.session_state:
    st.session_state.active_tab = load_active_tab()

# ==================== WELCOME PAGE ====================
if st.session_state.show_welcome:
    st.markdown("""
    <style>
    .welcome-container {
        text-align: center;
        padding: 0rem 2rem;
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #4a148c 0%, #7c3aed 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: slideDown 1s ease-out;
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .welcome-subtitle {
        font-size: 1.5rem;
        color: #5b21b6;
        font-weight: 500;
        margin-bottom: 1rem;
        animation: slideUp 1s ease-out 0.3s both;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 1rem auto;
        max-width: 1000px;
        animation: fadeIn 1s ease-out 0.6s both;
    }
    .feature-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(240,240,255,0.9) 100%);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.15);
        transition: all 0.4s ease;
        border: 2px solid rgba(124, 58, 237, 0.2);
    }
    .feature-card:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 15px 35px rgba(124, 58, 237, 0.3);
        border-color: #7c3aed;
    }
    .feature-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4a148c;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 1rem;
        color: #5b21b6;
        line-height: 1.6;
    }
    .get-started-btn {
        display: inline-block;
        margin-top: 0rem;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
    st.markdown('<div class="welcome-title">🎨 Colour & Convert</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-subtitle">Bring Your Old Photos Back to Life with Advanced Colorization</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🎨</div>
            <div class="feature-title">Smart Colorization</div>
            <div class="feature-desc">Transform black & white photos into vibrant colored images using advanced colorization technology</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <div class="feature-title">Format Converter</div>
            <div class="feature-desc">Convert images between PNG, JPEG, BMP, and WebP formats with quality controls</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Processing</div>
            <div class="feature-desc">Get results in under a second with optimized processing algorithms</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📂</div>
            <div class="feature-title">History Gallery</div>
            <div class="feature-desc">Keep track of your colorized images with automatic saving and history management</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get Started Button
    st.markdown("""
    <style>
    div[data-testid="column"]:has(button) button {
        font-size: 2.5rem !important;
        padding: 2rem 4rem !important;
        font-weight: 800 !important;
        height: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True, type="primary"):
            st.session_state.show_welcome = False
            save_welcome_status(True)
            st.session_state.active_tab = "🎨 Colorizer"
            save_active_tab("🎨 Colorizer")
            st.rerun()
    
    st.stop()

# ---------- Header ----------
st.markdown('<div class="app-title" style="text-align: center; margin: 0; padding: 0.5rem 0; line-height: 1.2; color: #2d0a5e;">🎨 Colour & Convert</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #4a148c; font-size: 1.2rem; font-weight: 500; margin: 0.2rem 0 1rem 0; padding: 0;">Colorize photos and convert image formats - all in one place.</p>', unsafe_allow_html=True)

# ---------- Main Tabs (Colorizer first) ----------
# Create tabs with persistence
tab_names = ["🎨 Colorizer", "🔄 Format Converter", "📂 History", "🎨 Themes"]

# Custom tab selector with persistence
selected_tab = st.radio(
    "Navigation",
    tab_names,
    index=tab_names.index(st.session_state.active_tab) if st.session_state.active_tab in tab_names else 0,
    horizontal=True,
    label_visibility="collapsed",
    key="tab_selector"
)

# Update active tab and save
if selected_tab != st.session_state.active_tab:
    st.session_state.active_tab = selected_tab
    save_active_tab(selected_tab)
    st.rerun()

# Apply theme CSS globally
theme = st.session_state.colorizer_theme
css = ""
if theme == "Light":
    css = """
    <style>
        .stApp {background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);} 
        .app-title {
            color: #2d0a5e !important;
        }
        .stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#7c3aed,#8b5cf6); color: white; border: 0;}
        body {color: #1f2937;}
        .stMarkdown {color: #1f2937;}
        div[role="radiogroup"] label {
            color: #4a148c !important;
            background: rgba(255,255,255,0.3) !important;
        }
        div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,0.6) !important;
            transform: scale(1.05) !important;
            color: #3b0764 !important;
        }
        div[role="radiogroup"] label:has(input[type="radio"]:checked) {
            color: #ffffff !important;
            background: linear-gradient(135deg, #4a148c 0%, #6b21a8 100%) !important;
            box-shadow: 0 8px 16px rgba(74, 20, 140, 0.4) !important;
            transform: scale(1.08) !important;
            border-bottom: 4px solid #3b0764 !important;
        }
    </style>
    """
elif theme == "Dark":
    css = """
    <style>
        .stApp {background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;} 
        .app-title {color: #a78bfa !important;}
        .stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#8b5cf6,#7c3aed); color: white; border: 0;}
        body {color: #f1f5f9 !important;}
        .stMarkdown {color: #f1f5f9 !important;}
        [data-testid="stHeader"] {background-color: #1e293b;}
        div[role="radiogroup"] label {
            color: #cbd5e1 !important;
            background: rgba(30, 41, 59, 0.5) !important;
        }
        div[role="radiogroup"] label:hover {
            background: rgba(30, 41, 59, 0.8) !important;
            transform: scale(1.05) !important;
            color: #e0e7ff !important;
        }
        div[role="radiogroup"] label:has(input[type="radio"]:checked) {
            color: #ffffff !important;
            background: linear-gradient(135deg, #6b21a8 0%, #7c3aed 100%) !important;
            box-shadow: 0 8px 16px rgba(139, 92, 246, 0.4) !important;
            transform: scale(1.08) !important;
            border-bottom: 4px solid #8b5cf6 !important;
        }
        p, span, div, label, h1, h2, h3, h4, h5, h6 {color: #f1f5f9 !important;}
        .stRadio label {color: #f1f5f9 !important;}
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.95) 100%) !important;
            border: 2px solid rgba(168, 85, 247, 0.5) !important;
        }
        [data-testid="stMetricLabel"] {color: #c4b5fd !important; font-weight: 700 !important; font-size: 1.1rem !important;}
        [data-testid="stMetricValue"] {color: #ffffff !important; font-weight: 800 !important; font-size: 2.2rem !important;}
        .stCaption {color: #cbd5e1 !important;}
        [data-testid="stFileUploader"] label {color: #f1f5f9 !important;}
        [data-testid="stFileUploader"] {color: #f1f5f9 !important; border-color: #a855f7 !important;}
        [data-testid="stFileUploader"] section {background-color: #1e293b !important;}
        input, textarea, select {background-color: #334155 !important; color: #f1f5f9 !important; border-color: #475569 !important;}
        [data-baseweb="select"] {background-color: #334155 !important;}
        [data-baseweb="select"] div {color: #f1f5f9 !important;}
    </style>
    """

if css:
    st.markdown(css, unsafe_allow_html=True)

# ==================== COLORIZER TAB ====================
if selected_tab == "🎨 Colorizer":
    render_colorizer_tab()

# ==================== CONVERTER TAB ====================
elif selected_tab == "🔄 Format Converter":
    render_converter_tab()

# ==================== HISTORY TAB ====================
elif selected_tab == "📂 History":
    st.subheader("Processing History")
    
    # Initialize confirmation state
    if "confirm_clear_history" not in st.session_state:
        st.session_state.confirm_clear_history = False
    
    # Clear history button
    col_header1, col_header2 = st.columns([3, 1])
    with col_header2:
        if st.session_state.history:
            if st.button("🗑️ Clear History"):
                st.session_state.confirm_clear_history = True
    
    # Show confirmation dialog as popup
    if st.session_state.confirm_clear_history:
        @st.dialog("Confirm Action")
        def confirm_clear_dialog():
            st.write("Are you sure you want to clear all history?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Clear All", type="primary", use_container_width=True):
                    st.session_state.history = []
                    save_history([])
                    st.session_state.confirm_clear_history = False
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_clear_history = False
                    st.rerun()
        
        confirm_clear_dialog()
    
    if not st.session_state.history:
        st.info("No results yet. Process an image to see your gallery here.")
    else:
        st.caption(f"Showing {len(st.session_state.history)} most recent items (max 10)")
        cols = st.columns(3)
        def _friendly_ratio_from_dims(w: int, h: int) -> str:
            """Return a friendly string like '16:9' or '4:3' for common aspect ratios."""
            if w == 0 or h == 0:
                return "N/A"
            ratio = w / h
            common = {16/9: "16:9", 4/3: "4:3", 1: "1:1", 3/2: "3:2"}
            for r, label in common.items():
                if abs(ratio - r) < 0.05:
                    return label
            return f"{w}:{h}"

        for idx, item in enumerate(st.session_state.history):
            with cols[idx % 3]:
                st.image(item["image"], use_column_width=True)
                when_str = item.get("when", "Unknown")
                st.caption(f"**{item['type']}** • {when_str}")
                
                # Handle size - it can be a tuple or a string
                size_info = item.get("size", (0, 0))
                if isinstance(size_info, tuple):
                    w, h = size_info
                else:
                    # Parse string like "1920×1080"
                    try:
                        parts = str(size_info).split('×')
                        w, h = int(parts[0]), int(parts[1])
                    except:
                        w, h = 0, 0
                
                aspect = _friendly_ratio_from_dims(w, h)
                col_info = item.get("colorfulness", "N/A")
                st.caption(f"📐 {w}x{h} ({aspect}) • 🎨 {col_info}")
                
                if st.button(f"💾 Download", key=f"dl_{idx}"):
                    full_img = item.get("full")
                    if full_img is not None:
                        dlbytes = to_download_bytes(full_img)
                        st.download_button(
                            "⬇️ Save Image",
                            data=dlbytes,
                            file_name=f"{item['type']}_{when_str}.png",
                            mime="image/png",
                            key=f"real_dl_{idx}"
                        )

# ==================== THEMES TAB ====================
elif selected_tab == "🎨 Themes":
    st.markdown('<h2 style="text-align: center; font-size: 2rem; font-weight: 700; color: #5b21b6; margin-bottom: 2rem;">🌓 Choose Your Theme</h2>', unsafe_allow_html=True)
    
    # Create two columns for theme selection
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Custom styled theme selector
        st.markdown("""
        <style>
        .theme-container {
            display: flex;
            gap: 2rem;
            justify-content: center;
            margin: 2rem 0;
        }
        .theme-card {
            flex: 1;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 3px solid transparent;
        }
        .theme-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        }
        .light-theme-card {
            background: linear-gradient(135deg, #ffffff 0%, #f0f0ff 100%);
            border-color: #e0e0e0;
        }
        .light-theme-card:hover {
            border-color: #7c3aed;
        }
        .dark-theme-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-color: #334155;
            color: white;
        }
        .dark-theme-card:hover {
            border-color: #8b5cf6;
        }
        .theme-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        .theme-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .theme-desc {
            font-size: 1rem;
            opacity: 0.8;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<p style="text-align: center; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">Select Mode</p>', unsafe_allow_html=True)
        
        col_light, col_dark = st.columns(2)
        
        with col_light:
            if st.button("☀️ Light Mode", key="light_btn", use_container_width=True, 
                        type="primary" if st.session_state.colorizer_theme == "Light" else "secondary"):
                st.session_state.colorizer_theme = "Light"
                st.rerun()
        
        with col_dark:
            if st.button("🌙 Dark Mode", key="dark_btn", use_container_width=True,
                        type="primary" if st.session_state.colorizer_theme == "Dark" else "secondary"):
                st.session_state.colorizer_theme = "Dark"
                st.rerun()
        
        theme_option = st.session_state.colorizer_theme
        
        # Display theme preview
        st.markdown("---")
        if theme_option == "Light":
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #ffffff 0%, #f7f9fc 100%); border-radius: 15px; border: 2px solid #7c3aed;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">☀️</div>
                <h3 style="color: #5b21b6; font-size: 1.5rem; font-weight: 700;">Light Mode Active</h3>
                <p style="color: #7c3aed; font-size: 1.1rem;">Bright and clear interface for comfortable viewing</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 15px; border: 2px solid #8b5cf6;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🌙</div>
                <h3 style="color: #c4b5fd; font-size: 1.5rem; font-weight: 700;">Dark Mode Active</h3>
                <p style="color: #a78bfa; font-size: 1.1rem;">Easy on the eyes for low-light environments</p>
            </div>
            """, unsafe_allow_html=True)
