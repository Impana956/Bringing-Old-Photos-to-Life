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

# History file path
HISTORY_FILE = os.path.join(os.path.dirname(__file__), ".app_history.pkl")

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
    .stApp {background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);}    
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

# ---------- Initialize session state ----------
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "target_format" not in st.session_state:
    st.session_state.target_format = "png"
if "uploaded_format" not in st.session_state:
    st.session_state.uploaded_format = None
if "original_colorized" not in st.session_state:
    st.session_state.original_colorized = None

# Auto-save history when it changes
if "last_history_len" not in st.session_state:
    st.session_state.last_history_len = len(st.session_state.history)
elif st.session_state.last_history_len != len(st.session_state.history):
    save_history(st.session_state.history)
    st.session_state.last_history_len = len(st.session_state.history)

# ---------- Header ----------
st.markdown('<div class="app-title" style="text-align: center;">🎨 Colour & Convert</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; margin-bottom: 2rem;">Colorize black & white photos and convert image formats - all in one place.</p>', unsafe_allow_html=True)

# Initialize theme
if "colorizer_theme" not in st.session_state:
    st.session_state.colorizer_theme = "Theme1"

# ---------- Main Tabs (including Themes tab) ----------
tab_themes, tab_colorizer, tab_converter, tab_history = st.tabs(["🎨 Themes", "🎨 Colorizer", "🔄 Format Converter", "📂 History"])

# Themes tab - shows all 3 theme options
with tab_themes:
    theme_option = st.radio(
        "Select Theme",
        ["Theme1", "Theme2", "Theme3"],
        index=["Theme1", "Theme2", "Theme3"].index(st.session_state.colorizer_theme),
        key="theme_selector"
    )
    st.session_state.colorizer_theme = theme_option

# Apply theme CSS globally
theme = st.session_state.colorizer_theme
css = ""
if theme == "Theme1":
    css = """
    <style>
        .stApp {background: linear-gradient(160deg, #ede9fe 0%, #f5f3ff 40%, #faf5ff 100%);} 
        .app-title {background: linear-gradient(90deg,#6d28d9,#8b5cf6,#a78bfa); -webkit-background-clip:text; background-clip:text; color: transparent;}
        .stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#7c3aed,#8b5cf6); color: white; border: 0;}
    </style>
    """
elif theme == "Theme2":
    css = """
    <style>
        .stApp {background: linear-gradient(145deg, #f0f9ff 0%, #fde68a 35%, #fbcfe8 100%);} 
        .app-title {background: linear-gradient(90deg,#f97316,#ef4444,#a855f7,#06b6d4); -webkit-background-clip:text; background-clip:text; color: transparent;}
        .stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#f97316,#ef4444); color: white; border: 0;}
    </style>
    """
elif theme == "Theme3":
    css = """
    <style>
        .stApp {background: linear-gradient(160deg, #ffedd5 0%, #fecaca 45%, #e9d5ff 100%);} 
        .app-title {background: linear-gradient(90deg,#fb923c,#f43f5e,#8b5cf6); -webkit-background-clip:text; background-clip:text; color: transparent;}
        .stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#fb923c,#f43f5e); color: white; border: 0;}
    </style>
    """

if css:
    st.markdown(css, unsafe_allow_html=True)

# ==================== COLORIZER TAB ====================
with tab_colorizer:
    render_colorizer_tab()

# ==================== CONVERTER TAB ====================
with tab_converter:
    render_converter_tab()

# ==================== HISTORY TAB ====================
with tab_history:
    st.subheader("Processing History")
    
    # Clear history button
    col_header1, col_header2 = st.columns([3, 1])
    with col_header2:
        if st.session_state.history:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                save_history([])
    
    if not st.session_state.history:
        st.info("No results yet. Process an image to see your gallery here.")
    else:
        st.caption(f"Showing {len(st.session_state.history)} most recent items (max 10)")
        cols = st.columns(3)
        for idx, item in enumerate(st.session_state.history):
            with cols[idx % 3]:
                # Build caption with all available details
                caption_parts = [item['when'], item['size']]
                
                if item['type'] == 'colorized':
                    caption_parts.append("🎨 Colorized")
                    if item.get('colorfulness'):
                        caption_parts.append(f"Color: {item['colorfulness']:.1f}")
                elif item['type'] == 'converted':
                    caption_parts.append(f"🔄 → {item.get('format', 'N/A')}")
                
                caption_text = " • ".join(caption_parts)
                
                st.image(item["image"], caption=caption_text, use_column_width=True)
                b = to_download_bytes(item["full"], fmt="PNG")
                st.download_button(
                    label="📥 Download",
                    data=b,
                    file_name=f"processed_{idx+1}.png",
                    mime="image/png",
                    key=f"dl_hist_{idx}",
                )
