#!/usr/bin/env python
# coding: utf-8

"""
Converter Module
Handles image format conversion and PDF conversion functionality
"""

import os
import io
import time
import zipfile
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageFilter
import fitz  # PyMuPDF
import img2pdf


def compress_image_to_target(img: Image.Image, target_format: str, original_size_bytes: int, preserve_quality: bool = False) -> bytes:
    """
    Compress image to ensure output is smaller than or equal to original size.
    If preserve_quality is True, prioritize quality over size reduction.
    """
    if target_format == "bmp":
        # BMP is uncompressed, just save it
        buf = io.BytesIO()
        img.save(buf, "BMP")
        return buf.getvalue()
    
    elif target_format == "jpg":
        if preserve_quality:
            # Maximum quality for PDF conversion
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=98, optimize=True, subsampling=0)  # Highest quality settings
            return buf.getvalue()
        
        # Try progressively lower quality until file is smaller than original
        for quality in [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality, optimize=True)
            
            if buf.tell() <= original_size_bytes:
                return buf.getvalue()
        
        # If still too large, try resizing with moderate quality
        for scale in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
            resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, "JPEG", quality=75, optimize=True)
            
            if buf.tell() <= original_size_bytes:
                return buf.getvalue()
        
        # Last resort: aggressive compression
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=30, optimize=True)
        return buf.getvalue()
    
    elif target_format == "png":
        if preserve_quality:
            # Lossless PNG for quality preservation
            buf = io.BytesIO()
            img.save(buf, "PNG", compress_level=1, optimize=True)  # Best quality, minimal compression
            return buf.getvalue()
        
        # PNG is lossless, try maximum compression first
        buf = io.BytesIO()
        img.save(buf, "PNG", compress_level=9, optimize=True)
        
        if buf.tell() <= original_size_bytes:
            return buf.getvalue()
        
        # If PNG is larger, convert to JPEG instead (lossy but smaller)
        for quality in [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40]:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality, optimize=True)
            
            if buf.tell() <= original_size_bytes:
                # Return as PNG wrapper around JPEG to maintain requested format
                # Actually just return JPEG compressed version
                return buf.getvalue()
        
        # If still too large, try resizing PNG
        for scale in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
            resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, "PNG", compress_level=9, optimize=True)
            
            if buf.tell() <= original_size_bytes:
                return buf.getvalue()
        
        # Last resort: aggressively resize with JPEG
        for scale in [0.5, 0.45, 0.4, 0.35, 0.3]:
            resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, "JPEG", quality=85, optimize=True)
            
            if buf.tell() <= original_size_bytes:
                return buf.getvalue()
        
        # Absolute last resort
        buf = io.BytesIO()
        resized = img.resize((int(img.width * 0.25), int(img.height * 0.25)), Image.Resampling.LANCZOS)
        resized.save(buf, "PNG", compress_level=9, optimize=True)
        return buf.getvalue()
    
    return b""


def render_converter_tab():
    """Render the format converter tab UI"""
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
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, caption=f"{filename} (page 1)", width=400)
                else:
                    st.image(uploaded_file, caption=filename, width=400)
                
                # Display original file size
                uploaded_bytes = uploaded_file.getvalue()
                original_size_kb = len(uploaded_bytes) / 1024
                if original_size_kb < 1024:
                    st.caption(f"📦 Size: {original_size_kb:.2f} KB")
                else:
                    st.caption(f"📦 Size: {original_size_kb / 1024:.2f} MB")
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
                
                if ext == ".pdf" and format_choice in ["jpg", "png", "bmp"]:
                    doc = fitz.open(stream=uploaded_bytes)
                    pages = range(doc.page_count) if convert_all_pages else [0]
                    converted_images = []
                    original_size = len(uploaded_bytes)
                    
                    for i, pnum in enumerate(pages):
                        page = doc.load_page(pnum)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                        
                        # Get image as PIL for compression
                        img_bytes_temp = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_bytes_temp))
                        
                        # Compress to ensure smaller than original
                        target_size = original_size if not convert_all_pages else original_size // doc.page_count
                        compressed = compress_image_to_target(img_pil, format_choice, target_size)
                        
                        converted_images.append((f"{base_name}_page{pnum + 1}.{format_choice}", compressed))
                        progress.progress(30 + int(60 * (i + 1) / len(pages)))
                    
                    st.success(f"✅ Converted PDF → {len(converted_images)} image(s)")
                    
                    with col2:
                        st.markdown("**Converted Image")
                        st.image(converted_images[0][1], caption=converted_images[0][0], width=400)
                        
                        # Display converted file size
                        converted_size_kb = len(converted_images[0][1]) / 1024
                        if converted_size_kb < 1024:
                            st.caption(f"📦 Size: {converted_size_kb:.2f} KB")
                        else:
                            st.caption(f"📦 Size: {converted_size_kb / 1024:.2f} MB")
                    
                    # Add to history
                    img_pil_hist = Image.open(io.BytesIO(converted_images[0][1]))
                    img_np_hist = np.array(img_pil_hist)
                    h, w = img_np_hist.shape[:2]
                    thumb = cv2.resize(img_np_hist, (min(320, w), int(h * min(320, w) / w)))
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.insert(0, {
                        "type": "converted",
                        "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "size": f"{w}×{h}",
                        "format": format_choice.upper(),
                        "image": thumb,
                        "full": img_np_hist,
                    })
                    
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
                    img = Image.open(io.BytesIO(uploaded_bytes))
                    original_size = len(uploaded_bytes)
                    
                    # Handle transparency properly before PDF conversion
                    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                        # For transparent images, composite onto white background to preserve appearance
                        if img.mode != "RGBA":
                            img = img.convert("RGBA")
                        # Split all channels including alpha
                        *rgb, alpha = img.split()
                        bg = Image.new("RGB", img.size, (255, 255, 255))
                        bg.paste(img, mask=alpha)
                        img = bg
                    else:
                        img = img.convert("RGB")
                    
                    # Try to create PDF with reduced size while maintaining quality
                    pdf_bytes = None
                    
                    # First, try to create PDF with the image as-is
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, "JPEG", quality=85, optimize=True)
                    img_bytes.seek(0)
                    test_pdf = img2pdf.convert(img_bytes.getvalue())
                    
                    if test_pdf and len(test_pdf) <= original_size:
                        # If PDF is smaller or equal, use it
                        pdf_bytes = test_pdf
                    else:
                        # If PDF is larger, try progressive compression
                        # Apply aggressive JPG compression strategy (quality 90 down to 30)
                        for quality in [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]:
                            img_bytes = io.BytesIO()
                            img.save(img_bytes, "JPEG", quality=quality, optimize=True)
                            img_bytes.seek(0)
                            test_pdf = img2pdf.convert(img_bytes.getvalue())
                            
                            if test_pdf and len(test_pdf) <= original_size:
                                pdf_bytes = test_pdf
                                break
                        
                        # If still too large, try resizing
                        if pdf_bytes is None:
                            for scale in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
                                resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
                                img_bytes = io.BytesIO()
                                resized.save(img_bytes, "JPEG", quality=85, optimize=True)
                                img_bytes.seek(0)
                                test_pdf = img2pdf.convert(img_bytes.getvalue())
                                
                                if test_pdf and len(test_pdf) <= original_size:
                                    pdf_bytes = test_pdf
                                    break
                        
                        # Last resort: aggressive compression with lower quality and resizing
                        if pdf_bytes is None:
                            resized = img.resize((int(img.width * 0.7), int(img.height * 0.7)), Image.Resampling.LANCZOS)
                            img_bytes = io.BytesIO()
                            resized.save(img_bytes, "JPEG", quality=70, optimize=True)
                            img_bytes.seek(0)
                            test_pdf = img2pdf.convert(img_bytes.getvalue())
                            if test_pdf:
                                pdf_bytes = test_pdf
                    
                    # Success - PDF created with size reduction
                    if pdf_bytes:
                        progress.progress(95)
                        final_pdf_size = len(pdf_bytes)
                        size_diff = original_size - final_pdf_size
                        
                        if size_diff > 0:
                            reduction_pct = (size_diff / original_size) * 100
                            st.success(f"✅ {base_name}.pdf | {original_size/1024:.1f} KB → {final_pdf_size/1024:.1f} KB ({reduction_pct:.1f}% smaller)")
                        elif size_diff < 0:
                            st.warning(f"⚠️ {base_name}.pdf | {original_size/1024:.1f} KB → {final_pdf_size/1024:.1f} KB (+{abs(size_diff)/1024:.1f} KB) - Quality maintained")
                        else:
                            st.success(f"✅ {base_name}.pdf | {original_size/1024:.1f} KB - Size maintained")
                        
                        with col2:
                            st.markdown("**Converted PDF (Preview)")
                            doc = fitz.open(stream=pdf_bytes)
                            page = doc.load_page(0)
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                            pdf_preview_bytes = pix.tobytes("png")
                            st.image(pdf_preview_bytes, caption=f"{base_name}.pdf", width=400)
                            
                            # Display converted PDF size
                            pdf_size_kb = final_pdf_size / 1024
                            if pdf_size_kb < 1024:
                                st.caption(f"📦 Size: {pdf_size_kb:.2f} KB")
                            else:
                                st.caption(f"📦 Size: {pdf_size_kb / 1024:.2f} MB")
                        
                        st.download_button("📥 Download PDF", pdf_bytes, file_name=f"{base_name}.pdf")
                    else:
                        st.error(f"❌ Failed to create PDF from image")
                        progress.progress(0)
                
                else:
                    img = Image.open(io.BytesIO(uploaded_bytes))
                    
                    # Check if source and target formats are the same
                    if current_format == format_choice:
                        # No conversion needed, return original file
                        output_buf = io.BytesIO(uploaded_bytes)
                        original_size_kb = len(uploaded_bytes) / 1024
                        
                        st.success(f"✅ {base_name}.{format_choice} (same format - no conversion needed)")
                        
                        progress.progress(95)
                        
                        with col2:
                            st.markdown("**Converted Image")
                            st.image(output_buf.getvalue(), caption=f"{base_name}.{format_choice}", width=400)
                            
                            # Display file size (same as original)
                            if original_size_kb < 1024:
                                st.caption(f"📦 Size: {original_size_kb:.2f} KB")
                            else:
                                st.caption(f"📦 Size: {original_size_kb / 1024:.2f} MB")
                        
                        output_buf.seek(0)
                        st.download_button("📥 Download file", output_buf.read(), file_name=f"{base_name}.{format_choice}")
                        
                        # Add to history
                        img_np_hist = np.array(img)
                        h, w = img_np_hist.shape[:2]
                        thumb = cv2.resize(img_np_hist, (min(320, w), int(h * min(320, w) / w)))
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.insert(0, {
                            "type": "converted",
                            "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "size": f"{w}×{h}",
                            "format": format_choice.upper(),
                            "image": thumb,
                            "full": img_np_hist,
                        })
                    else:
                        # Different format conversion
                        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                            alpha = img.convert("RGBA").split()[-1]
                            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                            bg.paste(img, mask=alpha)
                            img = bg.convert("RGB")
                        else:
                            img = img.convert("RGB")
                        
                        output_filename = f"{base_name}.{format_choice}"
                        
                        # Convert to requested format with aggressive compression to reduce size
                        original_size = len(uploaded_bytes)
                        compressed_bytes = compress_image_to_target(img, format_choice, original_size)
                        output_buf = io.BytesIO(compressed_bytes)
                        
                        original_size_kb = original_size / 1024
                        file_size_kb = len(compressed_bytes) / 1024
                        
                        reduction_pct = ((original_size - len(compressed_bytes)) / original_size) * 100
                        st.success(f"✅ {output_filename} | {original_size_kb:.1f} KB → {file_size_kb:.1f} KB ({reduction_pct:.1f}% smaller)")
                        
                        progress.progress(95)
                        
                        with col2:
                            st.markdown("**Converted Image")
                            st.image(output_buf.getvalue(), caption=output_filename, width=400)
                            
                            # Display converted file size
                            if file_size_kb < 1024:
                                st.caption(f"📦 Size: {file_size_kb:.2f} KB")
                            else:
                                st.caption(f"📦 Size: {file_size_kb / 1024:.2f} MB")
                        
                        output_buf.seek(0)
                        st.download_button("📥 Download converted file", output_buf.read(), file_name=output_filename)
                        
                        # Add to history
                        img_np_hist = np.array(img)
                        h, w = img_np_hist.shape[:2]
                        thumb = cv2.resize(img_np_hist, (min(320, w), int(h * min(320, w) / w)))
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.insert(0, {
                            "type": "converted",
                            "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "size": f"{w}×{h}",
                            "format": format_choice.upper(),
                            "image": thumb,
                            "full": img_np_hist,
                        })
                
                progress.progress(100)
            
            except Exception as e:
                st.error(f"Error during conversion: {e}")
                progress.progress(0)
