# Image Processing Suite

A comprehensive Streamlit application that combines image colorization and format conversion functionalities.

## Features

### 🎨 B&W Colorizer
- Transform black & white images into vibrant color photos
- Uses deep learning colorization model 
- Processing history with gallery view
- Performance metrics (processing time, colorfulness score)

### 🔄 Format Converter
- Convert between image formats: JPG, PNG, BMP, PDF
- PDF to Image conversion (single or all pages)
- Image to PDF conversion
- Adjustable JPEG quality and image dimensions
- Side-by-side preview of original and converted files
- Batch conversion support for multi-page PDFs


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage

Run the application using either:

**Method 1 - Using run_app.py:**
```bash
python run_app.py
```

**Method 2 - Direct Streamlit:**
```bash
streamlit run app.py
```

Then open your browser to the URL shown (typically http://localhost:8501)

## Project Structure

```
merged_image_app/
├── app.py                 # Main Streamlit application
├── run_app.py            # Convenience launcher
├── requirements.txt      # Python dependencies
├── models/               # Colorization model files
└── converted/            # Output folder for converted files
```

## Technologies Used

- **Streamlit** - Web interface framework
- **OpenCV** - Image processing and colorization
- **PyMuPDF (fitz)** - PDF processing
- **Pillow** - Image manipulation
- **img2pdf** - Image to PDF conversion
- **NumPy** - Numerical operations

## Notes

- The colorization model works best on grayscale or black & white images
- Supported image formats: PNG, JPG, JPEG, BMP, WEBP
- PDF conversion supports multi-page documents
- All processing is done locally - no data is sent to external servers
