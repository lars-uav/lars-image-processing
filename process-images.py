import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

def fix_white_balance(img_array):
    """Apply white balance correction to RGNir image"""
    # Convert to float for processing
    img_float = img_array.astype(float)
    
    corrected = np.zeros_like(img_float)
    
    # Process each channel
    for i in range(3):
        channel = img_float[:,:,i]
        p2, p98 = np.percentile(channel, (2, 98))
        corrected[:,:,i] = np.clip((channel - p2) / (p98 - p2) * 255, 0, 255)
    
    return corrected.astype(np.uint8)

def calculate_ndvi(img_array):
    """Calculate NDVI from RGNir image"""
    # Extract NIR (3rd channel) and Red (1st channel)
    nir = img_array[:,:,2].astype(float)
    red = img_array[:,:,0].astype(float)
    
    # Avoid division by zero
    denominator = nir + red
    ndvi = np.zeros_like(red)
    valid = denominator != 0
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    
    return ndvi

def create_ndvi_visualization(ndvi_array):
    """Create colorful visualization of NDVI values"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, label='NDVI')
    ax.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def analyze_ndvi(ndvi_array):
    """Calculate NDVI statistics"""
    stats = {
        'Mean NDVI': float(np.mean(ndvi_array)),
        'Median NDVI': float(np.median(ndvi_array)),
        'Min NDVI': float(np.min(ndvi_array)),
        'Max NDVI': float(np.max(ndvi_array)),
        'Vegetation Coverage (%)': float(np.mean(ndvi_array > 0.2) * 100)
    }
    return stats

def main():
    st.set_page_config(layout="wide", page_title="RGNir Image Analyzer")
    
    st.title("RGNir Image Analyzer")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload RGNir images", 
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Create columns for image display
        for uploaded_file in uploaded_files:
            st.header(uploaded_file.name)
            
            # Load and process image
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            
            # Apply corrections and calculate NDVI
            corrected_array = fix_white_balance(img_array)
            ndvi_array = calculate_ndvi(img_array)
            ndvi_viz = create_ndvi_visualization(ndvi_array)
            
            # Display images side by side
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Original")
                st.image(img, use_column_width=True)
            
            with col2:
                st.subheader("White Balance Corrected")
                st.image(Image.fromarray(corrected_array), use_column_width=True)
            
            with col3:
                st.subheader("NDVI")
                st.image(ndvi_viz, use_column_width=True)
            
            # Display statistics
            st.subheader("NDVI Statistics")
            stats = analyze_ndvi(ndvi_array)
            
            # Create 2 columns for stats
            stat_col1, stat_col2 = st.columns(2)
            
            # Display stats in metric boxes
            for i, (key, value) in enumerate(stats.items()):
                if i % 2 == 0:
                    with stat_col1:
                        st.metric(key, f"{value:.3f}")
                else:
                    with stat_col2:
                        st.metric(key, f"{value:.3f}")
            
            st.divider()

    # Instructions when no file is uploaded
    else:
        st.info("""
        Upload RGNir images to:
        - View the original image
        - See white balance corrected version
        - Calculate and visualize NDVI
        - Get vegetation statistics
        """)
        
        st.markdown("""
        ### Supported file formats:
        - TIFF (.tif, .tiff)
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        """)

if __name__ == "__main__":
    main()