import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime

def fix_white_balance(img_array):
    """Apply white balance correction to RGNir image"""
    img_float = img_array.astype(float)
    corrected = np.zeros_like(img_float)
    
    for i in range(3):
        channel = img_float[:,:,i]
        p2, p98 = np.percentile(channel, (2, 98))
        corrected[:,:,i] = np.clip((channel - p2) / (p98 - p2) * 255, 0, 255)
    
    return corrected.astype(np.uint8)

def calculate_ndvi(img_array):
    """Calculate NDVI from RGNir image"""
    nir = img_array[:,:,2].astype(float)
    red = img_array[:,:,0].astype(float)
    
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

def create_image_gallery(images_dict):
    """Create a gallery view of uploaded images"""
    cols = st.columns(4)  # Show 4 images per row
    for idx, (name, img) in enumerate(images_dict.items()):
        with cols[idx % 4]:
            st.image(img, caption=name, use_column_width=True)
            if st.button(f"Analyze {name}", key=f"btn_{name}"):
                st.session_state.selected_image = name

def main():
    st.set_page_config(layout="wide", page_title="RGNir Image Analyzer")
    
    st.title("RGNir Image Analyzer")
    
    # Initialize session state
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = {}
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload RGNir images", 
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    # Process and store uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_images:
                img = Image.open(uploaded_file)
                st.session_state.processed_images[uploaded_file.name] = {
                    'original': img,
                    'array': np.array(img),
                    'timestamp': datetime.now()
                }
    
    # Display gallery if there are images
    if st.session_state.processed_images:
        st.header("Image Gallery")
        create_image_gallery({name: data['original'] 
                            for name, data in st.session_state.processed_images.items()})
        
        # Display analysis for selected image
        if st.session_state.selected_image:
            st.header(f"Analysis: {st.session_state.selected_image}")
            
            # Get image data
            img_data = st.session_state.processed_images[st.session_state.selected_image]
            img_array = img_data['array']
            
            # Process image
            corrected_array = fix_white_balance(img_array)
            ndvi_array = calculate_ndvi(img_array)
            ndvi_viz = create_ndvi_visualization(ndvi_array)
            
            # NDVI toggle
            show_ndvi = st.toggle("Show NDVI Visualization", value=False)
            
            # Display images based on toggle
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original")
                st.image(img_data['original'], use_column_width=True)
            
            with col2:
                if show_ndvi:
                    st.subheader("NDVI")
                    st.image(ndvi_viz, use_column_width=True)
                else:
                    st.subheader("White Balance Corrected")
                    st.image(Image.fromarray(corrected_array), use_column_width=True)
            
            # Display statistics
            if show_ndvi:
                st.subheader("NDVI Statistics")
                stats = analyze_ndvi(ndvi_array)
                
                # Create columns for stats
                stat_cols = st.columns(len(stats))
                
                # Display stats in metric boxes
                for col, (key, value) in zip(stat_cols, stats.items()):
                    with col:
                        st.metric(key, f"{value:.3f}")
    
    # Instructions when no file is uploaded
    else:
        st.info("""
        Upload RGNir images to:
        - View them in a gallery format
        - Analyze individual images
        - Toggle between white balance correction and NDVI visualization
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