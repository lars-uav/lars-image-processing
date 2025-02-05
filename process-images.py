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

def calculate_index(img_array, index_type):
    """Calculate various vegetation/water indices"""
    # Convert to float for calculations
    img_float = img_array.astype(float)
    
    # Extract bands (R, G, NIR)
    red = img_float[:,:,0]
    green = img_float[:,:,1]
    nir = img_float[:,:,2]
    
    # Avoid division by zero
    epsilon = 1e-10
    
    if index_type == "NDVI":
        # Normalized Difference Vegetation Index
        numerator = nir - red
        denominator = nir + red + epsilon
        index = numerator / denominator
    
    elif index_type == "GNDVI":
        # Green Normalized Difference Vegetation Index
        numerator = nir - green
        denominator = nir + green + epsilon
        index = numerator / denominator
    
    elif index_type == "NDWI":
        # Normalized Difference Water Index
        numerator = green - nir
        denominator = green + nir + epsilon
        index = numerator / denominator
    
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    return np.clip(index, -1, 1)

def create_index_visualization(index_array, index_type):
    """Create colorful visualization of index values"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Choose colormap based on index type
    if index_type == "NDWI":
        cmap = 'RdYlBu'  # Blue for water
    else:
        cmap = 'RdYlGn'  # Green for vegetation
    
    im = ax.imshow(index_array, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, label=index_type)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def analyze_index(index_array, index_type):
    """Calculate statistics for the given index"""
    threshold = 0.2  # Default threshold for vegetation/water detection
    
    if index_type == "NDWI":
        feature_name = "Water"
        threshold = 0.0  # Different threshold for water detection
    else:
        feature_name = "Vegetation"
    
    stats = {
        f'Mean {index_type}': float(np.mean(index_array)),
        f'Median {index_type}': float(np.median(index_array)),
        f'Min {index_type}': float(np.min(index_array)),
        f'Max {index_type}': float(np.max(index_array)),
        f'{feature_name} Coverage (%)': float(np.mean(index_array > threshold) * 100)
    }
    return stats

def create_image_gallery(images_dict):
    """Create a gallery view of uploaded images"""
    cols = st.columns(4)  # Show 4 images per row
    for idx, (name, img) in enumerate(images_dict.items()):
        with cols[idx % 4]:
            st.image(img, caption=name, use_container_width=True)
            if st.button(f"Analyze {name}", key=f"btn_{name}"):
                st.session_state.selected_image = name

def main():
    st.set_page_config(layout="wide", page_title="LARS Image Analyzer")
    
    st.title("LARS Image Analyzer")
    
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
                img_array = np.array(img)
                # Apply white balance correction immediately upon loading
                corrected_array = fix_white_balance(img_array)
                st.session_state.processed_images[uploaded_file.name] = {
                    'original': img,
                    'array': img_array,
                    'corrected_array': corrected_array,
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
            corrected_array = img_data['corrected_array']
            
            # Index selection
            selected_indices = st.multiselect(
                "Select Indices to Display",
                ["NDVI", "GNDVI", "NDWI"],
                default=[]
            )
            
            # Display images
            if selected_indices:
                # Calculate number of columns needed (original + white balance + selected indices)
                num_cols = 2 + len(selected_indices)
                cols = st.columns(num_cols)
                
                # Display original and white balance
                with cols[0]:
                    st.subheader("Original")
                    st.image(img_data['original'], use_container_width=True)
                
                with cols[1]:
                    st.subheader("White Balance Corrected")
                    st.image(Image.fromarray(corrected_array), use_container_width=True)
                
                # Display selected indices (calculated on white-balanced image)
                for idx, index_type in enumerate(selected_indices, 2):
                    with cols[idx]:
                        st.subheader(index_type)
                        # Calculate index on white-balanced image
                        index_array = calculate_index(corrected_array, index_type)
                        index_viz = create_index_visualization(index_array, index_type)
                        st.image(index_viz, use_container_width=True)
                        
                        # Display statistics for each index
                        st.write(f"{index_type} Statistics")
                        stats = analyze_index(index_array, index_type)
                        for key, value in stats.items():
                            st.metric(key, f"{value:.3f}")
            else:
                # Show only original and white balance when no indices are selected
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(img_data['original'], use_container_width=True)
                
                with col2:
                    st.subheader("White Balance Corrected")
                    st.image(Image.fromarray(corrected_array), use_container_width=True)
    
    # Instructions when no file is uploaded
    else:
        st.info("""
        Upload RGNir images to:
        - View them in a gallery format
        - Analyze individual images
        - Calculate multiple vegetation and water indices (NDVI, GNDVI, NDWI)
        - Get detailed statistics for each index
        
        Note: All indices are calculated on the white-balanced image for better accuracy.
        """)
        
        st.markdown("""
        ### Available Indices:
        - NDVI (Normalized Difference Vegetation Index)
        - GNDVI (Green Normalized Difference Vegetation Index)
        - NDWI (Normalized Difference Water Index)
        
        ### Supported file formats:
        - TIFF (.tif, .tiff)
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        """)

if __name__ == "__main__":
    main()