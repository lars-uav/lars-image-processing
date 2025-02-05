import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from datetime import datetime
import json
from pymongo import MongoClient
from bson.binary import Binary
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB setup
@st.cache_resource
def init_connection():
    """Initialize MongoDB connection"""
    try:
        # For local development, use .env file
        mongodb_uri = os.getenv("MONGODB_URI")
        # For Streamlit Cloud, use secrets
        if not mongodb_uri and hasattr(st.secrets, "MONGODB_URI"):
            mongodb_uri = st.secrets.MONGODB_URI
            
        if not mongodb_uri:
            raise ValueError("MongoDB URI not found in environment or secrets")
            
        client = MongoClient(mongodb_uri)
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

@st.cache_data
def get_stored_images():
    """Retrieve list of stored images from MongoDB"""
    try:
        client = init_connection()
        if not client:
            return []
            
        db = client.rgnir_analyzer
        return list(db.images.find({}, {'metadata': 1}).sort('metadata.timestamp', -1))
    except Exception as e:
        st.error(f"Failed to retrieve images: {str(e)}")
        return []

def save_image_to_db(uploaded_file, timestamp):
    """Save image and metadata to MongoDB"""
    try:
        # Check file size
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
        if file_size > 16:
            st.error(f"File size ({file_size:.1f}MB) exceeds MongoDB document limit (16MB). Please resize the image before uploading.")
            return None

        client = init_connection()
        if not client:
            return None
            
        db = client.rgnir_analyzer
        
        try:
            # Verify image can be opened
            img = Image.open(uploaded_file)
            # Reset file pointer
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Invalid image file: {str(e)}")
            return None
            
        # Prepare image data
        img_bytes = uploaded_file.getvalue()
        
        # Create document
        document = {
            'metadata': {
                'filename': uploaded_file.name,
                'timestamp': timestamp,
                'upload_date': datetime.now(),
                'file_size_mb': file_size,
                'image_dimensions': img.size
            },
            'image_data': Binary(img_bytes)  # Store as binary data
        }
        
        # Insert into MongoDB
        try:
            result = db.images.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            if "document too large" in str(e).lower():
                st.error(f"File size ({file_size:.1f}MB) is too large for MongoDB. Please resize the image.")
            else:
                st.error(f"Database error: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Failed to process image: {str(e)}")
        return None

def remove_image_from_db(image_id):
    """Remove image from MongoDB"""
    try:
        client = init_connection()
        if not client:
            return False
            
        db = client.rgnir_analyzer
        result = db.images.delete_one({'_id': ObjectId(image_id)})
        return result.deleted_count > 0
    except Exception as e:
        st.error(f"Failed to remove image: {str(e)}")
        return False

def load_image_from_db(image_id):
    """Load image data from MongoDB"""
    try:
        client = init_connection()
        if not client:
            return None
            
        db = client.rgnir_analyzer
        document = db.images.find_one({'_id': ObjectId(image_id)})
        
        if document:
            img_bytes = document['image_data']
            img = Image.open(io.BytesIO(img_bytes))
            return {
                'original': img,
                'array': np.array(img),
                'metadata': document['metadata']
            }
        return None
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        return None
    
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

def create_image_gallery(stored_images):
    """Create a gallery view of saved images"""
    if not stored_images:
        st.info("No images found in the database")
        return
        
    cols = st.columns(4)
    displayed_image_ids = set()  # Track displayed image IDs to prevent duplicates
    
    for idx, doc in enumerate(stored_images):
        # Skip if this image has already been displayed
        if doc['_id'] in displayed_image_ids:
            continue
        
        with cols[idx % 4]:
            # Load image data only when needed
            image_data = load_image_from_db(doc['_id'])
            if image_data:
                st.image(
                    image_data['original'],
                    caption=image_data['metadata']['filename'],
                    use_container_width=True
                )
                st.caption(f"Uploaded: {image_data['metadata']['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Create columns for Analyze and Remove buttons
                button_cols = st.columns(2)
                with button_cols[0]:
                    if st.button(f"Analyze", key=f"btn_analyze_{doc['_id']}"):
                        st.session_state.selected_image = str(doc['_id'])
                with button_cols[1]:
                    if st.button("Remove", key=f"btn_remove_{doc['_id']}", type="secondary"):
                        if remove_image_from_db(str(doc['_id'])):
                            st.success("Image removed successfully")
                            # Clear cache to refresh image list
                            get_stored_images.clear()
                            # Reset selected image if it was the one removed
                            if st.session_state.selected_image == str(doc['_id']):
                                st.session_state.selected_image = None
                            # Trigger rerun to refresh the page
                            st.rerun()
                        else:
                            st.error("Failed to remove image")
                
                # Add the image ID to displayed images to prevent duplicates
                displayed_image_ids.add(doc['_id'])
def main():
    st.set_page_config(layout="wide", page_title="RGNir Image Analyzer")
    st.title("RGNir Image Analyzer")
    
    # Initialize MongoDB connection
    if not init_connection():
        st.error("Failed to connect to database. Please check your connection settings.")
        return
    
    # Initialize session state
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    
    # Add expander for database management
    with st.expander("Database Management"):
        if st.button("Clear All Images", type="secondary"):
            if st.button("Confirm Delete All Images?", type="primary"):
                try:
                    client = init_connection()
                    if client:
                        db = client.rgnir_analyzer
                        db.images.delete_many({})
                        st.success("All images removed successfully")
                        get_stored_images.clear()
                        st.session_state.selected_image = None
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear database: {str(e)}")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload RGNir images", 
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing uploaded images..."):
            for uploaded_file in uploaded_files:
                timestamp = datetime.now()
                if save_image_to_db(uploaded_file, timestamp):
                    st.success(f"Successfully uploaded {uploaded_file.name}")
                else:
                    st.error(f"Failed to upload {uploaded_file.name}")
        # Clear cache to refresh image list
        get_stored_images.clear()
        st.rerun()
    
    # Display gallery
    st.header("Image Gallery")
    stored_images = get_stored_images()
    create_image_gallery(stored_images)
    
    # Display analysis for selected image
    if st.session_state.selected_image:
        image_data = load_image_from_db(st.session_state.selected_image)
        if image_data:
            st.header(f"Analysis: {image_data['metadata']['filename']}")
            
            # Process image for analysis
            corrected_array = fix_white_balance(image_data['array'])
            
            # Index selection
            selected_indices = st.multiselect(
                "Select Indices to Display",
                ["NDVI", "GNDVI", "NDWI"],
                default=[]
            )
            
            # Display images
            if selected_indices:
                num_cols = 2 + len(selected_indices)
                cols = st.columns(num_cols)
                
                with cols[0]:
                    st.subheader("Original")
                    st.image(image_data['original'], use_container_width=True)
                
                with cols[1]:
                    st.subheader("White Balance Corrected")
                    st.image(Image.fromarray(corrected_array), use_container_width=True)
                
                for idx, index_type in enumerate(selected_indices, 2):
                    with cols[idx]:
                        st.subheader(index_type)
                        index_array = calculate_index(corrected_array, index_type)
                        index_viz = create_index_visualization(index_array, index_type)
                        st.image(index_viz, use_container_width=True)
                        
                        st.write(f"{index_type} Statistics")
                        stats = analyze_index(index_array, index_type)
                        for key, value in stats.items():
                            st.metric(key, f"{value:.3f}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(image_data['original'], use_container_width=True)
                with col2:
                    st.subheader("White Balance Corrected")
                    st.image(Image.fromarray(corrected_array), use_container_width=True)

if __name__ == "__main__":
    main()