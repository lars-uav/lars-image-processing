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
import hashlib

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

def compute_file_hash(file_content):
    """Compute a hash of the file content to detect duplicates"""
    return hashlib.md5(file_content).hexdigest()

def get_stored_images():
    """Retrieve list of stored images from MongoDB"""
    try:
        client = init_connection()
        if not client:
            return []
            
        db = client.rgnir_analyzer
        # Retrieve images sorted by timestamp, most recent first
        images = list(db.images.find({}, {
            'metadata': 1, 
            '_id': 1
        }).sort('metadata.upload_date', -1))
        
        return images
    except Exception as e:
        st.error(f"Failed to retrieve images: {str(e)}")
        return []

def save_image_to_db(uploaded_file, timestamp):
    """Save image and metadata to MongoDB, preventing duplicates"""
    try:
        # Check file size
        file_content = uploaded_file.getvalue()
        file_size = len(file_content) / (1024 * 1024)  # Size in MB
        if file_size > 16:
            st.error(f"File size ({file_size:.1f}MB) exceeds MongoDB document limit (16MB). Please resize the image before uploading.")
            return None

        client = init_connection()
        if not client:
            return None
            
        db = client.rgnir_analyzer
        
        # Compute file hash to check for duplicates
        file_hash = compute_file_hash(file_content)
        
        # Check if image with same hash already exists
        existing_image = db.images.find_one({'metadata.file_hash': file_hash})
        if existing_image:
            st.warning(f"Image {uploaded_file.name} appears to be a duplicate and was not uploaded.")
            return None
        
        try:
            # Verify image can be opened
            img = Image.open(io.BytesIO(file_content))
        except Exception as e:
            st.error(f"Invalid image file: {str(e)}")
            return None
            
        # Create document
        document = {
            'metadata': {
                'filename': uploaded_file.name,
                'timestamp': timestamp,
                'upload_date': datetime.now(),
                'file_size_mb': file_size,
                'image_dimensions': img.size,
                'file_hash': file_hash  # Store hash to prevent duplicates
            },
            'image_data': Binary(file_content)  # Store as binary data
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
    
    # Limit to 20 most recent images to prevent overwhelming the page
    stored_images = stored_images[:20]
    
    # Create a dynamic number of columns based on image count
    num_cols = min(4, max(1, len(stored_images)))
    cols = st.columns(num_cols)
    
    for idx, doc in enumerate(stored_images):
        with cols[idx % num_cols]:
            # Load image data 
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
                    if st.button(f"Analyze_{str(doc['_id'])}"):
                        st.session_state.selected_image = str(doc['_id'])
                
                with button_cols[1]:
                    if st.button(f"Remove_{str(doc['_id'])}", type="secondary"):
                        if remove_image_from_db(str(doc['_id'])):
                            st.success("Image removed successfully")

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
    
    # Initialize stored images in session state if not exists
    if 'stored_images' not in st.session_state:
        st.session_state.stored_images = []
    
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
    
    # Refresh Database Button
    if st.button("Refresh Database"):
        st.session_state.stored_images = get_stored_images()
    
    # Display gallery
    st.header("Image Gallery")
    create_image_gallery(st.session_state.stored_images)
    
    # Database Management Expander
    with st.expander("Database Management"):
        if st.button("Clear All Images", type="secondary"):
            # Add a confirmation step
            if st.button("Confirm Delete All Images?", type="primary"):
                try:
                    client = init_connection()
                    if client:
                        db = client.rgnir_analyzer
                        db.images.delete_many({})
                        st.success("All images removed successfully")
                        st.session_state.stored_images = []
                except Exception as e:
                    st.error(f"Failed to clear database: {str(e)}")

if __name__ == "__main__":
    main()