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
import pandas as pd
import cv2
from skimage.registration import phase_cross_correlation
from scipy import ndimage

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

def remove_duplicate_images():
    """Remove duplicate images from the database"""
    try:
        client = init_connection()
        if not client:
            return 0
            
        db = client.rgnir_analyzer
        
        # Group images by their file hash
        pipeline = [
            {"$group": {
                "_id": "$metadata.file_hash",
                "count": {"$sum": 1},
                "ids": {"$push": "$_id"}
            }},
            {"$match": {"count": {"$gt": 1}}}
        ]
        
        duplicates = list(db.images.aggregate(pipeline))
        
        # Remove duplicate images, keeping the first one
        total_removed = 0
        for duplicate_group in duplicates:
            # Keep the first ID, remove the rest
            ids_to_remove = duplicate_group['ids'][1:]
            result = db.images.delete_many({"_id": {"$in": ids_to_remove}})
            total_removed += result.deleted_count
        
        return total_removed
    
    except Exception as e:
        st.error(f"Failed to remove duplicate images: {str(e)}")
        return 0

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

def save_image_to_db(uploaded_file):
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
            st.warning(f"Image {uploaded_file.name} is a duplicate and was not uploaded.")
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
                st.caption(f"Uploaded: {image_data['metadata']['upload_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Create columns for Analyze and Remove buttons
                button_cols = st.columns(2)
                with button_cols[0]:
                    if st.button(f"Analyze_{str(doc['_id'])}"):
                        st.session_state.selected_image = str(doc['_id'])
                
                with button_cols[1]:
                    if st.button(f"Remove_{str(doc['_id'])}", type="secondary"):
                        if remove_image_from_db(str(doc['_id'])):
                            st.success("Image removed successfully")
                            # Update stored_images in session state before rerun
                            st.session_state.stored_images = get_stored_images()
                            st.experimental_rerun()

def download_processed_images(image_data, corrected_array, selected_indices):
    """
    Create a zip file of processed images for download.
    
    Args:
        image_data (dict): Original image data from database
        corrected_array (numpy.ndarray): White-balanced image array
        selected_indices (list): List of selected vegetation/water indices
    
    Returns:
        bytes: Zip file content containing processed images
    """
    import zipfile
    import io
    
    # Create a bytes IO object for the zip file
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save white balance corrected image
        corrected_img = Image.fromarray(corrected_array)
        corrected_buffer = io.BytesIO()
        corrected_img.save(corrected_buffer, format='PNG')
        zipf.writestr('white_balanced.png', corrected_buffer.getvalue())
        
        # Save index visualizations
        for index_type in selected_indices:
            index_array = calculate_index(corrected_array, index_type)
            index_viz = create_index_visualization(index_array, index_type)
            index_buffer = io.BytesIO()
            index_viz.save(index_buffer, format='PNG')
            zipf.writestr(f'{index_type}_visualization.png', index_buffer.getvalue())
    
    # Reset buffer position
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_comparison_view(image_data_list, index_type=None):
    """Create a side-by-side comparison view of multiple images
    
    Args:
        image_data_list (list): List of image data dictionaries
        index_type (str, optional): Type of index to compare (NDVI, GNDVI, NDWI)
        
    Returns:
        tuple: Combined figure and statistics dictionary
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    
    n_images = len(image_data_list)
    if n_images == 0:
        return None, {}
        
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 4))
    if n_images == 1:
        axes = [axes]
    
    all_stats = {}
    
    for idx, (ax, image_data) in enumerate(zip(axes, image_data_list)):
        # Get image data
        if index_type:
            # Calculate vegetation/water index
            corrected_array = fix_white_balance(image_data['array'])
            index_array = calculate_index(corrected_array, index_type)
            
            # Create visualization
            if index_type == "NDWI":
                cmap = 'RdYlBu'
            else:
                cmap = 'RdYlGn'
            
            im = ax.imshow(index_array, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label=index_type)
            
            # Calculate statistics
            stats = analyze_index(index_array, index_type)
            all_stats[image_data['metadata']['filename']] = stats
        else:
            # Show original or white-balanced image
            corrected_array = fix_white_balance(image_data['array'])
            ax.imshow(corrected_array)
        
        # Set title with filename
        ax.set_title(image_data['metadata']['filename'], fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    buf.seek(0)
    comparison_image = Image.open(buf)
    
    return comparison_image, all_stats

# Time Series Monitoring Functions
def create_monitoring_site(site_name, description=None, coordinates=None):
    """Create a new monitoring site in the database"""
    try:
        client = init_connection()
        if not client:
            return None
            
        db = client.rgnir_analyzer
        
        # Check if site already exists
        existing_site = db.monitoring_sites.find_one({'name': site_name})
        if existing_site:
            st.warning(f"Site '{site_name}' already exists.")
            return str(existing_site['_id'])
        
        # Create site document
        site_document = {
            'name': site_name,
            'description': description,
            'coordinates': coordinates,
            'created_date': datetime.now(),
            'last_updated': datetime.now()
        }
        
        result = db.monitoring_sites.insert_one(site_document)
        return str(result.inserted_id)
    
    except Exception as e:
        st.error(f"Failed to create monitoring site: {str(e)}")
        return None

def get_all_monitoring_sites():
    """Retrieve all monitoring sites from the database"""
    try:
        client = init_connection()
        if not client:
            return []
            
        db = client.rgnir_analyzer
        sites = list(db.monitoring_sites.find().sort('name', 1))
        return sites
    
    except Exception as e:
        st.error(f"Failed to retrieve monitoring sites: {str(e)}")
        return []

def assign_image_to_site(image_id, site_id):
    """Associate an image with a monitoring site"""
    try:
        client = init_connection()
        if not client:
            return False
            
        db = client.rgnir_analyzer
        
        # Update the image document to include site reference
        result = db.images.update_one(
            {'_id': ObjectId(image_id)},
            {'$set': {
                'metadata.site_id': site_id,
                'metadata.assigned_to_site_date': datetime.now()
            }}
        )
        
        # Update the site's last_updated field
        db.monitoring_sites.update_one(
            {'_id': ObjectId(site_id)},
            {'$set': {'last_updated': datetime.now()}}
        )
        
        return result.modified_count > 0
    
    except Exception as e:
        st.error(f"Failed to assign image to site: {str(e)}")
        return False

def get_site_images(site_id):
    """Retrieve all images associated with a specific monitoring site"""
    try:
        client = init_connection()
        if not client:
            return []
            
        db = client.rgnir_analyzer
        images = list(db.images.find(
            {'metadata.site_id': site_id},
            {'metadata': 1, '_id': 1}
        ).sort('metadata.upload_date', 1))  # Sort by date, oldest first
        
        return images
    
    except Exception as e:
        st.error(f"Failed to retrieve site images: {str(e)}")
        return []

def align_images(fixed_img, moving_img):
    """
    Align moving image to fixed image using phase correlation
    
    Args:
        fixed_img (numpy.ndarray): Reference image
        moving_img (numpy.ndarray): Image to be aligned
        
    Returns:
        numpy.ndarray: Aligned version of moving_img
    """
    # Convert to grayscale for registration
    if fixed_img.ndim == 3:
        fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY)
    else:
        fixed_gray = fixed_img
        
    if moving_img.ndim == 3:
        moving_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
    else:
        moving_gray = moving_img
    
    # Calculate shift using phase correlation
    shift, error, diffphase = phase_cross_correlation(fixed_gray, moving_gray)
    
    # Apply shift to moving image
    aligned_img = ndimage.shift(moving_img, shift, order=3, mode='reflect')
    
    return aligned_img, shift

def create_time_series_plot(image_data_list, index_type):
    """
    Create a time series plot of a vegetation/water index for multiple dates
    
    Args:
        image_data_list (list): List of image data dictionaries, sorted by date
        index_type (str): Type of index to plot (NDVI, GNDVI, NDWI)
        
    Returns:
        PIL.Image: Plot image
    """
    # Extract dates and calculate index values
    dates = []
    mean_values = []
    max_values = []
    min_values = []
    
    # Process each image
    for img_data in image_data_list:
        # Get date from metadata
        date = img_data['metadata']['upload_date']
        dates.append(date)
        
        # Calculate index
        corrected_array = fix_white_balance(img_data['array'])
        index_array = calculate_index(corrected_array, index_type)
        
        # Calculate statistics
        mean_values.append(float(np.mean(index_array)))
        max_values.append(float(np.max(index_array)))
        min_values.append(float(np.min(index_array)))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean values with error bars
    ax.errorbar(dates, mean_values, 
                yerr=[np.array(mean_values) - np.array(min_values), 
                      np.array(max_values) - np.array(mean_values)],
                fmt='o-', capsize=5, label=f'Mean {index_type}')
    
    # Add horizontal line at threshold
    threshold = 0.2 if index_type != "NDWI" else 0.0
    ax.axhline(y=threshold, color='r', linestyle='--', 
               label=f'{"Vegetation" if index_type != "NDWI" else "Water"} Threshold')
    
    # Format plot
    ax.set_title(f'{index_type} Time Series')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{index_type} Value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Create image from plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def create_change_detection_visualization(image_pair, index_type):
    """
    Create visualization showing changes between two dates
    
    Args:
        image_pair (tuple): Tuple of (earlier_image_data, later_image_data)
        index_type (str): Type of index to compare (NDVI, GNDVI, NDWI)
        
    Returns:
        PIL.Image: Change detection visualization
    """
    early_img_data, late_img_data = image_pair
    
    # Calculate indices for both images
    early_corrected = fix_white_balance(early_img_data['array'])
    late_corrected = fix_white_balance(late_img_data['array'])
    
    # Try to align images first
    aligned_late, shift = align_images(early_corrected, late_corrected)
    
    # Calculate indices
    early_index = calculate_index(early_corrected, index_type)
    late_index = calculate_index(aligned_late, index_type)
    
    # Calculate difference
    diff = late_index - early_index
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Choose colormap based on index type
    if index_type == "NDWI":
        cmap = 'RdYlBu'
    else:
        cmap = 'RdYlGn'
    
    # Plot early image
    im1 = axes[0].imshow(early_index, cmap=cmap, vmin=-1, vmax=1)
    axes[0].set_title(f"Early: {early_img_data['metadata']['upload_date'].strftime('%Y-%m-%d')}")
    plt.colorbar(im1, ax=axes[0], label=index_type)
    axes[0].axis('off')
    
    # Plot late image
    im2 = axes[1].imshow(late_index, cmap=cmap, vmin=-1, vmax=1)
    axes[1].set_title(f"Late: {late_img_data['metadata']['upload_date'].strftime('%Y-%m-%d')}")
    plt.colorbar(im2, ax=axes[1], label=index_type)
    axes[1].axis('off')
    
    # Plot difference
    im3 = axes[2].imshow(diff, cmap='bwr', vmin=-0.5, vmax=0.5)
    axes[2].set_title(f"Change in {index_type}")
    plt.colorbar(im3, ax=axes[2], label=f'Δ{index_type}')
    axes[2].axis('off')
    
    # Layout and save
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def calculate_index_statistics_by_timeframe(image_data_list, index_type):
    """
    Calculate index statistics for each image in the time series
    
    Args:
        image_data_list (list): List of image data dictionaries, sorted by date
        index_type (str): Type of index to analyze (NDVI, GNDVI, NDWI)
        
    Returns:
        pandas.DataFrame: DataFrame with index statistics for each date
    """
    results = []
    
    # Process each image
    for img_data in image_data_list:
        # Get date from metadata
        date = img_data['metadata']['upload_date']
        
        # Calculate index
        corrected_array = fix_white_balance(img_data['array'])
        index_array = calculate_index(corrected_array, index_type)
        
        # Calculate statistics
        threshold = 0.2 if index_type != "NDWI" else 0.0
        feature_name = "Water" if index_type == "NDWI" else "Vegetation"
        
        stats = {
            'Date': date,
            'Mean': float(np.mean(index_array)),
            'Median': float(np.median(index_array)),
            'Min': float(np.min(index_array)),
            'Max': float(np.max(index_array)),
            f'{feature_name} Coverage (%)': float(np.mean(index_array > threshold) * 100)
        }
        
        results.append(stats)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df

# UI function for time series analysis
def time_series_analysis_ui():
    """UI for time series analysis of monitoring sites"""
    st.header("Time Series Monitoring")
    
    # Side-by-side layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Monitoring Sites")
        
        # Create new site form
        with st.expander("Create New Monitoring Site"):
            site_name = st.text_input("Site Name")
            site_desc = st.text_area("Description (optional)")
            
            # Optional coordinates
            include_coords = st.checkbox("Include Coordinates")
            lat, lng = None, None
            if include_coords:
                col_lat, col_lng = st.columns(2)
                with col_lat:
                    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
                with col_lng:
                    lng = st.number_input("Longitude", min_value=-180.0, max_value=180.0, format="%.6f")
                
            if st.button("Create Site"):
                if not site_name:
                    st.error("Site name is required")
                else:
                    coordinates = {"lat": lat, "lng": lng} if include_coords else None
                    site_id = create_monitoring_site(site_name, site_desc, coordinates)
                    if site_id:
                        st.success(f"Site '{site_name}' created successfully!")
                        # Force refresh the site list
                        st.session_state.monitoring_sites = get_all_monitoring_sites()
        
        # Display available sites
        monitoring_sites = get_all_monitoring_sites()
        if not monitoring_sites:
            st.info("No monitoring sites found. Create one to get started.")
        else:
            selected_site = st.selectbox(
                "Select Monitoring Site",
                options=monitoring_sites,
                format_func=lambda x: x['name'],
                key="site_selector"
            )
            
            if selected_site:
                st.write(f"**Description:** {selected_site.get('description', 'No description')}")
                if 'coordinates' in selected_site and selected_site['coordinates']:
                    st.write(f"**Coordinates:** Lat: {selected_site['coordinates']['lat']}, Lng: {selected_site['coordinates']['lng']}")
                
                st.write(f"**Created:** {selected_site['created_date'].strftime('%Y-%m-%d')}")
                st.write(f"**Last Updated:** {selected_site['last_updated'].strftime('%Y-%m-%d')}")
    
    # If a site is selected, show the image assignment panel and time series visualization
    if 'site_selector' in st.session_state and st.session_state.site_selector:
        selected_site = st.session_state.site_selector
        
        with col2:
            st.subheader(f"Time Series for {selected_site['name']}")
            
            # Get images already assigned to this site
            site_images = get_site_images(str(selected_site['_id']))
            
            # Show option to assign new images to site
            with st.expander("Assign Images to Site"):
                # Get all images not already assigned to this site
                unassigned_images = [img for img in st.session_state.stored_images 
                                    if not any(site_img['_id'] == img['_id'] for site_img in site_images)]
                
                if not unassigned_images:
                    st.info("No unassigned images available.")
                else:
                    # Create multiselect for unassigned images
                    image_options = {str(img['_id']): img['metadata']['filename'] for img in unassigned_images}
                    
                    selected_images = st.multiselect(
                        "Select Images to Assign",
                        options=list(image_options.keys()),
                        format_func=lambda x: image_options[x]
                    )
                    
                    if selected_images and st.button("Assign to Site"):
                        with st.spinner("Assigning images..."):
                            success_count = 0
                            for img_id in selected_images:
                                if assign_image_to_site(img_id, str(selected_site['_id'])):
                                    success_count += 1
                            
                            if success_count > 0:
                                st.success(f"Successfully assigned {success_count} images to {selected_site['name']}")
                                # Refresh site images
                                site_images = get_site_images(str(selected_site['_id']))
            
            # Display time series analysis for site images
            if not site_images:
                st.info(f"No images assigned to {selected_site['name']} yet. Assign images to begin analysis.")
            else:
                # Load image data for all site images
                image_data_list = []
                with st.spinner("Loading site images..."):
                    for img in site_images:
                        img_data = load_image_from_db(img['_id'])
                        if img_data:
                            image_data_list.append(img_data)
                
                # Select index for analysis
                index_type = st.selectbox(
                    "Select Index for Analysis",
                    ["NDVI", "GNDVI", "NDWI"],
                    key="ts_index_selector"
                )
                
                # Check if we have enough images for analysis
                if len(image_data_list) < 2:
                    st.warning("Need at least 2 images for time series analysis.")
                else:
                    # Time series plot
                    st.subheader(f"{index_type} Time Series")
                    ts_plot = create_time_series_plot(image_data_list, index_type)
                    st.image(ts_plot, use_container_width=True)
                    
                    # Statistics table
                    st.subheader("Statistics Over Time")
                    stats_df = calculate_index_statistics_by_timeframe(image_data_list, index_type)
                    st.dataframe(stats_df)
                    
                    # Change detection - compare first and last images
                    if len(image_data_list) >= 2:
                        st.subheader("Change Detection")
                        first_img = image_data_list[0]
                        last_img = image_data_list[-1]
                        
                        change_viz = create_change_detection_visualization(
                            (first_img, last_img), 
                            index_type
                        )
                        st.image(change_viz, use_container_width=True)
                        
                        # Add download button for change detection report
                        first_date = first_img['metadata']['upload_date'].strftime('%Y%m%d')
                        last_date = last_img['metadata']['upload_date'].strftime('%Y%m%d')
                        
                        # Create in-memory buffer for change report
                        report_buffer = io.BytesIO()
                        change_viz.save(report_buffer, format='PNG')
                        report_buffer.seek(0)
                        
                        st.download_button(
                            label=f"Download Change Report ({first_date} to {last_date})",
                            data=report_buffer,
                            file_name=f"change_report_{index_type}_{first_date}_to_{last_date}.png",
                            mime="image/png"
                        )

def main():
    """Main function with time series monitoring enhancement"""
    st.set_page_config(layout="wide", page_title="RGNir Image Analyzer")
    
    # Create tabs for different functionality
    tab1, tab2 = st.tabs(["Image Analysis", "Time Series Monitoring"])
    
    with tab1:
        st.title("RGNir Image Analyzer")
        
        # Initialize MongoDB connection
        if not init_connection():
            st.error("Failed to connect to database. Please check your connection settings.")
            return
        
        # Initialize session state
        if 'selected_images' not in st.session_state:
            st.session_state.selected_images = []
        if 'stored_images' not in st.session_state:
            st.session_state.stored_images = get_stored_images()
        if 'monitoring_sites' not in st.session_state:
            st.session_state.monitoring_sites = get_all_monitoring_sites()
        
        # File uploader section
        uploaded_files = st.file_uploader(
            "Upload RGNir images", 
            type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="file_uploader_unique"
        )
        
        # Process uploaded files
        if uploaded_files:
            uploaded_hashes = set()
            with st.spinner("Processing uploaded images..."):
                for uploaded_file in uploaded_files:
                    file_hash = compute_file_hash(uploaded_file.getvalue())
                    if file_hash in uploaded_hashes:
                        st.warning(f"Skipping duplicate image: {uploaded_file.name}")
                        continue
                    uploaded_hashes.add(file_hash)
                    if save_image_to_db(uploaded_file):
                        st.success(f"Successfully uploaded {uploaded_file.name}")
                        st.session_state.stored_images = get_stored_images()
        
        # Database Management section
        with st.expander("Database Management"):
            st.write("Database Maintenance Tools")
            
            if st.button("Remove Duplicate Images"):
                removed_count = remove_duplicate_images()
                if removed_count > 0:
                    st.success(f"Removed {removed_count} duplicate images")
                    st.session_state.stored_images = get_stored_images()
                else:
                    st.info("No duplicate images found")
            
            if st.button("Clear All Images", type="secondary"):
                if st.button("Confirm Delete All Images?", type="primary"):
                    try:
                        client = init_connection()
                        if client:
                            db = client.rgnir_analyzer
                            db.images.delete_many({})
                            st.success("All images removed successfully")
                            st.session_state.stored_images = get_stored_images()
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to clear database: {str(e)}")
        
        # Refresh Database Button
        if st.button("Refresh Database", key="refresh_button_unique"):
            st.session_state.stored_images = get_stored_images()
        
        # Gallery Section with Multi-select
        st.header("Image Gallery")
        if not st.session_state.stored_images:
            st.info("No images found in the database")
        else:
            # Create columns for gallery
            num_cols = min(4, max(1, len(st.session_state.stored_images)))
            cols = st.columns(num_cols)
            
            for idx, doc in enumerate(st.session_state.stored_images[:20]):
                with cols[idx % num_cols]:
                    image_data = load_image_from_db(doc['_id'])
                    if image_data:
                        st.image(
                            image_data['original'],
                            caption=image_data['metadata']['filename'],
                            use_container_width=True
                        )
                        st.caption(f"Uploaded: {image_data['metadata']['upload_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Add checkbox for selection
                        if st.checkbox(f"Select for comparison {str(doc['_id'])}", 
                                     value=str(doc['_id']) in st.session_state.selected_images):
                            if str(doc['_id']) not in st.session_state.selected_images:
                                st.session_state.selected_images.append(str(doc['_id']))
                        else:
                            if str(doc['_id']) in st.session_state.selected_images:
                                st.session_state.selected_images.remove(str(doc['_id']))
                        
                        # Remove button
                        if st.button(f"Remove_{str(doc['_id'])}", type="secondary"):
                            if remove_image_from_db(str(doc['_id'])):
                                st.success("Image removed successfully")
                                if str(doc['_id']) in st.session_state.selected_images:
                                    st.session_state.selected_images.remove(str(doc['_id']))
                                st.session_state.stored_images = get_stored_images()
                                st.experimental_rerun()
        
        # Comparison Analysis Section
        if st.session_state.selected_images:
            st.header("Image Comparison Analysis")
            
            # Load all selected images
            image_data_list = [load_image_from_db(img_id) for img_id in st.session_state.selected_images]
            image_data_list = [img for img in image_data_list if img is not None]
            
            if not image_data_list:
                st.warning("No valid images selected for comparison")
                return
                
            # Show original images comparison
            st.subheader("Original Images")
            comparison_image, _ = create_comparison_view(image_data_list)
            if comparison_image:
                st.image(comparison_image, use_container_width=True)
            
            # Show white-balanced comparison
            st.subheader("White Balance Corrected")
            for img_data in image_data_list:
                img_data['array'] = fix_white_balance(img_data['array'])
            comparison_image, _ = create_comparison_view(image_data_list)
            if comparison_image:
                st.image(comparison_image, use_container_width=True)
            
            # Index selection and comparison
            selected_indices = st.multiselect(
                "Select Indices to Compare",
                ["NDVI", "GNDVI", "NDWI"],
                default=[]
            )
            
            # Process selected indices
            for index_type in selected_indices:
                st.subheader(f"{index_type} Comparison")
                comparison_image, stats = create_comparison_view(image_data_list, index_type)
                if comparison_image:
                    st.image(comparison_image, use_container_width=True)
                
                # Display statistics in columns
                if stats:
                    st.write(f"{index_type} Statistics")
                    cols = st.columns(len(stats))
                    for idx, (filename, img_stats) in enumerate(stats.items()):
                        with cols[idx]:
                            st.write(f"**{filename}**")
                            for key, value in img_stats.items():
                                st.metric(key, f"{value:.3f}")
            
            # Download option
            if selected_indices:
                st.download_button(
                    label="Download Comparison Report",
                    data=download_processed_images(image_data_list[0], image_data_list[0]['array'], selected_indices),
                    file_name="comparison_report.zip",
                    mime="application/zip"
                )
    
    with tab2:
        # Initialize MongoDB connection if not already done
        if not init_connection():
            st.error("Failed to connect to database. Please check your connection settings.")
            return
            
        # Make sure session state is initialized
        if 'stored_images' not in st.session_state:
            st.session_state.stored_images = get_stored_images()
        if 'monitoring_sites' not in st.session_state:
            st.session_state.monitoring_sites = get_all_monitoring_sites()
            
        # Display time series UI
        time_series_analysis_ui()

if __name__ == "__main__":
    main()