import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from datetime import datetime
from pymongo import MongoClient
from bson.binary import Binary
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv
import hashlib
import pandas as pd
from skimage.registration import phase_cross_correlation
from scipy import ndimage
from skimage.color import rgb2gray
import zipfile
import gc

# Load environment variables
load_dotenv()

# MongoDB setup
@st.cache_resource
def init_connection():
    """Initialize MongoDB connection with connection pooling and timeouts"""
    try:
        # For local development, use .env file
        mongodb_uri = os.getenv("MONGODB_URI")
        # For Streamlit Cloud, use secrets
        if not mongodb_uri and hasattr(st.secrets, "MONGODB_URI"):
            mongodb_uri = st.secrets.MONGODB_URI
            
        if not mongodb_uri:
            raise ValueError("MongoDB URI not found in environment or secrets")
            
        # Add connection pool settings to reduce connection overhead
        if '?' not in mongodb_uri:
            mongodb_uri += '?'
        else:
            mongodb_uri += '&'
        mongodb_uri += 'maxPoolSize=3&maxIdleTimeMS=30000'
            
        client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,  # 5 second timeout for server selection
            connectTimeoutMS=10000,         # 10 second timeout for initial connection
            socketTimeoutMS=30000           # 30 second timeout for operations
        )
        
        # Test connection
        client.admin.command('ping')
        
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

def get_stored_images(limit=12, skip=0, include_total=False):
    """Retrieve list of stored images from MongoDB with pagination
    
    Args:
        limit: Maximum number of images to retrieve
        skip: Number of images to skip (for pagination)
        include_total: Whether to return total count of images
    
    Returns:
        tuple: (list of images, total count) if include_total=True
               list of images otherwise
    """
    try:
        client = init_connection()
        if not client:
            return [] if not include_total else ([], 0)
            
        db = client.rgnir_analyzer
        
        # Get total count if needed
        total = None
        if include_total:
            total = db.images.count_documents({})
        
        # Only fetch metadata fields, not full image data
        projection = {
            'metadata.filename': 1, 
            'metadata.upload_date': 1,
            'metadata.image_dimensions': 1,
            'metadata.file_size_mb': 1,
            '_id': 1,
            # Explicitly exclude the binary data
            'image_data': 0
        }
        
        # Use cursor to avoid loading all results into memory at once
        cursor = db.images.find(
            {}, 
            projection
        ).sort('metadata.upload_date', -1).skip(skip).limit(limit)
        
        # Convert cursor to list (only converts the limited results)
        images = list(cursor)
        
        if include_total:
            return images, total
        return images
    except Exception as e:
        st.error(f"Failed to retrieve images: {str(e)}")
        return [] if not include_total else ([], 0)

def load_image_from_db(image_id, thumbnail=False):
    """Load image data from MongoDB with memory optimizations
    
    Args:
        image_id: The MongoDB ObjectId of the image
        thumbnail: If True, create a smaller version for gallery view
    """
    try:
        client = init_connection()
        if not client:
            return None
            
        db = client.rgnir_analyzer
        
        # First just retrieve metadata to check file size
        document = db.images.find_one(
            {'_id': ObjectId(image_id)},
            {'metadata': 1}  # Only get metadata first
        )
        
        if not document:
            return None
            
        # Now get the image data separately
        result = {
            'metadata': document['metadata'],
            'original': None,
            'array': None
        }
        
        # Get image data field
        img_doc = db.images.find_one(
            {'_id': ObjectId(image_id)},
            {'image_data': 1}
        )
        
        if img_doc and 'image_data' in img_doc:
            img_bytes = img_doc['image_data']
            img = Image.open(io.BytesIO(img_bytes))
            
            # Create thumbnail if requested (saves memory)
            if thumbnail:
                # Resize to smaller size for gallery view
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                result['original'] = img
            else:
                # Store original image and convert to array for analysis
                result['original'] = img
                result['array'] = np.array(img)
                
        return result
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        return None

def save_image_to_db(uploaded_file):
    """Save image with optimized storage and preprocessing"""
    try:
        # Check file size first
        file_content = uploaded_file.getvalue()
        file_size = len(file_content) / (1024 * 1024)  # Size in MB
        
        if file_size > 16:
            st.error(f"File size ({file_size:.1f}MB) exceeds MongoDB document limit (16MB). Please resize the image before uploading.")
            return None

        # Compute hash before connecting to DB
        file_hash = compute_file_hash(file_content)
            
        client = init_connection()
        if not client:
            return None
            
        db = client.rgnir_analyzer
        
        # Check if image with same hash already exists
        existing_image = db.images.find_one({'metadata.file_hash': file_hash}, {'_id': 1})
        if existing_image:
            st.warning(f"Image {uploaded_file.name} is a duplicate and was not uploaded.")
            return None
        
        try:
            # Open the image to validate and get properties
            img = Image.open(io.BytesIO(file_content))
            
            # For very large images, resize before storing
            max_dimension = 2048  # Max dimension for stored images
            width, height = img.size
            
            # Only resize if needed
            if max(width, height) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert back to bytes
                buffer = io.BytesIO()
                img.save(buffer, format=img.format or 'PNG')
                file_content = buffer.getvalue()
                file_size = len(file_content) / (1024 * 1024)
                
                # Generate new hash for resized image
                file_hash = compute_file_hash(file_content)
            
            # Create document with optimized structure
            document = {
                'metadata': {
                    'filename': uploaded_file.name,
                    'upload_date': datetime.now(),
                    'file_size_mb': file_size,
                    'image_dimensions': img.size,
                    'file_hash': file_hash
                },
                'image_data': Binary(file_content)
            }
            
            # Insert into MongoDB
            result = db.images.insert_one(document)
            return str(result.inserted_id)
                
        except Exception as e:
            st.error(f"Invalid image file: {str(e)}")
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
            {'metadata': 1, '_id': 1}  # Only metadata, not full images
        ).sort('metadata.upload_date', 1))  # Sort by date, oldest first
        
        return images
    
    except Exception as e:
        st.error(f"Failed to retrieve site images: {str(e)}")
        return []
    
def preprocess_large_image(img_array, max_dimension=1024):
    """Resize large images to reduce memory usage during analysis"""
    # Skip if image is already small enough
    if img_array is None or img_array.size == 0:
        return None
        
    h, w = img_array.shape[:2]
    
    # Skip if image is already small enough
    if max(h, w) <= max_dimension:
        return img_array
        
    # Calculate new dimensions
    if h > w:
        new_h = max_dimension
        new_w = int(w * (max_dimension / h))
    else:
        new_w = max_dimension
        new_h = int(h * (max_dimension / w))
    
    # Resize using PIL (more memory efficient than skimage)
    pil_img = Image.fromarray(img_array)
    resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(resized_img)
    
def fix_white_balance(img_array):
    """Apply white balance correction to RGNir image with memory optimization"""
    # None check to avoid errors
    if img_array is None or img_array.size == 0:
        return None
    
    # Convert to float32 instead of float64 to use half the memory
    img_float = img_array.astype(np.float32)
    corrected = np.zeros_like(img_float)
    
    # Process each channel separately
    for i in range(3):
        channel = img_float[:,:,i]
        p2, p98 = np.percentile(channel, (2, 98))
        corrected[:,:,i] = np.clip((channel - p2) / (p98 - p2) * 255, 0, 255)
    
    # Convert back to uint8 to save memory
    result = corrected.astype(np.uint8)
    
    # Clear temporary variables
    img_float = None
    corrected = None
    
    return result

def calculate_index(img_array, index_type):
    """Calculate various vegetation/water indices with memory optimization"""
    # None check
    if img_array is None or img_array.size == 0:
        return None
    
    # Convert to float32 instead of float64 to save memory
    img_float = img_array.astype(np.float32)
    
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
    
    # Clean up temporary arrays to free memory
    img_float = None
    
    return np.clip(index, -1, 1)

def analyze_index(index_array, index_type):
    """Calculate statistics for the given index"""
    # None check
    if index_array is None or index_array.size == 0:
        return {}
    
    threshold = 0.2  # Default threshold for vegetation detection
    
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

def align_images(fixed_img, moving_img):
    """
    Align moving image to fixed image using phase correlation
    
    Args:
        fixed_img (numpy.ndarray): Reference image
        moving_img (numpy.ndarray): Image to be aligned
        
    Returns:
        numpy.ndarray: Aligned version of moving_img
    """
    # None checks
    if fixed_img is None or moving_img is None:
        return moving_img, np.array([0, 0])
    
    # Resize large images before alignment to save memory
    max_dim = 1024
    if fixed_img.shape[0] > max_dim or fixed_img.shape[1] > max_dim:
        fixed_img = preprocess_large_image(fixed_img, max_dim)
    
    if moving_img.shape[0] > max_dim or moving_img.shape[1] > max_dim:
        moving_img = preprocess_large_image(moving_img, max_dim)
    
    # Convert to grayscale for registration
    if fixed_img.ndim == 3:
        fixed_gray = rgb2gray(fixed_img)
    else:
        fixed_gray = fixed_img
        
    if moving_img.ndim == 3:
        moving_gray = rgb2gray(moving_img)
    else:
        moving_gray = moving_img
    
    # Calculate shift using phase correlation
    shift, error, diffphase = phase_cross_correlation(fixed_gray, moving_gray)
    
    # If the input is RGB but the correlation was done on grayscale,
    # we need to extend the shift vector for all dimensions
    if moving_img.ndim == 3 and len(shift) == 2:
        # For RGB images, extend the shift to include 0 shift for color channels
        shift = np.append(shift, 0)
    
    # Apply shift to moving image
    aligned_img = ndimage.shift(moving_img, shift, order=1, mode='reflect')
    
    # Clear temporary variables
    fixed_gray = None
    moving_gray = None
    
    return aligned_img, shift

def download_processed_images(image_data, corrected_array, selected_indices):
    """
    Create a zip file of processed images for download with memory optimization.
    
    Args:
        image_data (dict): Original image data from database
        corrected_array (numpy.ndarray): White-balanced image array
        selected_indices (list): List of selected vegetation/water indices
    
    Returns:
        bytes: Zip file content containing processed images
    """
    # Create a bytes IO object for the zip file
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save white balance corrected image
        corrected_img = Image.fromarray(corrected_array)
        corrected_buffer = io.BytesIO()
        corrected_img.save(corrected_buffer, format='PNG')
        zipf.writestr('white_balanced.png', corrected_buffer.getvalue())
        corrected_buffer.close()
        
        # Process one index at a time
        for index_type in selected_indices:
            # Calculate index
            index_array = calculate_index(corrected_array, index_type)
            
            # Create visualization
            index_viz = create_index_visualization(index_array, index_type)
            
            # Save to zip
            index_buffer = io.BytesIO()
            index_viz.save(index_buffer, format='PNG')
            zipf.writestr(f'{index_type}_visualization.png', index_buffer.getvalue())
            index_buffer.close()
            
            # Clear variables to free memory
            index_array = None
            index_viz = None
            
            # Force garbage collection
            gc.collect()
    
    # Get the zip content
    zip_content = zip_buffer.getvalue()
    
    # Close the buffer
    zip_buffer.close()
    
    return zip_content

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
        
        # Get corrected array (if already computed) or calculate it
        if 'corrected_array' in img_data and img_data['corrected_array'] is not None:
            corrected_array = img_data['corrected_array']
        else:
            corrected_array = fix_white_balance(img_data['array'])
        
        # Calculate index
        index_array = calculate_index(corrected_array, index_type)
        
        # Calculate statistics
        threshold = 0.2 if index_type != "NDWI" else 0.0
        feature_name = "Water" if index_type == "NDWI" else "Vegetation"
        
        if index_array is not None:
            stats = {
                'Date': date,
                'Mean': float(np.mean(index_array)),
                'Median': float(np.median(index_array)),
                'Min': float(np.min(index_array)),
                'Max': float(np.max(index_array)),
                f'{feature_name} Coverage (%)': float(np.mean(index_array > threshold) * 100)
            }
            
            results.append(stats)
        
        # Clear variables to free memory
        index_array = None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df

def create_index_visualization(index_array, index_type):
    """Create colorful visualization of index values with memory optimization"""
    # None check
    if index_array is None or index_array.size == 0:
        return None
    
    # Use lower DPI to reduce memory usage
    plt.rcParams['figure.dpi'] = 100
    
    # Turn off interactive mode
    plt.ioff()
    
    # Create figure with direct Figure/Canvas approach (more memory efficient)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    fig = Figure(figsize=(10, 8))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Choose colormap based on index type
    if index_type == "NDWI":
        cmap = 'RdYlBu'  # Blue for water
    else:
        cmap = 'RdYlGn'  # Green for vegetation
    
    im = ax.imshow(index_array, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(im, label=index_type)
    ax.axis('off')
    
    # Save figure to buffer
    buf = io.BytesIO()
    canvas.print_figure(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    
    # Make a copy of the image before closing buffer
    img_copy = img.copy()
    
    # Clean up
    plt.close('all')
    buf.close()
    img.close()
    
    # Force garbage collection
    gc.collect()
    
    return img_copy

def create_comparison_view(image_data_list, index_type=None):
    """Create a side-by-side comparison view with memory optimization"""
    # None check
    if not image_data_list:
        return None, {}
    
    n_images = len(image_data_list)
    
    # Set matplotlib to lower DPI and turn off interactive mode
    plt.rcParams['figure.dpi'] = 100
    plt.ioff()
    
    # Create figure with direct Figure/Canvas approach
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    # Create figure with appropriate size (smaller for memory efficiency)
    fig = Figure(figsize=(4*n_images, 4))
    canvas = FigureCanvas(fig)
    
    all_stats = {}
    
    # Process each image
    for idx in range(n_images):
        image_data = image_data_list[idx]
        ax = fig.add_subplot(1, n_images, idx + 1)
        
        # Check what type of data to display
        if index_type:
            # Get image array (either original or processed)
            if 'array' in image_data and image_data['array'] is not None:
                img_array = image_data['array']
            else:
                # Fall back to original array if no processed data
                img_array = np.array(image_data['original'])
            
            # Create visualization
            if index_type == "NDWI":
                cmap = 'RdYlBu'
            else:
                cmap = 'RdYlGn'
            
            im = ax.imshow(img_array, cmap=cmap, vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax, label=index_type)
            
            # Calculate statistics
            stats = analyze_index(img_array, index_type)
            all_stats[image_data['metadata']['filename']] = stats
        else:
            # Simply show the image
            if 'array' in image_data and image_data['array'] is not None:
                ax.imshow(image_data['array'])
            else:
                # Fall back to original image if no array
                ax.imshow(np.array(image_data['original']))
        
        # Set title with filename
        if 'metadata' in image_data and 'filename' in image_data['metadata']:
            ax.set_title(image_data['metadata']['filename'], fontsize=8)
        ax.axis('off')
    
    # Adjust layout and save
    fig.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    canvas.print_figure(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
    buf.seek(0)
    comparison_image = Image.open(buf)
    
    # Make a copy before closing
    img_copy = comparison_image.copy()
    
    # Clean up
    plt.close('all')
    buf.close()
    comparison_image.close()
    
    # Force garbage collection
    gc.collect()
    
    return img_copy, all_stats

def create_time_series_plot(image_data_list, index_type):
    """Create a time series plot with memory optimization"""
    # None check
    if not image_data_list or len(image_data_list) < 2:
        return None
    
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
        
        # Get corrected array if already computed, otherwise calculate it
        if 'corrected_array' in img_data and img_data['corrected_array'] is not None:
            corrected_array = img_data['corrected_array']
        else:
            corrected_array = fix_white_balance(img_data['array'])
        
        # Calculate index
        index_array = calculate_index(corrected_array, index_type)
        
        if index_array is not None:
            # Calculate statistics
            mean_values.append(float(np.mean(index_array)))
            max_values.append(float(np.max(index_array)))
            min_values.append(float(np.min(index_array)))
        
        # Clear variables to free memory
        index_array = None
    
    # Create plot with direct Figure/Canvas approach
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    fig = Figure(figsize=(10, 6), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
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
    
    # Save to buffer
    buf = io.BytesIO()
    canvas.print_figure(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    
    # Make a copy before closing
    img_copy = img.copy()
    
    # Clean up
    plt.close('all')
    buf.close()
    img.close()
    
    # Force garbage collection
    gc.collect()
    
    return img_copy

def create_change_detection_visualization(image_pair, index_type):
    """Create change detection visualization with memory optimization"""
    # None check
    if not image_pair or len(image_pair) != 2:
        return None
    
    early_img_data, late_img_data = image_pair
    
    # Get corrected arrays if already computed, otherwise calculate them
    if 'corrected_array' in early_img_data and early_img_data['corrected_array'] is not None:
        early_corrected = early_img_data['corrected_array']
    else:
        early_corrected = fix_white_balance(early_img_data['array'])
    
    if 'corrected_array' in late_img_data and late_img_data['corrected_array'] is not None:
        late_corrected = late_img_data['corrected_array']
    else:
        late_corrected = fix_white_balance(late_img_data['array'])
    
    # Try to align images if they're both valid
    if early_corrected is not None and late_corrected is not None:
        aligned_late, shift = align_images(early_corrected, late_corrected)
    else:
        aligned_late = late_corrected
    
    # Calculate indices if arrays are valid
    if early_corrected is not None:
        early_index = calculate_index(early_corrected, index_type)
    else:
        early_index = None
    
    if aligned_late is not None:
        late_index = calculate_index(aligned_late, index_type)
    else:
        late_index = None
    
    # Calculate difference if both indices are valid
    if early_index is not None and late_index is not None:
        diff = late_index - early_index
    else:
        return None
    
    # Create visualization with direct Figure/Canvas approach
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    fig = Figure(figsize=(15, 5), dpi=100)
    canvas = FigureCanvas(fig)
    
    # Choose colormap based on index type
    if index_type == "NDWI":
        cmap = 'RdYlBu'
    else:
        cmap = 'RdYlGn'
    
    # Plot early image
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(early_index, cmap=cmap, vmin=-1, vmax=1)
    ax1.set_title(f"Early: {early_img_data['metadata']['upload_date'].strftime('%Y-%m-%d')}")
    fig.colorbar(im1, ax=ax1, label=index_type)
    ax1.axis('off')
    
    # Plot late image
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(late_index, cmap=cmap, vmin=-1, vmax=1)
    ax2.set_title(f"Late: {late_img_data['metadata']['upload_date'].strftime('%Y-%m-%d')}")
    fig.colorbar(im2, ax=ax2, label=index_type)
    ax2.axis('off')
    
    # Plot difference
    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(diff, cmap='bwr', vmin=-0.5, vmax=0.5)
    ax3.set_title(f"Change in {index_type}")
    fig.colorbar(im3, ax=ax3, label=f'Δ{index_type}')
    ax3.axis('off')
    
    # Layout
    fig.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    canvas.print_figure(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    
    # Make a copy before closing
    img_copy = img.copy()
    
    # Clean up
    plt.close('all')
    buf.close()
    img.close()
    
    # Clear arrays to free memory
    early_index = None
    late_index = None
    diff = None
    early_corrected = None
    late_corrected = None
    aligned_late = None
    
    # Force garbage collection
    gc.collect()
    
    return img_copy

# Here's the updated time_series_analysis_ui function with st.rerun() instead of st.experimental_rerun()

def time_series_analysis_ui():
    """Memory-optimized UI for time series analysis of monitoring sites"""
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
                        st.rerun()  # Changed from st.experimental_rerun()
        
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
            
            # Get images already assigned to this site (metadata only)
            site_images = get_site_images(str(selected_site['_id']))
            
            # Show option to assign new images to site
            with st.expander("Assign Images to Site"):
                # Get all available images for assignment
                if 'available_images' not in st.session_state or st.button("Refresh Available Images"):
                    st.session_state.available_images = get_stored_images(limit=100)  # Get up to 100 images
                
                # Filter out images already assigned to this site
                unassigned_images = [img for img in st.session_state.available_images 
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
                                st.rerun()  # Changed from st.experimental_rerun()
            
            # Display time series analysis for site images
            if not site_images:
                st.info(f"No images assigned to {selected_site['name']} yet. Assign images to begin analysis.")
            else:
                # First just show how many images are available
                st.info(f"{len(site_images)} images available for analysis.")
                
                # Select index for analysis before loading any data
                index_type = st.selectbox(
                    "Select Index for Analysis",
                    ["NDVI", "GNDVI", "NDWI"],
                    key="ts_index_selector"
                )
                
                # Check if we have enough images for analysis
                if len(site_images) < 2:
                    st.warning("Need at least 2 images for time series analysis.")
                else:
                    # Only load and process images when requested
                    if st.button("Generate Time Series Analysis"):
                        with st.spinner("Loading and processing site images..."):
                            # Create progress bar
                            progress_bar = st.progress(0)
                            
                            # Load images one by one with progress tracking
                            image_data_list = []
                            for i, img in enumerate(site_images):
                                # Update progress
                                progress = (i + 1) / len(site_images)
                                progress_bar.progress(progress)
                                
                                # Load image data with full resolution for analysis
                                img_data = load_image_from_db(img['_id'], thumbnail=False)
                                if img_data:
                                    # Resize large images to save memory
                                    img_data['array'] = preprocess_large_image(img_data['array'])
                                    # Pre-compute white balance to save memory later
                                    img_data['corrected_array'] = fix_white_balance(img_data['array'])
                                    # Clear original array to save memory
                                    img_data['array'] = None
                                    image_data_list.append(img_data)
                            
                            # Force garbage collection after loading
                            gc.collect()
                        
                        # Now create visualizations if we have enough images
                        if len(image_data_list) >= 2:
                            # Time series plot
                            st.subheader(f"{index_type} Time Series")
                            ts_plot = create_time_series_plot(image_data_list, index_type)
                            if ts_plot:
                                st.image(ts_plot, use_container_width=True)
                            
                            # Statistics table
                            st.subheader("Statistics Over Time")
                            stats_df = calculate_index_statistics_by_timeframe(image_data_list, index_type)
                            st.dataframe(stats_df)
                            
                            # Change detection - compare first and last images
                            st.subheader("Change Detection")
                            first_img = image_data_list[0]
                            last_img = image_data_list[-1]
                            
                            with st.spinner("Generating change detection visualization..."):
                                change_viz = create_change_detection_visualization(
                                    (first_img, last_img), 
                                    index_type
                                )
                                
                                if change_viz:
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
                                    
                                    # Clean up
                                    report_buffer.close()
                                    
                            # Clear all image data from memory
                            image_data_list = None
                            first_img = None
                            last_img = None
                            change_viz = None
                            ts_plot = None
                            
                            # Force garbage collection
                            gc.collect()
                        else:
                            st.warning("Not enough valid images for analysis. Need at least 2 images.")

def main():
    """Main function with memory optimizations"""
    st.set_page_config(layout="wide", page_title="RGNir Image Analyzer")
    
    # Create tabs for different functionality
    tab1, tab2 = st.tabs(["Image Analysis", "Time Series Monitoring"])
    
    # Add a memory cleanup button in sidebar
    if st.sidebar.button("💾 Clear Memory Cache"):
        for key in list(st.session_state.keys()):
            if key not in ['selected_images', 'page_number', 'tab_index']:
                # Keep minimal UI state, clear everything else
                del st.session_state[key]
        
        # Force garbage collection
        gc.collect()
        st.sidebar.success("Memory cache cleared")
    
    # Image Analysis Tab
    with tab1:
        st.title("RGNir Image Analyzer")
        
        # Initialize MongoDB connection
        if not init_connection():
            st.error("Failed to connect to database. Please check your connection settings.")
            return
        
        # Initialize session state
        if 'selected_images' not in st.session_state:
            st.session_state.selected_images = []
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 0
            
        # Set images per page
        IMAGES_PER_PAGE = 12
            
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
            
            # Clear the file uploader after processing
            st.rerun()
        
        # Database Management section
        with st.expander("Database Management"):
            st.write("Database Maintenance Tools")
            
            if st.button("Remove Duplicate Images"):
                removed_count = remove_duplicate_images()
                if removed_count > 0:
                    st.success(f"Removed {removed_count} duplicate images")
                    # Force refresh
                    if 'stored_images' in st.session_state:
                        del st.session_state.stored_images
                    st.rerun()
                else:
                    st.info("No duplicate images found")
            
            danger_col1, danger_col2 = st.columns(2)
            with danger_col1:
                if st.button("Clear All Images", type="secondary"):
                    st.session_state.confirm_delete = True
                    
            with danger_col2:
                if 'confirm_delete' in st.session_state and st.session_state.confirm_delete:
                    if st.button("⚠️ Confirm Delete All Images?", type="primary"):
                        try:
                            client = init_connection()
                            if client:
                                db = client.rgnir_analyzer
                                db.images.delete_many({})
                                st.success("All images removed successfully")
                                # Clear session state
                                for key in list(st.session_state.keys()):
                                    if key not in ['page_number', 'tab_index']:
                                        del st.session_state[key]
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to clear database: {str(e)}")
        
        # Image Gallery with Pagination
        st.header("Image Gallery")
        
        # Only load image list when needed, not full images
        if 'stored_images' not in st.session_state or st.button("Refresh Image List"):
            # Get total count of images
            client = init_connection()
            if client:
                db = client.rgnir_analyzer
                total_count = db.images.count_documents({})
                
                # Only fetch current page
                skip = st.session_state.page_number * IMAGES_PER_PAGE
                limit = IMAGES_PER_PAGE
                
                # Get images for current page with metadata only
                st.session_state.stored_images = list(db.images.find(
                    {}, 
                    {'metadata': 1, '_id': 1}
                ).sort('metadata.upload_date', -1).skip(skip).limit(limit))
                
                st.session_state.total_images = total_count
                st.session_state.total_pages = (total_count + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
                
                # Force garbage collection
                gc.collect()
        
        # Show pagination controls
        if 'total_pages' in st.session_state and st.session_state.total_pages > 1:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("Previous Page", disabled=st.session_state.page_number == 0):
                    st.session_state.page_number -= 1
                    # Clear stored images to force reload on next run
                    if 'stored_images' in st.session_state:
                        del st.session_state.stored_images
                    st.rerun()
            
            with col2:
                st.write(f"Page {st.session_state.page_number + 1} of {st.session_state.total_pages} • {st.session_state.total_images} Total Images")
                
            with col3:
                if st.button("Next Page", disabled=st.session_state.page_number >= st.session_state.total_pages - 1):
                    st.session_state.page_number += 1
                    # Clear stored images to force reload on next run
                    if 'stored_images' in st.session_state:
                        del st.session_state.stored_images
                    st.rerun()
        
        # Display gallery with current page images
        if 'stored_images' in st.session_state:
            if not st.session_state.stored_images:
                st.info("No images found in the database")
            else:
                # Create columns for gallery
                num_cols = min(3, max(1, len(st.session_state.stored_images)))
                cols = st.columns(num_cols)
                
                # Display images from current page
                for idx, doc in enumerate(st.session_state.stored_images):
                    img_id = str(doc['_id'])
                    with cols[idx % num_cols]:
                        # Load the image if not already loaded
                        if f"img_{img_id}" not in st.session_state:
                            # Load image from database with thumbnail size
                            image_data = load_image_from_db(img_id, thumbnail=True)
                            if image_data and image_data['original']:
                                st.session_state[f"img_{img_id}"] = image_data['original']
                        
                        # Display image if available
                        if f"img_{img_id}" in st.session_state:
                            st.image(
                                st.session_state[f"img_{img_id}"],
                                caption=doc['metadata']['filename'],
                                use_container_width=True
                            )
                        else:
                            # Fallback if image couldn't be loaded
                            st.write(f"**{doc['metadata']['filename']}**")
                            
                        st.caption(f"Uploaded: {doc['metadata']['upload_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Add checkbox for selection
                        if st.checkbox(f"Select for comparison {img_id}", 
                                     value=img_id in st.session_state.selected_images):
                            if img_id not in st.session_state.selected_images:
                                st.session_state.selected_images.append(img_id)
                        else:
                            if img_id in st.session_state.selected_images:
                                st.session_state.selected_images.remove(img_id)
                        
                        # Remove button
                        if st.button(f"Remove_{img_id}", type="secondary"):
                            if remove_image_from_db(img_id):
                                st.success("Image removed successfully")
                                if img_id in st.session_state.selected_images:
                                    st.session_state.selected_images.remove(img_id)
                                # Clean up session state
                                if f"img_{img_id}" in st.session_state:
                                    del st.session_state[f"img_{img_id}"]
                                # Force refresh
                                if 'stored_images' in st.session_state:
                                    del st.session_state.stored_images
                                st.rerun()
        
        # Comparison Analysis Section - Only load when needed
        if st.session_state.selected_images:
            st.header("Image Comparison Analysis")
            
            # Check if we have at least one image selected
            if len(st.session_state.selected_images) == 0:
                st.warning("No images selected for comparison")
            else:
                # Initialize analysis state if not exists
                if 'analysis_complete' not in st.session_state:
                    st.session_state.analysis_complete = False
                if 'analyzed_image_ids' not in st.session_state:
                    st.session_state.analyzed_image_ids = []
                if 'analysis_results' not in st.session_state:
                    st.session_state.analysis_results = {}
                    
                # Check if we need to run analysis (images changed or first time)
                need_analysis = not st.session_state.analysis_complete or \
                                sorted(st.session_state.selected_images) != sorted(st.session_state.analyzed_image_ids)
                
                # Let user trigger the analysis explicitly
                if need_analysis and st.button("Generate Comparison Analysis"):
                    with st.spinner("Loading and processing selected images..."):
                        # Reset analysis state
                        st.session_state.analysis_results = {
                            'image_data_list': [],
                            'original_comparison': None,
                            'wb_comparison': None,
                            'indices': {}
                        }
                        
                        # Load images one by one
                        image_data_list = []
                        progress_bar = st.progress(0)
                        
                        for i, img_id in enumerate(st.session_state.selected_images):
                            # Update progress
                            progress = (i + 1) / len(st.session_state.selected_images)
                            progress_bar.progress(progress)
                            
                            # Load full image data for analysis
                            img_data = load_image_from_db(img_id, thumbnail=False)
                            if img_data:
                                # Resize very large images before analysis
                                img_data['array'] = preprocess_large_image(img_data['array'])
                                image_data_list.append(img_data)
                    
                        if not image_data_list:
                            st.warning("No valid images selected for comparison")
                        else:                   
                            # Show original images comparison
                            comparison_image, _ = create_comparison_view(image_data_list)
                            if comparison_image:
                                st.session_state.analysis_results['original_comparison'] = comparison_image
                            
                            # Pre-compute white balance for all images to save memory
                            for img_data in image_data_list:
                                img_data['corrected_array'] = fix_white_balance(img_data['array'])
                                # We don't need the original array anymore
                                img_data['array'] = None
                            
                            # Force garbage collection
                            gc.collect()
                            
                            # Show white-balanced comparison
                            # Create a simplified list using only corrected arrays
                            wb_list = [{
                                'array': img['corrected_array'],  
                                'metadata': img['metadata']
                            } for img in image_data_list]
                            
                            comparison_image, _ = create_comparison_view(wb_list)
                            if comparison_image:
                                st.session_state.analysis_results['wb_comparison'] = comparison_image
                            
                            # Store the processed image data for future index calculations
                            st.session_state.analysis_results['image_data_list'] = image_data_list
                            
                            # Mark analysis as complete and store which images were analyzed
                            st.session_state.analysis_complete = True
                            st.session_state.analyzed_image_ids = st.session_state.selected_images.copy()
                            
                            # Force a rerun to display results
                            st.rerun()
                
                # Display analysis results if available
                if st.session_state.analysis_complete and 'analysis_results' in st.session_state:
                    results = st.session_state.analysis_results
                    
                    # Show original comparison
                    if 'original_comparison' in results and results['original_comparison'] is not None:
                        st.subheader("Original Images")
                        st.image(results['original_comparison'], use_container_width=True)
                    
                    # Show white-balanced comparison
                    if 'wb_comparison' in results and results['wb_comparison'] is not None:
                        st.subheader("White Balance Corrected")
                        st.image(results['wb_comparison'], use_container_width=True)
                    
                    # Only show index selection if we have images loaded
                    if 'image_data_list' in results and results['image_data_list']:
                        # Index selection - this won't reset state now
                        selected_indices = st.multiselect(
                            "Select Indices to Compare",
                            ["NDVI", "GNDVI", "NDWI"],
                            default=[]
                        )
                        
                        # Process selected indices one at a time
                        for index_type in selected_indices:
                            # Check if we already calculated this index
                            if index_type not in results['indices']:
                                st.subheader(f"{index_type} Comparison")
                                
                                # Process this index
                                with st.spinner(f"Processing {index_type}..."):
                                    # Use the corrected arrays calculated earlier
                                    index_data_list = []
                                    all_stats = {}
                                    
                                    for img_data in results['image_data_list']:
                                        # Calculate index
                                        index_array = calculate_index(img_data['corrected_array'], index_type)
                                        
                                        # Calculate statistics
                                        stats = analyze_index(index_array, index_type)
                                        all_stats[img_data['metadata']['filename']] = stats
                                        
                                        # Add to list for visualization
                                        index_data_list.append({
                                            'array': index_array,
                                            'metadata': img_data['metadata']
                                        })
                                    
                                    # Create visualization
                                    comparison_image, stats = create_comparison_view(index_data_list, index_type)
                                    
                                    # Store results
                                    results['indices'][index_type] = {
                                        'comparison': comparison_image,
                                        'stats': stats
                                    }
                                    
                                    # Clear data after processing
                                    index_data_list = None
                                    
                                    # Force garbage collection
                                    gc.collect()
                            
                            # Display the index visualization and stats
                            if index_type in results['indices']:
                                index_result = results['indices'][index_type]
                                
                                st.subheader(f"{index_type} Comparison")
                                if 'comparison' in index_result and index_result['comparison'] is not None:
                                    st.image(index_result['comparison'], use_container_width=True)
                                
                                # Display statistics in columns
                                if 'stats' in index_result and index_result['stats']:
                                    st.write(f"{index_type} Statistics")
                                    cols = st.columns(len(index_result['stats']))
                                    for idx, (filename, img_stats) in enumerate(index_result['stats'].items()):
                                        with cols[idx]:
                                            st.write(f"**{filename}**")
                                            for key, value in img_stats.items():
                                                st.metric(key, f"{value:.3f}")
                        
                        # Add download option if indices were processed
                        if selected_indices and results['image_data_list']:
                            # Create ZIP file with processed images
                            if st.button("Prepare Download Package"):
                                with st.spinner("Creating download package..."):
                                    # Get the first image's corrected array
                                    first_img = results['image_data_list'][0]
                                    corrected_array = first_img['corrected_array']
                                    
                                    # Create download package
                                    zip_data = download_processed_images(
                                        first_img, 
                                        corrected_array, 
                                        selected_indices
                                    )
                                    
                                    # Offer download
                                    st.download_button(
                                        label="Download Processed Images",
                                        data=zip_data,
                                        file_name="processed_images.zip",
                                        mime="application/zip"
                                    )
                        
                # Add a button to reset analysis if needed
                if st.session_state.analysis_complete and st.button("Reset Analysis"):
                    # Clear analysis state
                    st.session_state.analysis_complete = False
                    st.session_state.analyzed_image_ids = []
                    st.session_state.analysis_results = {}
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Rerun to update UI
                    st.rerun()
    
    # Time Series Monitoring Tab
    with tab2:
        # Only load time series data when on this tab
        if 'monitoring_sites' not in st.session_state or st.button("Refresh Sites"):
            st.session_state.monitoring_sites = get_all_monitoring_sites()
            gc.collect()
            
        # Use the optimized time series UI
        time_series_analysis_ui()

if __name__ == "__main__":
    main()