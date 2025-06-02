# Multi-Altitude UAV Image Processing for Agricultural Applications
# This notebook explores methods for enhancing keypoint identification in high-altitude imagery 
# using low-altitude UAV imagery for agricultural applications

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, segmentation, color, filters, feature, io, img_as_float, measure, util
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage as ndi
from skimage import draw
import warnings
warnings.filterwarnings('ignore')

# Display configurations
plt.rcParams['figure.figsize'] = (15, 10)

# Utility functions to load and display images
def display_images(images, titles=None, figsize=(15, 10), cmaps=None):
    """
    Display multiple images in a single figure
    """
    n = len(images)
    if titles is None:
        titles = ['Image %d' % (i+1) for i in range(n)]
    if cmaps is None:
        cmaps = ['viridis' for _ in range(n)]
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for i, (image, title, cmap) in enumerate(zip(images, titles, cmaps)):
        if len(image.shape) == 3 and cmap == 'viridis':
            axes[i].imshow(image)
        else:
            axes[i].imshow(image, cmap=cmap)
        axes[i].set_title(title, fontsize=14)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Multi-Scale Keypoint Integration Methods
def create_multiscale_descriptor_database(images, altitudes, method='sift'):
    """
    Create a multi-scale descriptor database from images at different altitudes
    This allows for storing feature descriptors with altitude information for later matching
    """
    database = {
        'descriptors': [],
        'keypoints': [],
        'altitudes': [],
        'image_indices': []
    }
    
    # Extract keypoints from all images
    for i, img in enumerate(images):
        kps, des, _ = detect_keypoints(img, method=method)
        
        if des is not None:
            # Store each descriptor with its altitude and image index
            for j, (kp, descriptor) in enumerate(zip(kps, des)):
                database['descriptors'].append(descriptor)
                database['keypoints'].append((i, j, kp))  # Store image index, keypoint index, and keypoint
                database['altitudes'].append(altitudes[i])
                database['image_indices'].append(i)
    
    return database

def match_keypoint_across_altitudes(database, query_descriptor, altitude_range=None, k=5):
    """
    Match a query descriptor against all descriptors in the database,
    optionally filtering by altitude range
    """
    if not database['descriptors']:
        return []
    
    # Convert descriptors to numpy array
    descriptors = np.array(database['descriptors'], dtype=np.float32)
    
    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Convert query to proper format
    if len(query_descriptor.shape) == 1:
        query_descriptor = query_descriptor.reshape(1, -1)
    
    # Find k nearest matches
    matches = flann.knnMatch(query_descriptor, descriptors, k=k)
    
    # Filter matches by altitude if range is specified
    if altitude_range is not None:
        min_alt, max_alt = altitude_range
        filtered_matches = []
        
        for match_group in matches:
            valid_matches = []
            for m in match_group:
                alt = database['altitudes'][m.trainIdx]
                if min_alt <= alt <= max_alt:
                    valid_matches.append(m)
            
            if valid_matches:
                filtered_matches.append(valid_matches)
        
        matches = filtered_matches
    
    return matches

def track_keypoint_across_altitudes(images, altitudes, keypoint_idx, image_idx, method='sift'):
    """
    Track a specific keypoint across different altitude images
    
    Parameters:
    - images: List of images at different altitudes
    - altitudes: List of altitude values for the images
    - keypoint_idx: Index of the keypoint to track in the specified image
    - image_idx: Index of the image containing the keypoint to track
    - method: Keypoint detection method (default: 'sift')
    
    Returns:
    - Dictionary containing tracked keypoint information
    """
    if image_idx >= len(images) or image_idx < 0:
        raise ValueError("Invalid image index")
    
    # Get keypoints from the source image
    source_kps, source_des, _ = detect_keypoints(images[image_idx], method=method)
    
    if keypoint_idx >= len(source_kps) or keypoint_idx < 0:
        raise ValueError("Invalid keypoint index")
    
    # Extract the query keypoint and descriptor
    query_kp = source_kps[keypoint_idx]
    query_des = source_des[keypoint_idx]
    
    # Create a database of all keypoints
    db = create_multiscale_descriptor_database(images, altitudes, method=method)
    
    # Match the query descriptor across all altitudes
    matches = match_keypoint_across_altitudes(db, query_des)
    
    # Organize matches by altitude
    tracked_points = {}
    for match_group in matches:
        for m in match_group:
            target_img_idx = db['image_indices'][m.trainIdx]
            target_kp_info = db['keypoints'][m.trainIdx]
            target_alt = db['altitudes'][m.trainIdx]
            
            if target_alt not in tracked_points:
                tracked_points[target_alt] = []
            
            tracked_points[target_alt].append({
                'image_idx': target_img_idx,
                'keypoint': target_kp_info[2],  # The actual keypoint object
                'distance': m.distance  # Match quality
            })
    
    return {
        'source_keypoint': query_kp,
        'source_altitude': altitudes[image_idx],
        'source_image_idx': image_idx,
        'tracked_points': tracked_points
    }

def visualize_tracked_keypoint(images, tracking_result):
    """
    Visualize keypoint tracking across different altitude images
    """
    # Extract tracking data
    source_kp = tracking_result['source_keypoint']
    source_img_idx = tracking_result['source_image_idx']
    source_alt = tracking_result['source_altitude']
    tracked_points = tracking_result['tracked_points']
    
    # Sort altitudes for consistent display
    altitudes = sorted(list(tracked_points.keys()))
    
    # Create visualizations
    result_images = []
    result_titles = []
    
    # Add source image with keypoint
    source_img = images[source_img_idx].copy()
    cv2.drawKeypoints(
        img_as_ubyte(source_img), 
        [source_kp], 
        source_img, 
        color=(0, 255, 0),  # Green color for source keypoint
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    result_images.append(source_img)
    result_titles.append(f'Source (Alt: {source_alt})')
    
    # Add target images with matched keypoints
    for alt in altitudes:
        if alt == source_alt:
            continue  # Skip source altitude
            
        # Get best match at this altitude
        matches = tracked_points[alt]
        if not matches:
            continue
            
        # Sort by match quality (lower distance is better)
        best_match = min(matches, key=lambda x: x['distance'])
        
        # Draw this keypoint on its image
        target_img = images[best_match['image_idx']].copy()
        cv2.drawKeypoints(
            img_as_ubyte(target_img), 
            [best_match['keypoint']], 
            target_img, 
            color=(0, 0, 255),  # Red color for target keypoints
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        result_images.append(target_img)
        result_titles.append(f'Altitude: {alt} (dist: {best_match["distance"]:.2f})')
    
    # Display all images
    display_images(result_images, result_titles)
    
    return result_images, result_titles

# Advanced Image Enhancement Methods
def adaptive_detail_enhancement(target_img, reference_img, patch_size=15, steps=3):
    """
    Enhance details in the target image guided by the reference image using an adaptive approach
    Works by progressively refining details at different scales
    """
    # Convert to float
    target = img_as_float(target_img)
    reference = img_as_float(reference_img)
    
    # Initialize result with target image
    result = target.copy()
    
    # Apply multiple passes with decreasing patch sizes for multi-scale enhancement
    for i in range(steps):
        # Calculate patch size for this step
        current_patch_size = max(3, patch_size - i*4)
        epsilon = 0.1 / (i+1)  # Decrease epsilon for finer details in later steps
        
        # Apply guided filter
        result = guided_filter(reference, result, radius=current_patch_size, epsilon=epsilon)
    
    return result

def cascade_detail_transfer(images, altitudes):
    """
    Perform a cascading detail transfer from lowest to highest altitude
    
    This approach progressively transfers details from the lowest altitude image
    to higher altitude images in sequence, preserving the fine details across
    the entire altitude range.
    """
    if len(images) < 2:
        print("Need at least 2 images for cascade transfer!")
        return None
    
    # Start with the lowest altitude image
    result = images[0].copy()
    
    # Progressively transfer details
    for i in range(1, len(images)):
        print(f"Cascading details from altitude {altitudes[i-1]} to {altitudes[i]}...")
        
        # Apply guided filtering for detail transfer
        result = guided_filter(result, images[i], radius=10, epsilon=0.1)
    
    return result

def hierarchical_detail_transfer(images, altitudes):
    """
    Combine details from all altitudes using a hierarchical approach
    
    This approach uses a binary tree structure to combine images in a hierarchical way,
    which may better preserve features at different scales.
    """
    if len(images) < 2:
        print("Need at least 2 images for hierarchical transfer!")
        return None
    
    # Make a copy of images to avoid modifying the original
    working_images = images.copy()
    
    # Continue until we've combined everything into one image
    while len(working_images) > 1:
        next_level = []
        
        # Process pairs of images
        for i in range(0, len(working_images) - 1, 2):
            # Combine this pair using Laplacian pyramid
            combined = transfer_details_pyramid(working_images[i], working_images[i+1])
            next_level.append(combined)
        
        # If there's an odd number of images, keep the last one
        if len(working_images) % 2 == 1:
            next_level.append(working_images[-1])
        
        # Update for next iteration
        working_images = next_level
    
    # The final result is the single remaining image
    return working_images[0]

def spatial_frequency_fusion(low_alt_img, high_alt_img):
    """
    Fuse images based on their spatial frequency content in the Fourier domain
    This preserves high-frequency details from the low altitude image while maintaining
    the global structure from the high altitude image
    """
    # Convert to grayscale
    if len(low_alt_img.shape) == 3:
        low_gray = color.rgb2gray(low_alt_img)
    else:
        low_gray = low_alt_img.copy()
        
    if len(high_alt_img.shape) == 3:
        high_gray = color.rgb2gray(high_alt_img)
    else:
        high_gray = high_alt_img.copy()
        
    # Apply Fourier transform
    low_fft = np.fft.fft2(low_gray)
    low_fft_shifted = np.fft.fftshift(low_fft)
    
    high_fft = np.fft.fft2(high_gray)
    high_fft_shifted = np.fft.fftshift(high_fft)
    
    # Create a filter to blend the frequency content
    rows, cols = low_gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a high-pass filter for low altitude image (high frequencies)
    high_pass = np.ones((rows, cols), dtype=np.float32)
    r = min(rows, cols) // 4  # Filter radius
    center = [crow, ccol]
    y, x = np.ogrid[:rows, :cols]
    mask_area = (y - center[0])**2 + (x - center[1])**2 <= r*r
    high_pass[mask_area] = 0
    
    # Create a low-pass filter for high altitude image (low frequencies)
    low_pass = 1 - high_pass
    
    # Apply filters and blend frequency components
    low_high_freq = low_fft_shifted * high_pass
    high_low_freq = high_fft_shifted * low_pass
    
    # Combine frequency components
    combined_fft_shifted = low_high_freq + high_low_freq
    
    # Inverse FFT to get the spatial image back
    combined_fft = np.fft.ifftshift(combined_fft_shifted)
    combined_spatial = np.fft.ifft2(combined_fft)
    combined_spatial = np.abs(combined_spatial)
    
    # Normalize to 0-1 range
    combined_spatial = (combined_spatial - np.min(combined_spatial)) / (np.max(combined_spatial) - np.min(combined_spatial))
    
    return combined_spatial

# Additional Crop Analysis Functions
def analyze_crop_rows(image, alt_level, min_length=100, max_gap=20):
    """
    Detect and analyze crop rows in the image
    Returns information about row spacing, direction, and uniformity
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.copy()
    
    # Calculate vegetation index to isolate crops
    indices = calculate_vegetation_indices(image)
    veg_mask = indices['vegetation_mask']
    
    # Apply edge detection to find boundaries
    edges = feature.canny(veg_mask, sigma=2)
    
    # Use Hough transform to find lines representing crop rows
    lines = probabilistic_hough_line(edges, line_length=min_length, line_gap=max_gap)
    
    # Analyze line properties
    angles = []
    lengths = []
    spacing = []
    row_count = len(lines)
    
    for line in lines:
        p0, p1 = line
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = np.sqrt(dx**2 + dy**2)
        angle = np.degrees(np.arctan2(dy, dx)) % 180
        
        angles.append(angle)
        lengths.append(length)
    
    # If we have at least 2 lines, try to estimate row spacing
    if len(lines) >= 2:
        # Group lines by similar angles
        angle_threshold = 10  # degrees
        grouped_lines = {}
        
        for i, angle in enumerate(angles):
            angle_key = int(angle / angle_threshold) * angle_threshold
            if angle_key not in grouped_lines:
                grouped_lines[angle_key] = []
            grouped_lines[angle_key].append(lines[i])
        
        # Find the most common direction
        most_common_angle = max(grouped_lines, key=lambda k: len(grouped_lines[k]))
        parallel_lines = grouped_lines[most_common_angle]
        
        # Sort these lines by their position (perpendicular to line direction)
        perp_angle = (most_common_angle + 90) % 180
        perp_rad = np.radians(perp_angle)
        perp_vec = np.array([np.cos(perp_rad), np.sin(perp_rad)])
        
        line_positions = []
        for line in parallel_lines:
            p0, p1 = line
            mid_point = np.array([(p0[0] + p1[0])/2, (p0[1] + p1[1])/2])
            position = np.dot(mid_point, perp_vec)
            line_positions.append((position, line))
        
        # Sort by position
        line_positions.sort(key=lambda x: x[0])
        
        # Calculate spacing between adjacent lines
        for i in range(1, len(line_positions)):
            space = line_positions[i][0] - line_positions[i-1][0]
            spacing.append(space)
    
    # Compile results
    result = {
        'altitude': alt_level,
        'row_count': row_count,
        'lines': lines,
        'mean_angle': np.mean(angles) if angles else 0,
        'angle_std': np.std(angles) if angles else 0,
        'mean_length': np.mean(lengths) if lengths else 0,
        'mean_spacing': np.mean(spacing) if spacing else 0,
        'spacing_std': np.std(spacing) if spacing else 0
    }
    
    return result

def probabilistic_hough_line(edge_image, line_length=50, line_gap=10):
    """
    Helper function to call Hough transform for line detection
    """
    # Use skimage's probabilistic Hough transform
    try:
        from skimage.transform import probabilistic_hough_line
        return probabilistic_hough_line(edge_image, 
                                       line_length=line_length,
                                       line_gap=line_gap)
    except ImportError:
        # Fallback to OpenCV if skimage's version is not available
        return cv2.HoughLinesP(img_as_ubyte(edge_image), 
                              rho=1, 
                              theta=np.pi/180, 
                              threshold=10, 
                              minLineLength=line_length, 
                              maxLineGap=line_gap)

def visualize_crop_rows(image, row_analysis):
    """
    Visualize detected crop rows on the image
    """
    # Create a color visualization
    if len(image.shape) == 2 or image.shape[2] == 1:
        viz = color.gray2rgb(image)
    else:
        viz = image.copy()
    
    # Draw detected lines
    for line in row_analysis['lines']:
        if hasattr(line, '__iter__') and len(line) == 2:  # skimage format: [(x0, y0), (x1, y1)]
            p0, p1 = line
            rr, cc = draw.line(p0[1], p0[0], p1[1], p1[0])
        else:  # opencv format: [x0, y0, x1, y1]
            rr, cc = draw.line(line[1], line[0], line[3], line[2])
        
        # Make sure coordinates are within image bounds
        in_bounds = (rr >= 0) & (rr < viz.shape[0]) & (cc >= 0) & (cc < viz.shape[1])
        viz[rr[in_bounds], cc[in_bounds]] = [1, 0, 0]  # Red lines
    
    # Add text with statistics
    text_info = [
        f"Altitude: {row_analysis['altitude']}",
        f"Row count: {row_analysis['row_count']}",
        f"Mean angle: {row_analysis['mean_angle']:.1f}°",
        f"Row spacing: {row_analysis['mean_spacing']:.1f} px"
    ]
    
    # Create figure with the visualization and text info
    plt.figure(figsize=(12, 8))
    plt.imshow(viz)
    plt.title("Crop Row Detection", fontsize=14)
    
    # Add text info to the upper right corner
    text = '\n'.join(text_info)
    plt.text(viz.shape[1] - 10, 20, text, 
             horizontalalignment='right',
             color='white', fontsize=12,
             bbox=dict(facecolor='black', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return viz

# ==============================
# 1. IMAGE PREPROCESSING
# ==============================

def load_all_altitude_images(base_path="", img_pattern="image*.JPG"):
    """
    Load all UAV images of increasing altitude from a directory
    Returns them sorted by image number (assuming naming convention image1.jpg, image2.jpg, etc.)
    """
    import glob
    import os
    import re
    
    # Find all matching image files
    image_files = glob.glob(os.path.join(base_path, img_pattern))
    
    # Extract image numbers for sorting
    def extract_number(filename):
        match = re.search(r'image(\d+)', os.path.basename(filename).lower())
        if match:
            return int(match.group(1))
        return float('inf')  # Put files without numbers at the end
    
    # Sort image files by their number
    image_files.sort(key=extract_number)
    
    if not image_files:
        raise ValueError(f"No images found matching pattern {img_pattern} in {base_path}")
    
    # Load all images
    images = []
    altitudes = []
    
    for i, img_file in enumerate(image_files):
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not load image {img_file}")
            continue
            
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        
        # Assign estimated altitudes based on order (or extract from filename/metadata if available)
        # Here we're just using the image number as a proxy for altitude
        altitude = extract_number(img_file)
        altitudes.append(altitude)
        
        print(f"Loaded image: {os.path.basename(img_file)} (Altitude level: {altitude})")
    
    return images, altitudes

def preprocess_image(image, denoise=True, enhance_contrast=True):
    """
    Preprocess an image with denoising and contrast enhancement options
    """
    # Convert to float for processing
    img_float = img_as_float(image)
    
    # Apply denoising if requested
    if denoise:
        # For color images
        if len(image.shape) == 3:
            # Non-local means denoising
            img_float = cv2.fastNlMeansDenoisingColored(
                img_as_ubyte(img_float), 
                None, 10, 10, 7, 21
            )
            img_float = img_as_float(img_float)
        else:
            # For grayscale images
            img_float = filters.gaussian(img_float, sigma=1)
    
    # Enhance contrast if requested
    if enhance_contrast:
        # Adaptive histogram equalization for better local contrast
        if len(image.shape) == 3:
            # Process each channel separately for color images
            img_enhanced = np.zeros_like(img_float)
            for i in range(3):
                img_enhanced[:,:,i] = exposure.equalize_adapthist(
                    img_float[:,:,i], clip_limit=0.03
                )
            img_float = img_enhanced
        else:
            img_float = exposure.equalize_adapthist(img_float, clip_limit=0.03)
    
    return img_float

def calculate_vegetation_indices(image):
    """
    Calculate common vegetation indices from an RGB image
    This is a simplified version - more accurate indices require proper
    multispectral imagery with NIR bands
    """
    # Ensure the image is in float format
    img = img_as_float(image)
    
    # Extract RGB channels
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    # Simplified indices (note: these are approximations as true indices need NIR)
    
    # Visible Atmospherically Resistant Index (VARI)
    # Good for RGB-only vegetation analysis
    epsilon = 1e-8  # Small number to avoid division by zero
    vari = (g - r) / (g + r - b + epsilon)
    vari = np.clip(vari, -1, 1)  # Clip to standard range
    
    # Excess Green Index (ExG)
    exg = 2*g - r - b
    
    # Normalized Green-Red Difference Index (NGRDI)
    ngrdi = (g - r) / (g + r + epsilon)
    ngrdi = np.clip(ngrdi, -1, 1)
    
    # Create a simple RGB visualization of vegetation
    # Higher values of ExG indicate more vegetation
    vegetation_mask = exg > np.mean(exg) + 0.5 * np.std(exg)
    
    return {
        'vari': vari,
        'exg': exg,
        'ngrdi': ngrdi,
        'vegetation_mask': vegetation_mask
    }

# ==============================
# 2. KEYPOINT DETECTION AND MATCHING
# ==============================

def detect_keypoints(image, method='sift', max_features=5000):
    """
    Detect keypoints in an image using different methods
    """
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_RGB2GRAY)
    else:
        gray = img_as_ubyte(image)
    
    # Initialize detector based on method
    if method.lower() == 'sift':
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif method.lower() == 'orb':
        detector = cv2.ORB_create(nfeatures=max_features)
    elif method.lower() == 'akaze':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unsupported keypoint detection method: {method}")
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    return keypoints, descriptors, gray

def match_keypoints(descriptors1, descriptors2, method='flann', ratio_thresh=0.75):
    """
    Match keypoints between two images
    """
    # Choose matching method
    if method.lower() == 'flann' and descriptors1 is not None and descriptors2 is not None:
        if descriptors1.dtype != np.float32:
            descriptors1 = np.float32(descriptors1)
        if descriptors2.dtype != np.float32:
            descriptors2 = np.float32(descriptors2)
            
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    else:
        # Brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = bf.match(descriptors1, descriptors2)
        # Sort matches by distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    return good_matches

def visualize_matches(image1, keypoints1, image2, keypoints2, matches, max_matches=100):
    """
    Visualize keypoint matches between two images
    """
    # Convert images to uint8 format
    img1 = img_as_ubyte(image1)
    img2 = img_as_ubyte(image2)
    
    # Limit number of matches to visualize
    matches = matches[:min(max_matches, len(matches))]
    
    # Create match visualization
    matched_img = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Convert from BGR to RGB for matplotlib
    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    
    return matched_img

def find_homography(keypoints1, keypoints2, matches, ransac_thresh=5.0):
    """
    Find the homography matrix between two images
    """
    if len(matches) < 4:
        return None, []
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_thresh)
    
    # Get only inlier matches
    inlier_matches = [matches[i] for i, val in enumerate(mask) if val == 1]
    
    return H, inlier_matches

def warp_image(image, homography, target_shape):
    """
    Warp an image using a homography matrix
    """
    return cv2.warpPerspective(
        img_as_ubyte(image), homography, 
        (target_shape[1], target_shape[0])
    )

# ==============================
# 3. SUPERPIXEL SEGMENTATION AND DETAIL TRANSFER
# ==============================

def generate_superpixels(image, n_segments=300, compactness=10):
    """
    Generate superpixels for an image using SLIC
    """
    segments = slic(
        image, n_segments=n_segments, compactness=compactness, 
        start_label=1, sigma=1, convert2lab=True
    )
    
    # Visualize superpixel boundaries
    marked = mark_boundaries(image, segments, color=(1, 0, 0))
    
    return segments, marked

def analyze_superpixels(image, segments):
    """
    Analyze properties of superpixels
    """
    # Convert to LAB color space for better color analysis
    if len(image.shape) == 3:
        lab_img = color.rgb2lab(image)
    else:
        lab_img = color.gray2rgb(image)
        lab_img = color.rgb2lab(lab_img)
    
    # Measure properties of regions
    props = measure.regionprops(segments, intensity_image=lab_img)
    
    # Extract features for each superpixel
    superpixel_features = []
    for prop in props:
        # Mean color (L, a, b)
        mean_color = prop.mean_intensity
        
        # Shape features
        area = prop.area
        perimeter = prop.perimeter
        eccentricity = prop.eccentricity
        
        superpixel_features.append({
            'label': prop.label,
            'mean_color': mean_color,
            'area': area,
            'perimeter': perimeter,
            'eccentricity': eccentricity,
            'centroid': prop.centroid
        })
    
    return superpixel_features

def transfer_details_pyramid(low_alt_img, high_alt_img, levels=6):
    """
    Transfer details from low altitude to high altitude image using Laplacian pyramid
    """
    # Convert to float and grayscale
    if len(low_alt_img.shape) == 3:
        low_alt_gray = color.rgb2gray(low_alt_img)
    else:
        low_alt_gray = low_alt_img.copy()
        
    if len(high_alt_img.shape) == 3:
        high_alt_gray = color.rgb2gray(high_alt_img)
    else:
        high_alt_gray = high_alt_img.copy()
    
    # Build Gaussian pyramids
    low_alt_pyramid = [low_alt_gray]
    high_alt_pyramid = [high_alt_gray]
    
    for i in range(levels-1):
        low_alt_pyramid.append(cv2.pyrDown(low_alt_pyramid[i]))
        high_alt_pyramid.append(cv2.pyrDown(high_alt_pyramid[i]))
    
    # Build Laplacian pyramids
    low_alt_laplacian = []
    high_alt_laplacian = []
    
    for i in range(levels-1):
        low_expanded = cv2.pyrUp(low_alt_pyramid[i+1])
        high_expanded = cv2.pyrUp(high_alt_pyramid[i+1])
        
        # Resize expanded image to match the original size if needed
        if low_expanded.shape != low_alt_pyramid[i].shape:
            low_expanded = cv2.resize(low_expanded, 
                                     (low_alt_pyramid[i].shape[1], low_alt_pyramid[i].shape[0]))
        if high_expanded.shape != high_alt_pyramid[i].shape:
            high_expanded = cv2.resize(high_expanded, 
                                      (high_alt_pyramid[i].shape[1], high_alt_pyramid[i].shape[0]))
        
        low_alt_laplacian.append(low_alt_pyramid[i] - low_expanded)
        high_alt_laplacian.append(high_alt_pyramid[i] - high_expanded)
    
    low_alt_laplacian.append(low_alt_pyramid[-1])
    high_alt_laplacian.append(high_alt_pyramid[-1])
    
    # Combine pyramids: keep high frequencies from low altitude image,
    # and low frequencies from high altitude image
    merged_laplacian = []
    for i in range(levels-1):
        # Use detail from low altitude image (high frequencies)
        merged_laplacian.append(low_alt_laplacian[i])
    
    # Use base from high altitude image (low frequencies)
    merged_laplacian.append(high_alt_laplacian[-1])
    
    # Reconstruct image
    reconstructed = merged_laplacian[-1]
    for i in range(levels-2, -1, -1):
        expanded = cv2.pyrUp(reconstructed)
        if expanded.shape != merged_laplacian[i].shape:
            expanded = cv2.resize(expanded, 
                                 (merged_laplacian[i].shape[1], merged_laplacian[i].shape[0]))
        reconstructed = expanded + merged_laplacian[i]
    
    return reconstructed

import cv2.ximgproc
def guided_filter(guide_img, input_img, radius=5, epsilon=0.1):
    """
    Edge-preserving filter that can transfer details from a guide image
    """
    # Ensure both images are float
    guide = img_as_float(guide_img)
    input_img = img_as_float(input_img)
    
    # Handle color images
    if len(guide.shape) == 3:
        result = np.zeros_like(input_img)
        for i in range(3):
            result[:,:,i] = cv2.ximgproc.guidedFilter(
                guide[:,:,i], input_img[:,:,i], radius, epsilon
            )
        return result
    else:
        return cv2.ximgproc.guidedFilter(guide, input_img, radius, epsilon)

# ==============================
# 4. MAIN ANALYSIS FUNCTIONS
# ==============================

def full_pipeline_analysis(images, altitudes):
    """
    Run the complete analysis pipeline on multiple images with increasing altitude
    
    Parameters:
    - images: List of UAV images sorted by increasing altitude
    - altitudes: List of corresponding altitude values/levels
    """
    n_images = len(images)
    if n_images < 2:
        print("Need at least 2 images for comparison!")
        return
        
    # 1. Preprocess all images
    print("Step 1: Preprocessing images...")
    processed_images = [preprocess_image(img) for img in images]
    
    # Display original and processed images
    all_display_images = []
    all_titles = []
    
    for i, (orig, proc) in enumerate(zip(images, processed_images)):
        all_display_images.extend([orig, proc])
        all_titles.extend([f'Altitude {altitudes[i]} Original', 
                          f'Altitude {altitudes[i]} Processed'])
    
    # Display in batches if there are many images
    max_display = 6
    for i in range(0, len(all_display_images), max_display):
        display_images(
            all_display_images[i:i+max_display],
            all_titles[i:i+max_display]
        )
    
    # 2. Calculate vegetation indices for all images
    print("\nStep 2: Calculating vegetation indices...")
    all_indices = [calculate_vegetation_indices(img) for img in processed_images]
    
    # Display ExG (Excess Green) index for all images
    exg_images = [indices['exg'] for indices in all_indices]
    exg_titles = [f'Altitude {alt} ExG' for alt in altitudes]
    
    for i in range(0, len(exg_images), max_display):
        display_images(
            exg_images[i:i+max_display],
            exg_titles[i:i+max_display],
            cmaps=['viridis'] * min(max_display, len(exg_images) - i)
        )
    
    # 3. Detect keypoints in all images
    print("\nStep 3: Detecting keypoints in all images...")
    # Try different keypoint detection methods
    methods = ['sift', 'orb']
    
    keypoints_data = {}  # Store keypoint data for all methods and images
    
    for method in methods:
        print(f"\nUsing {method.upper()} keypoint detection...")
        
        # Detect keypoints in all images
        all_kps = []
        all_des = []
        all_gray = []
        all_kp_imgs = []
        
        for i, img in enumerate(processed_images):
            kps, des, gray = detect_keypoints(img, method=method)
            all_kps.append(kps)
            all_des.append(des)
            all_gray.append(gray)
            
            print(f"Detected {len(kps)} keypoints in altitude {altitudes[i]} image")
            
            # Create keypoint visualization
            kp_img = cv2.drawKeypoints(
                img_as_ubyte(gray), kps, None, 
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            all_kp_imgs.append(cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB))
        
        # Store for later use
        keypoints_data[method] = {
            'keypoints': all_kps,
            'descriptors': all_des,
            'gray_images': all_gray
        }
        
        # Display keypoint visualizations
        kp_titles = [f'Altitude {alt} {method.upper()} Keypoints' for alt in altitudes]
        
        for i in range(0, len(all_kp_imgs), max_display):
            display_images(
                all_kp_imgs[i:i+max_display],
                kp_titles[i:i+max_display]
            )
            
        # 4. Match keypoints between consecutive altitude images
        print(f"\nMatching {method.upper()} keypoints between consecutive altitudes...")
        
        for i in range(n_images - 1):
            lower_idx = i
            higher_idx = i + 1
            
            lower_kps = all_kps[lower_idx]
            lower_des = all_des[lower_idx]
            lower_gray = all_gray[lower_idx]
            
            higher_kps = all_kps[higher_idx]
            higher_des = all_des[higher_idx]
            higher_gray = all_gray[higher_idx]
            
            if lower_des is not None and higher_des is not None:
                matches = match_keypoints(lower_des, higher_des)
                
                if len(matches) > 0:
                    print(f"Found {len(matches)} matches between altitude {altitudes[lower_idx]} and {altitudes[higher_idx]}")
                    
                    # Visualize matches
                    matched_img = visualize_matches(
                        lower_gray, lower_kps, higher_gray, higher_kps, matches
                    )
                    
                    plt.figure(figsize=(15, 10))
                    plt.imshow(matched_img)
                    plt.title(f"{method.upper()} Keypoint Matches: Altitude {altitudes[lower_idx]} → {altitudes[higher_idx]}", fontsize=14)
                    plt.axis('off')
                    plt.show()
                    
                    # Find homography for image registration
                    H, inliers = find_homography(lower_kps, higher_kps, matches)
                    
                    if H is not None:
                        print(f"Found homography with {len(inliers)} inlier matches")
                        
                        # Warp lower altitude image to align with higher altitude image
                        warped_lower = warp_image(
                            processed_images[lower_idx], H, processed_images[higher_idx].shape
                        )
                        
                        display_images(
                            [processed_images[lower_idx], warped_lower, processed_images[higher_idx]],
                            [f'Altitude {altitudes[lower_idx]}', 
                             f'Warped Altitude {altitudes[lower_idx]}', 
                             f'Altitude {altitudes[higher_idx]}']
                        )
                    else:
                        print("Could not find a valid homography")
                else:
                    print(f"No matches found between altitude {altitudes[lower_idx]} and {altitudes[higher_idx]}")
    
    # 5. Superpixel segmentation
    print("\nStep 5: Generating superpixels for all images...")
    all_segments = []
    all_marked = []
    all_features = []
    
    for i, img in enumerate(processed_images):
        segments, marked = generate_superpixels(img)
        features = analyze_superpixels(img, segments)
        
        all_segments.append(segments)
        all_marked.append(marked)
        all_features.append(features)
        
        print(f"Analyzed {len(features)} superpixels in altitude {altitudes[i]} image")
    
    # Display superpixel visualizations
    marked_titles = [f'Altitude {alt} Superpixels' for alt in altitudes]
    
    for i in range(0, len(all_marked), max_display):
        display_images(
            all_marked[i:i+max_display],
            marked_titles[i:i+max_display]
        )
    
    # 6. Progressive detail transfer from low to high altitude
    print("\nStep 6: Applying progressive detail transfer from low to high altitude...")
    
    # Laplacian pyramid blending between consecutive images
    pyramid_results = []
    guided_results = []
    
    for i in range(n_images - 1):
        lower_idx = i
        higher_idx = i + 1
        
        print(f"Transferring details from altitude {altitudes[lower_idx]} to {altitudes[higher_idx]}...")
        
        # Apply Laplacian pyramid detail transfer
        pyramid_result = transfer_details_pyramid(
            processed_images[lower_idx], processed_images[higher_idx]
        )
        pyramid_results.append(pyramid_result)
        
        # Apply guided filtering
        guided_result = guided_filter(
            processed_images[lower_idx], processed_images[higher_idx], radius=10, epsilon=0.1
        )
        guided_results.append(guided_result)
        
        # Display results for this pair
        display_images(
            [processed_images[lower_idx], processed_images[higher_idx], 
             pyramid_result, guided_result],
            [f'Altitude {altitudes[lower_idx]}', f'Altitude {altitudes[higher_idx]}',
             f'Laplacian Pyramid Transfer {altitudes[lower_idx]}→{altitudes[higher_idx]}', 
             f'Guided Filter Transfer {altitudes[lower_idx]}→{altitudes[higher_idx]}']
        )
    
    # 7. Final visualization with vegetation enhancement
    print("\nStep 7: Creating final vegetation-enhanced visualizations...")
    
    # Enhance vegetation in all images
    enhanced_viz = []
    viz_titles = []
    
    for i, (img, indices) in enumerate(zip(processed_images, all_indices)):
        veg_mask = indices['vegetation_mask']
        
        # Create color-coded visualization
        viz = img.copy()
        
        # Enhance vegetation in green
        for j in range(3):
            viz[:,:,j] = np.where(
                veg_mask, 
                viz[:,:,j] * (2.0 if j == 1 else 0.5),  # Boost green channel
                viz[:,:,j]
            )
        
        # Clip values to valid range
        viz = np.clip(viz, 0, 1)
        
        enhanced_viz.append(viz)
        viz_titles.append(f'Altitude {altitudes[i]} Vegetation Enhanced')
    
    # Display enhanced visualizations
    for i in range(0, len(enhanced_viz), max_display):
        display_images(
            enhanced_viz[i:i+max_display],
            viz_titles[i:i+max_display]
        )
    
    # 8. Generate cross-altitude vegetation analysis
    print("\nStep 8: Generating cross-altitude vegetation analysis...")
    
    # Compare ExG values across altitudes
    mean_exg_values = [np.mean(indices['exg']) for indices in all_indices]
    
    plt.figure(figsize=(10, 6))
    plt.plot(altitudes, mean_exg_values, 'o-', linewidth=2)
    plt.xlabel('Altitude Level')
    plt.ylabel('Mean Excess Green (ExG) Value')
    plt.title('Vegetation Index Variation by Altitude')
    plt.grid(True)
    plt.show()
    
    # Calculate vegetation coverage percentage
    veg_coverage = [np.mean(indices['vegetation_mask']) * 100 for indices in all_indices]
    
    plt.figure(figsize=(10, 6))
    plt.plot(altitudes, veg_coverage, 'o-', linewidth=2)
    plt.xlabel('Altitude Level')
    plt.ylabel('Vegetation Coverage (%)')
    plt.title('Vegetation Coverage by Altitude')
    plt.grid(True)
    plt.show()
    
    print("\nAnalysis complete!")


# Example workflow function to run all analyses
def complete_workflow_example():
    """
    Example function demonstrating a complete workflow for multi-altitude UAV image analysis
    """
    print("Starting complete multi-altitude UAV image analysis workflow...")
    
    # 1. Load all images
    print("\n1. Loading all altitude images...")
    images, altitudes = load_all_altitude_images(base_path="/Users/ananyashukla/Desktop/Brown_Research/lesion_segmentation/keypoint_agri")
    
    if len(images) < 2:
        print("Error: Need at least 2 images of different altitudes!")
        return
    
    # Display original images
    all_titles = [f'Altitude {alt} UAV Image' for alt in altitudes]
    max_display = min(6, len(images))
    
    display_images(images[:max_display], all_titles[:max_display])
    
    # 2. Preprocess all images
    print("\n2. Preprocessing images...")
    processed_images = [preprocess_image(img) for img in images]
    
    # 3. Calculate vegetation indices
    print("\n3. Calculating vegetation indices...")
    all_indices = [calculate_vegetation_indices(img) for img in processed_images]
    
    # Display ExG index for first few images
    exg_images = [indices['exg'] for indices in all_indices]
    exg_titles = [f'Altitude {alt} ExG' for alt in altitudes]
    
    display_images(
        exg_images[:max_display],
        exg_titles[:max_display],
        cmaps=['viridis'] * max_display
    )
    
    # 4. Detect keypoints using SIFT
    print("\n4. Detecting keypoints using SIFT...")
    all_kps = []
    all_des = []
    all_gray = []
    
    for i, img in enumerate(processed_images):
        kps, des, gray = detect_keypoints(img, method='sift')
        all_kps.append(kps)
        all_des.append(des)
        all_gray.append(gray)
        
        print(f"  Detected {len(kps)} keypoints in altitude {altitudes[i]} image")
    
    # 5. Match keypoints between consecutive altitudes
    print("\n5. Matching keypoints between consecutive altitudes...")
    
    for i in range(len(images) - 1):
        lower_idx = i
        higher_idx = i + 1
        
        matches = match_keypoints(all_des[lower_idx], all_des[higher_idx])
        
        if len(matches) > 0:
            print(f"  Found {len(matches)} matches between altitude {altitudes[lower_idx]} and {altitudes[higher_idx]}")
            
            # Visualize matches (only for first few pairs)
            if i < 2:  # Only show first few visualizations to avoid too many plots
                matched_img = visualize_matches(
                    all_gray[lower_idx], all_kps[lower_idx], 
                    all_gray[higher_idx], all_kps[higher_idx], 
                    matches
                )
                
                plt.figure(figsize=(15, 10))
                plt.imshow(matched_img)
                plt.title(f"SIFT Keypoint Matches: Altitude {altitudes[lower_idx]} → {altitudes[higher_idx]}", fontsize=14)
                plt.axis('off')
                plt.show()
    
    # 6. Create multi-scale descriptor database
    print("\n6. Creating multi-scale descriptor database...")
    db = create_multiscale_descriptor_database(processed_images, altitudes, method='sift')
    print(f"  Database contains {len(db['descriptors'])} descriptors from {len(images)} images")
    
    # 7. Track a specific keypoint across all altitudes
    print("\n7. Tracking keypoints across altitudes...")
    
    # Use a keypoint from the lowest altitude image as reference
    lowest_idx = 0
    if len(all_kps[lowest_idx]) > 0:
        # Use a distinctive keypoint with high response value
        sorted_kps = sorted(enumerate(all_kps[lowest_idx]), key=lambda x: x[1].response, reverse=True)
        kp_idx, _ = sorted_kps[0]  # Get the keypoint with highest response
        
        print(f"  Tracking keypoint {kp_idx} from altitude {altitudes[lowest_idx]} image...")
        tracking_result = track_keypoint_across_altitudes(
            processed_images, altitudes, kp_idx, lowest_idx, method='sift'
        )
        
        # Visualize the tracking result
        visualize_tracked_keypoint(processed_images, tracking_result)
    
    # 8. Analyze crop rows from different altitudes
    print("\n8. Analyzing crop rows from different altitudes...")
    
    for i, (img, alt) in enumerate(zip(processed_images, altitudes)):
        # Only analyze a few images to avoid too many plots
        if i >= 3:
            continue
            
        print(f"  Analyzing crop rows for altitude {alt}...")
        row_analysis = analyze_crop_rows(img, alt)
        visualize_crop_rows(img, row_analysis)
        
        print(f"  Row count: {row_analysis['row_count']}")
        print(f"  Mean row spacing: {row_analysis['mean_spacing']:.2f} pixels")
        print(f"  Mean row angle: {row_analysis['mean_angle']:.2f} degrees")
    
    # 9. Apply detail transfer between altitudes
    print("\n9. Applying detail transfer methods...")
    
    # Try different detail transfer methods between consecutive altitudes
    if len(processed_images) >= 2:
        # Use the lowest and highest altitude images for comparison
        lowest_img = processed_images[0]
        highest_img = processed_images[-1]
        
        print("  Applying standard guided filtering...")
        guided_result = guided_filter(lowest_img, highest_img)
        
        print("  Applying adaptive detail enhancement...")
        adaptive_result = adaptive_detail_enhancement(highest_img, lowest_img)
        
        print("  Applying spatial frequency fusion...")
        frequency_result = spatial_frequency_fusion(lowest_img, highest_img)
        
        # Display results
        display_images(
            [lowest_img, highest_img, guided_result, adaptive_result, frequency_result],
            ['Lowest Altitude', 'Highest Altitude', 
             'Guided Filter Transfer', 'Adaptive Enhancement', 'Frequency Fusion']
        )
    
    # 10. Apply advanced multi-scale approaches for all images
    if len(processed_images) >= 3:
        print("\n10. Applying advanced multi-scale approaches...")
        
        print("  Applying cascade detail transfer...")
        cascade_result = cascade_detail_transfer(processed_images, altitudes)
        
        print("  Applying hierarchical detail transfer...")
        hierarchical_result = hierarchical_detail_transfer(processed_images, altitudes)
        
        # Display results
        display_images(
            [processed_images[0], processed_images[-1], cascade_result, hierarchical_result],
            ['Lowest Altitude', 'Highest Altitude', 
             'Cascade Detail Transfer', 'Hierarchical Detail Transfer']
        )
    
    print("\nComplete workflow analysis finished!")

# If this script is run directly, execute the complete workflow
if __name__ == "__main__":
    complete_workflow_example()