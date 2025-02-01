import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_ndvi(image_path, save_path=None, visualize=True):
    """
    Calculate NDVI values for each pixel in an RGNir image.
    
    Args:
        image_path (str): Path to input RGNir image
        save_path (str, optional): Path to save NDVI visualization
        visualize (bool): Whether to display the NDVI heatmap
    
    Returns:
        numpy.ndarray: Array of NDVI values (-1 to 1)
    """
    # Load the image
    img = np.array(Image.open(image_path)).astype(float)
    
    # Extract NIR and Red bands
    nir = img[:,:,2]  # NIR is in the third channel
    red = img[:,:,0]  # Red is in the first channel
    
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-10
    
    # Calculate NDVI
    ndvi = (nir - red) / (nir + red + epsilon)
    
    # Clip values to valid NDVI range (-1 to 1)
    ndvi = np.clip(ndvi, -1, 1)
    
    if visualize or save_path:
        # Create colormap visualization
        plt.figure(figsize=(12, 8))
        
        # Use RdYlGn colormap for NDVI visualization
        ndvi_plot = plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(ndvi_plot, label='NDVI')
        plt.title('NDVI Values')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        if visualize:
            plt.show()
    
    return ndvi

def analyze_ndvi_statistics(ndvi_array):
    """
    Calculate basic statistics for NDVI values.
    
    Args:
        ndvi_array (numpy.ndarray): Array of NDVI values
    
    Returns:
        dict: Dictionary containing NDVI statistics
    """
    stats = {
        'mean_ndvi': float(np.mean(ndvi_array)),
        'median_ndvi': float(np.median(ndvi_array)),
        'min_ndvi': float(np.min(ndvi_array)),
        'max_ndvi': float(np.max(ndvi_array)),
        'std_ndvi': float(np.std(ndvi_array))
    }
    
    # Calculate vegetation coverage (percentage of pixels with NDVI > 0.2)
    vegetation_pixels = np.sum(ndvi_array > 0.2)
    total_pixels = ndvi_array.size
    stats['vegetation_coverage'] = float(vegetation_pixels / total_pixels * 100)
    
    return stats

def generate_ndvi_report(image_path, output_dir):
    """
    Generate comprehensive NDVI analysis with visualization and statistics.
    
    Args:
        image_path (str): Path to input RGNir image
        output_dir (str): Directory to save outputs
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate NDVI and save visualization
    ndvi_viz_path = os.path.join(output_dir, 'ndvi_visualization.png')
    ndvi_array = calculate_ndvi(image_path, ndvi_viz_path, visualize=False)
    
    # Calculate statistics
    stats = analyze_ndvi_statistics(ndvi_array)
    
    # Create histogram of NDVI values
    plt.figure(figsize=(10, 6))
    plt.hist(ndvi_array.flatten(), bins=50, range=(-1, 1))
    plt.title('Distribution of NDVI Values')
    plt.xlabel('NDVI')
    plt.ylabel('Pixel Count')
    plt.savefig(os.path.join(output_dir, 'ndvi_histogram.png'))
    plt.close()
    
    # Save statistics to text file
    with open(os.path.join(output_dir, 'ndvi_statistics.txt'), 'w') as f:
        f.write('NDVI Statistics:\n')
        for key, value in stats.items():
            f.write(f'{key}: {value:.4f}\n')
    
    return ndvi_array, stats

# Example usage:
if __name__ == "__main__":
    input_path = "mapir-test-images/2024_0921_011803_022.JPG_corrected.jpg"
    output_directory = "ndvi_analysis"
    
    # Generate complete NDVI analysis
    ndvi_values, statistics = generate_ndvi_report(input_path, output_directory)
    
    # Print summary
    print("\nNDVI Analysis Summary:")
    for key, value in statistics.items():
        print(f"{key}: {value:.4f}")