import numpy as np
from PIL import Image

def fix_white_balance_rgnir(image_path, save_path=None):
    """
    Fix white balance of an RGNir image by applying histogram stretching
    and normalization to each channel independently.
    
    Args:
        image_path (str): Path to input RGNir image
        save_path (str, optional): Path to save corrected image. If None, 
                                 returns the processed array
    
    Returns:
        numpy.ndarray or None: Processed image array if save_path is None
    """
    # Load the image
    img = np.array(Image.open(image_path)).astype(float)
    
    # Split into R, G, NIR channels
    r = img[:,:,0]
    g = img[:,:,1]
    nir = img[:,:,2]
    
    def stretch_channel(channel):
        """Stretch histogram of a single channel"""
        # Remove outliers by clipping at 2nd and 98th percentiles
        p2, p98 = np.percentile(channel, (2, 98))
        channel_stretched = np.clip(channel, p2, p98)
        
        # Normalize to 0-255 range
        channel_normalized = ((channel_stretched - p2) / (p98 - p2) * 255)
        return np.clip(channel_normalized, 0, 255)
    
    # Apply histogram stretching to each channel
    r_corrected = stretch_channel(r)
    g_corrected = stretch_channel(g)
    nir_corrected = stretch_channel(nir)
    
    # Recombine channels
    corrected_img = np.dstack((r_corrected, g_corrected, nir_corrected))
    
    # Convert back to 8-bit unsigned integer
    corrected_img = corrected_img.astype(np.uint8)
    
    if save_path:
        Image.fromarray(corrected_img).save(save_path)
    else:
        return corrected_img

def visualize_correction(original_path, corrected_path):
    """
    Create a side-by-side visualization of original and corrected images
    
    Args:
        original_path (str): Path to original image
        corrected_path (str): Path to corrected image
    """
    # Load images
    original = Image.open(original_path)
    corrected = Image.open(corrected_path)
    
    # Create side-by-side comparison
    comparison = Image.new('RGB', (original.width * 2, original.height))
    comparison.paste(original, (0, 0))
    comparison.paste(corrected, (original.width, 0))
    
    return comparison

# Example usage:
if __name__ == "__main__":
    input_path = "mapir-test-images/2024_0921_011803_022.JPG"
    output_path = "mapir-test-images/2024_0921_011803_022.JPG_corrected.jpg"
    
    # Fix white balance
    fix_white_balance_rgnir(input_path, output_path)
    
    # Create visualization
    comparison = visualize_correction(input_path, output_path)
    comparison.save("comparison.png")