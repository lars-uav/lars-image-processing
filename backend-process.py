import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def fix_white_balance(img_array):
    img_float = img_array.astype(float)
    corrected = np.zeros_like(img_float)
    
    for i in range(3):
        channel = img_float[:,:,i]
        p2, p98 = np.percentile(channel, (2, 98))
        corrected[:,:,i] = np.clip((channel - p2) / (p98 - p2) * 255, 0, 255)
    
    return corrected.astype(np.uint8)

def calculate_index(img_array, index_type):
    img_float = img_array.astype(float)
    red = img_float[:,:,0]
    green = img_float[:,:,1]
    nir = img_float[:,:,2]
    epsilon = 1e-10
    
    if index_type == "NDVI":
        return np.clip((nir - red) / (nir + red + epsilon), -1, 1)
    elif index_type == "GNDVI":
        return np.clip((nir - green) / (nir + green + epsilon), -1, 1)
    elif index_type == "NDWI":
        return np.clip((green - nir) / (green + nir + epsilon), -1, 1)

def create_index_visualization(index_array, index_type):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = 'RdYlBu' if index_type == "NDWI" else 'RdYlGn'
    im = ax.imshow(index_array, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, label=index_type)
    ax.axis('off')
    return fig

def process_image(image_path, output_dir):
    # Create output directories
    subdirs = ['white_balanced', 'NDVI', 'GNDVI', 'NDWI']
    for subdir in subdirs:
        Path(output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Load and process image
    img = Image.open(image_path)
    img_array = np.array(img)
    img_name = Path(image_path).stem
    
    # White balance correction
    corrected_array = fix_white_balance(img_array)
    Image.fromarray(corrected_array).save(
        output_dir / 'white_balanced' / f'{img_name}_wb.tif'
    )
    
    # Calculate and save indices
    for index_type in ['NDVI', 'GNDVI', 'NDWI']:
        # Calculate index on white-balanced image
        index_array = calculate_index(corrected_array, index_type)
        
        # Save index visualization
        fig = create_index_visualization(index_array, index_type)
        fig.savefig(
            output_dir / index_type / f'{img_name}_{index_type.lower()}.png',
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)
        
        # Save raw index values as numpy array
        np.save(
            output_dir / index_type / f'{img_name}_{index_type.lower()}_values.npy',
            index_array
        )

def batch_process(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Process all supported image files
    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    image_files = [f for f in input_path.glob('*') if f.suffix.lower() in extensions]
    
    for image_file in image_files:
        try:
            process_image(image_file, output_path)
            print(f"Processed: {image_file.name}")
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch process RGNir images")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory for output files")
    args = parser.parse_args()
    
    batch_process(args.input_dir, args.output_dir)