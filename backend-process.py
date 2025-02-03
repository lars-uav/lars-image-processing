import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Define paths
INPUT_DIR = "mapir-test-images/Feb Photos/3rd Feb"
OUTPUT_DIR = "mapir-test-images/Feb Photos/Processed"

# Configure processing options
PROCESS_WB = True
PROCESS_NDVI = True
PROCESS_GNDVI = False
PROCESS_NDWI = False

def fix_white_balance(img):
    img_array = np.array(img, dtype=np.float32)
    corrected = np.zeros_like(img_array)
    
    for i in range(3):
        channel = img_array[:,:,i]
        p2, p98 = np.percentile(channel, (2, 98))
        corrected[:,:,i] = np.clip((channel - p2) / (p98 - p2) * 255, 0, 255)
    
    return Image.fromarray(corrected.astype(np.uint8))

def calculate_index(red, green, nir, index_type):
    epsilon = 1e-10
    
    if index_type == "NDVI":
        index = (nir - red) / (nir + red + epsilon)
    elif index_type == "GNDVI":
        index = (nir - green) / (nir + green + epsilon)
    elif index_type == "NDWI":
        index = (green - nir) / (green + nir + epsilon)
    
    return np.clip(index, -1, 1)

def create_index_visualization(index_array, index_type, output_path):
    plt.figure(figsize=(10, 8), dpi=100)
    cmap = 'RdYlBu' if index_type == "NDWI" else 'RdYlGn'
    plt.imshow(index_array, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label=index_type)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_image(image_path, output_dir, process_wb=False, indices=None):
    img_name = Path(image_path).stem
    
    img = Image.open(image_path)
    
    if process_wb:
        Path(output_dir / 'white_balanced').mkdir(parents=True, exist_ok=True)
        corrected = fix_white_balance(img)
        corrected.save(output_dir / 'white_balanced' / f'{img_name}_wb.tif')
    else:
        corrected = fix_white_balance(img)
    
    if indices:
        img_array = np.array(corrected, dtype=np.float32)
        red = img_array[:,:,0].copy()
        green = img_array[:,:,1].copy()
        nir = img_array[:,:,2].copy()
        del img_array
        
        for index_type in indices:
            Path(output_dir / index_type).mkdir(parents=True, exist_ok=True)
            index_array = calculate_index(red, green, nir, index_type)
            viz_path = output_dir / index_type / f'{img_name}_{index_type.lower()}.png'
            create_index_visualization(index_array, index_type, viz_path)
            del index_array

def batch_process():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Collect requested indices
    indices = []
    if PROCESS_NDVI:
        indices.append("NDVI")
    if PROCESS_GNDVI:
        indices.append("GNDVI")
    if PROCESS_NDWI:
        indices.append("NDWI")
    
    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    image_files = [f for f in input_path.glob('*') if f.suffix.lower() in extensions]
    total = len(image_files)
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            print(f"Processing {idx}/{total}: {image_file.name}")
            process_image(image_file, output_path, PROCESS_WB, indices if indices else None)
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")

if __name__ == "__main__":
    batch_process()