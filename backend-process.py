import os
import numpy as np
from PIL import Image
from pathlib import Path

def fix_white_balance(img):
    """Memory efficient white balance correction"""
    img_array = np.array(img, dtype=np.float32)
    corrected = np.zeros_like(img_array)
    
    for i in range(3):
        channel = img_array[:,:,i]
        p2, p98 = np.percentile(channel, (2, 98))
        corrected[:,:,i] = np.clip((channel - p2) / (p98 - p2) * 255, 0, 255)
    
    return Image.fromarray(corrected.astype(np.uint8))

def process_image(image_path, output_dir):
    # Create output directory
    Path(output_dir / 'white_balanced').mkdir(parents=True, exist_ok=True)
    
    # Load and process image
    img = Image.open(image_path)
    img_name = Path(image_path).stem
    
    # White balance correction
    corrected = fix_white_balance(img)
    corrected.save(output_dir / 'white_balanced' / f'{img_name}_wb.jpg')

def batch_process(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    image_files = [f for f in input_path.glob('*') if f.suffix.lower() in extensions]
    total = len(image_files)
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            print(f"Processing {idx}/{total}: {image_file.name}")
            process_image(image_file, output_path)
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch process RGNir images")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory for output files")
    args = parser.parse_args()
    
    batch_process(args.input_dir, args.output_dir)