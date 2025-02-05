# lars-image-processing

## Overview

A poweerful application designed for analyzing multispectral (RGNir) images. This tool provides advanced image processing capabilities, allowing users to upload, store, and analyze images with features like white balance correction and vegetation/water index calculations.

## Features

### Image Management

- Upload multiple image formats (TIFF, PNG, JPG, JPEG)
- Store images in MongoDB database
- Remove duplicate images

### Image Analysis

- White balance correction
- Vegetation and Water Index Calculations:
  - Normalized Difference Vegetation Index (NDVI)
  - Green Normalized Difference Vegetation Index (GNDVI)
  - Normalized Difference Water Index (NDWI)
- Detailed statistical analysis of indices
- Colorful index visualizations

## Prerequisites

### Software Requirements

- Python 3.8+
- Streamlit
- MongoDB

### Required Python Packages

- streamlit
- numpy
- pillow (PIL)
- matplotlib
- pymongo
- python-dotenv

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/rgnir-image-analyzer.git
cd rgnir-image-analyzer
```

2. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Set up MongoDB

- Create a MongoDB Atlas account or set up a local MongoDB instance
- Create a `.env` file in the project root with your MongoDB connection string:

```
MONGODB_URI=your_mongodb_connection_string
```

## Running the Application

```bash
streamlit run process-images.py
```

## Usage

### Uploading Images

1. Click on the file uploader
2. Select one or multiple RGNir images
3. Images will be checked for duplicates and stored in the database

### Analyzing Images

1. Click "Refresh Database" to view stored images
2. Select an image by clicking "Analyze"
3. Choose vegetation/water indices to visualize
4. View original, white-balanced, and index-mapped images
5. Examine statistical information about the selected indices

## Database Management

- Remove Duplicate Images: Eliminates redundant image entries
- Clear All Images: Completely wipes the image database

## Image Requirements

- Supported Formats: TIFF, PNG, JPG, JPEG
- Recommended: Multispectral images with Red, Green, and Near-Infrared bands

## Troubleshooting

- Ensure MongoDB connection is valid
- Check internet connectivity
- Verify Python and package versions
- Refer to error messages in the Streamlit interface

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Copyright 2025 Ananya Shukla, Jia Bhargava, Tanay Srinivasa, Tanmay Nanda

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Contact

Your Name - tanmaynanda360@gmail.com

Project Link: [https://github.com/lars-uav/lars-image-processing.git](https://github.com/lars-uav/lars-image-processing.git)

## Acknowledgments

- Streamlit
- MongoDB
- NumPy
- Matplotlib
