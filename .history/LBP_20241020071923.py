import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
import os

# Function to apply LBP on a grayscale image
def apply_lbp(image, radius=3, n_points=24):
    # LBP method 'uniform' ensures a less blurred output
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp_image

# Function to process a dataset of images
def process_dataset(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
            image_path = os.path.join(input_dir, filename)
            
            # Load the grayscale image
            grayscale_image = io.imread(image_path, as_gray=True)
            
            # Apply LBP on the grayscale image
            lbp_result = apply_lbp(grayscale_image)
            
            # Save both original and LBP processed images
            original_output_path = os.path.join(output_dir, f"original_{filename}")
            lbp_output_path = os.path.join(output_dir, f"lbp_{filename}")
            
            # Save the original image (converted to unsigned 8-bit integer format)
            io.imsave(original_output_path, img_as_ubyte(grayscale_image))
            
            # Save the LBP processed image
            io.imsave(lbp_output_path, img_as_ubyte(lbp_result))
            
            print(f"Processed and saved: {filename}")

# Paths to your dataset directories
input_directory = r"C:\xampp\htdocs\ThesisPrototype\Grayscaledatasets\Original\Acne"  # Replace with your input directory path
output_directory = r"C:\xampp\htdocs\ThesisPrototype\DatasetswithLBP\OriginalDatasetswithLBP\OriginalAcnewithLBP"        # Replace with your desired output directory

# Process the dataset
process_dataset(input_directory, output_directory)
