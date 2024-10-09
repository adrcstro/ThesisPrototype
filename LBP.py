import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.exposure import rescale_intensity

def local_binary_pattern(image, P=8, R=1):
    """Calculate the LBP of an image."""
    gray_image = color.rgb2gray(image)
    gray_image = (gray_image * 70).astype(np.uint8)

    lbp_image = np.zeros_like(gray_image, dtype=np.uint8)
    padded_image = np.pad(gray_image, ((1, 1), (1, 1)), mode='constant')

    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            window = padded_image[i-1:i+2, j-1:j+2]
            center_value = window[1, 1]
            lbp_value = 0
            for k in range(P):
                theta = 2 * np.pi * k / P
                x = int(1 + R * np.cos(theta))
                y = int(1 + R * np.sin(theta))
                lbp_value |= (window[y, x] > center_value) << k
            lbp_image[i-1, j-1] = lbp_value
            
    return lbp_image

def process_image_dataset(input_dir, output_dir, P=8, R=1):
    """Process a directory of images and save LBP results to another directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = io.imread(image_path)
            lbp_result = local_binary_pattern(image, P, R)
            
            # Normalize the LBP image for better visibility
            lbp_result_normalized = rescale_intensity(lbp_result, in_range='image', out_range=(0, 255)).astype(np.uint8)
            
            # Save the normalized LBP image
            lbp_image_path = os.path.join(output_dir, f'lbp_{filename}')
            plt.imsave(lbp_image_path, lbp_result_normalized, cmap='gray')
            print(f'Saved LBP image to: {lbp_image_path}')

# Example usage
input_directory = r"C:\Users\User-PC\Documents\GitHub\thesisprototype2\asdasd"  # Replace with your input directory
output_directory = r"C:\Users\User-PC\Documents\GitHub\thesisprototype2\asdasd222"  # Replace with your output directory
process_image_dataset(input_directory, output_directory)
