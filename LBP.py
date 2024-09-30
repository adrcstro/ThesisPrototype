import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Parameters for LBP
radius = 1  # LBP radius
n_points = 8 * radius  # Number of points for LBP

# Function to apply LBP to an image
def apply_lbp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')  # Apply LBP
    return lbp

# Function to enhance a dataset folder with LBP
def enhance_dataset_with_lbp(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming dataset contains images
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Apply LBP
            lbp_image = apply_lbp(image)

            # Normalize to range 0-255 for visualization
            lbp_image = np.uint8(255 * lbp_image / np.max(lbp_image))

            # Save the LBP-enhanced image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, lbp_image)

            # Display one sample image (optional)
            plt.imshow(lbp_image, cmap='gray')
            plt.title('LBP Image')
            plt.show()

# Example usage


input_dir = r"C:\Users\andre\Documents\originaldatasets"  # Replace with the path to your dataset
output_dir = r"C:\Users\andre\Documents\datasetswithlbp"  # Replace with the path to save synthetic images


enhance_dataset_with_lbp(input_dir, output_dir)
