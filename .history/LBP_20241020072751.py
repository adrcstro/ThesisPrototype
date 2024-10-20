import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.exposure import rescale_intensity

def local_binary_pattern(image, P=8, R=1, smooth=False, sigma=0.5):
    """Calculate the LBP of an image with optional smoothing."""
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.shape[2] != 3:
        raise ValueError("Input image should be in RGB format or grayscale.")
    
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)
    gray_image = (gray_image * 255).astype(np.uint8)
    
    # Smooth the grayscale image if specified
    if smooth:
        # Reduce sigma to avoid too much blurring
        gray_image = filters.gaussian(gray_image, sigma=sigma, mode='reflect') * 255
        gray_image = gray_image.astype(np.uint8)

    lbp_image = np.zeros_like(gray_image, dtype=np.uint8)
    padded_image = np.pad(gray_image, ((R, R), (R, R)), mode='constant')

    for i in range(R, padded_image.shape[0] - R):
        for j in range(R, padded_image.shape[1] - R):
            center_value = padded_image[i, j]
            lbp_value = 0
            for k in range(P):
                theta = 2 * np.pi * k / P
                dx = round(R * np.cos(theta))
                dy = round(R * np.sin(theta))
                neighbor_value = padded_image[i + dy, j + dx]
                lbp_value |= (neighbor_value > center_value) << k
            lbp_image[i-R, j-R] = lbp_value
            
    return lbp_image

def process_image_dataset(input_dir, output_dir, P=8, R=1, smooth=False, sigma=0.5):
    """Process a directory of images and save LBP results to another directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_dir, filename)
            image = io.imread(image_path)
            lbp_result = local_binary_pattern(image, P, R, smooth, sigma)
            
            # Normalize the LBP image to improve visibility
            lbp_result_normalized = rescale_intensity(lbp_result, in_range='image', out_range=(0, 255)).astype(np.uint8)
            
            # Save the normalized LBP image at a higher resolution if needed
            lbp_image_path = os.path.join(output_dir, f'lbp_{filename}')
            plt.imsave(lbp_image_path, lbp_result_normalized, cmap='gray', dpi=300)
            print(f'Saved LBP image to: {lbp_image_path}')

# Example usage
input_directory = r"C:\xampp\htdocs\ThesisPrototype\Grayscaledatasets\Original\Acne"  # Replace with your input directory
output_directory = r"C:\xampp\htdocs\ThesisPrototype\DatasetswithLBP\OriginalDatasetswithLBP\OriginalAcnewithLBP"  # Replace with your output directory
process_image_dataset(input_directory, output_directory, smooth=False)  # Disable smoothing for clarity
