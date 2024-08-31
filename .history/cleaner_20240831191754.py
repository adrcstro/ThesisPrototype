import os
import cv2
import hashlib
from PIL import Image
from collections import defaultdict

def md5_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def check_blurriness(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def clean_image_dataset(input_directory, output_directory, expected_dim=(128, 128), blur_threshold=100, output_format='jpeg'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_hashes = defaultdict(list)
    
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_path = os.path.join(root, file)
                
                # Step 1: Check for Duplicates
                hash_value = md5_hash(image_path)
                if hash_value in image_hashes:
                    print(f"Duplicate images found: {image_path} and {image_hashes[hash_value][0]}")
                    continue
                image_hashes[hash_value].append(image_path)
                
                # Step 2: Verify Image Quality (Blurriness)
                image = cv2.imread(image_path)
                if check_blurriness(image, blur_threshold):
                    print(f"Image is blurry: {image_path}")
                    continue
                
                # Step 3: Ensure Consistent Image Dimensions (128x128 pixels)
                if image.shape[:2] != expected_dim:
                    print(f"Resizing image: {image_path}")
                    image = cv2.resize(image, expected_dim)
                
                # Step 4: Check for Noise (basic checks like extreme brightness/darkness)
                if image.mean() < 10 or image.mean() > 245:  # very dark or very bright images
                    print(f"Image may contain noise: {image_path} - mean pixel value: {image.mean()}")
                    continue
                
                # Step 5: Standardize Image Formats and Save to Output Directory
                new_filename = os.path.splitext(file)[0] + f'.{output_format}'
                new_path = os.path.join(output_directory, new_filename)
                
                # Convert and save the image in the desired format
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                image_pil.save(new_path, 'JPEG')  # Corrected to 'JPEG'

# Run the script
input_directory = r'C:\Users\andre\Downloads\7Dataset-20240717T141942Z-001\Final Datasets\melanoma'
output_directory = r'C:\Users\andre\Downloads\7Dataset-20240717T141942Z-001\CleanedDatasets\melanoma'
clean_image_dataset(input_directory, output_directory)
