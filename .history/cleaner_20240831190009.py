import os
import cv2
import hashlib
import shutil
from PIL import Image
from collections import defaultdict

def md5_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def check_blurriness(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def standardize_image_format(image_path, output_format='jpg'):
    image = Image.open(image_path)
    image = image.convert('RGB')
    new_path = os.path.splitext(image_path)[0] + f'.{output_format}'
    image.save(new_path, output_format)
    return new_path

def clean_image_dataset(input_directory, output_directory, expected_dim=(128, 128), blur_threshold=100, output_format='jpg'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Step 1: Check for Duplicates
    image_hashes = defaultdict(list)
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_path = os.path.join(root, file)
                hash_value = md5_hash(image_path)
                image_hashes[hash_value].append(image_path)
    
    duplicates = {hash_val: paths for hash_val, paths in image_hashes.items() if len(paths) > 1}
    for hash_val, paths in duplicates.items():
        print(f"Duplicate images found: {paths}")
        for path in paths[1:]:
            os.remove(path)
    
    # Step 2: Verify Image Quality (Blurriness)
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if check_blurriness(image, blur_threshold):
                    print(f"Image is blurry: {image_path}")
                    os.remove(image_path)
    
    # Step 3: Ensure Consistent Image Dimensions (128x128 pixels)
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image.shape[:2] != expected_dim:
                    print(f"Image has incorrect dimensions: {image_path} - {image.shape[:2]}")
                    os.remove(image_path)
    
    # Step 4: Manual Review for Mislabelled or Irrelevant Images
    # This step requires manual inspection, so you might want to load the images for visualization
    
    # Step 5: Check for Noise (basic checks like extreme brightness/darkness)
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image.mean() < 10 or image.mean() > 245:  # very dark or very bright images
                    print(f"Image may contain noise: {image_path} - mean pixel value: {image.mean()}")
                    os.remove(image_path)
    
    # Step 6: Standardize Image Formats and Copy to Output Directory
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                if image.size != expected_dim:
                    print(f"Resizing image: {image_path}")
                    image = image.resize(expected_dim)
                
                new_path = os.path.join(output_directory, file)
                image = image.convert('RGB')  # Ensures all images are in RGB format
                image.save(new_path, output_format)

# Run the script
input_directory = 'C:\Users\andre\Downloads\7Dataset-20240717T141942Z-001\Final Datasets\Acnecleaned'
output_directory = 'C:\Users\andre\Downloads\7Dataset-20240717T141942Z-001\CleanedDatasets\Acne'
clean_image_dataset(input_directory, output_directory)
