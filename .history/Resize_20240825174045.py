import os
from PIL import Image

def resize_images(input_dir, output_dir, size=(128, 128)):
    """
    Resizes all images in the input directory to the specified size and saves them to the output directory.
    
    Parameters:
    - input_dir: Directory containing the original images.
    - output_dir: Directory where resized images will be saved.
    - size: Tuple specifying the target size (width, height). Default is (128, 128).
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img = img.resize(size, Image.ANTIALIAS)  # Resize image

                # Save the resized image in the output directory with the same folder structure
                rel_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, rel_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                img.save(os.path.join(save_dir, file))
                print(f"Resized and saved {file} to {save_dir}")

# Example usage:
input_directory = 'C:\Users\andre\Downloads\7 Dataset-20240717T141942Z-001\Final Datasets\Chickenpoxoriginalimages'  # Replace with your dataset's root directory
output_directory = 'C:\Users\andre\Downloads\7 Dataset-20240717T141942Z-001\Final Datasets\outputforchikenfox'  # Replace with the directory where you want to save resized images

resize_images(input_directory, output_directory)
