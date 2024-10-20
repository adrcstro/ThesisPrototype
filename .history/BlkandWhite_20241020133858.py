import os
from skimage import io, color
from skimage.util import img_as_ubyte

# Function to convert images to grayscale
def convert_to_grayscale(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all the files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Read the image
            image_path = os.path.join(input_dir, filename)
            image = io.imread(image_path)

            # Convert the image to grayscale
            grayscale_image = color.rgb2gray(image)

            # Convert grayscale image to 8-bit unsigned integers
            grayscale_image_ubyte = img_as_ubyte(grayscale_image)

            # Save the grayscale image to the output directory
            output_path = os.path.join(output_dir, filename)
            io.imsave(output_path, grayscale_image_ubyte)
            print(f"Converted {filename} to grayscale and saved to {output_path}")

# Paths
input_directory = r"C:\xampp\htdocs\ThesisPrototype\CleanedOriginalDatasets\Vitiligo"  # Replace with the path to your image datasets
output_directory = r"C:\xampp\htdocs\ThesisPrototype\Grayscaledatasets\Original\Vitiligo"  # Replace with the path where you want to save grayscale images

# Convert images in the directory
convert_to_grayscale(input_directory, output_directory)
