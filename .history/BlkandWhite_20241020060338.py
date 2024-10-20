import os
from PIL import Image

def convert_images_to_bw(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Construct full file path
            file_path = os.path.join(input_dir, filename)
            # Open image
            with Image.open(file_path) as img:
                # Convert image to black and white
                bw_img = img.convert('L')
                # Save the black and white image to the output directory
                bw_img.save(os.path.join(output_dir, filename))

            print(f'Converted {filename} to black and white.')

if __name__ == '__main__':
    input_directory = r"C:\Users\User-PC\Documents\GitHub\thesisprototype2\syntheticDataset - Copy\Warts" 
    output_directory = r"C:\Users\User-PC\Documents\GitHub\thesisprototype2\LBP synthetic Input\Warts"  # Change to your output directory
    convert_images_to_bw(input_directory, output_directory)
