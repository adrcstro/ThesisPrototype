import os
import shutil

def rename_dataset_files(input_dir, output_dir, prefix='image_'):
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found!")
        return
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created!")

    # Loop through all files in the input directory
    for count, filename in enumerate(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, filename)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Get the file extension
            file_extension = os.path.splitext(filename)[1]
            
            # Create the new file name
            new_filename = f"{prefix}{count + 1}{file_extension}"
            
            # Define the new file path in the output directory
            new_file_path = os.path.join(output_dir, new_filename)
            
            # Copy the file to the output directory with the new name
            shutil.copy(file_path, new_file_path)
            print(f"Copied and Renamed: {filename} -> {new_filename}")
    
    print("Renaming and copying complete!")

# Example usage
input_directory = 'path_to_your_input_dataset'   # Replace with your input dataset directory path
output_directory = 'path_to_your_output_dataset' # Replace with your output dataset directory path

rename_dataset_files(input_directory, output_directory)
