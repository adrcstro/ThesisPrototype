import os

def rename_dataset_files(input_dir, prefix='image_'):
    # Check if the directory exists
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' not found!")
        return
    
    # Loop through all files in the directory
    for count, filename in enumerate(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, filename)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Get the file extension
            file_extension = os.path.splitext(filename)[1]
            
            # Create the new file name
            new_filename = f"{prefix}{count + 1}{file_extension}"
            
            # Define the new file path
            new_file_path = os.path.join(input_dir, new_filename)
            
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")
    
    print("Renaming complete!")

# Example usage
input_directory = 'path_to_your_dataset'  # Replace with your dataset directory path
rename_dataset_files(input_directory)
