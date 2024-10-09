import os

def rename_images(directory, prefix):
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            new_name = f"{prefix}_{filename}"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

# Usage
directory_path = r"C:\Users\User-PC\Documents\GitHub\thesisprototype2\syntheticDataset - Copy\Warts"# Change this to your directory
rename_images(directory_path, 'Warts')
