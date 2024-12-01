import os
import shutil

def create_new_folder(folder_path):
    # Check if the folder already exists
    if os.path.exists(folder_path):
        # Delete the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Deleted existing folder: {folder_path}")

    # Create the new folder
    os.makedirs(folder_path)
    print(f"Created new folder: {folder_path}")