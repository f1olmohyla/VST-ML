import os
import shutil


def create_new_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted existing folder: {folder_path}")

    os.makedirs(folder_path)
    print(f"Created new folder: {folder_path}")