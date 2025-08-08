#!/usr/bin/env python3
"""
Script to clean up dataset folder by removing directories and keeping only image files at root level
"""

import os
import shutil

def cleanup_dataset_folder(dataset_path):
    """Clean up dataset folder by removing directories and keeping only image files"""
    print(f"Cleaning up dataset folder: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset folder not found: {dataset_path}")
        return False
    
    # List all items
    items = os.listdir(dataset_path)
    print(f"Found items: {items}")
    
    # Separate files and directories
    files = []
    directories = []
    
    for item in items:
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            directories.append(item)
        else:
            files.append(item)
    
    print(f"Files: {files}")
    print(f"Directories: {directories}")
    
    # Remove directories
    for directory in directories:
        dir_path = os.path.join(dataset_path, directory)
        print(f"Removing directory: {directory}")
        try:
            shutil.rmtree(dir_path)
            print(f"Successfully removed: {directory}")
        except Exception as e:
            print(f"Error removing {directory}: {e}")
    
    # List final contents
    final_items = os.listdir(dataset_path)
    print(f"Final contents: {final_items}")
    
    return True

if __name__ == "__main__":
    # Clean up the specific dataset folder
    dataset_path = "/runpod-volume/datasets/test_flux_dreambooth_1000"
    cleanup_dataset_folder(dataset_path) 