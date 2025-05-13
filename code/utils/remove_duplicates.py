import os
import glob
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import shutil


def get_label_content(label_path):
    """Read and return the content of label file"""
    with open(label_path, 'r') as f:
        return f.read().strip()


def get_image_hash(image_path):
    """Calculate image hash to compare images"""
    try:
        with Image.open(image_path) as img:
            # Convert to numpy array and flatten
            img_array = np.array(img)
            return hashlib.md5(img_array.tobytes()).hexdigest()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def remove_duplicates(dataset_root):
    """Remove duplicate images and labels based on label content"""

    # Dictionary to store label content and corresponding files
    label_dict = {}
    duplicates_found = 0

    # Get all label files
    label_files = glob.glob(os.path.join(dataset_root, '*.txt'))
    total_files = len(label_files)

    # Create backup directories
    backup_dir = os.path.join(dataset_root, 'backup')
    backup_images = os.path.join(backup_dir)
    backup_labels = os.path.join(backup_dir)

    os.makedirs(backup_images, exist_ok=True)
    os.makedirs(backup_labels, exist_ok=True)

    for idx, label_path in enumerate(label_files, 1):
        if idx % 100 == 0:
            print(f"Progress: {idx}/{total_files} files processed")

        label_content = get_label_content(label_path)
        if not label_content:
            continue

        # Get corresponding image path
        label_name = Path(label_path).stem
        image_path = os.path.join(dataset_root, f"{label_name}.jpg")

        if not os.path.exists(image_path):
            print(f"Warning: No image file found for {label_name}")
            continue

        # Calculate image hash
        img_hash = get_image_hash(image_path)
        if img_hash is None:
            continue

        # Create a unique key combining label content and image hash
        unique_key = f"{label_content}"

        if unique_key in label_dict:
            # This is a duplicate
            duplicates_found += 1

            # Move duplicates to backup instead of deleting
            shutil.move(image_path, os.path.join(backup_images, f"{label_name}.jpg"))
            shutil.move(label_path, os.path.join(backup_labels, f"{label_name}.txt"))

            print(f"Duplicate found: {label_name}")
            print(f"  Original: {label_dict[unique_key]['name']}")
        else:
            # This is a new unique file
            label_dict[unique_key] = {
                'name': label_name,
                'image_path': image_path,
                'label_path': label_path
                }

    print(f"\nSummary:")
    print(f"Total duplicates found and moved to backup: {duplicates_found}")
    print(f"Unique files remaining: {len(label_dict)}")
    print(f"\nBackup directory: {backup_dir}")


if __name__ == "__main__":
    # Replace this path with your dataset root directory
    dataset_root = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\robo"

    # Confirm before proceeding
    print("This script will move duplicate files to a backup directory.")
    print(f"Dataset path: {dataset_root}")

    remove_duplicates(dataset_root)