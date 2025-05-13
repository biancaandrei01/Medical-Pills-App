import glob
import os
import shutil
from pathlib import Path


def get_label_content(label_path):
    """Read and return the content of label file"""
    with open(label_path, 'r') as f:
        return f.read().strip()


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

        # Create a unique key combining label content and label name
        unique_key = f"{label_content}_{label_name.split('--')[0]}"

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
    dataset_root = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\robo_nefiltrat"

    # Confirm before proceeding
    print("This script will move duplicate files to a backup directory.")
    print(f"Dataset path: {dataset_root}")

    remove_duplicates(dataset_root)