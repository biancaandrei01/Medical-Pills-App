import os
import glob
from pathlib import Path
import shutil


def get_class_from_label(label_path):
    """Read and return the class index and full content of label file"""
    with open(label_path, 'r') as f:
        content = f.readlines()
        if content:
            class_idx = int(content[0].split()[0])
            return class_idx, content
    return None, None


def update_label_content(content, old_class):
    """Update class index in label content"""
    updated_lines = []
    for line in content:
        parts = line.split()
        if parts:
            class_idx = int(parts[0])
            if class_idx == old_class:  # If it's Volta class
                # Decrease the class index by 1 (because Unknown class is removed)
                parts[0] = str(class_idx - 1)
            line = ' '.join(parts)
        updated_lines.append(line)
    return updated_lines


def remove_unknown_class(dataset_root):
    """Remove Unknown class images and update Volta class numbers"""
    # Class names from the YAML
    class_names = [
        'Amoxicillin', 'Antacill', 'Burajel', 'CaRBon', 'Celebrex',
        'CentrumMultivitamin', 'Cetrizin', 'Chlorpheniramine', 'Crabocysteine',
        'Dafomin', 'ParacetamolDecolgen', 'Difelene', 'Doxycycline',
        'Fenafex', 'Flunarizine', 'Gaszym', 'Glutamax', 'HandyHerb',
        'Heromycin', 'IbumanPlus', 'Ibuprofen', 'Imodium', 'Lamotrigine',
        'MefenamicAcid', 'Miracid', 'Norxacin', 'Noxa', 'ParacetamolSara',
        'Simvastatin', 'Unknown', 'Volta'
    ]

    # Find indices for Unknown and Volta classes
    unknown_idx = class_names.index('Unknown')
    volta_idx = class_names.index('Volta')

    # Create backup directory
    backup_dir = os.path.join(dataset_root, 'backup_unknown_class')
    os.makedirs(backup_dir, exist_ok=True)

    # Process all image files
    image_files = glob.glob(os.path.join(dataset_root, '*.jpg'))

    unknown_count = 0
    volta_count = 0
    updated_count = 0

    print("Processing files...")
    for img_path in image_files:
        img_name = Path(img_path).stem
        label_path = os.path.join(dataset_root, f"{img_name}.txt")

        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_name}")
            continue

        class_idx, content = get_class_from_label(label_path)
        if class_idx is None:
            print(f"Warning: Invalid label file {label_path}")
            continue

        if class_idx == unknown_idx:
            # Move Unknown class files to backup
            unknown_count += 1
            shutil.move(img_path, os.path.join(backup_dir, f"{img_name}.jpg"))
            shutil.move(label_path, os.path.join(backup_dir, f"{img_name}.txt"))
            print(f"Moved Unknown class file: {img_name}")

        elif class_idx == volta_idx:
            # Update Volta class number
            volta_count += 1
            updated_content = update_label_content(content, volta_idx)

            # Write updated content back to file
            with open(label_path, 'w') as f:
                f.writelines([line + '\n' for line in updated_content])
            updated_count += 1
            print(f"Updated Volta class file: {img_name}")

    print("\nOperation complete!")
    print(f"Unknown class files moved to backup: {unknown_count}")
    print(f"Volta class files found: {volta_count}")
    print(f"Files updated: {updated_count}")
    print(f"\nBackup directory: {backup_dir}")

    # Reminder to update YAML file
    print("\nIMPORTANT: Don't forget to update your data.yaml file:")
    print("1. Remove 'Unknown' from the names list")
    print("2. Update 'nc' (number of classes) to 30")


if __name__ == "__main__":
    # Replace this path with your dataset root directory
    dataset_root = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\robo"

    # Confirm before proceeding
    print("This script will:")
    print("1. Move all Unknown class images and labels to a backup directory")
    print("2. Update class numbers for Volta class (decrease by 1)")
    print(f"\nDataset path: {dataset_root}")

    remove_unknown_class(dataset_root)