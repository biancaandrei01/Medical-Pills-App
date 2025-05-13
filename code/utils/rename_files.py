import os
import glob
from pathlib import Path


def get_class_from_label(label_path):
    """Read the first number from label file which represents the class index"""
    with open(label_path, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            return int(first_line.split()[0])
    return None


def rename_dataset_files(dataset_root):
    """Rename all images and their corresponding label files"""
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


    # Get all image files
    image_files = glob.glob(os.path.join(dataset_root, '*.jpg'))

    for img_path in image_files:
        img_name = Path(img_path).stem
        label_path = os.path.join(dataset_root, f"{img_name}.txt")

        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_name}")
            continue

        # Get class index from label file
        class_idx = get_class_from_label(label_path)
        if class_idx is None or class_idx >= len(class_names):
            print(f"Warning: Invalid class index in {label_path}")
            continue

        # Generate new names
        class_name = class_names[class_idx]
        # Count existing files with same class name to generate unique index
        existing_count = len(glob.glob(os.path.join(dataset_root, f"{class_name}_*.jpg")))
        new_name = f"{class_name}_{existing_count + 1}"

        # New file paths
        new_img_path = os.path.join(dataset_root, f"{new_name}.jpg")
        new_label_path = os.path.join(dataset_root, f"{new_name}.txt")

        # Rename files
        try:
            os.rename(img_path, new_img_path)
            os.rename(label_path, new_label_path)
            print(f"Renamed: {img_name} → {new_name}")
        except OSError as e:
            print(f"Error renaming {img_name}: {e}")


if __name__ == "__main__":
    # Replace this path with your dataset root directory
    dataset_root = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\robo"
    rename_dataset_files(dataset_root)