import os
import glob
from pathlib import Path
import shutil


def shift_labels_and_copy_dataset(source_root, class_shift=35):
    """
    Shift class labels and copy dataset to new location with modified labels
    starting from specified number (default 35)
    """

    # Create new directory structure
    target_root = os.path.join(os.path.dirname(source_root), 'shifted35_robo')
    os.makedirs(target_root, exist_ok=True)

    # Counter for processed files
    processed_images = 0
    processed_labels = 0

    print(f"Starting to process dataset...")
    print(f"Source directory: {source_root}")
    print(f"Target directory: {target_root}")
    print(f"Class shift: {class_shift}")

    # Process all image files
    image_files = glob.glob(os.path.join(source_root, '*.jpg'))
    total_files = len(image_files)

    for img_path in image_files:
        img_name = Path(img_path).stem
        label_path = os.path.join(source_root, f"{img_name}.txt")

        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_name}")
            continue

        # Copy image file
        new_img_path = os.path.join(target_root, f"{img_name}.jpg")
        shutil.copy2(img_path, new_img_path)
        processed_images += 1

        # Read and modify label file
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    # Shift class number
                    old_class = int(parts[0])
                    new_class = old_class + class_shift
                    parts[0] = str(new_class)
                    modified_lines.append(' '.join(parts))

            # Write modified label file
            new_label_path = os.path.join(target_root, f"{img_name}.txt")
            with open(new_label_path, 'w') as f:
                for line in modified_lines:
                    f.write(f"{line}\n")
            processed_labels += 1

            # Progress update
            if processed_images % 10 == 0:
                print(f"Processed {processed_images}/{total_files} files...")

        except Exception as e:
            print(f"Error processing {label_path}: {e}")


    print("\nProcessing complete!")
    print(f"Images processed: {processed_images}")
    print(f"Labels processed: {processed_labels}")
    print(f"\nNew dataset location: {target_root}")
    print("\nClass mapping example:")
    print(f"Old class 0 (Amoxicillin) -> New class {class_shift}")
    print(f"Old class 1 (Antacill) -> New class {class_shift + 1}")
    print("...")
    print(f"Old class 29 (Volta) -> New class {class_shift + 29}")


if __name__ == "__main__":
    # Source dataset path
    source_root = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\robo"

    print("This script will:")
    print("1. Create a new 'shifted35_robo' directory")
    print("2. Copy all images")
    print("3. Modify labels to start from class 35")
    print("4. Create an updated data.yaml file")

    shift_labels_and_copy_dataset(source_root, class_shift=35)