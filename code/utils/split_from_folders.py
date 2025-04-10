import os
import shutil
import random
from glob import glob

if __name__ == "__main__":
    # Define the source folders with your images and labels
    source_folders = ['C:\\Users\\Bianca\\PycharmProjects\\Medical-Pills-App\\datasets\\poze_pastile_dataset']

    # Define the output folders for train, test, and validation splits
    output_dir = 'C:\\Users\\Bianca\\PycharmProjects\\Medical-Pills-App\\datasets\\splitted_poze_pastile_dataset'
    # Create directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

    # Define split ratios (e.g., 70% train, 20% validation, 10% test)
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # Collect all images and their corresponding label files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    all_images = []

    for folder in source_folders:
        for ext in image_extensions:
            all_images.extend(glob(os.path.join(folder, 'images', ext)))

    # Shuffle the images to ensure random split
    random.shuffle(all_images)

    # Split the dataset into train, validation, and test sets
    total_images = len(all_images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    train_images = all_images[:train_count]
    val_images = all_images[train_count:train_count + val_count]
    test_images = all_images[train_count + val_count:]

    for image_file in train_images:
        # Get the corresponding label file (replace image extension with .txt)
        label_path = os.path.splitext(image_file)[0] + '.txt'
        label_path = label_path.replace('images', 'labels')

        # Copy image and label to the train folder
        shutil.copy(image_file, os.path.join(output_dir, 'images', 'train'))
        shutil.copy(label_path, os.path.join(output_dir, 'labels', 'train'))

    for image_file in val_images:
        # Get the corresponding label file (replace image extension with .txt)
        label_path = os.path.splitext(image_file)[0] + '.txt'
        label_path = label_path.replace('images', 'labels')

        # Copy image and label to the train folder
        shutil.copy(image_file, os.path.join(output_dir, 'images', 'val'))
        shutil.copy(label_path, os.path.join(output_dir, 'labels', 'val'))


    for image_file in test_images:
        # Get the corresponding label file (replace image extension with .txt)
        label_path = os.path.splitext(image_file)[0] + '.txt'
        label_path = label_path.replace('images', 'labels')

        # Copy image and label to the train folder
        shutil.copy(image_file, os.path.join(output_dir, 'images', 'test'))
        shutil.copy(label_path, os.path.join(output_dir, 'labels', 'test'))

    print(f"Dataset created! Train: {train_count}, Val: {val_count}, Test: {test_count}")
