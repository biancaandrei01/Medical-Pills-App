import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    source_folder = r'D:\~Pastile\toate imaginile complete\toate imaginile complete'
    output_dir = r'D:\~Pastile\toate imaginile complete'

    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    all_images = []
    all_labels = []

    for ext in image_extensions:
        images = glob(os.path.join(source_folder, ext))
        all_images.extend(images)

    if not all_images:
        raise ValueError("No images found! Check source folder paths and extensions.")

    for img in all_images:
        basename = os.path.basename(img)
        label = basename.split('_')[1]  # Extract label from filename
        all_labels.append(label)

    images_train, images_val, labels_train, labels_val = train_test_split(
        all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42)

    def copy_files(image_list, split_name):
        for image_file in image_list:
            shutil.copy(image_file, os.path.join(output_dir, 'images', split_name))
            label_file = os.path.splitext(image_file)[0] + '.txt'
            if os.path.exists(label_file):
                shutil.copy(label_file, os.path.join(output_dir, 'labels', split_name))

    copy_files(images_train, 'train')
    copy_files(images_val, 'val')

    print(f"Dataset created! Train: {len(images_train)}, Val: {len(images_val)}")