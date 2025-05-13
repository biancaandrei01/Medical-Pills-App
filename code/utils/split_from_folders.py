import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    source_folder = r'C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\shifted35_robo'
    source_folder_2 = r'C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\lab'
    output_dir = r'C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\splitted_lab_robo'

    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    all_images = []
    all_images_2 = []
    all_labels = []

    images = glob(os.path.join(source_folder, '*.jpg'))
    all_images.extend(images)

    images_2 = glob(os.path.join(source_folder_2, '*.jpg'))
    all_images_2.extend(images_2)

    if not all_images:
        raise ValueError("No images found! Check source folder paths and extensions.")

    for img in all_images:
        basename = os.path.basename(img)
        label = basename.split('_')[0]  # Extract label from filename !!! [1] for Lab dataset
        all_labels.append(label)

    for img in all_images_2:
        basename = os.path.basename(img)
        label_2 = basename.split('_')[1]  # Extract label from filename !!! [0] for RoboFlow dataset
        all_labels.append(label_2)

    all_images.extend(all_images_2)

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