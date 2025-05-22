import os
import cv2
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
INPUT_BASE = r'C:\Users\Sebi\Desktop\train_robo\splitted_lab'
OUTPUT_BASE = r'C:\Users\Sebi\Desktop\train_robo\preprocessed_lab_pur'
YAML_PATH = os.path.join(INPUT_BASE, 'data.yaml')

# Create output directories
os.makedirs(os.path.join(OUTPUT_BASE, 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, 'test'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, 'csv'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, 'plots'), exist_ok=True)

# Load class names from YAML
with open(YAML_PATH, 'r') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']


def process_split(split_type):
    df = pd.DataFrame(columns=['filename', 'class'])
    image_dir = os.path.join(INPUT_BASE, 'images', split_type)
    label_dir = os.path.join(INPUT_BASE, 'labels', split_type)

    for image_file in tqdm(os.listdir(image_dir), desc=f'Processing {split_type}'):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        base_name = os.path.splitext(image_file)[0]

        # Load corresponding label
        label_path = os.path.join(label_dir, f'{base_name}.txt')
        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h

            # Calculate coordinates
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # Crop image
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            # Save cropped image
            output_name = f'{base_name}_{idx}.jpg'
            output_path = os.path.join(OUTPUT_BASE, 'test' if split_type == 'val' else 'train', output_name)
            cv2.imwrite(output_path, cropped)

            # Add to DataFrame with numeric class ID
            df = pd.concat([df, pd.DataFrame([{
                'filename': output_name,
                'class': class_id  # Changed from class_names[class_id]
            }])], ignore_index=True)

    return df


# Process both splits
train_df = process_split('train')
test_df = process_split('val')

# Save CSVs
train_df.to_csv(os.path.join(OUTPUT_BASE, 'csv', 'train.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_BASE, 'csv', 'test.csv'), index=False)

# Create class distribution
combined_df = pd.concat([train_df, test_df])
class_dist = combined_df['class'].value_counts().sort_index()

# Save distribution plot
plt.figure(figsize=(12, 6))
class_dist.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class ID')
plt.ylabel('Count')
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, 'plots', 'class_distribution.png'))
plt.close()

# Save distribution report
with open(os.path.join(OUTPUT_BASE, 'plots', 'class_distribution.txt'), 'w') as f:
    f.write("Class Distribution Report\n")
    f.write("=========================\n")
    f.write("Class ID\tClass Name\tCount\n")
    for class_id, count in class_dist.items():
        f.write(f"{class_id}\t\t{class_names[class_id]}\t\t{count}\n")

print("Processing completed successfully!")
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Class distribution saved in output directory")