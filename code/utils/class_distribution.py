import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np


def get_class_from_label(label_path):
    """Read the first number from label file which represents the class index"""
    with open(label_path, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            return int(first_line.split()[0])
    return None


def create_distribution_plot(dataset_root):
    """Create and save class distribution plot"""
    # Class names from the YAML (excluding Unknown class)
    class_names = [
        'Amoxicillin', 'Antacill', 'Burajel', 'Ca_R_Bon', 'Celebrex',
        'Centrum_Multivitamin', 'Cetrizin', 'Chlorpheniramine', 'Crabocysteine',
        'Dafomin', 'Decolgen_Paracetamol', 'Difelene', 'Doxycycline',
        'Fenafex', 'Flunarizine', 'Gaszym', 'Glutamax', 'HandyHerb',
        'Heromycin', 'Ibuman_plus', 'Ibuprofen', 'Imodium', 'Lamotrigine',
        'Mefenamic_acid', 'Miracid', 'Norxacin', 'Noxa', 'Sara_Paracetamol',
        'Simvastatin', 'Volta'
    ]

    # Dictionary to store class counts
    class_counts = Counter()

    # Process all label files
    label_files = glob.glob(os.path.join(dataset_root, '*.txt'))

    print("Analyzing dataset...")
    for label_path in label_files:
        class_idx = get_class_from_label(label_path)
        if class_idx is not None and class_idx < len(class_names):
            class_counts[class_names[class_idx]] += 1

    # Prepare data for plotting
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Create figure with larger size
    plt.figure(figsize=(15, 10))

    # Create bar plot
    colors = sns.color_palette("husl", len(classes))
    bars = plt.bar(range(len(classes)), counts, color=colors)

    # Customize plot
    plt.title('Class Distribution in Dataset', fontsize=16, pad=20)
    plt.xlabel('Class Names', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(dataset_root, 'class_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_path}")

    # Print statistics
    total_images = sum(counts)
    print("\nDataset Statistics:")
    print(f"Total number of images: {total_images}")
    print("\nClass distribution:")
    for class_name, count in class_counts.most_common():
        percentage = (count / total_images) * 100
        print(f"{class_name}: {count} images ({percentage:.1f}%)")

    # Calculate imbalance metrics
    min_count = min(counts)
    max_count = max(counts)
    mean_count = np.mean(counts)
    std_count = np.std(counts)

    print("\nImbalance Metrics:")
    print(f"Minimum class count: {min_count}")
    print(f"Maximum class count: {max_count}")
    print(f"Mean class count: {mean_count:.1f}")
    print(f"Standard deviation: {std_count:.1f}")
    print(f"Imbalance ratio (max/min): {max_count / min_count:.1f}")

    plt.show()


if __name__ == "__main__":
    # Replace this path with your dataset root directory
    dataset_root = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\robo"

    # Check if required packages are available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        create_distribution_plot(dataset_root)
    except ImportError:
        print("Error: Required packages not found. Please install matplotlib and seaborn:")
        print("pip install matplotlib seaborn")