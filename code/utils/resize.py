import os
from PIL import Image


def resize_image(input_path, output_path):
    """
    Resize an image while maintaining aspect ratio.

    :param input_path: Path to the original image.
    :param output_path: Path to save the resized image.
    :param target_size: Desired maximum width or height to resize (while maintaining aspect ratio).
    """
    with Image.open(input_path) as img:
        # Get original dimensions
        original_width, original_height = img.size
        print(f"Original size: {original_width}x{original_height}")


        # Resize image using high-quality resampling (LANCZOS)
        resized_img = img.resize((800, 400), Image.LANCZOS)

        # Save the resized image with 95% quality
        resized_img.save(output_path, quality=100)
        print(f"Resized image saved as {output_path} with size: {800}x{400}")


def process_images(input_folder, output_folder):
    """
    Process all images in the input folder, resize them, and save to the output folder.

    :param input_folder: Folder containing the images to resize.
    :param output_folder: Folder where resized images will be saved.
    :param target_size: Maximum width or height for resizing.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image (you can extend this list if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path)


# Example usage:
input_folder = ''  # Add your path to the folder containing original images
output_folder = ''  # Add your path to the folder where resized images will be saved

process_images(input_folder, output_folder)
