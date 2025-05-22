from ultralytics import YOLO
import os

if __name__ == "__main__":
    # Load a model
    model = YOLO(r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\checkpoints\lab_yolo_Adam_augColGeoS\weights\best.pt")
    # Define the path to the test images folder
    test_images_folder = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\splitted_lab\images\val"

    # Get all image file paths in the test set folder (supports .jpg, .jpeg, .png)
    supported_formats = ('.JPG', '.jpeg', '.png')
    test_images = [os.path.join(test_images_folder, img) for img in os.listdir(test_images_folder) if
                   img.endswith(supported_formats)]

    # Run inference on each image in the test set
    for image_path in test_images:
        # Perform inference
        results = model.predict(source=image_path, conf=0.5, iou=0.45, device='cuda:0')

        # Process the results
        for result in results:
            # result.show()  # Display result on the screen (optional)
            # Save the result to disk (optional)
            output_filename = os.path.join(
                r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\inference",
                os.path.basename(image_path))
            result.save(filename=output_filename)

        print(f"Inference done for {image_path}")