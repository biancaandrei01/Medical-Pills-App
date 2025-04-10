import cv2
import matplotlib.pyplot as plt


def draw_yolo_labels(image_path, labels_path, output_path=None):
    # Load image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Read labels
    with open(labels_path, 'r') as file:
        labels = file.readlines()

    # Iterate over each label
    for label in labels:
        label_parts = label.strip().split()
        x_center, y_center, width, height = map(float, label_parts[1:])

        # Convert YOLO format to pixel values
        x_center_pixel = int(x_center * image_width)
        y_center_pixel = int(y_center * image_height)
        width_pixel = int(width * image_width)
        height_pixel = int(height * image_height)

        # Calculate top-left and bottom-right coordinates
        top_left = (x_center_pixel - width_pixel // 2, y_center_pixel - height_pixel // 2)
        bottom_right = (x_center_pixel + width_pixel // 2, y_center_pixel + height_pixel // 2)

        # Draw rectangle on the image
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 255), 2)

    # Save or display the image
    if output_path:
        cv2.imwrite(output_path, image)
    else:
        # cv2.imshow('Image with YOLO Labels', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plt.figure(), plt.imshow(image), plt.show()


# Example usage
draw_yolo_labels('C:\\Users\\Bianca\\PycharmProjects\\Medical-Pills-App\\datasets\\poze_pastile_dataset\\images\\DSC02374.JPG',
                 'C:\\Users\\Bianca\\PycharmProjects\\Medical-Pills-App\\datasets\\poze_pastile_dataset\\labels\\DSC02374.txt') # add your image and label path
