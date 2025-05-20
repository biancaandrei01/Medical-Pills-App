import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms, models
from ultralytics import YOLO

class_names = [
        'Amoxicillin', 'Antacill', 'Burajel', 'CaRBon', 'Celebrex',
        'CentrumMultivitamin', 'Cetrizin', 'Chlorpheniramine', 'Crabocysteine',
        'Dafomin', 'ParacetamolDecolgen', 'Difelene', 'Doxycycline',
        'Fenafex', 'Flunarizine', 'Gaszym', 'Glutamax', 'HandyHerb',
        'Heromycin', 'IbumanPlus', 'Ibuprofen', 'Imodium', 'Lamotrigine',
        'MefenamicAcid', 'Miracid', 'Norxacin', 'Noxa', 'ParacetamolSara',
        'Simvastatin', 'Volta'
    ]

# Define the Combined Model
class YoloResNetClassifier(torch.nn.Module):
    def __init__(self, yolo_model_path, resnet_model_path, num_classes):
        super(YoloResNetClassifier, self).__init__()
        # Load the YOLO model (pretrained on object detection)
        self.yolo = YOLO(yolo_model_path)
        self.yolo.to('cuda')
        self.yolo.fuse()  # Fuse layers for faster inference

        # Load your pretrained ResNet classification model
        resnet = models.resnet50(pretrained=False)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)  # Update the final layer
        self.classifier = resnet
        self.classifier.load_state_dict(torch.load(resnet_model_path, map_location='cuda')['model_state_dict'])
        self.classifier.to('cuda')
        self.classifier.eval()

    def forward(self, image):
        """Perform a forward pass through YOLO and the ResNet classifier"""
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_crop = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        # YOLO Predictions (bounding boxes + features)
        results = self.yolo.predict(source=image, conf=0.5, iou=0.45, device='cuda', save=False)

        out = []
        for result in results:
            bboxes = result.boxes

            if result.boxes.shape[0] == 1:
                # Crop objects from image based on bounding boxes
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[0].tolist())  # Extract bounding box coordinates
                # Load and preprocess image
                img = Image.open(image).convert('RGB')
                img = transform(img)
                cropped_img = img[:, y1:y2, x1:x2]  # Crop bounding box region (assumes image is a tensor)
                cropped_img = transform_crop(cropped_img)

                # for plotting crops, you need to remove normalising
                # pl = cropped_img.cpu().numpy().transpose(1, 2, 0)
                # plt.figure(), plt.imshow(pl), plt.show()

                # Run cropped images through ResNet for classification
                cropped_images_tensor = cropped_img.unsqueeze(0).to('cuda')  # Convert list to tensor
                # Make prediction
                with torch.no_grad():
                    outputs = self.classifier(cropped_images_tensor)
                    _, predicted = torch.max(outputs, 1)
                    class_idx = predicted.item()

                out.append((class_names[int (bboxes.cls.cpu().numpy()[0])], class_names[class_idx]))  # Save bounding box and classification label
            elif result.boxes.shape[0] > 0:
                for box in result.boxes:
                    # Crop objects from image based on bounding boxes
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract bounding box coordinates
                    # Load and preprocess image
                    img = Image.open(image).convert('RGB')
                    img = transform(img)
                    cropped_img = img[:, y1:y2, x1:x2]  # Crop bounding box region (assumes image is a tensor)
                    cropped_img = transform_crop(cropped_img)

                    # for plotting crops, you need to remove normalising
                    # pl = cropped_img.cpu().numpy().transpose(1, 2, 0)
                    # plt.figure(), plt.imshow(pl), plt.show()

                    # Run cropped images through ResNet for classification
                    cropped_images_tensor = cropped_img.unsqueeze(0).to('cuda')  # Convert list to tensor
                    # Make prediction
                    with torch.no_grad():
                        outputs = self.classifier(cropped_images_tensor)
                        _, predicted = torch.max(outputs, 1)
                        class_idx = predicted.item()

                    out.append((class_names[int(box.cls.cpu().numpy()[0])],
                                class_names[class_idx]))  # Save bounding box and classification label
            else:
                out.append(("No Detection", "-"))

        return out


if __name__ == "__main__":
    # Paths to the models and dataset
    yolo_model_path = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\checkpoints\yolo_exp\robo_yolo_Adam_augColGeoS\weights\best.pt"
    resnet_model_path = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\checkpoints\classificator_exp\robo_resNet_augColGeoS_best.pth"  # Replace with your ResNet model path
    test_images_folder = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\splitted_robo\images\val"

    # Initialize the combined model
    num_classes = 30  # Replace with the number of classes in your dataset
    combined_model = YoloResNetClassifier(yolo_model_path, resnet_model_path, num_classes)
    yolo_good_cls = []
    yolo_bad_cls = []
    resnet_good_cls = []
    resnet_bad_cls = []

    # Loop through test images for inference
    for image in os.listdir(test_images_folder):
        # Run inference with the combined model
        outputs = combined_model(os.path.join(test_images_folder, image))

        # Postprocessing and results
        for yolo_label, resnet_label in outputs:

            if yolo_label == image.split("_")[0]:
                yolo_good_cls.append(image)
            elif (yolo_label != image.split("_")[0]) and (yolo_label != "No Detection"):
                yolo_bad_cls.append(image)
            if resnet_label == image.split("_")[0]:
                resnet_good_cls.append(image)
            elif (resnet_label != image.split("_")[0]) and (resnet_label != "-"):
                resnet_bad_cls.append(image)

            print(f"Yolo: {yolo_label}, Clasificator: {resnet_label}")
    print(f">>> Good classification for YOLO: {np.size(yolo_good_cls)}/{np.size(os.listdir(test_images_folder))}")
    print(f">>> Bad classification for YOLO: {np.size(yolo_bad_cls)}/{np.size(os.listdir(test_images_folder))}")
    print(f">>> Good classification for ResNet: {np.size(resnet_good_cls)}/{np.size(os.listdir(test_images_folder))}")
    print(f">>> Bad classification for ResNet: {np.size(resnet_bad_cls)}/{np.size(os.listdir(test_images_folder))}\n")
    print(f"Bad classified ResNet files: {resnet_bad_cls}")

