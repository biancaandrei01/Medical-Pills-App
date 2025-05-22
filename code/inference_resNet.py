import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import argparse
import cv2
import numpy as np
import os
import pandas as pd
from skimage import io

def load_model(model_path, num_classes):
    # Initialize model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:  # For older model formats
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']


def preprocess_image(image_path):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def predict(image_path, model_path, yaml_path):
    # Load class names from YAML
    class_names = load_class_names(yaml_path)
    num_classes = len(class_names)

    # Load model
    model = load_model(model_path, num_classes)

    # Preprocess image
    input_tensor = preprocess_image(image_path)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
    # image = Image.open(image_path).convert("RGB")
    # plt.imshow(image)
    # plt.title(f"Predicted: {class_names[class_idx]}")
    # plt.axis('off')
    # plt.show()
    return class_idx, class_names[class_idx]


if __name__ == "__main__":

    #pt lab
    # Image_path= r'C:\Users\Sebi\Desktop\train_robo\preprocessed_lab_pur\test\3_Ibusinus_Sus_Alb_BecBlitz_0.jpg'
    # yaml_path= r'C:\Users\Sebi\Desktop\train_robo\splitted_lab\data.yaml'
    # model_path= r'C:\Users\Sebi\Desktop\train_robo\preprocessed_lab_pur\best_robo_model.pth'

    #pt robo
    Image_path = r'C:\Users\Sebi\Desktop\train_robo\preprocessed_robo_pur\test\ParacetamolDecolgen_15_0.jpg'
    yaml_path = r'C:\Users\Sebi\Desktop\train_robo\splitted_robo\data.yaml'
    model_path = r'C:\Users\Sebi\Desktop\train_robo\preprocessed_robo_pur\best_robo_model_full_aug.pth'

    path = r'C:\Users\Sebi\Desktop\train_robo\preprocessed_robo_pur\test'
    poze = os.listdir(path)

    results=[]
    for img_path in poze[:]:
        img = io.imread(os.path.join(path, img_path))
        class_id, class_name = predict(os.path.join(path, img_path), model_path, yaml_path)
        #print(img_path, " rezultat", class_id+1)
        results.append({"img_path": img_path, "class_name": class_name})
    #pt robo_lab
    # Image_path = r'C:\Users\Sebi\Desktop\train_robo\preprocessed_robo\test\Gaszym_4_0.jpg'
    # yaml_path = r'C:\Users\Sebi\Desktop\train_robo\splitted_lab_robo\data.yaml'
    # model_path = r'C:\Users\Sebi\Desktop\train_robo\preprocessed_robo\best_robo_model.pth'

    df = pd.DataFrame(results)
    df.to_csv("predictii_nume.csv", index=False)
    try:
        class_id, class_name = predict(Image_path,  model_path, yaml_path)
        print(f"Predicted Class ID: {class_id}")
        print(f"Predicted Class Name: {class_name}")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        exit(1)