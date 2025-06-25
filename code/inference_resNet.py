import os

import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from skimage import io
from torchvision import transforms, models

from utils import load_config


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

    return class_idx, class_names[class_idx]


if __name__ == "__main__":
    config = load_config.load_config()

    # for lab dataset
    # Image_path= os.path.join(config["datasets"]["lab_robo"],"val" , "3_Ibusinus_Sus_Alb_BecBlitz_0.jpg")
    # yaml_path= os.path.join(config["datasets"]["lab"], "data.yaml")
    # model_path= config["checkpoints"]["lab_resnet"]

    # for robo dataset
    Image_path = os.path.join(config["datasets"]["robo"], "val" , "ParacetamolDecolgen_15_0.jpg")
    yaml_path = os.path.join(config["datasets"]["robo"], "data.yaml")
    model_path = config["checkpoints"]["robo_resnet"]

    # for robo_lab dataset
    # Image_path = os.path.join(config["datasets"]["lab_robo"], "val" , "Gaszym_4_0.jpg")
    # yaml_path = os.path.join(config["datasets"]["lab_robo"], "data.yaml")
    # model_path = config["checkpoints"]["lab_robo_resnet"]

    path = os.path.join(config["datasets"]["robo"], "val")
    val_images = os.listdir(path)

    results=[]
    for img_path in val_images[:]:
        img = io.imread(os.path.join(path, img_path))
        class_id, class_name = predict(os.path.join(path, img_path), model_path, yaml_path)
        results.append({"img_path": img_path, "class_name": class_name})

    df = pd.DataFrame(results)
    df.to_csv("name_predictions.csv", index=False)
    try:
        class_id, class_name = predict(Image_path,  model_path, yaml_path)
        print(f"Predicted Class ID: {class_id}")
        print(f"Predicted Class Name: {class_name}")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        exit(1)