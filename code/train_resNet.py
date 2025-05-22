import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import wandb
import seaborn as sns
import yaml
import cv2

# Configurații
DATA_DIR = r"C:\Users\Sebi\Desktop\train_robo\preprocessed_lab_pur"
YAML_PATH = os.path.join(r"C:\Users\Sebi\Desktop\train_robo\splitted_lab", 'data.yaml')
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
IMG_SIZE = 224

# Încarcă numele claselor din YAML
with open(YAML_PATH, 'r') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']
NUM_CLASSES = len(class_names)


# Transformare HSV personalizată
class HSVTransform:
    def __call__(self, img):
        img = np.array(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = hsv[..., 1] * 0.3  # Saturation
        hsv[..., 2] = hsv[..., 2] * 0.3  # Value
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(img)


# Configurare Wandb
config = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "image_size": IMG_SIZE,
    "num_classes": NUM_CLASSES,
    "augmentation": "Full: Rotation(180)+Flip+HSV(s=0.3,v=0.3)"
}

wandb.init(
    entity="transformers_3",
    project="Medical Pills App",
    name="ResNet-50-Lab-Full-Aug",
    config=config
)


class RoboDataset(Dataset):
    def __init__(self, csv_path, split_type, transform=None):
        self.df = pd.read_csv(csv_path)
        self.split_type = split_type
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        label = self.df.iloc[idx]['class']

        img_path = os.path.join(DATA_DIR, self.split_type, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def train():
    # Transformări cu toate augmentările
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomApply([HSVTransform()], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Încarcă date
    train_dataset = RoboDataset(
        csv_path=os.path.join(DATA_DIR, 'csv', 'train.csv'),
        split_type='train',
        transform=train_transform
    )

    test_dataset = RoboDataset(
        csv_path=os.path.join(DATA_DIR, 'csv', 'test.csv'),
        split_type='test',
        transform=val_transform
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # Optimizer și Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # Validation loop
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculează metrici
        train_loss = train_loss / len(train_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        scheduler.step(f1)

        # Logare Wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Salvare model
        if f1 > best_f1:
            best_f1 = f1
            model_path = os.path.join(DATA_DIR, 'best_robo_model_full_aug.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, model_path)
            wandb.save(model_path)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print("-" * 50)

    # Evaluare finală
    checkpoint = torch.load(os.path.join(DATA_DIR, 'best_robo_model_full_aug.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Full Augmentation')
    wandb.log({"Confusion Matrix": wandb.Image(plt)})
    plt.close()

    # Raport clasificare
    # class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    # wandb.log({"Classification Report": class_report})

    wandb.finish()


if __name__ == "__main__":
    train()