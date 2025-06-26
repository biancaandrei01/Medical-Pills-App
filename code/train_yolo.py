import os.path

import torch
import wandb
from ultralytics import YOLO
from utils import load_config


def train():
    config = load_config.load_config()

    # Initialize your Weights and Biases project
    wandb.init(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],  # Project name in wandb
        name=config["wandb"]["name"],  # Name of the run
        config={
            "dataset": "lab+robo",
            "model": "yolo11n.pt",
            "pretrained": True,
            "epochs": 100,
            "batch_size": 10,
            "workers": 2,
            "optimizer": "Adam",
            "img_size": 800,
            "augmentation": "hsv_s=0.3, hsv_v=0.3"
        }
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the pre-trained model
    model = YOLO(config["models"]["yolo"])
    model.to(device)

    # Start training
    results = model.train(
        # Path to the dataset.yaml file
        data=os.path.join(config["datasets"]["lab_robo"], "data.yaml"),
        # resume=True, # Continue training from checkpoint
        project=config["checkpoints"]["root"],  # Directory where the model checkpoints will be saved
        name=config["wandb"]["name"],  # Name of the training run folder
        device=device,
        verbose=True,  # Display training progress
        save=True,  # Save checkpoints after each epoch
        pretrained=True,  # Determines whether to start training from a pretrained model
        epochs=100,  # Number of epochs
        batch=10,
        # set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
        workers=2,  # Number of data loading workers
        optimizer='Adam',
        # Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration
        imgsz=800,
        close_mosaic=0,
        # --- HSV Color-Space Augmentation ---
        hsv_h=0,  # Hue modification (disabled, as color is important)
        hsv_s=0.3,  # Saturation modification (e.g., up to +/- 0.5)
        hsv_v=0.3,  # Value (brightness) modification (e.g., up to +/- 0.5)
        # --- Geometric Augmentation ---
        degrees=0,  # Image rotation (+/- 180 degrees)
        translate=0,  # Image translation (disabled, could lose the pill)
        scale=0,  # Image scaling (disabled, could lose the pill)
        shear=0,  # Shear transformation (can be enabled)
        perspective=0,  # Perspective transformation (can be enabled)
        flipud=0,  # Probability of vertical flip (upside-down)
        fliplr=0,  # Probability of horizontal flip (left-right)
        mosaic=0,  # Mosaic (disabled, only one pill per image)
        erasing=0  # Random erasing (disabled, could remove the pill)
    )

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    train()