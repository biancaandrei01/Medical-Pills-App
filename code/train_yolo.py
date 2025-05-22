import torch
import wandb
from ultralytics import YOLO

if __name__ == "__main__":
    # Initialize your Weights and Biases project
    wandb.init(
        entity="transformers_3",
        project="Medical Pills App",  # Project name in wandb
        name="lab_robo_yolo_Adam_augCol",  # Name of the run
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
    model = YOLO("yolo11n.pt")
    model.to(device)

    # Start training
    results = model.train(
        # Path to the dataset.yaml file
        data=r'C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\splitted_lab_robo\data.yaml',
        # resume=True, # Continue training from checkpoint
        project="../checkpoints", # Directory where the model checkpoints will be saved
        name="lab_robo_yolo_Adam_augCol",  # Name of the training run folder
        device=device,
        verbose=True,  # Display training progress
        save=True,  # Save checkpoints after each epoch
        pretrained=True,  # Determines whether to start training from a pretrained model
        epochs=100,  # Number of epochs
        batch=10,  # set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
        workers=2,  # Number of data loading workers
        optimizer='Adam',  # Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration
        imgsz=800,
        close_mosaic=0,
        # Color space augumentations
        hsv_h=0, # nu modificam, e imp nuanta
        hsv_s=0.3, # putem modifica putin saturatia un 0.5 max (-0.5, 0.5)
        hsv_v=0.3, # putem modifica putin luminozitatea un 0.5 max (-0.5, 0.5)
        # Geometric transformations
        degrees=0, # putem pune 180 (img rotita intre -180 si 180)
        translate=0, # nu e de modificat, poate disparea pastila din img
        scale=0, # nu e de modificat, poate disparea pastila din img
        shear=0, # merge modificat
        perspective=0, # merge modificat
        flipud=0, # valoarea e prob de a fi flipped upside->down imaginea
        fliplr=0, # valoarea e prob de a fi flipped left->right imaginea
        mosaic=0, # nu e de modificat, noi avem o pastila per imagine
        erasing=0 # nu e de modificat, poate disparea pastila din img
    )

    # Finish the wandb run
    wandb.finish()

