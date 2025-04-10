import torch
import wandb
from ultralytics import YOLO

if __name__ == "__main__":
    # Initialize your Weights and Biases project
    wandb.init(
        entity="transformers_3",
        project="Medical Pills App",  # Project name in wandb
        name="poze_pastile1",  # Name of the run
        config={
            "epochs": 100,
            "batch_size": 10,
            "model": "yolo11n.pt",  # You can change this to other YOLOv8 variants
        }
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the pre-trained model
    model = YOLO("yolo11n.pt")
    model.to(device)

    # Start training
    results = model.train(
        # Path to the dataset.yaml file
        data='C:\\Users\\Bianca\\PycharmProjects\\Medical-Pills-App\\datasets\\splitted_poze_pastile_dataset\\data.yaml',
        epochs=100,  # Number of epochs
        batch=5,  # Batch size
        save=True,  # Save checkpoints after each epoch
        device=device,
        # Directory where the model checkpoints will be saved
        project="../checkpoints",
        name="poze_pastile1",  # Name of the training run folder
        optimizer='Adam',  # Optional: you can specify optimizer
        workers=2,  # Number of data loading workers
        verbose=True,  # Display training progress
        imgsz=800
    )

    # Finish the wandb run
    wandb.finish()

