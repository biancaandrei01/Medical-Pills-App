import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
import torch
from utils import load_config

def load_model(checkpoint_path):
    yolo_path = checkpoint_path.get("YOLO")
    yolo = YOLO(yolo_path)
    yolo.to('cuda')
    yolo.fuse()

    resNet_path = checkpoint_path.get("ResNet")
    resNet = models.resnet50(weights = None)
    resNet.fc = nn.Linear(resNet.fc.in_features, checkpoint_path.get("num_classes"))  # Update the final layer
    resNet.load_state_dict(torch.load(resNet_path, map_location='cuda')['model_state_dict'])
    resNet.to('cuda')
    resNet.eval()
    return yolo, resNet

def inference(yolo, input_image_path):
    results = yolo.predict(source=input_image_path, conf=0.5, iou=0.45, device='cuda', save=False)
    return results


class MedicalPillsApp:
    def __init__(self, root):
        self.output_label = None
        self.image_label = None
        self.photo = None
        self.output_image = None
        self.loading_label = None
        self.root = root
        self.root.title("Medical Pills App")
        self.yolo = None
        self.resNet = None
        self.original_image = None
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.geometry("300x150")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        ttk.Button(frame, text="Load Image", command=self.open_image).grid(row=0, column=0, sticky="w")

        self.image_label = ttk.Label(frame)
        self.image_label.grid(row=1, column=0, sticky="nsew", pady=5)

        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        config = load_config.load_config()

        self.experiment_checkpoints = {
            "lab dataset": {
                "YOLO": config["checkpoints"]["lab_yolo"],
                "ResNet" : config["checkpoints"]["lab_resnet"],
                "num_classes": config["num_classes"]["lab"]
            },
            "robo dataset": {
                "YOLO": config["checkpoints"]["robo_yolo"],
                "ResNet": config["checkpoints"]["robo_resnet"],
                "num_classes": config["num_classes"]["robo"]
            },
            "lab & robo dataset": {
                "YOLO": config["checkpoints"]["lab_robo_yolo"],
                "ResNet": config["checkpoints"]["lab_robo_resnet"],
                "num_classes": config["num_classes"]["lab_robo"]
            }
        }

        self.selected_experiment = tk.StringVar()
        self.selected_experiment.set("lab & robo dataset")

        ttk.Label(frame, text="Select dataset used for training model:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Combobox(frame, textvariable=self.selected_experiment,
                     values=list(self.experiment_checkpoints.keys())).grid(row=3, column=0, sticky="w", pady=5)

        ttk.Button(frame, text="Apply Model", command=self.load_and_apply_model).grid(row=4, column=0, sticky="w",
                                                                                      pady=5)

        self.loading_label = ttk.Label(frame, text="Loading...")
        self.loading_label.grid(row=5, column=0, sticky="w", pady=5)
        self.loading_label.grid_remove()

        self.output_label = ttk.Label(frame, text="", wraplength=600, justify="left")
        self.output_label.grid(row=6, column=0, columnspan=4, sticky="w", pady=10)
        self.output_label.grid_remove()

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.output_label.grid_remove()
            self.image_path = file_path
            self.original_image = Image.open(file_path).convert("RGB")
            self.display_image(self.original_image)

    def display_image(self, image):
        max_size = (800, 600)
        image_for_display = image.copy()
        image_for_display.thumbnail(size=max_size, resample=Image.Resampling.BICUBIC, reducing_gap=2.0)
        self.photo = ImageTk.PhotoImage(image_for_display)
        self.image_label.configure(image=self.photo)
        self.image_label.image = self.photo  # keep a reference

        # Calculate window size based on image dimensions
        # Add padding for buttons and other widgets
        window_width = image_for_display.width + 40  # 20px padding on each side
        window_height = image_for_display.height + 200  # Extra space for buttons and labels

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Ensure window size doesn't exceed screen size
        window_width = min(window_width, screen_width)
        window_height = min(window_height, screen_height)

        # Calculate center position
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")


    def load_and_apply_model(self):
        if not self.original_image:
            return
        self.output_label.grid_remove()
        self.loading_label.grid(row=5, column=0, sticky="ew", pady=5)
        self.root.update()

        checkpoint_key = self.selected_experiment.get()
        checkpoint_path = self.experiment_checkpoints.get(checkpoint_key)

        try:
            self.yolo, self.resNet = load_model(checkpoint_path)
            self.update_image()
        except Exception as e:
            print(f"Failed to load model: {e}")
        finally:
            self.loading_label.grid_remove()

    def update_image(self):
        try:
            # Define transformations
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            transform_crop = transforms.Compose([
                transforms.Resize((224, 224)),
            ])

            results = inference(self.yolo, self.image_path)
            image = self.original_image.copy()
            draw = ImageDraw.Draw(image)

            output_lines = []
            for result in results:
                boxes = result.boxes
                names = result.names

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract bounding box coordinates
                    img = transform(image)
                    cropped_img = img[:, y1:y2, x1:x2]  # Crop bounding box region (assumes image is a tensor)
                    cropped_img = transform_crop(cropped_img)
                    # for plotting crops, you need to remove normalising
                    # pl = cropped_img.cpu().numpy().transpose(1, 2, 0)
                    # plt.figure(), plt.imshow(pl), plt.show()

                    # Run cropped images through ResNet for classification
                    cropped_images_tensor = cropped_img.unsqueeze(0).to('cuda')  # Convert list to tensor
                    # Make prediction
                    with torch.no_grad():
                        outputs = self.resNet(cropped_images_tensor)
                        _, predicted = torch.max(outputs, 1)
                        class_idx = predicted.item()

                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"YOLO >> Class: {names[cls_id]}, Confidence: {conf:.2f}\n\nResNet >> Class: {names[class_idx]}"

                    line_width = max(1, int(min(image.width,
                                                image.height) * 0.001))  # Adapt line width based on image size
                    draw.rectangle([x1, y1, x2, y2], outline='green', width=line_width)
                    output_lines.append(label)

            self.display_image(image)

            if output_lines:
                label_text = "\n".join(output_lines)
                self.output_label.config(text=label_text)
                self.output_label.grid()
            else:
                self.output_label.config(text="No pills detected. Try a different image or experiment.")
                self.output_label.grid()

        except Exception as e:
            print(f"Failed to apply model: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalPillsApp(root)
    root.mainloop()
