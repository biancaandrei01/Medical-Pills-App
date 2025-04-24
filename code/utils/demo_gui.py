import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO

#todo load_model for other models
def load_model(checkpoint_path, model_type):
    if model_type == 'YOLO':
        return YOLO(checkpoint_path)
    elif model_type == '?Sebi':
        return None
    elif model_type == '?Adelina':
        return None

#todo inference for other models
def inference(model_type, model, input_image_path):
    if model_type == 'YOLO':
        results = model.predict(source=input_image_path, conf=0.5, iou=0.45, device='cuda:0')
        return results
    elif model_type == '?Sebi':
        return None
    elif model_type == '?Adelina':
        return None


class MedicalPillsApp:
    def __init__(self, root):
        self.output_label = None
        self.image_label = None
        self.photo = None
        self.output_image = None
        self.loading_label = None
        self.root = root
        self.root.title("Medical Pills App")
        self.model_type = tk.StringVar(value="YOLO")
        self.model = None
        self.original_image = None
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.grid(row=0, column=0, sticky="nsew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        ttk.Button(frame, text="Load Image", command=self.open_image).grid(row=0, column=0, sticky="w", pady=5)


        self.image_label = ttk.Label(frame)
        self.image_label.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=5)

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)
        frame.grid_columnconfigure(3, weight=1)

        ttk.Label(frame, text="Model Type:").grid(row=2, column=0, sticky="w", pady=5)
        for i, model in enumerate(['YOLO', '?Sebi', '?Adelina']):
            ttk.Radiobutton(frame, text=model, variable=self.model_type, value=model).grid(row=2, column=i + 1, sticky="ew", pady=5)

        ttk.Button(frame, text="Apply Model", command=self.load_and_apply_model).grid(row=3, column=0, sticky="w", pady=5)

        self.loading_label = ttk.Label(frame, text="Loading...")
        self.loading_label.grid(row=4, column=0, sticky="w", pady=5)
        self.loading_label.grid_remove()

        self.output_label = ttk.Label(frame, text="", wraplength=600, justify="left")
        self.output_label.grid(row=5, column=0, columnspan=4, sticky="w", pady=10)
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
        image_for_display.thumbnail(max_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image_for_display)
        self.image_label.configure(image=self.photo)
        self.image_label.image = self.photo  # keep a reference

    def load_and_apply_model(self):
        if not self.original_image:
            return
        self.loading_label.grid(row=4, column=0, sticky="ew", pady=5)
        self.root.update()

        model_type = self.model_type.get()
        #todo edit checkpoint_path
        checkpoint_path = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\checkpoints\pastile_complet\weights\best.pt"

        try:
            self.model = load_model(checkpoint_path, model_type)
            self.update_image()
        except Exception as e:
            print(f"Failed to load model: {e}")
        finally:
            self.loading_label.grid_remove()

    def update_image(self):
        try:
            # todo process results different for other models
            results = inference(self.model_type.get(), self.model, self.image_path)
            image = self.original_image.copy()
            draw = ImageDraw.Draw(image)

            output_lines = []
            for result in results:
                boxes = result.boxes
                names = result.names

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{names[cls_id]}: {conf:.2f}"

                    draw.rectangle([x1, y1, x2, y2], outline='green', width=10)
                    output_lines.append(label)

            self.display_image(image)

            if output_lines:
                label_text = "\n".join(output_lines)
                self.output_label.config(text=label_text)
                self.output_label.grid()
            else:
                self.output_label.grid_remove()

        except Exception as e:
            print(f"Failed to apply model: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalPillsApp(root)
    root.mainloop()
