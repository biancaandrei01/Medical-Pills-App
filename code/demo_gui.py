import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO

def load_model(checkpoint_path):
    return YOLO(checkpoint_path)

def inference(model, input_image_path):
    results = model.predict(source=input_image_path, conf=0.5, iou=0.45, device='cuda:0')
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
        self.model = None
        self.original_image = None
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.geometry("300x150")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        ttk.Button(frame, text="Load Image", command=self.open_image).grid(row=0, column=0, sticky="w", pady=5)


        self.image_label = ttk.Label(frame)
        self.image_label.grid(row=1, column=0, sticky="nsew", pady=5)
        self.image_label.grid_propagate(True)

        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ttk.Button(frame, text="Apply Model", command=self.load_and_apply_model).grid(row=2, column=0, sticky="w", pady=5)

        self.loading_label = ttk.Label(frame, text="Loading...")
        self.loading_label.grid(row=3, column=0, sticky="w", pady=5)
        self.loading_label.grid_remove()

        self.output_label = ttk.Label(frame, text="", wraplength=600, justify="left")
        self.output_label.grid(row=4, column=0, columnspan=4, sticky="w", pady=10)
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
        
        # Calculate window size based on image dimensions
        # Add padding for buttons and other widgets
        window_width = image_for_display.width + 40  # 20px padding on each side
        window_height = image_for_display.height + 150  # Extra space for buttons and labels
        
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
        self.loading_label.grid(row=4, column=0, sticky="ew", pady=5)
        self.root.update()

        checkpoint_path = r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\checkpoints\lab_yolo_Adam_augColGeoS\weights\best.pt"

        try:
            self.model = load_model(checkpoint_path)
            self.update_image()
        except Exception as e:
            print(f"Failed to load model: {e}")
        finally:
            self.loading_label.grid_remove()

    def update_image(self):
        try:
            results = inference(self.model, self.image_path)
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
