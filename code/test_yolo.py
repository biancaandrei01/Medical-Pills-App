from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Users\Bianca\PycharmProjects\Medical-Pills-App\checkpoints\lab_yolo_Adam_augColGeoS\weights\best.pt")
    results = model.val(data=r'C:\Users\Bianca\PycharmProjects\Medical-Pills-App\datasets\splitted_lab\data.yaml', split='val')
