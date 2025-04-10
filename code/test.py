from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("C:\\Users\\Bianca\\PycharmProjects\\Medical-Pills-App\\checkpoints\\test\\weights\\best.pt")
    results = model.val(data='C:\\Users\\Bianca\\PycharmProjects\\Medical-Pills-App\\datasets\\pills-dataset\\data.yaml', split='val')
