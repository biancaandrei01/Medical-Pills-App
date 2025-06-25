from ultralytics import YOLO
import os.path

from utils import load_config


def test():
    config = load_config.load_config()

    model = YOLO(config["checkpoints"]["lab_yolo"])
    results = model.val(data=os.path.join(config["datasets"]["lab"], "data.yaml"), split='val')

if __name__ == "__main__":
    test()
