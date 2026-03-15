from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(data="training.yaml", epochs=100, imgsz=1024)

    model = YOLO("runs/detect/train/weights/best.pt")
    model.predict("dataset/images/val", save=True, conf=0.05)

if __name__ == "__main__":
    main()
