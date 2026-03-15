from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")   

    model.train( 
        data="dataset/data.yaml",
        epochs=10,
        imgsz=640,
        device=0,
        batch=-1,
        workers=8,
        cache=True,     
        patience=10     
    )

if __name__ == "__main__":
    main()
