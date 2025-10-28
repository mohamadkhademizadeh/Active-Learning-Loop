import argparse, os
from ultralytics import YOLO
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_yaml', default='scripts/yolo_data.yaml')
    ap.add_argument('--weights', default='yolov8n.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--project', default='runs/train')
    args=ap.parse_args()
    model=YOLO(args.weights)
    model.train(data=args.data_yaml, epochs=args.epochs, imgsz=args.imgsz, project=args.project)
    best=os.path.join(args.project,'weights','best.pt')
    if os.path.exists(best):
        os.makedirs('models', exist_ok=True)
        import shutil; shutil.copy2(best, os.path.join('models','best.pt'))
        print('Saved best weights to models/best.pt')
