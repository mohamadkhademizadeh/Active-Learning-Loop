import argparse, os, json
from ultralytics import YOLO
def run(weights, images, out):
    os.makedirs(out, exist_ok=True)
    model = YOLO(weights if (weights and os.path.exists(weights)) else 'yolov8n.pt')
    res = model.predict(source=images, stream=True, save=False, conf=0.1, max_det=100)
    for r in res:
        boxes=[]
        if hasattr(r,'boxes') and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
                boxes.append({'xyxy':[float(x1),float(y1),float(x2),float(y2)], 'conf':float(c), 'cls':int(k)})
        meta={'image_path': getattr(r,'path',''), 'boxes': boxes}
        name = os.path.splitext(os.path.basename(getattr(r,'path','img')))[0]
        with open(os.path.join(out, f"{name}.json"), 'w') as f: json.dump(meta, f)
    print("Wrote predictions to", out)
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--images", default="data/unlabeled/images")
    ap.add_argument("--out", default="data/unlabeled/preds")
    args=ap.parse_args(); run(args.weights, args.images, args.out)
