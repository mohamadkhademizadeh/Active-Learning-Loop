import argparse, json, os, shutil
from utils.ls_convert import convert
def write_yolo(out_dir, mapping, images_root, split='train'):
    img_out = os.path.join(out_dir, 'images', split)
    lab_out = os.path.join(out_dir, 'labels', split)
    os.makedirs(img_out, exist_ok=True); os.makedirs(lab_out, exist_ok=True)
    for img_name, boxes in mapping.items():
        src = os.path.join(images_root, img_name)
        if not os.path.exists(src): 
            print('missing image', src); continue
        shutil.copy2(src, os.path.join(img_out, img_name))
        stem,_ = os.path.splitext(img_name)
        with open(os.path.join(lab_out, stem+'.txt'),'w') as f:
            for (cls,cx,cy,bw,bh,_) in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--export_json', required=True)
    ap.add_argument('--images_root', default='data/unlabeled/images')
    ap.add_argument('--out_dir', default='data/labeled')
    ap.add_argument('--split', default='train')
    ap.add_argument('--classes', default='{"part":0,"defect":1}')
    args=ap.parse_args()
    klass=json.loads(args.classes)
    data=json.load(open(args.export_json,'r'))
    mapping=convert(data, klass)
    write_yolo(args.out_dir, mapping, args.images_root, split=args.split)
    print('Wrote labeled YOLO files into', args.out_dir)
