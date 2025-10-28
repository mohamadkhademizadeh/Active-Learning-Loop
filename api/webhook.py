from fastapi import FastAPI, Request
import os, shutil
from utils.ls_convert import convert
app = FastAPI(title="Active Learning Webhook")
@app.post("/labelstudio/webhook")
async def ls_webhook(request: Request):
    body = await request.json()
    classes = {"part":0,"defect":1}
    mapping = convert(body, classes)
    out_dir='data/labeled'; img_root='data/unlabeled/images'
    img_out=os.path.join(out_dir,'images','train')
    lab_out=os.path.join(out_dir,'labels','train')
    os.makedirs(img_out, exist_ok=True); os.makedirs(lab_out, exist_ok=True)
    for img_name, boxes in mapping.items():
        src=os.path.join(img_root, img_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(img_out, img_name))
        stem,_=os.path.splitext(img_name)
        with open(os.path.join(lab_out, stem+'.txt'),'w') as f:
            for (cls,cx,cy,bw,bh,_) in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return {"ok": True, "count": len(mapping)}
