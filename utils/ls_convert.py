# Convert Label Studio object detection export JSON to YOLOv5/v8 TXT
import json, os
def to_yolo_bbox(w, h, x1, y1, x2, y2):
    bw = x2 - x1; bh = y2 - y1
    cx = x1 + bw/2.0; cy = y1 + bh/2.0
    return cx/w, cy/h, bw/w, bh/h

def convert(ls_json, classes):
    """
    Convert Label Studio export JSON to a mapping usable for YOLO training.

    Args:
        ls_json (dict or list): Parsed Label Studio export (list of tasks or dict containing them).
        classes (dict): Mapping from label name to class id, e.g., {"part":0, "defect":1}.

    Returns:
        dict: {image_filename: [(cls_id, cx, cy, bw, bh, image_name), ...]}
    """
    out = {}
    tasks = ls_json if isinstance(ls_json, list) else ls_json.get('tasks') or ls_json.get('data') or []
    for t in tasks:
        img = t.get('data', {}).get('image') or t.get('image') or t.get('file') or t.get('img') or ""
        image_name = os.path.basename(img)
        anns = t.get('annotations') or t.get('result') or t.get('completions') or []
        boxes = []
        for a in anns:
            results = a.get('result') or a.get('value') or []
            if isinstance(results, dict): results=[results]
            for r in results:
                if {'x','y','width','height'}.issubset(r.keys()):
                    W = t.get('width') or r.get('original_width') or r.get('image_width') or 0
                    H = t.get('height') or r.get('original_height') or r.get('image_height') or 0
                    if W==0 or H==0: continue
                    x = float(r['x'])/100.0 * W
                    y = float(r['y'])/100.0 * H
                    w = float(r['width'])/100.0 * W
                    h = float(r['height'])/100.0 * H
                    cls_name = (r.get('rectanglelabels') or r.get('labels') or ['object'])[0]
                    cls_id = classes.get(cls_name, 0)
                    cx, cy, bw, bh = to_yolo_bbox(W,H,x,y,x+w,y+h)
                    boxes.append((cls_id, cx, cy, bw, bh, image_name))
        if boxes:
            out[image_name]=boxes
    return out
