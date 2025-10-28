import os, json, numpy as np
from sklearn.cluster import KMeans
import cv2
def color_hist_embedding(img_path, bins=16):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return np.zeros((bins*3,), dtype=np.float32)
    feats=[]
    for ch in range(3):
        hist = cv2.calcHist([img],[ch],None,[bins],[0,256]).flatten()
        hist = hist / (hist.sum()+1e-6); feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0)
def image_uncertainty(pred_json, min_conf=0.25):
    if not pred_json or ('boxes' not in pred_json) or len(pred_json['boxes'])==0:
        return 1.0
    confs = [float(b['conf']) for b in pred_json['boxes'] if float(b['conf'])>=min_conf]
    if not confs: return 1.0
    return float(1.0 - max(confs))
def select_batch(pred_dir, k=20, diversity_weight=0.3, min_conf=0.25, hist_bins=16):
    files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    if not files: return []
    infos=[]; paths=[]
    for jf in files:
        p = os.path.join(pred_dir, jf)
        meta = json.load(open(p,'r'))
        img_path = meta.get('image_path') or meta.get('path') or meta.get('im_path') or ""
        u = image_uncertainty(meta, min_conf=min_conf)
        infos.append((img_path, u)); paths.append(img_path)
    if diversity_weight>0:
        X = np.array([color_hist_embedding(p, bins=hist_bins) for p,_ in infos])
        if len(X)>=k:
            kmeans=KMeans(n_clusters=k, n_init=5, random_state=7).fit(X)
            chosen=[]
            for c in range(k):
                idxs = np.where(kmeans.labels_==c)[0]
                if len(idxs)==0: continue
                best = sorted(idxs, key=lambda i: infos[i][1], reverse=True)[0]
                chosen.append(infos[best][0])
            if len(chosen)<k:
                rest = sorted([p for p,_ in infos if p not in chosen], key=lambda p: dict(infos)[p], reverse=True)
                chosen += rest[:(k-len(chosen))]
            return chosen[:k]
    infos_sorted = sorted(infos, key=lambda t: t[1], reverse=True)
    return [p for p,_ in infos_sorted[:k]]
