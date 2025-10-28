import argparse, yaml
from utils.selection import select_batch
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--preds", default="data/unlabeled/preds")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--diversity_weight", type=float, default=None)
    ap.add_argument("--min_conf", type=float, default=None)
    ap.add_argument("--hist_bins", type=int, default=None)
    ap.add_argument("--out", default="data/selection.txt")
    ap.add_argument("--config", default="configs/active.yaml")
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config,'r'))
    k = args.k or cfg['selection']['k']
    dw = cfg['selection']['diversity_weight'] if args.diversity_weight is None else args.diversity_weight
    mc = cfg['selection']['min_confidence'] if args.min_conf is None else args.min_conf
    hb = cfg['selection']['hist_bins'] if args.hist_bins is None else args.hist_bins
    chosen=select_batch(args.preds, k=k, diversity_weight=dw, min_conf=mc, hist_bins=hb)
    with open(args.out,'w') as f:
        for p in chosen: f.write(p+"\n")
    print("Selected", len(chosen), "images ->", args.out)
