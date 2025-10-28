import os, glob, shutil
def list_images(folder):
    paths = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
        paths += glob.glob(os.path.join(folder, ext))
    return sorted(paths)
def move_files(paths, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    out = []
    for p in paths:
        dst = os.path.join(dst_folder, os.path.basename(p))
        shutil.copy2(p, dst); out.append(dst)
    return out
