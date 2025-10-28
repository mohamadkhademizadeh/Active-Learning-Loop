import streamlit as st, os, subprocess
from utils.dataset import list_images
st.set_page_config(page_title="Active Learning Loop", layout="wide")
st.title("🔁 Active Learning Loop — YOLOv8")
st.header("1) Unlabeled pool"); imgs=list_images('data/unlabeled/images'); st.write(f"{len(imgs)} images")
if st.button("Run predictions"):
    cmd="python scripts/predict.py --images data/unlabeled/images --out data/unlabeled/preds"
    st.code(cmd); st.write(subprocess.getoutput(cmd))
st.header("2) Select batch"); k=st.number_input("k",1,1000,20)
if st.button("Select"):
    cmd=f"python scripts/select_batch.py --preds data/unlabeled/preds --k {int(k)} --out data/selection.txt"
    st.code(cmd); st.write(subprocess.getoutput(cmd))
    if os.path.exists("data/selection.txt"): st.text(open("data/selection.txt").read())
st.header("3) Labeling"); st.write("Start webhook: `uvicorn api.webhook:app --port 8080` and POST Label Studio JSON.")
uploaded=st.file_uploader("Upload LS export JSON", type=['json'])
if uploaded:
    tmp="ls_export.json"; open(tmp,"wb").write(uploaded.read())
    cmd=f"python scripts/convert_labelstudio.py --export_json {tmp} --images_root data/unlabeled/images --out_dir data/labeled --split train"
    st.code(cmd); st.write(subprocess.getoutput(cmd))
st.header("4) Train"); epochs=st.number_input("epochs",1,300,20)
if st.button("Train YOLO"):
    cmd=f"python scripts/train.py --data_yaml scripts/yolo_data.yaml --epochs {int(epochs)}"
    st.code(cmd); st.write(subprocess.getoutput(cmd))
st.success("Loop ready. Repeat steps as needed.")
