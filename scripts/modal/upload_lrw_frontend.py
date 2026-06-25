"""Phase-A7 (ops, one-time): download the pretrained LRW lip-reading checkpoint and store it on the
shared volume at /mnt/pretrained/, then INSPECT it (auto-detect relu_type, confirm frontend3D+trunk
keys + dims). License: research / non-commercial (comparative-benchmark) — thesis use complies; cite
mpc001/Lipreading_using_Temporal_Convolutional_Networks.

Default = resnet18_mstcn_video (LRW 88.9%, 139MB). Scoped: writes only the NEW /mnt/pretrained/ dir.

  modal run scripts/modal/upload_lrw_frontend.py            # fetch + inspect (prints relu_type)
"""
import sys
from pathlib import Path

import modal

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import ML_TRAIN_IMAGE, get_volume

APP_NAME = "avsr-upload-lrw"
app = modal.App(APP_NAME)
volume = get_volume()
@app.function(image=ML_TRAIN_IMAGE, volumes={"/mnt": volume}, timeout=1800)
def fetch(file_id: str, dest: str):
    import os
    import subprocess
    import sys

    # gdown isn't in the base image; install at runtime (one-time fetch) to avoid an image rebuild
    # after add_local_* (which Modal forbids).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
    import gdown
    import torch

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"already present: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
    else:
        gdown.download(id=file_id, output=dest, quiet=False)
    print(f"saved {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")

    ckpt = torch.load(dest, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    keys = [k.replace("module.", "", 1) for k in sd.keys()]
    fe = [k for k in keys if k.startswith("frontend3D.")]
    tr = [k for k in keys if k.startswith("trunk.")]
    tcn = [k for k in keys if k.startswith("tcn")]
    # PReLU has learnable weights (frontend3D[2] activation, trunk ...reluN.weight); else relu/swish.
    prelu = [k for k in keys if k.endswith(("relu1.weight", "relu2.weight")) or k == "frontend3D.2.weight"]
    relu_type = "prelu" if prelu else "relu_or_swish (no activation params)"
    print(f"keys: total={len(keys)} frontend3D={len(fe)} trunk={len(tr)} tcn={len(tcn)}")
    print(f"relu_type → {relu_type} (prelu-marker keys={len(prelu)})")
    for probe in ("frontend3D.0.weight", "trunk.layer1.0.conv1.weight", "trunk.layer4.1.conv2.weight"):
        shp = next((tuple(v.shape) for k, v in sd.items() if k.replace("module.", "", 1) == probe), None)
        print(f"  {probe}: {shp}")
    volume.commit()
    return {"relu_type": relu_type, "frontend3D": len(fe), "trunk": len(tr), "tcn": len(tcn)}


@app.local_entrypoint()
def main(
    file_id: str = "1vqMpxZ5LzJjg50HlZdj_QFJGm2gQmDUD",  # resnet18_mstcn_video (LRW 88.9%, 139MB)
    dest: str = "/mnt/pretrained/lrw_resnet18_frontend.pth",
):
    info = fetch.remote(file_id=file_id, dest=dest)
    print("INSPECT:", info)
    print(f"\n→ set configs/phase_lipvis.yaml  model.visual_frontend_relu: {info['relu_type'].split()[0]}")
