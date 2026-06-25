"""Inspect a checkpoint ON the Modal volume (no local download) — verify epoch / step /
best_metric (val WER) and param count. Use to confirm `best_model.pt` after a restore.

  modal run scripts/modal/inspect_ckpt.py                                   # phase1/best_model.pt
  modal run scripts/modal/inspect_ckpt.py --path /mnt/checkpoints/phase1/epoch_25.pt
"""
import sys
from pathlib import Path

import modal

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import ML_TRAIN_IMAGE, get_volume

app = modal.App("avsr-inspect-ckpt")
volume = get_volume()


@app.function(image=ML_TRAIN_IMAGE, volumes={"/mnt": volume}, timeout=300)
def inspect(path: str = "/mnt/checkpoints/phase1/best_model.pt"):
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", {})
    print(f"path:        {path}")
    print(f"epoch:       {ck.get('epoch')}")
    print(f"step:        {ck.get('step')}")
    print(f"best_metric: {ck.get('best_metric')}  (val WER %, lower = better)")
    print(f"model:       {len(sd)} tensors, {sum(v.numel() for v in sd.values()):,} params")


@app.local_entrypoint()
def main(path: str = "/mnt/checkpoints/phase1/best_model.pt"):
    inspect.remote(path=path)
