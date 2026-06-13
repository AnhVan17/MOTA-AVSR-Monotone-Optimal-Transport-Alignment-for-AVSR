import modal
import sys
from pathlib import Path

# Make `src` importable. Locally: add repo root (for app-build). In Modal containers the
# script is flattened to /root and src lives at /root/src, so parents[N] would be out of range.
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import ML_TRAIN_IMAGE, get_volume

APP_NAME = "avsr-train-phase2-mqot"

app = modal.App(APP_NAME)
volume = get_volume()

@app.function(
    image=ML_TRAIN_IMAGE,
    volumes={"/mnt": volume},
    gpu="A10G",
    timeout=7200
)
def train_remote(config_path: str = None):
    # THIN Modal wrapper: set up container paths, then call the pure training logic
    # (src/training/run.py).
    sys.path.append("/root")
    from src.utils.config_utils import load_config
    from src.training.run import run_training

    # Load config (inheritance). use_mqot=True lives in phase2.yaml; pass --config-path
    # /root/configs/phase2_smoke.yaml for the frame-shard smoke.
    final_config_path = config_path if config_path else "/root/configs/phase2.yaml"
    config = load_config(final_config_path)
    # run_training selects device (cuda on Modal GPU); the webdataset path needs no manifest.
    run_training(config)

@app.local_entrypoint()
def main(config_path: str = None):
    train_remote.remote(config_path=config_path)
