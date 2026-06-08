import modal
import sys
from pathlib import Path

# Repo root on path so we can import shared Modal image definitions locally (at app-build time).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
def train_remote():
    # THIN Modal wrapper: set up container paths, then call the pure training logic
    # (src/training/run.py).
    sys.path.append("/root")
    from src.utils.config_utils import load_config
    from src.training.run import run_training

    # Load config (inheritance). use_mqot=True lives in phase2_mqot.yaml.
    config = load_config("/root/configs/phase2_mqot.yaml")
    # run_training validates the manifest in config + selects device (cuda on Modal GPU).
    run_training(config)

@app.local_entrypoint()
def main():
    train_remote.remote()
