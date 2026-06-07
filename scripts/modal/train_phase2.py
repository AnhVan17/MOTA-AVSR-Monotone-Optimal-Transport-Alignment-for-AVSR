import modal
import sys

# --- Config ---
APP_NAME = "avsr-train-phase2-mqot"
VOLUME_NAME = "avsr-volume"

# --- Image ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "numpy<2",
        "jiwer",
        "matplotlib",
        "soundfile",
        "opencv-python-headless"
    )
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="A10G",
    timeout=7200
)
def train_remote():
    # Vỏ Modal MỎNG: setup path container rồi gọi logic train THUẦN (src/training/run.py).
    sys.path.append("/root")
    from src.utils.config_utils import load_config
    from src.training.run import run_training

    # Load config (inheritance). use_mqot=True nằm trong phase2_mqot.yaml.
    config = load_config("/root/configs/phase2_mqot.yaml")
    # run_training tự validate manifest trong config + chọn device (cuda trên Modal GPU).
    run_training(config)

@app.local_entrypoint()
def main():
    train_remote.remote()
