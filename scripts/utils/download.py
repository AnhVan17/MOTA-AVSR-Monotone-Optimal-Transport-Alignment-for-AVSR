import modal
import os
import time
import logging

# Configure basic logging since this script acts as a standalone entry point
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("download_script")

APP_NAME = "avsr-dataset-downloader"
VOLUME_NAME = "avsr-dataset-volume"
VOL_MOUNT_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    # hf_transfer giúp tải siêu nhanh
    .pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})     
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={VOL_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name("hf-token")],
    cpu=0.5,      
    memory=1024,   
    timeout=86400  
)
def download_dataset_snapshot():
    from huggingface_hub import snapshot_download

    logger.info("Start downloading dataset...")
    logger.info("Mirror mode")

    download_path = f"{VOL_MOUNT_PATH}/raw_mirror"
    os.makedirs(download_path, exist_ok=True)

    try:
        snapshot_download(
            repo_id="nguyenvulebinh/ViCocktail",
            repo_type="dataset",
            local_dir=download_path,
            local_dir_use_symlinks=False, 
            resume_download=True,        
            max_workers=4                 
        )
        logger.info(f"Download completed! Dataset is stored at: {download_path}")
        volume.commit()
        logger.info("💾 Volume committed.")

    except Exception as e:
        logger.error(f"Error: {e}")
        volume.commit()


@app.local_entrypoint()
def main():
    download_dataset_snapshot.remote()
