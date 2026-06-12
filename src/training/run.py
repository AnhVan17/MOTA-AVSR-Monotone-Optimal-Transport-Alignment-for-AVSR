"""Logic huấn luyện THUẦN — không phụ thuộc Modal, không hardcode path container.

Gọi được từ cả 2 phía:
  - scripts/modal/*  (cloud): dựng config với path /mnt rồi gọi run_training(config)
  - scripts/local/*  (local): dựng config với path local rồi gọi run_training(config)

Mọi đường dẫn lấy TỪ config (không gắn cứng /mnt, /root) để chạy được ở bất kỳ đâu.
"""
import os
from typing import Dict

from src.training.trainer import Trainer
from src.utils.logging_utils import setup_logger

logger = setup_logger("train.run")


def run_training(config: Dict) -> Trainer:
    """Chạy training từ một config đã load đầy đủ.

    Args:
        config: dict cấu hình (đã resolve inheritance). Đường dẫn data/manifest
                nằm trong config['data'].

    Returns:
        Trainer sau khi train xong (tiện cho test/inspection).
    """
    data_format = str(config["data"].get("format", "jsonl")).lower()
    train_manifest = config["data"].get("train_manifest")
    # WebDataset (frame shards) reads data.train_shards/val_shards — there is no jsonl manifest,
    # so skip the manifest existence check (train_manifest may still be inherited from base.yaml).
    if data_format != "webdataset" and train_manifest and not os.path.exists(train_manifest):
        logger.error(f"Train manifest không tồn tại: {train_manifest}")
        raise FileNotFoundError(train_manifest)

    source = config["data"].get("train_shards") if data_format == "webdataset" else train_manifest
    logger.info(f"Device-agnostic training | format={data_format} | source={source}")
    trainer = Trainer(config)
    trainer.train()
    return trainer
