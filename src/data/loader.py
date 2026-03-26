"""
DataLoader factory for AVSR training.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

from .datasets.grid import GridDataset
from .datasets.vicocktail import ViCocktailDataset
from .collate import Collator
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def build_dataloader(
    config: Dict,
    tokenizer,
    mode: str = "train",
) -> DataLoader:
    """
    Factory function to build DataLoaders for AVSR.

    Supports auto-detecting dataset type from manifest filename.
    Raw video loading is NOT supported — use pre-computed .pt features only.

    Args:
        config: Hydra/OmegaConf config dict.
                Required keys: 'data.train_manifest', 'data.data_root'.
        tokenizer: WhisperTokenizer instance.
        mode: 'train', 'val', or 'test'.

    Returns:
        DataLoader instance.
    """
    data_cfg = config.get('data', config)

    # --- Manifest path ---
    manifest_key = f"{mode}_manifest" if mode == "train" else f"{mode}_manifest"
    manifest_path = data_cfg.get(manifest_key)
    if manifest_path is None:
        manifest_path = config.get(manifest_key)

    if manifest_path is None:
        raise ValueError(f"Manifest not found for mode='{mode}' in config")

    # --- Shuffle / drop_last ---
    shuffle = (mode == "train")
    drop_last = (mode == "train")

    logger.info(f"Building DataLoader [{mode}]")
    logger.debug(f"  Manifest: {manifest_path}")
    logger.debug(f"  Input: pre-computed .pt features only")

    # --- Auto-detect dataset type ---
    dataset_type = _detect_dataset_type(manifest_path)

    # --- Build dataset ---
    if dataset_type == "grid":
        dataset = GridDataset(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            data_root=data_cfg.get('data_root'),
            max_samples=data_cfg.get('max_samples', None),
            augment=(mode == "train"),
            aug_cfg=config.get('augmentation', None),
        )
    elif dataset_type == "vicocktail":
        dataset = ViCocktailDataset(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            data_root=data_cfg.get('data_root'),
            max_samples=data_cfg.get('max_samples', None),
            augment=(mode == "train"),
            aug_cfg=config.get('augmentation', None),
        )
    else:
        raise ValueError(
            f"Unknown dataset type for manifest '{manifest_path}'. "
            f"Supported: 'grid', 'vicocktail'. "
            f"Rename manifest file to include 'grid' or 'vicocktail'."
        )

    # --- Collator ---
    pad_id = getattr(tokenizer, 'eot_token_id', 50257)
    collator = Collator(pad_id=pad_id)

    # --- DataLoader ---
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get('batch_size', 32),
        shuffle=shuffle,
        num_workers=data_cfg.get('num_workers', 2),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )

    return loader


def _detect_dataset_type(manifest_path: str) -> str:
    """
    Auto-detect dataset type from manifest filename.
    Supports: 'grid', 'vicocktail'.

    Override via config['data']['dataset_type'] if needed.
    """
    stem = Path(manifest_path).stem.lower()
    if 'grid' in stem:
        return "grid"
    elif 'vicocktail' in stem:
        return "vicocktail"
    # Default fallback — try GridDataset
    return "grid"
