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

    # --- WebDataset (sharded .tar) path: streaming IterableDataset ---
    # Solves Modal Volume inode limit; see thesis/WEBDATASET_DESIGN.md.
    if str(data_cfg.get('format', 'jsonl')).lower() == 'webdataset':
        return _build_webdataset_loader(config, data_cfg, tokenizer, mode)

    # --- Manifest path ---
    manifest_key = f"{mode}_manifest"
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
    logger.debug("  Input: pre-computed .pt features only")

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


def _build_webdataset_loader(config: Dict, data_cfg: Dict, tokenizer, mode: str) -> DataLoader:
    """Build a DataLoader over WebDataset feature shards (streaming IterableDataset).

    Config keys (under 'data'): format: webdataset, and either '<mode>_shards' or 'shards'
    (brace/glob pattern or list of .tar paths). Worker-splitting is handled by WebDataset
    (single-node, multi-worker safe). Shuffle is internal (train only) — no DataLoader shuffle.
    """
    from src.data.shards import build_webdataset

    shards = data_cfg.get(f"{mode}_shards") or data_cfg.get("shards")
    if shards is None:
        raise ValueError(
            f"data.format=webdataset but no '{mode}_shards' or 'shards' in config"
        )

    is_train = (mode == "train")
    logger.info(f"Building WebDataset DataLoader [{mode}] from shards={shards}")

    dataset = build_webdataset(
        shards,
        tokenizer,
        train=is_train,
        augment=is_train or data_cfg.get("val_augment", False),  # val_augment → noise-aug on val too (robustness-aware best_model)
        aug_cfg=config.get("augmentation", None),
        shuffle_buffer=data_cfg.get("shuffle_buffer", 1000),
        # val/dev with augment → FROZEN per-sample noise (fixed noisy dev = reproducible selection);
        # train stays random (fresh noise every epoch).
        deterministic=(not is_train and data_cfg.get("val_augment", False)),
    )

    # Cheap smoke runs: cap the stream to N samples. Real training leaves this unset (unbounded).
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        dataset = dataset.slice(int(max_samples))

    pad_id = getattr(tokenizer, "eot_token_id", 50257)
    loader_kwargs = dict(
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=Collator(pad_id),
        pin_memory=torch.cuda.is_available(),
    )
    # prefetch_factor / persistent_workers are only valid with worker processes (num_workers > 0)
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = data_cfg.get("prefetch_factor", 4)
        loader_kwargs["persistent_workers"] = data_cfg.get("persistent_workers", True)
    return DataLoader(dataset, **loader_kwargs)


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
