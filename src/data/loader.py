import torch
from torch.utils.data import DataLoader
from typing import Dict

# Import your dataset and collate function
from .datasets.grid import GridDataset
from .collate import Collator
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def build_dataloader(
    config: Dict,
    tokenizer,
    mode: str = "train"
) -> DataLoader:
    """
    Factory function to build PyTorch DataLoaders for AVSR.
    
    Args:
        config (Dict): Configuration dictionary containing paths and hyperparameters.
                       Required keys: 'data_root', 'batch_size', 'num_workers'.
                       Conditional keys: 'train_manifest', 'val_manifest', 'test_manifest'.
        tokenizer: Instance of WhisperTokenizer.
        mode (str): 'train', 'val', or 'test'. Controls shuffling and data sources.
        
    Returns:
        DataLoader: A configured PyTorch DataLoader.
    """
    
    # 1. Configure Loader Parameters based on Mode
    # Handle nested 'data' config if present (Standard in base.yaml)
    data_cfg = config.get('data', config)
    
    if mode == "train":
        manifest_path = data_cfg.get('train_manifest')
        shuffle = True
        drop_last = True   # Good for BatchNorm stability during training
        # Check if we should use precomputed .npy features (Phase 1) or raw video (Phase 2)
        # Default to False (Raw Video) if not specified
        use_features = data_cfg.get('use_precomputed_features', False)
        
    elif mode in ["val", "test"]:
        if mode == "val":
            manifest_path = data_cfg.get('val_manifest')
        else:
            manifest_path = data_cfg.get('test_manifest')
            
        shuffle = False    # Never shuffle validation/test data (for consistent metrics)
        drop_last = False  # Keep all samples for evaluation
        # Validation usually runs on raw video to ensure end-to-end correctness,
        # unless you specifically want to validate on features.
        use_features = data_cfg.get('use_precomputed_features', False)
        
    else: 
        raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'val', or 'test'.")

    # Fallback if manifest not found in data_cfg (e.g. flat config override)
    if manifest_path is None:
         manifest_path = config.get('train_manifest' if mode == 'train' else 'val_manifest')

    logger.info(f"Building DataLoader [{mode}]")
    logger.debug(f"   - Manifest: {manifest_path}")
    logger.debug(f"   - Input Type: {'Precomputed Features (.npy)' if use_features else 'Raw Video (.mpg)'}")
    logger.debug(f"   - Batch Size: {data_cfg.get('batch_size', 32)}")

    # 3. Initialize Dataset
    dataset = GridDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        data_root=data_cfg.get('data_root'),
        use_precomputed_features=use_features,
        use_precropped=data_cfg.get('use_precropped', False), # Support pre-cropped data
        max_samples=data_cfg.get('max_samples', None), # Useful for debugging (quick run)
        augment=(mode == "train"),
        aug_cfg=config.get('augmentation', None) # Augmentation is usually top-level
    )

    # 4. Initialize Collator with Tokenizer's Pad ID (Fix for ! pollution)
    # Whisper usually repurposes EOT (50257) as PAD/Blank
    pad_id = getattr(tokenizer, 'eot_token_id', 50257) 
    collator = Collator(pad_id=pad_id)

    # 5. Initialize DataLoader
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get('batch_size', 32),
        shuffle=shuffle,
        num_workers=data_cfg.get('num_workers', 2),
        collate_fn=collator,  # Custom batch processing logic
        pin_memory=True,        # Faster data transfer to CUDA (GPU)
        drop_last=drop_last
    )

    # 3. Initialize DataLoader
    return loader