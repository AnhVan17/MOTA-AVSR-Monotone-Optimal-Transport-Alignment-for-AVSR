import torch
from torch.utils.data import DataLoader
from typing import Dict

# Import your dataset and collate function
from .datasets.grid import GridDataset
from .collate import collate_fn

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
    if mode == "train":
        manifest_path = config['train_manifest']
        shuffle = True
        drop_last = True   # Good for BatchNorm stability during training
        # Check if we should use precomputed .npy features (Phase 1) or raw video (Phase 2)
        # Default to False (Raw Video) if not specified
        use_features = config.get('use_precomputed_features', False)
        
    elif mode in ["val", "test"]:
        manifest_path = config['val_manifest'] if mode == "val" else config.get('test_manifest')
        shuffle = False    # Never shuffle validation/test data (for consistent metrics)
        drop_last = False  # Keep all samples for evaluation
        # Validation usually runs on raw video to ensure end-to-end correctness,
        # unless you specifically want to validate on features.
        use_features = config.get('use_precomputed_features', False)
        
    else: 
        raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'val', or 'test'.")

    print(f" Building DataLoader [{mode}]")
    print(f"   - Manifest: {manifest_path}")
    print(f"   - Input Type: {'Precomputed Features (.npy)' if use_features else 'Raw Video (.mpg)'}")
    print(f"   - Batch Size: {config['batch_size']}")

    # 2. Initialize Dataset
    dataset = GridDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        data_root=config['data_root'],
        use_precomputed_features=use_features,
        max_samples=config.get('max_samples', None) # Useful for debugging (quick run)
    )

    # 3. Initialize DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,  # Custom batch processing logic
        pin_memory=True,        # Faster data transfer to CUDA (GPU)
        drop_last=drop_last
    )
    
    return loader