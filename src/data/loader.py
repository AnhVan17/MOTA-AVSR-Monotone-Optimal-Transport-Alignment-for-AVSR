import torch
from torch.utils.data import DataLoader
from typing import Dict

# Import datasets
from .datasets.grid import GridDataset
from .datasets.vicocktail import VicocktailDataset
# Import new unified collate function
from .collate import avsr_collate_fn
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def build_dataloader(
    config: Dict,
    tokenizer,
    mode: str = "train"
) -> DataLoader:
    """
    Factory function to build PyTorch DataLoaders for AVSR.
    Compatible with both Grid and ViCocktail datasets.
    
    Args:
        config (Dict): Configuration dictionary. 
                       Must include 'dataset_name' ('grid' or 'vicocktail').
        tokenizer: Tokenizer instance.
        mode (str): 'train' | 'val' | 'test'.
        
    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    
    # 1. Select Dataset Class
    # Default to Vicocktail if not specified to maintain backward compatibility
    dataset_name = config.get('dataset_name', 'vicocktail').lower()
    
    if dataset_name == 'grid':
        DatasetClass = GridDataset
    elif dataset_name == 'vicocktail':
        DatasetClass = VicocktailDataset
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Supported: 'grid', 'vicocktail'")

    # 2. Configure Loader Parameters based on Mode
    if mode == "train":
        manifest_path = config['train_manifest']
        shuffle = True
        drop_last = True
        augment = True
        max_s = config.get('max_train_samples')
        
    elif mode in ["val", "test"]:
        manifest_path = config['val_manifest'] if mode == "val" else config.get('test_manifest')
        shuffle = False
        drop_last = False
        augment = False
        max_s = config.get('max_val_samples')
        
    else: 
        raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'val', or 'test'.")

    use_features = config.get('use_precomputed_features', True)

    logger.info(f"Building DataLoader [{mode}] for {dataset_name.upper()}")
    # logger.debug(f"   - Manifest: {manifest_path}")
    # logger.debug(f"   - Augment: {augment}")
    
    # 3. Initialize Dataset
    try:
        dataset = DatasetClass(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            data_root=config['data_root'],
            use_precomputed_features=use_features,
            max_samples=max_s,
            augment=augment,
            aug_cfg=config.get('aug_cfg', None) # Pass augmentation config if present
        )
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}")
        raise e

    # 4. Initialize DataLoader with Unified Collate
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config.get('num_workers', 4),
        collate_fn=avsr_collate_fn,  # Using the robust collate function
        pin_memory=True,
        drop_last=drop_last
    )
    
    return loader