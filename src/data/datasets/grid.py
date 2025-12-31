"""
GRID Dataset for AVSR Training
================================
Inherits from BaseDataset for clean architecture.
"""

import os
import torch
from .base import FeatureDataset
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class GridDataset(FeatureDataset):
    """
    Dataset for GRID Corpus.
    
    Inherits from FeatureDataset which handles .pt file loading.
    For Phase 2 (raw video), use RawVideoDataset or extend this class.
    """
    
    def __init__(
        self, 
        manifest_path: str, 
        tokenizer,
        data_root: str,
        use_precomputed_features: bool = True,
        max_samples: int = None,
        augment: bool = False,
        aug_cfg: dict = None
    ):
        """
        Args:
            manifest_path: Path to JSONL manifest
            tokenizer: WhisperTokenizer instance
            data_root: Root directory containing .pt files
            use_precomputed_features: Must be True for Phase 1
            max_samples: Limit samples for debugging
            augment: Whether to apply feature augmentation
            aug_cfg: Augmentation configuration dict
        """
        if not use_precomputed_features:
            raise NotImplementedError(
                "GridDataset currently only supports Phase 1 (precomputed features). "
                "Set use_precomputed_features=True."
            )
        
        super().__init__(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            data_root=data_root,
            use_precomputed_features=use_precomputed_features,
            max_samples=max_samples,
            augment=augment,
            aug_cfg=aug_cfg
        )
        
        logger.debug(f"GridDataset initialized with {len(self)} samples")