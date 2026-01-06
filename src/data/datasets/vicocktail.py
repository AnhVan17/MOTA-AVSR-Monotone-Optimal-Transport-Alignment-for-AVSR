"""
Vicocktail Dataset for AVSR Training
====================================
Inherits from FeatureDataset to support Phase 1 (Features) training
with standardized augmentation and loading logic.

FIXED: Removed _tokenize() override - now inherits correct encode_for_ctc() from base class
"""

import torch
from .base_dataset import FeatureDataset
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class VicocktailDataset(FeatureDataset):
    """
    Dataset for Vicocktail Corpus.
    
    Inherits from FeatureDataset.
    Supports loading precomputed features (.pt) and optional Augmentation.
    
    NOTE: _tokenize() is inherited from base class which correctly uses encode_for_ctc()
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
        if not use_precomputed_features:
            raise NotImplementedError(
                "VicocktailDataset currently only supports Phase 1 (precomputed features). "
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
        
        logger.debug(f"VicocktailDataset initialized with {len(self)} samples")
        
        # Verify tokenizer has encode_for_ctc
        if not hasattr(self.tokenizer, 'encode_for_ctc'):
            logger.warning("Tokenizer missing encode_for_ctc method!")
    
    # NO _tokenize() OVERRIDE!
    # Inherited from base class which correctly uses encode_for_ctc()
