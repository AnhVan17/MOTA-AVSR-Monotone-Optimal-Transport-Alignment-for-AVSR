"""
Vicocktail Dataset for AVSR Training
====================================
Inherits from FeatureDataset to support Phase 1 (Features) training
with standardized augmentation and loading logic.
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

    def _tokenize(self, text: str) -> torch.Tensor:
        """
        Override tokenize to add specific length constraints for Vicocktail if needed.
        Whisper Decoder limit is typically 448 tokens.
        """
        token_ids = self.tokenizer.encode(text)
        
        # Explicit truncation to 448 to match Whisper's max position embedding
        # (Though usually handling in Collate is better, safety here is okay)
        MAX_LEN = 448
        if len(token_ids) > MAX_LEN:
            token_ids = token_ids[:MAX_LEN]
            
        return torch.tensor(token_ids, dtype=torch.long)
