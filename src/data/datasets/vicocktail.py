"""
ViCocktail Dataset for AVSR Training
====================================
Loads pre-computed .pt features from preprocessing pipeline.
"""

from .base import FeatureDataset
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class ViCocktailDataset(FeatureDataset):
    """
    Dataset for ViCocktail benchmark — pre-computed features only.

    Preprocessing team cung cấp .pt files chứa audio/visual features.
    MOTA chỉ load .pt features, không xử lý video gốc.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        data_root: str,
        max_samples: int = None,
        augment: bool = False,
        aug_cfg: dict = None,
    ):
        super().__init__(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            data_root=data_root,
            use_precomputed_features=True,
            max_samples=max_samples,
            augment=augment,
            aug_cfg=aug_cfg,
        )

        logger.debug(f"ViCocktailDataset: {len(self)} samples (features only)")
