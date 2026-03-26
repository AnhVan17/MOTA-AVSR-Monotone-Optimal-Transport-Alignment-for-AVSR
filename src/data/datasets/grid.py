"""
GRID Dataset for AVSR Training
================================
Loads pre-computed .pt features from preprocessing pipeline.
Raw video loading (MediaPipe, face detection) is owned by the preprocessing team.
"""

from .base import FeatureDataset
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class GridDataset(FeatureDataset):
    """
    Dataset for GRID Corpus — pre-computed features only.

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
        """
        Args:
            manifest_path: Path to JSONL manifest
            tokenizer: WhisperTokenizer instance
            data_root: Root directory containing .pt feature files
            max_samples: Limit samples for debugging
            augment: Whether to apply augmentation
            aug_cfg: Augmentation configuration dict

        Note:
            use_precomputed_features=True is always enforced.
            Preprocessing team handles mouth cropping and feature extraction.
        """
        super().__init__(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            data_root=data_root,
            use_precomputed_features=True,
            max_samples=max_samples,
            augment=augment,
            aug_cfg=aug_cfg
        )

        logger.debug(f"GridDataset: {len(self)} samples (features only)")
