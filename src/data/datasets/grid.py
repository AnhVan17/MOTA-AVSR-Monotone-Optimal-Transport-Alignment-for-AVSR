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
        use_precropped: bool = False,
        max_samples: int = None,
        augment: bool = False,
        aug_cfg: dict = None
    ):
        """
        Args:
            manifest_path: Path to JSONL manifest
            tokenizer: WhisperTokenizer instance
            data_root: Root directory
            use_precomputed_features: True for .pt, False for raw .mpg/.mp4
            use_precropped: If True, skip face detection (assume video is mouth-cropped)
            max_samples: Limit samples for debugging
            augment: Whether to apply augmentation
            aug_cfg: Augmentation configuration dict
        """
        super().__init__(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            data_root=data_root,
            use_precomputed_features=use_precomputed_features,
            max_samples=max_samples,
            augment=augment,
            aug_cfg=aug_cfg
        )
        
        # Initialize processors for Phase 2 (Raw Video)
        if not use_precomputed_features:
            from src.data.preprocessors.base import VideoProcessor, AudioFeatureExtractor
            logger.info(f"Initializing On-the-fly Processors (Precropped: {use_precropped})...")
            self.video_processor = VideoProcessor(use_precropped=use_precropped)
            self.audio_extractor = AudioFeatureExtractor()
        else:
            self.video_processor = None
            self.audio_extractor = None

        logger.debug(f"GridDataset initialized with {len(self)} samples (Features: {use_precomputed_features})")

    def __getitem__(self, idx: int):
        if self.use_precomputed_features:
            return super().__getitem__(idx)
        else:
            return self._load_raw_video(idx)

    def _load_raw_video(self, idx: int):
        """Load and process raw video on-the-fly"""
        item = self.data[idx]
        rel_path = item['rel_path'] # e.g. "s1/bbaf2n.mpg"
        text = item.get('text', "")
        
        full_path = os.path.join(self.data_root, rel_path)
        
        # 1. Visual (Crop Mouth)
        # Returns [T_v, 1, 88, 88] (if grayscale) or [T_v, 3, 88, 88]
        # VideoProcessor returns [C, T, H, W] or [T, C, H, W]? 
        # Checked preprocessor: returns [K, C, H, W] (from process) or similar.
        # Let's verify output of VideoProcessor.process below.
        visual = self.video_processor.process(full_path) 
        
        # Handle Load Failure
        if visual is None:
             logger.warning(f"Failed to load visual: {full_path}")
             visual = torch.zeros((1, 1, 88, 88)) # Dummy

        # Check shape, MOTA expects [B, T, C, H, W] or [T, C, H, W].
        # VideoProcessor returns [T, C, H, W].
        # If it returns [T, 1, 88, 88], we are good.
        
        # 2. Audio (Extract -> Whisper Features)
        # Returns [T_a, 768]
        audio = self.audio_extractor.process_file(full_path)
        
        if audio is None:
             logger.warning(f"Failed to load audio: {full_path}")
             audio = torch.zeros((1, 768)) # Dummy

        # 3. Tokenize
        target = self._tokenize(text)
        
        return {
            'audio': audio,
            'visual': visual,
            'target': target,
            'text': text,
            'rel_path': rel_path
        }