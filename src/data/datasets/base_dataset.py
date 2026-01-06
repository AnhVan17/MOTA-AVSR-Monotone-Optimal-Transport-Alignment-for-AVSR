"""
Base Dataset Classes for AVSR Training
=======================================
This module contains ONLY dataset-loading logic.
Preprocessing logic is in src/data/preprocessors/
"""

import os
import json
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all AVSR datasets.
    Subclasses: GridDataset, ViCocktailDataset
    """
    
    def __init__(
        self, 
        manifest_path: str, 
        tokenizer,
        data_root: str,
        use_precomputed_features: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            manifest_path: Path to JSONL manifest file
            tokenizer: Tokenizer instance (e.g., WhisperTokenizer)
            data_root: Root directory for data files
            use_precomputed_features: If True, load .pt features. If False, load raw video.
            max_samples: Limit number of samples (for debugging)
        """
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.use_precomputed_features = use_precomputed_features
        
        # Load manifest
        self.data = self._load_manifest(manifest_path, max_samples)
        logger.info(f"Loaded {len(self.data)} samples from {manifest_path}")
    
    def _load_manifest(self, path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load JSONL manifest file"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        if max_samples is not None:
            data = data[:max_samples]
            logger.info(f"   (Limited to {max_samples} samples for debugging)")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Should return a dict with keys:
        - 'audio': Audio features tensor
        - 'visual': Visual features tensor  
        - 'target': Token IDs tensor
        - 'text': Raw text string
        - 'rel_path': Relative path for debugging
        """
        pass
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to tensor of token IDs for CTC training"""
        # Use encode_for_ctc to remove special tokens
        # CTC should only see content tokens, not <|startoftranscript|> etc.
        if hasattr(self.tokenizer, 'encode_for_ctc'):
            token_ids = self.tokenizer.encode_for_ctc(text)
        else:
            # Fallback for tokenizers without encode_for_ctc
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        return torch.tensor(token_ids, dtype=torch.long)


class FeatureDataset(BaseDataset):
    """
    Dataset that loads precomputed .pt feature files.
    Used for Phase 1 training (fast, memory efficient).
    """
    
    def __init__(self, *args, augment: bool = False, aug_cfg: Dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        if augment:
            from src.data.augmentations import FeatureAugmenter
            # Separate audio/visual configs if they exist in aug_cfg, otherwise pass empty dicts (defaults)
            ac = aug_cfg if aug_cfg else {}
            vc = aug_cfg if aug_cfg else {}
            self.augmenter = FeatureAugmenter(audio_conf=ac, visual_conf=vc) 
        else:
            self.augmenter = None
    
    def _get_actual_length(self, features: torch.Tensor, threshold: float = 1e-6) -> int:
        """
        Detect actual sequence length by finding where trailing zeros (padding) starts.
        
        Args:
            features: [T, D] tensor
            threshold: Values below this are considered padding
            
        Returns:
            Actual length (non-padded portion)
        """
        # Sum absolute values along feature dimension
        # Padding frames will have sum close to 0
        frame_energy = features.abs().sum(dim=-1)  # [T]
        
        # Find last non-zero frame
        non_zero_mask = frame_energy > threshold
        
        if not non_zero_mask.any():
            return 1  # At least 1 frame
        
        # Find the last True index
        actual_len = non_zero_mask.nonzero()[-1].item() + 1
        
        return actual_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        rel_path = item['rel_path']
        text = item.get('text', "")
        
        # Build full path (manifest usually has .pt paths)
        full_path = os.path.join(self.data_root, rel_path)
        
        # Ensure .pt extension
        if not full_path.endswith('.pt'):
            full_path = os.path.splitext(full_path)[0] + '.pt'
        
        # Tokenize
        target = self._tokenize(text)
        
        try:
            data = torch.load(full_path)
            audio = data['audio'].float()
            visual = data['visual'].float()
            
            # CRITICAL FIX: Compute actual lengths
            # Visual: Can detect trailing zeros (works well)
            visual_len = self._get_actual_length(visual)
            
            # Audio: Whisper encoder normalizes output, so no trailing zeros
            # SOLUTION: Estimate audio_len from visual_len using frame rate ratio
            # Typically: Audio ~50Hz (Whisper), Visual ~25fps → ratio ~2
            # But since audio is already 1500 frames for ~10 sec and visual ~150 frames
            # The ratio is closer to audio_total / visual_total
            audio_total = audio.size(0)
            visual_total = visual.size(0)
            
            if visual_total > 0 and visual_len > 0:
                # Scale audio_len proportionally to visual_len
                ratio = audio_total / visual_total
                audio_len = min(int(visual_len * ratio), audio_total)
            else:
                audio_len = audio_total
            
            # Apply Augmentation if enabled
            if self.augmenter is not None:
                audio, visual = self.augmenter(audio, visual)

            return {
                'audio': audio,   # [T_a, 768]
                'visual': visual, # [T_v, 512]
                'audio_len': audio_len,   # Estimated from visual
                'visual_len': visual_len, # Detected from zeros
                'target': target,
                'text': text,
                'rel_path': rel_path
            }
        except Exception as e:
            logger.error(f"Error loading {full_path}: {e}")
            # Return dummy data
            return {
                'audio': torch.zeros(300, 768),
                'visual': torch.zeros(75, 512),
                'audio_len': 300,
                'visual_len': 75,
                'target': target,
                'text': text,
                'rel_path': rel_path
            }


class RawVideoDataset(BaseDataset):
    """
    Dataset that loads raw/cropped video files.
    Used for Phase 2 training (end-to-end, higher quality).
    
    NOTE: This requires on-the-fly feature extraction if not using
    the preprocessed cropped videos.
    """
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        rel_path = item['rel_path']
        text = item.get('text', "")
        
        full_path = os.path.join(self.data_root, rel_path)
        target = self._tokenize(text)
        
        # TODO: Implement raw video loading with on-the-fly processing
        # This requires VideoProcessor and AudioExtractor
        raise NotImplementedError(
            "Phase 2 (Raw Video) loading requires on-the-fly feature extraction. "
            "Use FeatureDataset for Phase 1 training with precomputed .pt files."
        )