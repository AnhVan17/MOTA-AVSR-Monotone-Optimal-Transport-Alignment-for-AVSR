"""
Base Dataset Classes for AVSR Training
=======================================
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
    """Abstract base class for all AVSR datasets."""
    
    def __init__(
        self, 
        manifest_path: str, 
        tokenizer,
        data_root: str,
        use_precomputed_features: bool = True,
        max_samples: Optional[int] = None
    ):
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.use_precomputed_features = use_precomputed_features
        
        self.data = self._load_manifest(manifest_path, max_samples)
        logger.info(f"Loaded {len(self.data)} samples from {manifest_path}")
    
    def _load_manifest(self, path: str, max_samples: Optional[int] = None) -> List[Dict]:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        if max_samples is not None:
            data = data[:max_samples]
            logger.info(f"   (Limited to {max_samples} samples)")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text for CTC training (no special tokens)"""
        if hasattr(self.tokenizer, 'encode_for_ctc'):
            token_ids = self.tokenizer.encode_for_ctc(text)
        else:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        return torch.tensor(token_ids, dtype=torch.long)


class FeatureDataset(BaseDataset):
    """Dataset that loads precomputed .pt feature files."""
    
    def __init__(self, *args, augment: bool = False, aug_cfg: Dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        if augment:
            from src.data.augmentations import FeatureAugmenter
            ac = aug_cfg if aug_cfg else {}
            vc = aug_cfg if aug_cfg else {}
            self.augmenter = FeatureAugmenter(audio_conf=ac, visual_conf=vc) 
        else:
            self.augmenter = None
        
        self._debug_logged = False
    
    def _get_actual_length(self, features: torch.Tensor) -> int:
        """
        Detect actual length by finding where real content ends.
        
        KEY FIX: Use threshold = 0.01 (not 1e-6!)
        Whisper features have noise, never exactly 0.
        """
        T = features.size(0)
        
        # Energy per frame
        frame_energy = features.abs().sum(dim=-1)  # [T]
        
        # FIX: Higher threshold to detect real content vs noise/padding
        # Real features: energy > 0.01 (typically 0.5-5.0)
        # Padding/noise: energy < 0.01
        THRESHOLD = 0.01
        
        valid_mask = frame_energy > THRESHOLD
        
        if not valid_mask.any():
            return max(1, T // 10)
        
        # Find last valid frame
        actual_len = valid_mask.nonzero()[-1].item() + 1
        
        # Sanity: at least 10% of total
        return max(actual_len, T // 10)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        rel_path = item['rel_path']
        text = item.get('text', "")
        
        full_path = os.path.join(self.data_root, rel_path)
        if not full_path.endswith('.pt'):
            full_path = os.path.splitext(full_path)[0] + '.pt'
        
        target = self._tokenize(text)
        
        try:
            data = torch.load(full_path, map_location='cpu')
            audio = data['audio'].float()
            visual = data['visual'].float()
            
            # Get lengths - from file if available, else estimate
            if 'audio_len' in data and 'visual_len' in data:
                audio_len = data['audio_len']
                visual_len = data['visual_len']
            else:
                # Estimate from visual (visual has clear zero-padding)
                visual_len = self._get_actual_length(visual)
                
                # Audio len from visual ratio (audio ~2x visual frame rate)
                audio_total = audio.size(0)
                visual_total = visual.size(0)
                ratio = audio_total / visual_total if visual_total > 0 else 1
                audio_len = min(int(visual_len * ratio), audio_total)
            
            # Debug first sample
            if not self._debug_logged:
                print(f"\n📏 [Dataset] First sample lengths:")
                print(f"   Audio: {audio.size(0)} → actual: {audio_len}")
                print(f"   Visual: {visual.size(0)} → actual: {visual_len}")
                print(f"   Target: {len(target)} tokens")
                print(f"   Audio/Target ratio: {audio_len / max(1, len(target)):.1f}")
                self._debug_logged = True
            
            if self.augmenter is not None:
                audio, visual = self.augmenter(audio, visual)

            return {
                'audio': audio,
                'visual': visual,
                'audio_len': audio_len,
                'visual_len': visual_len,
                'target': target,
                'text': text,
                'rel_path': rel_path
            }
            
        except Exception as e:
            logger.error(f"Error loading {full_path}: {e}")
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
    """Dataset for raw video files (Phase 2)."""
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Use FeatureDataset for Phase 1 training.")