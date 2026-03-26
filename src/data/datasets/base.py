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
        """Tokenize text to tensor of token IDs"""
        token_ids = self.tokenizer.encode(text)
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
            
            # Apply Augmentation if enabled
            if self.augmenter is not None:
                audio, visual = self.augmenter(audio, visual)

            return {
                'audio': audio,   # [T_a, 768]
                'visual': visual, # [T_v, 512]
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
                'target': target,
                'text': text,
                'rel_path': rel_path
            }


