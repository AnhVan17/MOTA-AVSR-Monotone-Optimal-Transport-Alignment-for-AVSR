import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional, Any
from .tokenizers.whisper import WhisperProcessor

logger = logging.getLogger(__name__)

class FastAuroraDataset(Dataset):
    """
    Optimized AuroraXT Dataset
    - Fast loading from .pt files
    - Uses WhisperProcessor for decoding/encoding if needed
    """
    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        max_samples: Optional[int] = None
    ):
        self.data_root = Path(data_root)
        self.samples = []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        if max_samples:
            self.samples = self.samples[:max_samples]
            
        logger.info(f"✅ Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        pt_path = self.data_root / sample['path']
        
        try:
            # Optimized: load with weights_only=False for custom objects if any, 
            # but for tensors True is faster and safer
            data = torch.load(pt_path, map_location='cpu', weights_only=True)
            
            return {
                'audio': data['audio'],   # [T, 768] Whisper features
                'visual': data['visual'], # [T, 512]
                'tokens': data['tokens'], # [L]
                'text': data.get('text', ''),
                'id': sample['id']
            }
        except Exception as e:
            logger.warning(f"Failed to load {pt_path}: {e}")
            # Return a dummy sample or try next
            return self.__getitem__((idx + 1) % len(self))

def collate_fn(batch: List[Dict]) -> Dict:
    """Fast collation with efficient padding"""
    # 1. Audio features [T, 768] - Whisper encoder output
    # Pad variable-length sequences
    audio_list = [s['audio'] for s in batch]
    audio_batch = torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True)
    
    # Create audio mask
    audio_mask = torch.zeros(len(batch), audio_batch.size(1), dtype=torch.bool)
    for i, a in enumerate(audio_list):
        audio_mask[i, :a.size(0)] = True
        
    # 2. Visual features [T, 512]
    visual_list = [s['visual'] for s in batch]
    visual_batch = torch.nn.utils.rnn.pad_sequence(visual_list, batch_first=True)
    
    visual_mask = torch.zeros(len(batch), visual_batch.size(1), dtype=torch.bool)
    for i, v in enumerate(visual_list):
        visual_mask[i, :v.size(0)] = True
        
    # 3. Tokens [L]
    token_list = [s['tokens'] for s in batch]
    token_batch = torch.nn.utils.rnn.pad_sequence(token_list, batch_first=True, padding_value=-100)
    
    # Target mask for CE loss (ignore padding -100)
    target_mask = (token_batch != -100)
    
    return {
        'audio': audio_batch,
        'audio_mask': audio_mask,
        'visual': visual_batch,
        'visual_mask': visual_mask,
        'target': token_batch,
        'target_mask': target_mask,
        'ids': [s['id'] for s in batch]
    }

def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    batch_size: int = 32,
    num_workers: int = 4,
    data_root: str = "./data",
    **kwargs
) -> Dict[str, DataLoader]:
    
    train_ds = FastAuroraDataset(train_manifest, data_root, max_samples=kwargs.get('max_train_samples'))
    val_ds = FastAuroraDataset(val_manifest, data_root, max_samples=kwargs.get('max_val_samples'))
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return {'train': train_loader, 'val': val_loader}
