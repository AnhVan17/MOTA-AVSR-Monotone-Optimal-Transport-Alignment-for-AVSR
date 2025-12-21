import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class AuroraDataset(Dataset):
    """
    AURORA-XT Dataset
    
    Loads preprocessed features from .pt files
    Features:
    - Proper path resolution (no double "data/" issue)
    - Error handling for corrupt files
    - Data validation
    """
    
    def __init__(
        self,
        file_paths: List[str],
        data_root: Optional[str] = None
    ):
        """
        Args:
            file_paths: List of .pt file paths (relative or absolute)
            data_root: Root directory for relative paths
        """
        self.file_paths = file_paths
        self.data_root = Path(data_root).resolve() if data_root else None
    
    @classmethod
    def from_manifest(cls, manifest_path: str, data_root: str = "./data"):
        """Create dataset from manifest file"""
        file_paths = []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                file_paths.append(item['path'])
        
        return cls(file_paths, data_root=data_root)
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def _resolve_path(self, file_path: str) -> Path:
        """
        Resolve file path properly - avoid double data/ issue
        
        Handles cases:
        - Absolute path: /data/processed_features/...
        - Relative with data_root overlap: data/processed_features/...
        - Relative without overlap: processed_features/...
        """
        file_path = str(file_path)
        
        if Path(file_path).is_absolute():
            return Path(file_path)
        
        if self.data_root is None:
            return Path(file_path)

        data_root_name = self.data_root.name 
        if file_path.startswith(data_root_name + "/"):
            file_path = file_path[len(data_root_name) + 1:]
        elif file_path.startswith(data_root_name + "\\"):
            file_path = file_path[len(data_root_name) + 1:]
        return self.data_root / file_path
    
    def _load_sample(self, file_path: Path) -> Optional[Dict]:
        """Load sample với error handling"""
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            if not self._validate_sample(data, file_path):
                return None
            
            return data
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return None
    
    def _validate_sample(self, data: Dict, path: Path) -> bool:
        """Validate sample data"""
        required_keys = ['audio', 'visual', 'text', 'id']
        
        for key in required_keys:
            if key not in data:
                logger.warning(f"Missing key '{key}' in {path}")
                return False
        
        if torch.isnan(data['audio']).any():
            logger.warning(f"NaN in audio features: {path}")
            return False
        
        if torch.isnan(data['visual']).any():
            logger.warning(f"NaN in visual features: {path}")
            return False

        if data['audio'].abs().sum() < 1e-6:
            logger.warning(f"All-zero audio features: {path}")
            return False
        
        if data['visual'].abs().sum() < 1e-6:
            logger.warning(f"All-zero visual features: {path}")
            return False
        
        if data['audio'].dim() != 2 or data['audio'].size(-1) != 768:
            logger.warning(f"Invalid audio shape {data['audio'].shape}: {path}")
            return False
        
        if data['visual'].dim() != 2 or data['visual'].size(-1) != 512:
            logger.warning(f"Invalid visual shape {data['visual'].shape}: {path}")
            return False
        
        return True
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load sample với fallback to next sample nếu lỗi
        
        Returns:
            dict with:
                - audio: [T_a, 768] Whisper features
                - visual: [T_v, 512] ResNet18 features
                - target: [L] token IDs
                - id: sample ID
        """
        for attempt in range(10):
            actual_idx = (idx + attempt) % len(self.file_paths)
            file_path = self._resolve_path(self.file_paths[actual_idx])
            
            data = self._load_sample(file_path)
            if data is not None:
                return {
                    'audio': data['audio'].clone(),     
                    'visual': data['visual'].clone(),    
                    'target': data['text'].clone() if isinstance(data['text'], torch.Tensor) else torch.tensor(data['text'], dtype=torch.long),
                    'id': data['id']
                }
        
        logger.error(f"All attempts failed around idx {idx}, returning zeros")
        return {
            'audio': torch.zeros(450, 768),  
            'visual': torch.zeros(375, 512),  
            'target': torch.zeros(10, dtype=torch.long),
            'id': f"error_{idx}"
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function với padding
    
    Args:
        batch: List of samples
        
    Returns:
        Batched dict với padding và masks
    """
    max_audio_len = max(x['audio'].size(0) for x in batch)
    max_visual_len = max(x['visual'].size(0) for x in batch)
    max_target_len = max(x['target'].size(0) for x in batch)
    
    B = len(batch)
    audio_dim = batch[0]['audio'].size(-1)  
    visual_dim = batch[0]['visual'].size(-1)  
    
    audio_batch = torch.zeros(B, max_audio_len, audio_dim)
    visual_batch = torch.zeros(B, max_visual_len, visual_dim)
    target_batch = torch.zeros(B, max_target_len, dtype=torch.long) 

    audio_mask = torch.zeros(B, max_audio_len, dtype=torch.bool)
    visual_mask = torch.zeros(B, max_visual_len, dtype=torch.bool)
    target_mask = torch.zeros(B, max_target_len, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        audio_len = sample['audio'].size(0)
        visual_len = sample['visual'].size(0)
        target_len = sample['target'].size(0)
        
        audio_batch[i, :audio_len] = sample['audio']
        visual_batch[i, :visual_len] = sample['visual']
        target_batch[i, :target_len] = sample['target']
        
        audio_mask[i, :audio_len] = True
        visual_mask[i, :visual_len] = True
        target_mask[i, :target_len] = True
    
    return {
        'audio': audio_batch,         
        'visual': visual_batch,       
        'target': target_batch,        
        'audio_mask': audio_mask,      
        'visual_mask': visual_mask,    
        'target_mask': target_mask,   
        'ids': [x['id'] for x in batch]
    }


def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    batch_size: int = 32,
    num_workers: int = 4,
    data_root: str = "./data",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    **kwargs  
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_manifest: Path to train manifest
        val_manifest: Path to val manifest
        batch_size: Batch size
        num_workers: Number of workers
        data_root: Data root directory
        max_train_samples: Limit train samples (for testing)
        max_val_samples: Limit val samples
        
    Returns:
        Dict with 'train' and 'val' dataloaders
    """

    train_dataset = AuroraDataset.from_manifest(train_manifest, data_root)
    val_dataset = AuroraDataset.from_manifest(val_manifest, data_root)
    
    if max_train_samples:
        train_dataset.file_paths = train_dataset.file_paths[:max_train_samples]
    if max_val_samples:
        val_dataset.file_paths = val_dataset.file_paths[:max_val_samples]
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }

