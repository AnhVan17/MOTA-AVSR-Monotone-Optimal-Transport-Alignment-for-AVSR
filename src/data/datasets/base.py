import torch
import json
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):
    """
    Abstract base class for AVSR datasets.
    Ensures consistent data loading and error handling structure.
    """
    
    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            manifest_path: Path to .jsonl or .txt manifest file.
            tokenizer: Instance of WhisperTokenizer.
            max_samples: Limit number of samples (useful for debugging).
        """
        self.tokenizer = tokenizer
        self.manifest_path = manifest_path
        self.samples = self._load_manifest(manifest_path)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Dataset {self.__class__.__name__} loaded: {len(self.samples)} samples")
    
    def _load_manifest(self, path: str) -> List[Dict]:
        """Load manifest file (JSONL format expected)."""
        samples = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        samples.append(data)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Manifest file not found: {path}")
            return []
        return samples
    
    @abstractmethod
    def parse_sample(self, sample_data: Dict) -> Dict:
        """
        Abstract method to parse raw sample data.
        Must return a dictionary with:
            - 'audio': Tensor (80, 3000) -> Mel Spectrogram
            - 'visual': Tensor (T, 3, 224, 224) OR (T, 512)
            - 'text': str
        """
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves a sample with retry logic to prevent crashes on bad files.
        """
        max_retries = 5
        current_idx = idx
        
        for _ in range(max_retries):
            try:
                sample_data = self.samples[current_idx]
                
                # 1. Parse raw data (Audio/Visual processing happens here)
                parsed_sample = self.parse_sample(sample_data)
                
                # 2. Tokenize Text
                # Tokenizer wrapper adds SOT/EOT tokens automatically
                token_ids = self.tokenizer.encode(parsed_sample['text'])
                target = torch.tensor(token_ids, dtype=torch.long)
                
                return {
                    'audio': parsed_sample['audio'],
                    'visual': parsed_sample['visual'],
                    'target': target,
                    'text': parsed_sample['text']
                }
            
            except Exception as e:
                print(f"Error loading sample {current_idx}: {e}. Retrying next sample...")
                current_idx = (current_idx + 1) % len(self)
        
        raise RuntimeError(f"Failed to load samples after {max_retries} retries.")