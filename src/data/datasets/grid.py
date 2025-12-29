import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np

class GridDataset(Dataset):
    """
    Dataset for GRID Corpus.
    Supports two modes:
    1. Precomputed Features (Phase 1): Loads .pt files containing {audio, visual, text}.
    2. Raw Video (Phase 2): Loads .mpg files and extracts features on-the-fly (TODO).
    """
    def __init__(self, manifest_path, tokenizer, data_root, use_precomputed_features=False, max_samples=None):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.use_precomputed_features = use_precomputed_features
        
        # Load manifest
        print(f"📄 Loading manifest: {manifest_path}")
        self.data = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        if max_samples:
            self.data = self.data[:max_samples]
            print(f" Limiting to {max_samples} samples.")
            
        print(f"   Found {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        rel_path = item['rel_path'] # Could be .pt or .mpg depending on manifest
        full_path = os.path.join(self.data_root, rel_path)
        
        text = item.get('text', "")
        
        # Tokenize Text
        # Output: (L,) tensor
        token_ids = self.tokenizer.encode(text)
        target = torch.tensor(token_ids, dtype=torch.long)
        
        if self.use_precomputed_features:
            # Phase 1: Load .pt feature file
            # If manifest points to .mpg, swap ext
            if full_path.endswith('.mpg'):
                full_path = full_path.replace('.mpg', '.pt')
            
            try:
                data = torch.load(full_path)
                # data is dict: {'visual': (T, 512), 'audio': (T, 768), 'text': str, ...}
                
                visual = data['visual'].float() # [T_v, 512]
                audio = data['audio'].float()   # [T_a, 768]
                
                return {
                    'audio': audio,
                    'visual': visual,
                    'target': target,
                    'text': text,
                    'rel_path': rel_path
                }
            except Exception as e:
                print(f"❌ Error loading {full_path}: {e}")
                # Return dummy
                return {
                    'audio': torch.zeros(300, 768),
                    'visual': torch.zeros(75, 512),
                    'target': target,
                    'text': text,
                    'rel_path': rel_path
                }
        else:
            # Phase 2: Raw Video Loading (To be implemented or migrated from legacy)
            raise NotImplementedError("Phase 2 (Raw Video) loading not yet implemented in GridDataset.")