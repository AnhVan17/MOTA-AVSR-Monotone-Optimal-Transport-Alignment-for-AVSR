import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

class Collator:
    """
    Custom collator with configurable padding ID.

    Collates variable-length sequences into padded batches.
    Returns None for empty batches (caller handles gracefully).
    """

    def __init__(self, pad_id: int = 50257):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict]) -> Dict:
        # Guard: empty batch from DataLoader
        if not batch:
            return None

        batch_size = len(batch)
        
        # 1. Process Audio (variable length from Whisper encoder)
        audio_list = [s['audio'] for s in batch]
        audio_batch = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
        
        # Audio mask
        max_audio_len = audio_batch.size(1)
        audio_mask = torch.zeros((batch_size, max_audio_len), dtype=torch.bool)
        for i, a in enumerate(audio_list):
            audio_mask[i, :a.size(0)] = True
        
        # 2. Process Visual (variable length)
        visual_list = [s['visual'] for s in batch]
        visual_batch = pad_sequence(visual_list, batch_first=True, padding_value=0.0)
        
        # Visual mask
        max_visual_len = visual_batch.size(1)
        visual_mask = torch.zeros((batch_size, max_visual_len), dtype=torch.bool)
        for i, v in enumerate(visual_list):
            visual_mask[i, :v.size(0)] = True
            
        # 3. Process Targets (Labels)
        if 'target' in batch[0]:
            target_list = [s['target'] for s in batch]
            # Use Tokenizer's pad_id (e.g. 50257 for Whisper)
            target_batch = pad_sequence(target_list, batch_first=True, padding_value=self.pad_id)
            
            # Target mask (1 = valid, 0 = padding)
            target_mask = torch.zeros((batch_size, target_batch.size(1)), dtype=torch.bool)
            for i, t in enumerate(target_list):
                target_mask[i, :t.size(0)] = True
        else:
            target_batch = None
            target_mask = None
    
        # 4. Collect Metadata
        texts = [s.get('text', '') for s in batch]
        rel_paths = [s.get('rel_path', '') for s in batch]
        
        return {
            'audio': audio_batch,       # (B, T_a_max, 768)
            'visual': visual_batch,     # (B, T_v_max, 512)
            'audio_mask': audio_mask,   # (B, T_a_max)
            'visual_mask': visual_mask, # (B, T_v_max)
            'target': target_batch,     # (B, L_max)
            'target_mask': target_mask, # (B, L_max)
            'text': texts,              # List[str]
            'rel_paths': rel_paths      # List[str]
        }