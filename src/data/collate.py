
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

# SOTA standard: Use -100 for ignore_index in PyTorch Loss functions
IGNORE_INDEX = -100 

def avsr_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for AVSR training (Grid & ViCocktail).
    
    Handles:
    - Variable length Audio & Visual inputs (Padding)
    - Target Token Padding (using -100 for Ignore Index)
    - Attention Masks (Bool)
    - Sequence Lengths (Int) - needed for CTC Loss
    
    Compatible with:
    - FeatureDataset (Phase 1)
    - RawVideoDataset (Phase 2)
    """
    # Filter out None or invalid samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    # 1. Audio (Pad with 0.0)
    audio_list = [s['audio'] for s in batch]
    
    # CRITICAL FIX: Use actual lengths from dataset if available
    # The dataset now computes actual lengths by detecting trailing zeros
    if 'audio_len' in batch[0]:
        audio_lens = torch.tensor([s['audio_len'] for s in batch], dtype=torch.long)
    else:
        # Fallback: use tensor size (may include padding!)
        audio_lens = torch.tensor([a.size(0) for a in audio_list], dtype=torch.long)
    
    audio_batch = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    
    # Create Attention Mask (True = Valid, False = Pad)
    # [B, T_max]
    B, T_a_max = audio_batch.shape[:2]
    audio_mask = torch.arange(T_a_max).expand(B, T_a_max) < audio_lens.unsqueeze(1)

    # 2. Visual (Pad with 0.0)
    visual_list = [s['visual'] for s in batch]
    
    # CRITICAL FIX: Use actual lengths from dataset if available
    if 'visual_len' in batch[0]:
        visual_lens = torch.tensor([s['visual_len'] for s in batch], dtype=torch.long)
    else:
        visual_lens = torch.tensor([v.size(0) for v in visual_list], dtype=torch.long)
    
    visual_batch = pad_sequence(visual_list, batch_first=True, padding_value=0.0)
    
    B, T_v_max = visual_batch.shape[:2]
    visual_mask = torch.arange(T_v_max).expand(B, T_v_max) < visual_lens.unsqueeze(1)

    # 3. Targets (Pad with IGNORE_INDEX for Loss calculation)
    # Whisper pad token is usually 50257, but for Loss we want -100
    target_batch = None
    target_lens = None
    
    if 'target' in batch[0]:
        target_list = [s['target'] for s in batch]
        target_lens = torch.tensor([t.size(0) for t in target_list], dtype=torch.long)
        
        # IMPORTANT: Pad with -100 so CrossEntropyLoss ignores it automatically
        # This is CRITICAL for variable length transcriptions (Grid is fixed, but Vicocktail is NOT)
        target_batch = pad_sequence(target_list, batch_first=True, padding_value=IGNORE_INDEX)

    return {
        'audio': audio_batch,        # [B, T_a, D_a]
        'visual': visual_batch,      # [B, T_v, D_v]
        'audio_len': audio_lens,     # [B]
        'visual_len': visual_lens,   # [B]
        'audio_mask': audio_mask,    # [B, T_a] (Bool)
        'visual_mask': visual_mask,  # [B, T_v] (Bool)
        'target': target_batch,      # [B, L]
        'target_len': target_lens,   # [B]
        'text': [s.get('text', '') for s in batch],
        'rel_paths': [s.get('rel_path', '') for s in batch]
    }