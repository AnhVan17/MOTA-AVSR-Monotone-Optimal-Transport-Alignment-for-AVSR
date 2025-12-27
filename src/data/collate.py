import torch
from typing import List, Dict

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Optimized collate function for AVSR using Whisper Mel + Visual features.
    
    Args:
        batch (List[Dict]): A list of samples from the Dataset. 
    """
    
    # 1. Mel Spectrograms (Fixed size 80x3000)
    # We allow for [80, 3000] or [T, 768] (legacy)
    first_audio = batch[0]['audio']
    if first_audio.dim() == 2 and first_audio.size(0) == 80:
        audio_batch = torch.stack([s['audio'] for s in batch])
    else:
        # Pad legacy features [T, 768]
        audio_list = [s['audio'] for s in batch]
        audio_batch = torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True)
    
    # 2. Visual features [T, 512]
    visual_list = [s['visual'] for s in batch]
    visual_batch = torch.nn.utils.rnn.pad_sequence(visual_list, batch_first=True)
    
    # Visual mask (True for real frames, False for padding)
    visual_mask = torch.zeros(len(batch), visual_batch.size(1), dtype=torch.bool)
    for i, v in enumerate(visual_list):
        visual_mask[i, :v.size(0)] = True
        
    # 3. Targets (Tokens)
    # Use padding_value = -100 (PyTorch CrossEntropyLoss ignore_index)
    if 'tokens' in batch[0]:
        token_list = [s['tokens'] for s in batch]
        target_batch = torch.nn.utils.rnn.pad_sequence(token_list, batch_first=True, padding_value=-100)
    elif 'target' in batch[0]: # Legacy key
        token_list = [s['target'] for s in batch]
        target_batch = torch.nn.utils.rnn.pad_sequence(token_list, batch_first=True, padding_value=-100)
    else:
        target_batch = None

    return {
        'audio': audio_batch,
        'visual': visual_batch,
        'visual_mask': visual_mask,
        'target': target_batch,
        'ids': [s.get('id', '') for s in batch],
        'text': [s.get('text', '') for s in batch]
    }