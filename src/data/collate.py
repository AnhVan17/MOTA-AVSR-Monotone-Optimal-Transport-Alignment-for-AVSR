import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for AVSR (Audio-Visual Speech Recognition).
    Handles padding for variable-length audio, visual inputs and text targets.

    Args:
        batch (List[Dict]): A list of samples from the Dataset. 
                            Each sample dict contains:
                            - 'audio': (T_a, 768) - Whisper encoder features
                            - 'visual': (T_v, 512) - ResNet features
                            - 'target': (L,) -> Token IDs
                            - 'text': str
                            - 'rel_path': str

    Returns:
        Dict: A batched dictionary ready for the model.
    """
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
        target_batch = pad_sequence(target_list, batch_first=True, padding_value=0)
        
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