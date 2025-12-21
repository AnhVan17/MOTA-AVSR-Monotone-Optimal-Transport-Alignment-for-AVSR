import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for AVSR (Audio-Visual Speech Recognition).
    Handles padding for variable-length visual inputs and text targets.

    Args:
        batch (List[Dict]): A list of samples from the Dataset. 
                            Each sample dict contains:
                            - 'audio': (80, 3000)
                            - 'visual': (T, C, H, W) or (T, Feature_Dim)
                            - 'target': (L,) -> Token IDs
                            - 'text': str
                            - 'rel_path': str

    Returns:
        Dict: A batched dictionary ready for the model.
    """
    
    # 1. Process Audio
    # Since Whisper requires fixed input (30s), audio is already (80, 3000).
    # We just stack them: (Batch_Size, 80, 3000)
    audio_batch = torch.stack([s['audio'] for s in batch])
    
    # 2. Process Visual
    # Visual frames/features have variable length T. We need to pad them.
    visual_list = [s['visual'] for s in batch]
    
    # Pad sequence with 0.0. 
    # batch_first=True makes output (Batch, T_max, ...)
    visual_batch = pad_sequence(visual_list, batch_first=True, padding_value=0.0)
    
    # Create Attention Mask / Time Mask for Visual inputs
    # 1 (True) = Real frame, 0 (False) = Padding
    # Shape: (Batch_Size, T_max)
    batch_size = len(batch)
    max_visual_len = visual_batch.size(1)
    visual_mask = torch.zeros((batch_size, max_visual_len), dtype=torch.bool)
    
    for i, v in enumerate(visual_list):
        # Mark valid frames as True
        visual_mask[i, :v.size(0)] = True
        
    # 3. Process Targets (Labels)
    # Token sequences also have variable lengths.
    # Use padding_value = -100 (PyTorch CrossEntropyLoss ignore_index)
    if 'target' in batch[0]:
        target_list = [s['target'] for s in batch]
        target_batch = pad_sequence(target_list, batch_first=True, padding_value=-100)
    else:
        # Fallback if target is missing (e.g., inference mode without labels)
        target_batch = None

    # 4. Collect Metadata
    # Useful for debugging or calculating PER/WER later
    texts = [s.get('text', '') for s in batch]
    rel_paths = [s.get('rel_path', '') for s in batch]
    
    return {
        'audio': audio_batch,       # (B, 80, 3000)
        'visual': visual_batch,     # (B, T_max, C, H, W) or (B, T_max, 512)
        'visual_mask': visual_mask, # (B, T_max)
        'target': target_batch,     # (B, L_max)
        'text': texts,              # List[str]
        'rel_paths': rel_paths      # List[str]
    }