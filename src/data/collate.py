from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function to pad visual inputs and targets.
    Audio inputs are already fixed size (80, 3000).
    """
    # 1. Process Audio
    # Stack fixed-size Mel spectrograms: (B, 80, 3000)
    audio_batch = torch.stack([s['audio'] for s in batch])
    
    # 2. Process Visual
    # Pad variable length sequences (frames or features)
    # Output: (B, T_max, 3, 224, 224) OR (B, T_max, 512)
    visual_list = [s['visual'] for s in batch]
    visual_batch = pad_sequence(visual_list, batch_first=True, padding_value=0.0)
    
    # Create visual mask for Attention/M-QOT (B, T_max)
    # 1 = Real frame, 0 = Padding
    batch_size = len(batch)
    max_visual_len = visual_batch.size(1)
    visual_mask = torch.zeros((batch_size, max_visual_len), dtype=torch.bool)
    
    for i, v in enumerate(visual_list):
        visual_mask[i, :v.size(0)] = True
        
    # 3. Process Targets
    # Pad with -100 (Ignore Index for CrossEntropyLoss)
    target_list = [s['target'] for s in batch]
    target_batch = pad_sequence(target_list, batch_first=True, padding_value=-100)
    
    # 4. Collect Texts
    texts = [s['text'] for s in batch]
    
    return {
        'audio': audio_batch,
        'visual': visual_batch,
        'visual_mask': visual_mask,
        'target': target_batch,
        'text': texts
    }