import torch
from typing import List

class CTCDecoder:
    """
    Handles CTC Decoding strategies (Greedy, and potentially Beam Search).
    """
    
    def __init__(self, tokenizer, blank_id: int = 4):
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        
    def greedy_decode(self, logits: torch.Tensor) -> List[str]:
        """
        Greedy decode logits to text.
        
        Args:
            logits: [B, T, V]
            
        Returns:
            List of decoded strings
        """
        pred_ids = logits.argmax(dim=-1)
        decoded_texts = []
        
        for seq in pred_ids:
            unique_tokens = []
            prev = None
            for token in seq:
                token_id = token.item()
                if token_id != prev:
                    unique_tokens.append(token_id)
                    prev = token_id
            
            # Filter blank
            unique_tokens = [t for t in unique_tokens if t != self.blank_id]
            
            # Convert to string
            if len(unique_tokens) > 0:
                text = self.tokenizer.decode(unique_tokens, skip_special_tokens=True)
            else:
                text = ""
            decoded_texts.append(text.strip())
            
        return decoded_texts
        
    def decode_targets(self, targets: torch.Tensor) -> List[str]:
        """
        Decode target tensor to text.
        """
        decoded_texts = []
        for seq in targets:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            decoded_texts.append(text.strip())
        return decoded_texts
