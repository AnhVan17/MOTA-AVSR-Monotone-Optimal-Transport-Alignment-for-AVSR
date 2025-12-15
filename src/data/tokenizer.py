import torch
from typing import List, Union
import unicodedata


class VietnameseCharTokenizer:
    """
    Character-level tokenizer for Vietnamese AVSR
    
    Vocab size: 220
    - Special tokens: 5 (PAD, BOS, EOS, UNK, BLANK)
    - Characters: 215 (Vietnamese + punctuation)
    """
    
    def __init__(self):
        self.special_tokens = {
            '<PAD>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3,
            '<BLANK>': 4,  
        }

        chars = self._build_vietnamese_chars()
        self.vocab = list(self.special_tokens.keys()) + chars
        
        self.char2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2char = {i: c for c, i in self.char2id.items()}
        
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.blank_token_id = 4
        
        self.vocab_size = len(self.vocab)
        
        print(f"✅ VietnameseCharTokenizer initialized: {self.vocab_size} tokens")
    
    @staticmethod
    def _build_vietnamese_chars() -> List[str]:
        chars = []    

        chars.append(' ')      
        chars.extend(list('0123456789'))
        chars.extend(list('abcdefghijklmnopqrstuvwxyz'))
        

        chars.extend([
            'à', 'á', 'ả', 'ã', 'ạ', 
            'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',  
            'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ',  
        ])
        
        chars.extend([
            'è', 'é', 'ẻ', 'ẽ', 'ẹ',  
            'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ',  
        ])
        

        chars.extend(['ì', 'í', 'ỉ', 'ĩ', 'ị'])
        
        chars.extend([
            'ò', 'ó', 'ỏ', 'õ', 'ọ',  
            'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ',  
            'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ',  
        ])
        
        chars.extend([
            'ù', 'ú', 'ủ', 'ũ', 'ụ',  
            'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự',  
        ])
        
        chars.extend(['ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'])
        
        chars.append('đ')
        
        chars.extend(['.', ',', '!', '?', '-', ':', ';', '"', "'", '(', ')', '/'])
        
        return chars
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input Vietnamese text
            add_special_tokens: Add BOS/EOS
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        text = unicodedata.normalize('NFC', text.lower().strip())
        
        ids = []
        
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        for char in text:
            char_id = self.char2id.get(char, self.unk_token_id)
            ids.append(char_id)
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        return ids
    
    def decode(
        self, 
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text
        
        Args:
            ids: Token IDs (list or tensor)
            skip_special_tokens: Skip special tokens
            
        Returns:
            Decoded text
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        chars = []
        for token_id in ids:
            if skip_special_tokens and token_id < 5:
                continue
            
            char = self.id2char.get(token_id, '')
            if char and not char.startswith('<'):
                chars.append(char)
        
        return ''.join(chars)
    
    def batch_decode(
        self,
        batch_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Batch decode"""
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in batch_ids
        ]
