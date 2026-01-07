"""
Whisper Vocabulary Pruning for Vietnamese AVSR
===============================================
Reduces vocab from 51,865 -> ~5,000 tokens for faster training

BENEFITS:
- 10x faster training
- 10x less memory
- Still maintains subword advantages
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple
from transformers import WhisperTokenizer
import os

class VietnameseVocabPruner:
    """
    Prune Whisper vocabulary to Vietnamese-only tokens
    """
    
    def __init__(self, model_name: str = "openai/whisper-small"):
        """
        Args:
            model_name: Whisper model name
        """
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language="vi", task="transcribe")
        self.full_vocab = self.tokenizer.get_vocab()
        self.full_vocab_size = len(self.full_vocab)
        
        # Define Vietnamese character set
        self.vietnamese_chars = set(
            'aàáảãạăắằẳẵặâấầẩẫậ'
            'bcdđ'
            'eèéẻẽẹêếềểễệ'
            'ghiìíỉĩị'
            'klmnoòóỏõọôốồổỗộơớờởỡợ'
            'pqrstuùúủũụưứừửữự'
            'vxyỳýỷỹỵ'
            'AÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ'
            'BCDĐ'
            'EÈÉẺẼẸÊẾỀỂỄỆ'
            'GHIÌÍỈĨỊ'
            'KLMNOÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ'
            'PQRSTUÙÚỦŨỤƯỨỪỬỮỰ'
            'VXYỲÝỶỸỴ'
        )
        
        # Common characters to keep
        self.common_chars = set(' 0123456789.,!?\'-()[]{}:;/@#$%&*+=<>')
        
        # Common loan words (English/French) to keep
        self.loan_words = {
            # English
            'email', 'internet', 'computer', 'online', 'offline',
            'ok', 'okay', 'bye', 'hello', 'hi', 'yes', 'no',
            'facebook', 'youtube', 'google',
            # French
            'café', 'merci', 'au revoir',
            # Numbers (spelled out)
            'zero', 'one', 'two', 'three', 'four', 'five',
            'six', 'seven', 'eight', 'nine', 'ten'
        }
        
        print(f"📊 Original vocab size: {self.full_vocab_size}")
    
    def _is_vietnamese_token(self, token: str) -> bool:
        """
        Check if token contains Vietnamese characters
        
        Args:
            token: Token string (may have 'Ġ' prefix for space)
            
        Returns:
            True if token is relevant for Vietnamese
        """
        # Clean token (Whisper uses 'Ġ' for leading space)
        clean_token = token.replace('Ġ', '').lower()
        
        # Empty token
        if not clean_token:
            return True  # Keep space tokens
        
        # 1. Has Vietnamese characters
        if any(c in self.vietnamese_chars for c in clean_token):
            return True
        
        # 2. Is common punctuation/symbol
        if any(c in self.common_chars for c in clean_token):
            return True
        
        # 3. Is loan word
        if clean_token in self.loan_words:
            return True
        
        # 4. Is very short (likely syllable or character)
        if len(clean_token) <= 2:
            return True
        
        # 5. Is pure ASCII letters (might be loan word)
        if clean_token.isalpha() and clean_token.isascii() and len(clean_token) <= 6:
            return True
        
        return False
    
    def filter_vocab(self) -> Tuple[Dict[str, int], Dict[int, int]]:
        """
        Filter vocabulary to Vietnamese-relevant tokens
        
        Returns:
            filtered_vocab: New vocab dict {token: new_id}
            id_mapping: Old ID -> New ID mapping
        """
        print("\n🔍 Filtering vocabulary...")
        
        # Keep special tokens
        special_tokens = {
            '<|endoftext|>': 50256,
            '<|startoftranscript|>': 50257,
            '<|vi|>': 50268,  # Vietnamese language token
            '<|notimestamps|>': 50363,
            '<|transcribe|>': 50359,
        }
        
        vietnamese_tokens = {}
        
        # Add special tokens
        for token, old_id in special_tokens.items():
            # Check if token exists in full vocab (some special tokens might not be dict keys directly)
            # The tokenizer usually handles them separately, but get_vocab() includes them if added.
            # However, for whisper, some special tokens are outside of vocab keys but accessible via convert_tokens_to_ids.
            # We rely on passing the correct ID.
            vietnamese_tokens[token] = old_id
        
        # Filter content tokens
        for token, old_id in self.full_vocab.items():
            if old_id in special_tokens.values():
                continue  # Already added
            
            if self._is_vietnamese_token(token):
                vietnamese_tokens[token] = old_id
        
        # Create new vocab with sequential IDs
        filtered_vocab = {}
        id_mapping = {}
        
        # Sort by old ID for consistency
        sorted_tokens = sorted(vietnamese_tokens.items(), key=lambda x: x[1])
        
        for new_id, (token, old_id) in enumerate(sorted_tokens):
            filtered_vocab[token] = new_id
            id_mapping[old_id] = new_id
        
        new_vocab_size = len(filtered_vocab)
        reduction = (1 - new_vocab_size / self.full_vocab_size) * 100
        
        print(f"✅ Filtered vocab size: {new_vocab_size}")
        print(f"📉 Reduction: {reduction:.1f}%")
        print(f"⚡ Training speed: ~{self.full_vocab_size / new_vocab_size:.1f}x faster")
        print(f"💾 Memory: ~{self.full_vocab_size / new_vocab_size:.1f}x less")
        
        return filtered_vocab, id_mapping
    
    def save_mapping(self, output_dir: str):
        """
        Save filtered vocab and ID mapping
        
        Args:
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filtered_vocab, id_mapping = self.filter_vocab()
        
        # Save filtered vocab
        vocab_file = output_path / "vietnamese_vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_vocab, f, ensure_ascii=False, indent=2)
        
        # Save ID mapping
        mapping_file = output_path / "id_mapping.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump(id_mapping, f)
        
        # Save reverse mapping (for debugging)
        reverse_mapping = {v: k for k, v in filtered_vocab.items()}
        reverse_file = output_path / "reverse_vocab.json"
        with open(reverse_file, 'w', encoding='utf-8') as f:
            json.dump(reverse_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Saved to {output_dir}/")
        print(f"   - vietnamese_vocab.json (filtered vocab)")
        print(f"   - id_mapping.pkl (old->new ID mapping)")
        print(f"   - reverse_vocab.json (new_id->token)")
    
    def remap_tokens(self, token_ids: List[int], id_mapping: Dict[int, int]) -> List[int]:
        """
        Remap token IDs from old vocab to new vocab
        
        Args:
            token_ids: List of old token IDs
            id_mapping: Old ID -> New ID mapping
            
        Returns:
            List of new token IDs (OOV tokens removed)
        """
        new_ids = []
        for old_id in token_ids:
            if old_id in id_mapping:
                new_ids.append(id_mapping[old_id])
            # else: skip OOV token
        
        return new_ids
    
    def test_pruning(self):
        """Test pruning with example Vietnamese text"""
        print("\n🧪 Testing pruning...")
        
        test_texts = [
            "xin chào việt nam",
            "tôi tên là gì",
            "một hai ba bốn năm",
            "tôi gửi email cho bạn",  # loan word
            "ChatGPT rất hay"  # proper noun
        ]
        
        filtered_vocab, id_mapping = self.filter_vocab()
        
        for text in test_texts:
            # Encode with original tokenizer
            old_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Remap to new vocab
            new_ids = self.remap_tokens(old_ids, id_mapping)
            
            # Decode back
            old_tokens = [self.tokenizer.decode([id]) for id in old_ids]
            
            print(f"\n📝 Text: '{text}'")
            print(f"   Old IDs: {old_ids}")
            print(f"   New IDs: {new_ids}")
            print(f"   Tokens: {old_tokens}")
            print(f"   Coverage: {len(new_ids)}/{len(old_ids)} tokens kept")


def main():
    """Demo usage"""
    pruner = VietnameseVocabPruner(model_name="openai/whisper-small")
    
    # Test pruning
    pruner.test_pruning()
    
    # Save mapping
    # Ensure directory exists relative to project root or current working dir
    output_dir = "src/data/vocab_pruned"
    pruner.save_mapping(output_dir=output_dir)
    
    print(f"\n✅ Complete! Use 'id_mapping.pkl' from '{output_dir}' to remap your dataset.")


if __name__ == "__main__":
    main()
