"""
✅ CORRECT Vietnamese Vocabulary Creation
==========================================
Creates ACTUAL Vietnamese-only vocab from corpus

Key insight: We can't guess which tokens are Vietnamese.
We MUST analyze the actual training data to find which tokens appear.
"""

import json
import pickle
from pathlib import Path
from transformers import WhisperTokenizer
from collections import Counter
import re


def is_vietnamese_char(c: str) -> bool:
    """Check if character is valid Vietnamese"""
    # Vietnamese alphabet and diacritics
    vietnamese_chars = set(
        'aàáảãạăằắẳẵặâầấẩẫậ'
        'eèéẻẽẹêềếểễệ'
        'iìíỉĩị'
        'oòóỏõọôồốổỗộơờớởỡợ'
        'uùúủũụưừứửữự'
        'yỳýỷỹỵ'
        'đ'
        'AÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ'
        'EÈÉẺẼẸÊỀẾỂỄỆ'
        'IÌÍỈĨỊ'
        'OÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ'
        'UÙÚỦŨỤƯỪỨỬỮỰ'
        'YỲÝỶỸỴ'
        'Đ'
    )
    
    # Also allow: basic latin, numbers, punctuation, whitespace
    if c.isascii():
        return True
    
    return c in vietnamese_chars


def is_vietnamese_token(token: str) -> bool:
    """
    Check if a token is Vietnamese
    
    Returns True if:
    - Token contains ONLY Vietnamese-compatible characters
    - No foreign scripts (Korean, Cyrillic, CJK, Arabic, etc.)
    """
    if not token:
        return True
    
    # Decode BPE special chars
    clean_token = token.replace('Ġ', ' ').replace('Ċ', '\n')
    
    # Check each character
    for c in clean_token:
        # Whitespace and control chars are OK
        if c.isspace() or ord(c) < 32:
            continue
        
        if not is_vietnamese_char(c):
            return False
    
    return True


def create_vietnamese_vocab_from_corpus(
    manifest_paths: list,
    output_dir: str = "src/data/vocab_pruned",
    model_name: str = "openai/whisper-small",
    min_frequency: int = 1
):
    """
    Create Vietnamese vocab by analyzing actual transcripts
    
    This is the CORRECT approach:
    1. Load all transcripts from training data
    2. Tokenize each transcript
    3. Count token frequencies
    4. Keep ONLY tokens that actually appear in Vietnamese data
    
    Args:
        manifest_paths: List of JSONL manifest files
        output_dir: Where to save vocab mapping
        model_name: Whisper model name
        min_frequency: Minimum times a token must appear
    """
    print("="*80)
    print("✅ CREATING VIETNAMESE VOCAB FROM CORPUS")
    print("="*80)
    
    # Load tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        language="vi",
        task="transcribe"
    )
    
    full_vocab_size = tokenizer.vocab_size
    print(f"\n📊 Original vocab size: {full_vocab_size}")
    
    # Collect ALL tokens from corpus
    print(f"\n📖 Analyzing transcripts...")
    token_freq = Counter()
    total_texts = 0
    
    for manifest_path in manifest_paths:
        if not Path(manifest_path).exists():
            print(f"   ⚠️ Skipping (not found): {manifest_path}")
            continue
        
        print(f"   Processing: {manifest_path}")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', data.get('transcript', ''))
                    
                    if text:
                        # Tokenize WITHOUT special tokens (we want content only)
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        token_freq.update(tokens)
                        total_texts += 1
                except:
                    continue
    
    print(f"\n📊 Corpus stats:")
    print(f"   Total texts: {total_texts}")
    print(f"   Unique tokens: {len(token_freq)}")
    
    if len(token_freq) == 0:
        print("❌ ERROR: No tokens found in corpus!")
        print("   Check your manifest paths and text field names.")
        return None
    
    # Filter by frequency
    frequent_tokens = {tid for tid, count in token_freq.items() if count >= min_frequency}
    print(f"   Tokens with freq >= {min_frequency}: {len(frequent_tokens)}")
    
    # Get special tokens (ALWAYS include)
    special_token_ids = set()
    special_tokens = [
        tokenizer.pad_token,
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.unk_token,
        "<|startoftranscript|>",
        "<|endoftranscript|>",
        "<|notimestamps|>",
        "<|vi|>",
    ]
    
    for token in special_tokens:
        if token:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and token_id < full_vocab_size:
                    special_token_ids.add(token_id)
            except:
                pass
    
    print(f"   Special tokens: {len(special_token_ids)}")
    
    # Combine: corpus tokens + special tokens
    final_token_ids = frequent_tokens | special_token_ids
    
    # Additional safety: filter by Vietnamese character check
    vietnamese_only = set()
    non_vietnamese = []
    
    for tid in final_token_ids:
        token = tokenizer.convert_ids_to_tokens(tid)
        if is_vietnamese_token(token):
            vietnamese_only.add(tid)
        else:
            non_vietnamese.append((tid, token))
    
    if non_vietnamese:
        print(f"\n⚠️ Filtered out {len(non_vietnamese)} non-Vietnamese tokens:")
        for tid, token in non_vietnamese[:10]:
            print(f"      {tid}: '{token}'")
        if len(non_vietnamese) > 10:
            print(f"      ... and {len(non_vietnamese) - 10} more")
    
    # Create ID mapping (old_id -> new_id)
    # Sort to maintain consistency
    final_token_ids_sorted = sorted(vietnamese_only)
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(final_token_ids_sorted)}
    
    # Stats
    print(f"\n📝 Final vocabulary:")
    print(f"   Size: {len(id_mapping)} tokens")
    print(f"   Reduction: {(1 - len(id_mapping) / full_vocab_size) * 100:.1f}%")
    
    # Show top tokens
    print(f"\n   Top 20 most frequent tokens:")
    for tid, count in token_freq.most_common(20):
        token = tokenizer.convert_ids_to_tokens(tid)
        new_id = id_mapping.get(tid, "FILTERED")
        print(f"      {tid:5d} -> {str(new_id):5s} ({count:5d}x): '{token}'")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save pickle mapping
    mapping_file = output_path / "id_mapping.pkl"
    with open(mapping_file, 'wb') as f:
        pickle.dump(id_mapping, f)
    print(f"\n💾 Saved: {mapping_file}")
    
    # Save JSON for debugging
    json_file = output_path / "id_mapping.json"
    mapping_readable = {
        str(old_id): {
            "new_id": new_id,
            "token": tokenizer.convert_ids_to_tokens(old_id),
            "freq": token_freq.get(old_id, 0)
        }
        for old_id, new_id in id_mapping.items()
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_readable, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved: {json_file}")
    
    # Save metadata
    metadata = {
        'original_vocab_size': full_vocab_size,
        'pruned_vocab_size': len(id_mapping),
        'reduction_percent': (1 - len(id_mapping) / full_vocab_size) * 100,
        'model_name': model_name,
        'special_token_ids': list(special_token_ids),
        'total_texts_analyzed': total_texts,
        'min_frequency': min_frequency
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved: {metadata_file}")
    
    # Verification
    print(f"\n✅ VERIFICATION:")
    test_texts = [
        "xin chào việt nam",
        "để mà phát hiện ra",
        "cảm ơn bạn rất nhiều"
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        remapped = [id_mapping.get(t, -1) for t in tokens]
        all_valid = all(r >= 0 for r in remapped)
        status = "✅" if all_valid else "❌"
        print(f"   {status} '{text}'")
        print(f"      Original: {tokens}")
        print(f"      Remapped: {remapped}")
    
    print("="*80)
    print(f"✅ VOCAB CREATION COMPLETE: {len(id_mapping)} tokens")
    print("="*80)
    
    return id_mapping


if __name__ == "__main__":
    # Example usage for local development
    # For Modal, adjust paths accordingly
    
    import sys
    
    if len(sys.argv) > 1:
        manifest_paths = sys.argv[1:]
    else:
        # Default paths (adjust as needed)
        manifest_paths = [
            "data/manifests/train.jsonl",
            "data/manifests/val.jsonl",
            # Add more manifests here
        ]
    
    create_vietnamese_vocab_from_corpus(
        manifest_paths=manifest_paths,
        output_dir="src/data/vocab_pruned",
        min_frequency=1
    )
