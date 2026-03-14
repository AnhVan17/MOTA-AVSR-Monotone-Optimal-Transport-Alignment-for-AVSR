"""
Prune Whisper Vocabulary for Vietnamese - SUPER STRICT
========================================================
ONLY keeps tokens with:
1. Vietnamese diacritics (ă, ơ, ư, đ, ầ, etc.)
2. Explicit Vietnamese word whitelist
3. Punctuation/digits

Target: ~2,000-4,000 tokens (NOT 40,000!)
"""

import json
import pickle
from pathlib import Path
from transformers import WhisperTokenizer


# Vietnamese-SPECIFIC characters (unique to Vietnamese)
VIETNAMESE_DIACRITICS = set(
    'àáảãạăắằẳẵặâấầẩẫậ'
    'èéẻẽẹêếềểễệ'
    'ìíỉĩị'
    'òóỏõọôốồổỗộơớờởỡợ'
    'ùúủũụưứừửữự'
    'ỳýỷỹỵ'
    'đ'
    'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ'
    'ÈÉẺẼẸÊẾỀỂỄỆ'
    'ÌÍỈĨỊ'
    'ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ'
    'ÙÚỦŨỤƯỨỪỬỮỰ'
    'ỲÝỶỸỴ'
    'Đ'
)

# Common Vietnamese words WITHOUT special diacritics
VIETNAMESE_WHITELIST = {
    # Common words
    'la', 'va', 'cua', 'co', 'duoc', 'trong', 'nay', 'cho', 'voi', 'nhu',
    'nhung', 'cac', 've', 'de', 'theo', 'da', 'hoac', 'khong', 'nguoi',
    'hay', 'tu', 'ra', 'khi', 'den', 'con', 'neu', 'cung', 'thi', 'vi', 'ma',
    'ban', 'toi', 'em', 'anh', 'chi', 'ong', 'ba', 'chu',
    'xin', 'chao', 'cam', 'on', 'vang', 'dung', 'sai',
    'nhe', 'nha', 'nghe', 'biet', 'lam', 'di', 'gi', 'nao', 'the', 'cai',
    'noi', 'hoi', 'tra', 'loi', 'giup', 'can', 'muon', 'thich',
    # Numbers
    'khong', 'mot', 'hai', 'ba', 'bon', 'nam', 'sau', 'bay', 'tam', 'chin', 'muoi',
    'tram', 'nghin', 'ngan', 'trieu', 'ty',
    # Single consonants (for BPE)
    'b', 'c', 'd', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x',
    # Common syllables
    'an', 'am', 'ai', 'ao', 'au', 'ay', 'ang', 'anh', 'ach',
    'en', 'em', 'et', 'eng', 'enh', 'ech',
    'in', 'im', 'it', 'inh', 'ich',
    'on', 'om', 'oi', 'ong', 'onh', 'oc',
    'un', 'um', 'ui', 'ung', 'ut', 'uc',
    'yen', 'uyen', 'oan', 'uat',
}

# Allowed punctuation
PUNCTUATION = set(' .,!?;:\'-"()[]{}0123456789')


def has_vietnamese_diacritic(text: str) -> bool:
    """Check if text has Vietnamese diacritics"""
    return any(c in VIETNAMESE_DIACRITICS for c in text)


def has_foreign_script(text: str) -> bool:
    """Check for foreign scripts"""
    for c in text:
        # Korean
        if '\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff':
            return True
        # CJK
        if '\u4e00' <= c <= '\u9fff':
            return True
        # Japanese
        if '\u3040' <= c <= '\u30ff':
            return True
        # Cyrillic
        if '\u0400' <= c <= '\u04ff':
            return True
        # Arabic
        if '\u0600' <= c <= '\u06ff':
            return True
        # Thai
        if '\u0e00' <= c <= '\u0e7f':
            return True
        # Greek
        if '\u0370' <= c <= '\u03ff':
            return True
    return False


def is_vietnamese_token(token: str) -> bool:
    """
    SUPER STRICT check - only Vietnamese tokens
    
    Returns True ONLY if:
    1. Has Vietnamese diacritics, OR
    2. Is in explicit whitelist, OR  
    3. Is pure punctuation/digits
    """
    # Clean token
    clean = token.replace('Ġ', '').replace('Ċ', '').strip()
    lower = clean.lower()
    
    # Empty - keep
    if not clean:
        return True
    
    # Block foreign scripts immediately
    if has_foreign_script(clean):
        return False
    
    # Priority 1: Has Vietnamese diacritics - DEFINITELY keep
    if has_vietnamese_diacritic(clean):
        return True
    
    # Priority 2: In explicit whitelist
    if lower in VIETNAMESE_WHITELIST:
        return True
    
    # Priority 3: Pure punctuation/digits
    if all(c in PUNCTUATION for c in clean):
        return True
    
    # Priority 4: Single letter (for BPE tokenization)
    if len(clean) == 1 and clean.isalpha():
        return True
    
    # BLOCK everything else - including English words like "using", "white", "party"
    return False


def prune_whisper_vocab(
    model_name: str = "openai/whisper-small",
    output_dir: str = "src/data/vocab_pruned"
):
    """Prune Whisper vocabulary to Vietnamese-only"""
    print("="*80)
    print("🔪 PRUNING VOCABULARY (SUPER STRICT)")
    print("="*80)
    
    # Load tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name, language="vi", task="transcribe"
    )
    
    full_vocab = tokenizer.get_vocab()
    print(f"\n📊 Original vocab: {len(full_vocab)}")
    
    # Always keep these special tokens
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.unk_token_id,
        tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
        tokenizer.convert_tokens_to_ids("<|endoftranscript|>"),
        tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
        tokenizer.convert_tokens_to_ids("<|vi|>"),
    }
    special_ids = {x for x in special_ids if x is not None}
    print(f"   Special tokens: {len(special_ids)}")
    
    # Filter
    kept_ids = set(special_ids)
    blocked_samples = []
    kept_viet_samples = []
    
    for token, token_id in full_vocab.items():
        if token_id in special_ids:
            continue
        
        if is_vietnamese_token(token):
            kept_ids.add(token_id)
            if has_vietnamese_diacritic(token.replace('Ġ', '')) and len(kept_viet_samples) < 20:
                kept_viet_samples.append((token_id, token))
        else:
            if len(blocked_samples) < 30:
                blocked_samples.append((token_id, token))
    
    # Show blocked samples
    print(f"\n❌ Blocked samples:")
    for tid, tok in blocked_samples[:15]:
        print(f"   {tid:5d}: '{tok}'")
    
    # Show kept Vietnamese samples
    print(f"\n✅ Kept Vietnamese samples:")
    for tid, tok in kept_viet_samples[:15]:
        print(f"   {tid:5d}: '{tok}'")
    
    # Create mapping
    sorted_ids = sorted(kept_ids)
    id_mapping = {old: new for new, old in enumerate(sorted_ids)}
    
    print(f"\n📊 Final vocab: {len(id_mapping)}")
    print(f"   Reduction: {(1 - len(id_mapping)/len(full_vocab))*100:.1f}%")
    
    if len(id_mapping) > 10000:
        print("⚠️ WARNING: Still too large! Check filtering.")
    else:
        print("✅ Good size!")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "id_mapping.pkl", 'wb') as f:
        pickle.dump(id_mapping, f)
    
    metadata = {
        'original_size': len(full_vocab),
        'pruned_size': len(id_mapping),
        'reduction': f"{(1 - len(id_mapping)/len(full_vocab))*100:.1f}%"
    }
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Saved to: {output_dir}/")
    
    # Test
    print("\n🧪 Testing:")
    tests = [
        ("xin chào việt nam", True),
        ("một hai ba bốn", True),
        ("using white party", False),  # English - should be blocked
        ("Snapdragon", False),
    ]
    
    for text, expected in tests:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        in_vocab = sum(1 for t in tokens if t in id_mapping)
        ratio = in_vocab / len(tokens) if tokens else 0
        passed = (ratio > 0.5) == expected
        status = "✅" if passed else "❌"
        print(f"   {status} '{text}': {ratio*100:.0f}% (expected: {'IN' if expected else 'OUT'})")
    
    print("="*80)
    return id_mapping


if __name__ == "__main__":
    prune_whisper_vocab(
        model_name="openai/whisper-small",
        output_dir="/root/src/data/vocab_pruned"
    )