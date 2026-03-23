"""
Quick script to verify preprocessing data is correct.
Run: python scripts/verify_data.py
"""
import torch
import json
import sys
sys.path.insert(0, ".")

from src.data.tokenizers.whisper import WhisperTokenizer

# Paths - adjust if needed
MANIFEST_PATH = "path/to/train.jsonl"  # Update this
DATA_ROOT = "path/to/features"  # Update this

def main():
    print("=" * 60)
    print("🔍 Verifying Preprocessing Data")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = WhisperTokenizer()
    print(f"✅ Tokenizer loaded: vocab_size = {len(tokenizer)}")
    
    # Load a few samples from manifest
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()][:5]
    
    print(f"✅ Loaded {len(samples)} samples from manifest")
    
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i} ---")
        rel_path = sample['rel_path']
        text = sample.get('text', '')
        
        # Check text & tokens
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        print(f"   Text: {text}")
        print(f"   Tokens: {tokens[:10]}... (len={len(tokens)})")
        print(f"   Decoded: {decoded}")
        print(f"   Token range: min={min(tokens)}, max={max(tokens)}")
        
        # Load .pt file
        pt_path = f"{DATA_ROOT}/{rel_path}"
        if not pt_path.endswith('.pt'):
            pt_path = pt_path.rsplit('.', 1)[0] + '.pt'
        
        try:
            data = torch.load(pt_path)
            audio = data['audio']
            visual = data['visual']
            
            print(f"   Audio: shape={audio.shape}, dtype={audio.dtype}")
            print(f"          min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.4f}")
            print(f"   Visual: shape={visual.shape}, dtype={visual.dtype}")
            print(f"          min={visual.min():.4f}, max={visual.max():.4f}, mean={visual.mean():.4f}")
            
            # Check for NaN/Inf
            if torch.isnan(audio).any():
                print("   ⚠️ WARNING: Audio contains NaN!")
            if torch.isnan(visual).any():
                print("   ⚠️ WARNING: Visual contains NaN!")
            if torch.isinf(audio).any():
                print("   ⚠️ WARNING: Audio contains Inf!")
            if torch.isinf(visual).any():
                print("   ⚠️ WARNING: Visual contains Inf!")
                
        except Exception as e:
            print(f"   ❌ Error loading {pt_path}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Verification complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
