"""
Check processed data for invalid token IDs
"""
import torch
import json
from pathlib import Path

# Config
MANIFEST_PATH = "data/manifests/train.jsonl"
MAX_SAMPLES_CHECK = 10
VOCAB_SIZE = 50258

print("=" * 70)
print("Checking Processed Data for Invalid Tokens")
print("=" * 70)
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Valid token range: 0 to {VOCAB_SIZE - 1}")
print()

# Read manifest
if not Path(MANIFEST_PATH).exists():
    print(f"❌ Manifest not found: {MANIFEST_PATH}")
    print("   Run this on Modal volume or copy manifest locally")
    exit(1)

with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f if line.strip()]

print(f"📊 Found {len(samples)} samples in manifest")
print(f"Checking first {MAX_SAMPLES_CHECK} samples...\n")

invalid_found = False

for i, sample in enumerate(samples[:MAX_SAMPLES_CHECK]):
    pt_path = Path(sample['path'])
    
    if not pt_path.exists():
        print(f"Sample {i+1}: ⚠️  File not found: {pt_path}")
        continue
    
    try:
        data = torch.load(pt_path, map_location='cpu')
        tokens = data['tokens']
        
        # Check for invalid token IDs
        max_token = tokens.max().item()
        min_token = tokens.min().item()
        invalid_tokens = tokens[tokens >= VOCAB_SIZE]
        
        print(f"Sample {i+1}: {sample['id']}")
        print(f"  Token count: {len(tokens)}")
        print(f"  Min token ID: {min_token}")
        print(f"  Max token ID: {max_token}")
        
        if len(invalid_tokens) > 0:
            print(f"  ❌ INVALID TOKENS FOUND: {invalid_tokens.tolist()}")
            invalid_found = True
        else:
            print(f"  ✅ All tokens valid")
        print()
        
    except Exception as e:
        print(f"Sample {i+1}: ❌ Error loading: {e}\n")

print("=" * 70)
if invalid_found:
    print("❌ INVALID TOKENS DETECTED!")
    print("   You need to RE-PREPROCESS the data with fixed tokenizer")
else:
    print("✅ All checked samples have valid tokens")
print("=" * 70)
