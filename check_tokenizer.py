"""
Debug script: Check token IDs and vocab size consistency
"""

from transformers import WhisperProcessor
import torch

# Initialize Whisper tokenizer
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="vi", task="transcribe")
tokenizer = processor.tokenizer

print("=" * 70)
print("Whisper Tokenizer Analysis")
print("=" * 70)
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print(f"BOS token ID: {tokenizer.bos_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"UNK token ID: {tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else 'N/A'}")

# Special tokens
sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
eot = tokenizer.convert_tokens_to_ids("<|endoftranscript|>")
print(f"SOT (<|startoftranscript|>): {sot}")
print(f"EOT (<|endoftranscript|>): {eot}")

# Test encoding
test_texts = [
    "Xin chào",
    "Tôi là người Việt Nam",
    "This is a test",
    ""
]

print("\n" + "=" * 70)
print("Test Encoding")
print("=" * 70)

for text in test_texts:
    tokens_with_special = tokenizer.encode(text, add_special_tokens=True)
    tokens_without_special = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"\nText: '{text}'")
    print(f"  With special tokens: {tokens_with_special}")
    print(f"  Without special tokens: {tokens_without_special}")
    print(f"  Max token ID (with special): {max(tokens_with_special) if tokens_with_special else 'N/A'}")
    print(f"  Max token ID (without special): {max(tokens_without_special) if tokens_without_special else 'N/A'}")
    
    # Check if any token exceeds vocab size
    invalid_with = [t for t in tokens_with_special if t >= tokenizer.vocab_size]
    invalid_without = [t for t in tokens_without_special if t >= tokenizer.vocab_size]
    
    if invalid_with:
        print(f"  ⚠️  INVALID TOKENS (with special): {invalid_with}")
    if invalid_without:
        print(f"  ⚠️  INVALID TOKENS (without special): {invalid_without}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Valid token ID range: 0 to {tokenizer.vocab_size - 1}")
print(f"\n⚠️  If any token ID >= {tokenizer.vocab_size}, it will cause CUDA assert error!")
