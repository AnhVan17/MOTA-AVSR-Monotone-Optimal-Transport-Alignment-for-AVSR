import sys
import os
sys.path.append(os.getcwd())

from src.data.tokenizers.whisper import WhisperTokenizer
from transformers import WhisperTokenizer as HfWhisperTokenizer

def verify_vi_tokens():
    print("--- Verifying WhisperTokenizer (language='vi') ---")
    
    # Initialize with VI
    tokenizer = WhisperTokenizer(model="openai/whisper-small", language="vi")
    hf_tokenizer = tokenizer.tokenizer
    
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"EOT ID (used as blank_id): {tokenizer.eot_token_id}")
    print(f"SOT ID: {tokenizer.sot_token_id}")
    
    # Check Language Token
    lang_token = "<|vi|>"
    lang_id = hf_tokenizer.convert_tokens_to_ids(lang_token)
    print(f"Language Token '{lang_token}': {lang_id}")
    
    # Check stability
    if tokenizer.vocab_size != 51865:
         print("⚠️ WARNING: Vocab size changed from expected 51865!")
    else:
         print("✅ Vocab size stable.")

    # Check Encoding
    text = "xin chào"
    encoded = tokenizer.encode(text)
    print(f"\nEncoding '{text}': {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoding: '{decoded}'")
    
    # Verify First Token is SOT? 
    # Usually Whisper adds SOT, Lang, Task
    # Let's inspect the raw IDs
    print(f"IDs: {encoded}")
    
    # Check if first token is SOT
    if encoded[0] == tokenizer.sot_token_id:
        print("✅ Starts with SOT")
    
    # Check if language token is present
    if lang_id in encoded:
        print(f"✅ Contains Language Token ID {lang_id}")
    else:
        print(f"⚠️ Missing Language Token ID {lang_id}")

if __name__ == "__main__":
    verify_vi_tokens()
