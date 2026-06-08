import modal
import os

# Define the image with necessary dependencies
# Minimal image for transformers
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch"
)

app = modal.App("verify-vocab-vi", image=image)

# Reuse the wrapper logic, but directly inside the modal function to avoid 'src' dep
@app.function()
def verify_remote():
    print("--- Verifying WhisperTokenizer (language='vi') on Modal ---")
    from transformers import WhisperTokenizer
    
    # 1. Initialize Default (English)
    tok_en = WhisperTokenizer.from_pretrained("openai/whisper-small")
    
    # 2. Initialize Vietnamese
    tok_vi = WhisperTokenizer.from_pretrained("openai/whisper-small", language="vi", task="transcribe")
    
    print(f"\n[1] Vocab Size Comparison:")
    print(f"    EN: {tok_en.vocab_size}")
    print(f"    VI: {tok_vi.vocab_size}")
    
    if tok_en.vocab_size != tok_vi.vocab_size:
        print("⚠️ WARNING: VOCAB SIZE CHANGED! This might break the model classification layer.")
    else:
        print("✅ Vocab Size is STABLE (51865).")
        
    print(f"\n[2] Special Tokens (VI):")
    print(f"    EOT (End of Transcript): {tok_vi.eos_token_id}")
    print(f"    SOT (Start of Transcript): {tok_vi.convert_tokens_to_ids('<|startoftranscript|>')}")
    
    # Check for <|vi|>
    lang_token = "<|vi|>"
    try:
        lang_id = tok_vi.convert_tokens_to_ids(lang_token)
        print(f"    Language Token '{lang_token}': {lang_id}")
    except:
        print(f"    Language Token '{lang_token}' NOT FOUND!")

    print("\n[3] Decoding Check:")
    text = "xin chào việt nam"
    # Note: Transformers tokenizer might output different IDs than our wrapper depending on how special tokens are handled
    # But checking the decode looptrip works is key.
    encoded = tok_vi.encode(text)
    decoded = tok_vi.decode(encoded, skip_special_tokens=True)
    print(f"    Input: '{text}'")
    print(f"    Encoded: {encoded}")
    print(f"    Decoded: '{decoded}'")
    
    if text in decoded.lower():
        print("✅ Decoding Correct.")
    else:
        print("⚠️ Decoding Mismatch.")

    print("\n--- DONE ---")

@app.local_entrypoint()
def main():
    verify_remote.remote()
