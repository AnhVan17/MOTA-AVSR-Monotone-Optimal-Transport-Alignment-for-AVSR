import modal
import os

# Define the image with necessary dependencies
# We use the same base image as training to ensure consistency
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "soundfile",
    "librosa"
)

app = modal.App("debug-vocab-check", image=image)

@app.function()
def check_vocab():
    print("--- START VOCAB CHECK ---")
    from transformers import WhisperTokenizer
    
    # Load the exact model used in training
    model_name = "openai/whisper-small"
    print(f"Loading tokenizer: {model_name}")
    
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    
    print(f"1. Vocab Size (property): {tokenizer.vocab_size}")
    print(f"2. Len(tokenizer): {len(tokenizer)}")
    
    # Check Special Tokens
    print("\n--- Special Token IDs ---")
    print(f"EOS (<|endoftext|>) ID: {tokenizer.eos_token_id}")
    print(f"BOS (<|startoftranscript|>) ID: {tokenizer.bos_token_id}")
    print(f"PAD ID: {tokenizer.pad_token_id}")
    print(f"UNK ID: {tokenizer.unk_token_id}")
    
    # Verify specific IDs
    target_blank = 50257
    print(f"\n--- Verifying Candidate Blank ID: {target_blank} ---")
    try:
        decoded_50257 = tokenizer.decode([target_blank], skip_special_tokens=False)
        print(f"ID {target_blank} decodes to: '{decoded_50257}'")
    except Exception as e:
        print(f"Error decoding {target_blank}: {e}")
        
    # Check what was previously 4
    print(f"\n--- Checking Old Blank ID: 4 ---")
    decoded_4 = tokenizer.decode([4], skip_special_tokens=False)
    print(f"ID 4 decodes to: '{decoded_4}'")

    print("--- END VOCAB CHECK ---")

@app.local_entrypoint()
def main():
    check_vocab.remote()
