import os
import sys

sys.path.append(os.getcwd())

from src.data.tokenizers.whisper import WhisperTokenizer


def verify_vi_tokens():
    print("--- Verifying WhisperTokenizer (language='vi') ---")

    tokenizer = WhisperTokenizer(model="openai/whisper-small", language="vi")
    info = tokenizer.vocab_info()
    hf_tokenizer = tokenizer.tokenizer

    print(f"Total Vocab Size (len): {info['vocab_size']}")
    print(f"Model Vocab Size (HF base): {info['model_vocab_size']}")
    print(f"Max Token ID: {info['max_token_id']}")
    print(f"EOT ID (blank): {info['eot_token_id']}")
    print(f"SOT ID: {info['sot_token_id']}")

    # Check language token
    lang_token = "<|vi|>"
    lang_id = hf_tokenizer.convert_tokens_to_ids(lang_token)
    print(f"Language Token '{lang_token}': {lang_id}")

    # Stability checks for current codebase assumptions
    expected_total_vocab = 51865
    expected_blank = 50257
    expected_sot = 50258

    if info["vocab_size"] != expected_total_vocab:
        print(
            "⚠️ WARNING: total vocab size changed! "
            f"expected={expected_total_vocab}, got={info['vocab_size']}"
        )
    else:
        print("✅ Total vocab size stable (51865).")

    if info["eot_token_id"] != expected_blank:
        print(
            "⚠️ WARNING: EOT/blank changed! "
            f"expected={expected_blank}, got={info['eot_token_id']}"
        )
    else:
        print("✅ EOT/blank ID stable (50257).")

    if info["sot_token_id"] != expected_sot:
        print(
            "⚠️ WARNING: SOT changed! "
            f"expected={expected_sot}, got={info['sot_token_id']}"
        )
    else:
        print("✅ SOT ID stable (50258).")

    # Encode/decode sanity
    text = "xin chào"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"\nEncoding '{text}': {encoded}")
    print(f"Decoding: '{decoded}'")

    if encoded and encoded[0] == tokenizer.sot_token_id:
        print("✅ Starts with SOT")
    else:
        print("⚠️ Does not start with SOT")

    if lang_id in encoded:
        print(f"✅ Contains Language Token ID {lang_id}")
    else:
        print(f"⚠️ Missing Language Token ID {lang_id}")


if __name__ == "__main__":
    verify_vi_tokens()
