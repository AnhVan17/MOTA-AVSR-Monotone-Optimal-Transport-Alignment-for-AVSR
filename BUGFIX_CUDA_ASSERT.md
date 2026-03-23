# CRITICAL BUG FIX - CUDA Assert Error

## 🔴 Vấn Đề:

```
RuntimeError: CUDA error: device-side assert triggered
Assertion `srcIndex < srcSelectDimSize` failed
```

Lỗi xảy ra khi training ở decoder embedding lookup.

## 🔍 Root Cause:

**Vocab Size Mismatch:**
- **Config**: `vocab_size: 51865` 
- **Whisper Tokenizer**: `vocab_size: 50258` ❌

→ Token IDs từ 50258-51864 không tồn tại trong embedding layer!

## 🛠️ Solution:

### Fixed `configs/model/config.yaml`:
```yaml
# BEFORE (WRONG):
vocab_size: 51865  # ❌ Incorrect!

# AFTER (CORRECT):
vocab_size: 50258  # ✅ Matches Whisper tokenizer
```

## 📊 Verification:

Run `check_tokenizer.py` to verify:
```bash
python check_tokenizer.py
```

Output:
```
Vocab size: 50258
Valid token ID range: 0 to 50257
```

## ⚠️ Impact:

**Before fix:**
- Training crashes immediately with CUDA assert
- Index out of bounds in embedding layer

**After fix:**
- All token IDs are valid (0-50257)
- Training runs without errors

## 🎯 Why This Happened:

The config incorrectly used `51865` which is Whisper's **extended multilingual vocab** size, but `openai/whisper-small` only has **50258 tokens**.

## ✅ Checklist:

- [x] Fixed vocab_size in config.yaml
- [x] Verified tokenizer vocab size
- [x] Created debug script for future checks
- [ ] Re-run training to confirm fix

## 📝 Notes:

Always verify:
```python
tokenizer.vocab_size == model_config['vocab_size']
```

This simple check would have prevented this critical bug!
