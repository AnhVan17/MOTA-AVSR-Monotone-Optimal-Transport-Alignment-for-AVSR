# 🔧 BUG FIX DOCUMENTATION: CTC Training Failure

## Ngày: 2026-01-06
## Vấn đề: Model không học được gì (WER ~100%, predictions rỗng hoặc garbage)

---

## 🔴 TRIỆU CHỨNG

```
Epoch 1: WER 100%, Predictions: "" (rỗng)
Epoch 2: WER 99.88%, Predictions: "�" hoặc "à" (garbage)
```

Model chạy training nhưng:
- CTC loss cao (~200+)
- AR loss giảm nhưng không predict được
- Output là chuỗi rỗng hoặc ký tự rác

---

## 🔍 NGUYÊN NHÂN GỐC (3 Lỗi Chính)

### 1️⃣ **CTC Blank ID Conflict** ⚠️ CRITICAL

**Trước đây:**
```python
blank_id = 0  # CONFLICT với token thật!
```

**Vấn đề:**
- Whisper tokenizer: Token ID 0 = ký tự `!` (token thật!)
- CTC dùng blank_id = 0 → Model học predict 0 là "nothing"
- Nhưng 0 là token thật → Model confused → Output garbage

**Sau khi fix:**
```python
blank_id = 51865  # NGOÀI tất cả vocab, không conflict!
```

---

### 2️⃣ **Special Tokens trong CTC Targets** ⚠️

**Trước đây:**
```python
tokens = tokenizer.encode(text)  # Mặc định add_special_tokens=True
# Output: [50258, 50359, 50363, 87, 259, 417, 20807, 50257]
#          ↑ SOT  ↑ lang  ↑ task                     ↑ EOT
```

**Vấn đề:**
- CTC targets chứa special tokens (SOT, language, task, EOT)
- CTC loss cố gắng align speech với những meta-tokens này
- Không thể học được → loss cao

**Sau khi fix:**
```python
tokens = tokenizer.encode_for_ctc(text)  # add_special_tokens=False
# Output: [87, 259, 417, 20807]  ← Chỉ content tokens!
```

---

### 3️⃣ **CTC Head Output Size Sai** ⚠️

**Trước đây:**
```python
self.ctc_head = nn.Linear(d_model, vocab_size)  # Thiếu blank!
# Output: 51865 classes
```

**Vấn đề:**
- CTC cần `vocab_size + 1` outputs (tokens + blank)
- Model thiếu vị trí cho blank token
- Loss function không tính đúng

**Sau khi fix:**
```python
self.ctc_head = nn.Linear(d_model, vocab_size + 1)  # +1 cho blank
# Output: 51866 classes (51865 vocab + 1 blank)
```

---

## ✅ DANH SÁCH FILE ĐÃ SỬA

### 1. `src/data/tokenizers/whisper.py`

**Thay đổi:** Thêm method `encode_for_ctc()`

```python
def encode_for_ctc(self, text: str) -> List[int]:
    """Encode without special tokens for CTC training"""
    return self.tokenizer.encode(
        text,
        add_special_tokens=False  # KEY FIX
    )
```

**Lý do:** CTC targets phải là content-only, không có special tokens.

---

### 2. `src/data/datasets/base_dataset.py`

**Thay đổi:** Dùng `encode_for_ctc()` thay vì `encode()`

```python
def _tokenize(self, text: str) -> torch.Tensor:
    if hasattr(self.tokenizer, 'encode_for_ctc'):
        token_ids = self.tokenizer.encode_for_ctc(text)
    else:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(token_ids, dtype=torch.long)
```

**Lý do:** Đảm bảo training targets không có special tokens.

---

### 3. `src/training/losses.py`

**Thay đổi:** `blank_id = vocab_size` (ngoài vocab range)

```python
def __init__(self, vocab_size=51865, ...):
    self.blank_id = vocab_size  # 51865 - ngoài valid tokens
    
    self.ctc_loss = nn.CTCLoss(
        blank=self.blank_id,  # 51865
        ...
    )
```

**Lý do:** Blank token phải NGOÀI vocab để không conflict với tokens thật.

---

### 4. `src/models/layers/decoders.py`

**Thay đổi:** CTC head output `vocab_size + 1`

```python
self.ctc_head = nn.Sequential(
    nn.LayerNorm(d_model),
    nn.Linear(d_model, vocab_size + 1)  # +1 cho blank
)
# Output shape: [B, T, 51866]
```

**Lý do:** CTC cần một vị trí output riêng cho blank token.

---

### 5. `src/evaluation/metrics.py`

**Thay đổi:** Evaluator dùng đúng `blank_id = 51865`

```python
def __init__(self, tokenizer, blank_id: int = 51865):
    self.blank_id = blank_id

def ctc_greedy_decode(self, logits):
    # Filter: remove blank AND keep only valid tokens (< blank_id)
    unique_tokens = [t for t in tokens if t != self.blank_id and t < self.blank_id]
```

**Lý do:** Decode phải filter đúng blank_id để ra text.

---

### 6. `configs/vicocktail_phase1.yaml`

**Thay đổi:** `vocab_size = 51865`

```yaml
model:
  vocab_size: 51865  # Whisper full vocab (blank sẽ ở 51865)

loss:
  ctc_weight: 0.3
  ce_weight: 0.7
  # blank_id tự động = vocab_size trong code
```

---

### 7. `src/training/trainer.py`

**Thay đổi:** Pass đúng `blank_id` vào Evaluator

```python
blank_id = config['model']['vocab_size']  # 51865
self.evaluator = Evaluator(self.tokenizer, blank_id=blank_id)
```

---

## 📊 SO SÁNH TRƯỚC/SAU

| Aspect | TRƯỚC (Lỗi) | SAU (Đã Fix) |
|--------|-------------|--------------|
| vocab_size | 50259 hoặc 51865 | 51865 |
| blank_id | 0 (conflict!) | 51865 (ngoài vocab) |
| CTC head output | vocab_size | vocab_size + 1 |
| encode() | add_special_tokens=True | add_special_tokens=False |
| Token range targets | [50258, ..., 50257] | [87, 259, ...] |

---

## 📈 KẾT QUẢ MONG ĐỢI

### Debug output sẽ hiển thị:

```
Loss Configuration:
   Vocab size: 51865
   Blank ID: 51865

CTC Head initialized: 256 -> 51866 (vocab + blank)

[Evaluator] Initialized with blank_id=51865

[DEBUG] Raw argmax (first 30): [234, 567, 1234, ...]  ← Real tokens!
[DEBUG] After CTC decode: [234, 567, 1234, ...]
[DEBUG] Number of non-blank tokens: 25  ← Có tokens!
```

### Training progress:

```
Epoch 1: WER ~85%, Pred: "xin chào" ✅
Epoch 5: WER ~65%
Epoch 10: WER ~45%
Epoch 20: WER ~25-35%
```

---

## 🔑 KEY TAKEAWAYS

1. **CTC blank_id PHẢI nằm ngoài vocab range**
   - Whisper vocab: 0-51864
   - blank_id = 51865 ✅

2. **CTC targets KHÔNG được có special tokens**
   - Dùng `encode_for_ctc()` với `add_special_tokens=False`

3. **CTC head output = vocab_size + 1**
   - 51865 classes cho vocab + 1 class cho blank = 51866

4. **Tất cả components phải dùng cùng blank_id**
   - Loss: blank=51865
   - Decoder: output 51866 classes
   - Evaluator: filter blank_id=51865

---

## 🧪 CÁCH VERIFY FIX HOẠT ĐỘNG

### 1. Check logs khi training start:
```
Loss Configuration:
   Blank ID: 51865  ← Phải = 51865
   
CTC Head initialized: 256 -> 51866  ← Phải = 51866
```

### 2. Check debug output trong validation:
```
[DEBUG] Number of non-blank tokens: 25  ← Phải > 0
```

### 3. Check predictions:
```
[0] Pred: xin chào việt  ← Phải là text đọc được!
    Ref: xin chào việt nam
```

### 4. Check loss convergence:
```
Epoch 1: CTC ~150 → Epoch 5: CTC ~50  ← Phải giảm!
```

---

## 📞 TROUBLESHOOTING

### Vẫn predictions rỗng sau fix?
1. Kiểm tra Modal đã rebuild image mới chưa (APP_NAME đổi)
2. Kiểm tra logs có hiện đúng blank_id=51865
3. Đợi thêm epochs (model cần thời gian học)

### CUDA error: index out of range?
1. Token IDs > 51864 trong data
2. Kiểm tra `encode_for_ctc()` có filter đúng không

### Loss = NaN?
1. Input length < target length
2. Gradient explosion → giảm learning rate

---

## ✅ CHECKLIST TRƯỚC KHI TRAIN

- [ ] `whisper.py` có method `encode_for_ctc()`
- [ ] `base_dataset.py` dùng `encode_for_ctc()`
- [ ] `losses.py` có `blank_id = vocab_size`
- [ ] `decoders.py` có `vocab_size + 1` output
- [ ] `metrics.py` có `blank_id = 51865`
- [ ] `trainer.py` pass đúng `blank_id`
- [ ] Config có `vocab_size: 51865`
- [ ] APP_NAME đã đổi để force rebuild

---

**Tác giả:** AI Assistant  
**Ngày tạo:** 2026-01-06  
**Phiên bản:** v5 (Full Fix)
