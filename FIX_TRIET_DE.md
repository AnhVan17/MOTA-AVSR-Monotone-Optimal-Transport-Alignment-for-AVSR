# FIX TRIỆT ĐỂ - CUDA Assert Error

## 🔴 Vấn đề:
Lỗi vẫn xảy ra sau khi fix vocab_size trong config!

## 🔍 Root Cause (Real):

**2 nguyên nhân chính:**

### 1. Modal Image Cache ❌
- Modal đã cache Docker image với model **vocab_size cũ (51865)**  
- Mặc dù config đã sửa thành 50258, model trong container vẫn dùng cache cũ
- **Solution**: Đổi `APP_NAME` để force rebuild

### 2. Preprocessed Data có Invalid Tokens ❌  
- Nếu data đã được preprocess với tokenizer cũ
- Có thể có token IDs > 50257
- **Solution**: RE-PREPROCESS data

---

## ✅ GIẢI PHÁP TRIỆT ĐỂ:

### Bước 1: Force Rebuild Image ✅ (Đã làm)
```python
# training_modal.py
APP_NAME = "avsr-training-v4-fixed"  # Changed from v3
```

### Bước 2: Kiểm Tra Processed Data

**Option A: Check trên Modal volume**
```bash
# SSH vào Modal container hoặc dùng Modal function
```

**Option B: Check local (nếu có data)**
```bash
python check_processed_data.py
```

### Bước 3: Re-preprocess (NẾU cần)

Nếu tìm thấy invalid tokens:
```bash
# Xóa processed data cũ
modal volume rm -r avsr-dataset-volume:/data/processed_features

# Xóa manifests cũ
modal volume rm -r avsr-dataset-volume:/data/manifests

# Chạy lại preprocessing
modal run scripts/preprocessing_modal.py::main
```

### Bước 4: Training với image mới
```bash
modal run --detach scripts/training_modal.py
```

---

## 📊 Verification Checklist:

- [x] Config vocab_size = 50258
- [x] Changed APP_NAME to force rebuild
- [ ] Verified processed data has valid tokens (0-50257)
- [ ] Training runs without CUDA assert error

---

## 🎯 TỐI ƯU:

### Nếu muốn NHANH:

**Không cần check processed data**, chỉ cần:
1. ✅  Đã đổi APP_NAME → force rebuild
2. Run training: `modal run --detach scripts/training_modal.py`

Nếu vẫn lỗi → Chắc chắn là processed data có vấn đề → Re-preprocess

---

## 💡 Prevention:

Luôn verify trước khi preprocessing:
```python
assert tokenizer.vocab_size == config['model']['vocab_size']
```

Thêm validation trong preprocessing pipeline để reject invalid tokens!
