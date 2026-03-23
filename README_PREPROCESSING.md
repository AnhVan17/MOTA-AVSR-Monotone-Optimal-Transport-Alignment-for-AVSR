# Hướng Dẫn Sử Dụng Preprocessing Modal

## 📋 Tổng Quan

Script `preprocessing_modal.py` có **2 chức năng chính**:

### 1️⃣ **Preprocessing** (Tự động tạo manifest)
```bash
modal run scripts/preprocessing_modal.py::main
```

**Quá trình:**
1. Process TAR files → tạo `.pt` files
2. Lưu shard metadata (`_metadata.jsonl`) cho mỗi TAR
3. **Tự động tạo manifest** `train.jsonl`, `val.jsonl`, `test.jsonl` từ TARs vừa xử lý

### 2️⃣ **Regenerate Manifests** (Từ tất cả dữ liệu đã xử lý)
```bash
modal run scripts/preprocessing_modal.py::regenerate_manifests_only
```

**Quá trình:**
- Scan tất cả dữ liệu đã processed
- Tạo manifest tổng hợp cho **toàn bộ dataset**
- Tự động chọn strategy nhanh nhất

---

## ⚡ Performance Optimization

### Shard Metadata System

Khi preprocessing, script **tự động lưu shard metadata** (`_metadata.jsonl`) cho mỗi TAR:

```
/data/processed_features/
├── train_000/
│   ├── sample001.pt
│   ├── sample002.pt
│   └── _metadata.jsonl  ← id, path, text của tất cả samples
└── test_000/
    ├── sample001.pt
    └── _metadata.jsonl
```

### Regenerate Manifests - 2 Strategies:

#### **Strategy 1: FAST Mode** ⚡ (Recommended)
- **Khi nào**: Đã chạy preprocessing với code hiện tại (có `_metadata.jsonl`)
- **Tốc độ**: **~5-10 giây** cho 51K files
- **Cách hoạt động**: Đọc JSON metadata từ tất cả shards

#### **Strategy 2: PARALLEL Mode** 🔄 (Fallback)
- **Khi nào**: Chỉ có .pt files (preprocessing cũ, không có metadata)
- **Tốc độ**: **~3-6 phút** cho 51K files (200 workers song song)
- **Cách hoạt động**: Load .pt files parallel với 200 workers

**Script tự động chọn strategy phù hợp!**

---

## 🎯 Workflow Khuyến Nghị

### Scenario A: Xử lý incremental (từng phần)
```bash
# Chạy preprocessing nhiều lần
modal run scripts/preprocessing_modal.py::main  # Lần 1: 25 train + 5 test TARs
modal run scripts/preprocessing_modal.py::main  # Lần 2: 25 train + 5 test TARs khác
...

# Sau khi xử lý đủ → tạo manifest tổng hợp
modal run scripts/preprocessing_modal.py::regenerate_manifests_only
# → FAST mode: 5-10 giây! ⚡
```

### Scenario B: Xử lý toàn bộ dataset cùng lúc
```bash
# Sửa MAX_TRAIN_TARS = None, MAX_TEST_TARS = None
# Rồi chạy:
modal run scripts/preprocessing_modal.py::main
# → Tự động tạo manifest cho toàn bộ dataset
```

---

## 📊 Output Structure

### Sau preprocessing:
```
/data/
├── processed_features/
│   ├── train_000/
│   │   ├── sample001.pt
│   │   ├── sample002.pt
│   │   └── _metadata.jsonl  ← Shard metadata
│   └── test_000/
│       └── _metadata.jsonl
│
└── manifests/
    ├── train.jsonl  ← Manifest của TARs vừa xử lý
    ├── val.jsonl
    └── test.jsonl
```

### Sau regenerate_manifests_only:
```
/data/manifests/
├── train.jsonl  ← Manifest tổng hợp TẤT CẢ
├── val.jsonl
└── test.jsonl
```

---

## ⚙️ Cấu Hình

```python
# Giới hạn xử lý (trong preprocessing_modal.py)
MAX_TRAIN_TARS = 25         # None = xử lý tất cả
MAX_TEST_TARS = 5           # None = xử lý tất cả
SAMPLES_PER_TAR = None      # None = xử lý tất cả samples

# Train/Val split
VAL_SPLIT_RATIO = 0.1       # 10% validation
RANDOM_SEED = 42            # Reproducibility
```

---

## 🔍 Kiểm Tra Kết Quả

### Xem số lượng samples đã xử lý
```bash
modal run scripts/preprocessing_modal.py::regenerate_manifests_only
```

**Output với FAST mode:**
```
✨ Found 100 shard metadata files
⚡ Using FAST mode (reading JSON metadata)

✅ Loaded metadata in <5 seconds!
  Train/Val: 46200
  Test:      5108

📈 Final Statistics:
  Train: 41580
  Val:   4620
  Test:  5108
  Total: 51308
```

**Output với PARALLEL mode (fallback):**
```
⚠️  No shard metadata found
⚡ Using PARALLEL mode (loading .pt files)

📊 Found 51308 .pt files
  Workers: 200
  Batches: 200 x ~256 files each

  Progress: 10/200 batches (5.0%)
  Progress: 20/200 batches (10.0%)
  ...
  Progress: 200/200 batches (100.0%)

✅ Loaded 51308 files
```

---

## 💡 Tips

1. **Incremental Processing**: Lần preprocessing đầu tiên sẽ tạo `_metadata.jsonl`. Từ đó về sau, `regenerate_manifests_only` chỉ mất 5-10 giây! ⚡

2. **Validation Split**: Để thay đổi tỷ lệ train/val, sửa `VAL_SPLIT_RATIO` rồi chạy `regenerate_manifests_only`.

3. **Kiểm tra Volume**:
   ```bash
   modal volume ls avsr-dataset-volume:/data/processed_features
   modal volume ls avsr-dataset-volume:/data/manifests
   ```

4. **Debugging**: Nếu manifest không có dữ liệu:
   - Kiểm tra `.pt` files có tồn tại không
   - Kiểm tra `_metadata.jsonl` files
   - Xem logs của preprocessing

---

## 🚀 Quick Start

```bash
# 1. Chạy preprocessing lần đầu (tạo ~30 TARs)
modal run scripts/preprocessing_modal.py::main

# 2. Chạy lại để process thêm TARs
modal run scripts/preprocessing_modal.py::main

# 3. Tạo manifest tổng hợp (5-10 giây!)
modal run scripts/preprocessing_modal.py::regenerate_manifests_only
```
