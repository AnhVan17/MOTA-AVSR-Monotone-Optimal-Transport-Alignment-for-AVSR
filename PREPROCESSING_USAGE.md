# Hướng Dẫn Sử Dụng Preprocessing Modal

## 📋 Tổng Quan

Script `preprocessing_modal.py` có **2 chế độ chạy**:

### 1️⃣ **Chế độ Preprocessing** (Mặc định)
Xử lý TAR files và tạo manifest từ những TAR vừa được xử lý.

```bash
modal run scripts/preprocessing_modal.py
```

**Khi nào sử dụng:**
- Khi bạn muốn xử lý dữ liệu từ TAR files
- Lần đầu tiên setup dataset
- Processing incremental batches

**Giới hạn hiện tại:**
```python
MAX_TRAIN_TARS = 1  # Xử lý 1 train TAR random
MAX_TEST_TARS = 1   # Xử lý 1 test TAR random
```

---

### 2️⃣ **Chế độ Regenerate Manifests** (Mới)
Tạo manifest từ **TẤT CẢ** .pt files đã được xử lý trong các lần chạy trước.

```bash
modal run scripts/preprocessing_modal.py::regenerate_manifests_only
```

**Khi nào sử dụng:**
- Sau khi đã chạy preprocessing nhiều lần với các TAR files khác nhau
- Muốn tạo manifest tổng hợp cho toàn bộ dataset
- Thay đổi `VAL_SPLIT_RATIO` và muốn regenerate manifests
- Kiểm tra tổng số samples đã được xử lý

---

## 🎯 Workflow Khuyến Nghị

### Scenario A: Xử lý toàn bộ dataset cùng lúc
```bash
# Bước 1: Đặt MAX_TRAIN_TARS và MAX_TEST_TARS = None
# Sửa trong preprocessing_modal.py:
# MAX_TRAIN_TARS = None
# MAX_TEST_TARS = None

# Bước 2: Chạy preprocessing
modal run scripts/preprocessing_modal.py
```

### Scenario B: Xử lý incremental (từng phần)
```bash
# Bước 1: Chạy preprocessing nhiều lần (mỗi lần 1-2 TARs)
modal run scripts/preprocessing_modal.py  # Lần 1
modal run scripts/preprocessing_modal.py  # Lần 2
modal run scripts/preprocessing_modal.py  # Lần 3
# ... (cho đến khi xử lý đủ)

# Bước 2: Regenerate manifest cho toàn bộ
modal run scripts/preprocessing_modal.py::regenerate_manifests_only
```

---

## 📊 Output

### Preprocessing Mode
```
/data/processed_features/
├── train_000/
│   ├── sample001.pt
│   ├── sample002.pt
│   └── ...
└── test_000/
    ├── sample001.pt
    └── ...

/data/manifests/
├── train.jsonl  ← Chỉ từ TARs vừa xử lý
├── val.jsonl
└── test.jsonl
```

### Regenerate Manifests Mode
```
/data/manifests/
├── train.jsonl  ← Từ TẤT CẢ processed files
├── val.jsonl
└── test.jsonl
```

---

## ⚙️ Cấu Hình

```python
# Giới hạn xử lý
MAX_TRAIN_TARS = 1          # None = xử lý tất cả
MAX_TEST_TARS = 1           # None = xử lý tất cả
SAMPLES_PER_TAR = None      # None = xử lý tất cả samples

# Train/Val split
VAL_SPLIT_RATIO = 0.1       # 10% validation
RANDOM_SEED = 42            # Reproducibility
```

---

## ⚡ Performance Optimization

### Parallel Processing cho Regenerate Manifests

Khi chạy `regenerate_manifests_only`, script sử dụng **parallel processing** để tăng tốc độ:

**Cách hoạt động:**
- Chia 51,308 files thành ~40 batches
- Mỗi batch được xử lý bởi một worker riêng (parallel)
- Aggregate kết quả từ tất cả workers

**Performance:**
| Method | 51,308 files | Số workers | Thời gian |
|--------|-------------|-----------|----------|
| **Sequential** (cũ) | ❌ 1 worker | 1 | ~15-30 phút |
| **Parallel** (mới) | ✅ 40 workers | 40 | ~2-5 phút |

**Lưu ý:**
- Lần đầu có thể chậm hơn do cold start của containers
- Các lần sau sẽ nhanh hơn nhờ warm containers

---

## 🔍 Kiểm Tra Kết Quả


### Xem số lượng samples đã xử lý
```bash
modal run scripts/preprocessing_modal.py::regenerate_manifests_only
```

Output sẽ hiển thị:
```
📊 Found 42000 processed files
  Train/Val: 38000
  Test:      4000

📈 Final Statistics:
  Train: 34200
  Val:   3800
  Test:  4000
  Total: 42000
```

---

## 💡 Tips

1. **Incremental Processing**: Nếu bạn chạy preprocessing nhiều lần, mỗi lần sẽ chọn TAR files ngẫu nhiên (random seed = timestamp). Sau đó dùng `regenerate_manifests_only` để tạo manifest tổng hợp.

2. **Validation Split**: `VAL_SPLIT_RATIO = 0.1` có nghĩa 10% train data sẽ được dùng làm validation. Nếu muốn thay đổi, sửa config rồi chạy `regenerate_manifests_only`.

3. **Kiểm tra Volume**: 
   ```bash
   modal volume ls avsr-dataset-volume:/data/processed_features
   ```

4. **Debugging**: Nếu manifest không có dữ liệu, kiểm tra:
   - `.pt` files có tồn tại không?
   - Folder structure: `processed_features/train_XXX/*.pt` hoặc `processed_features/test_XXX/*.pt`
   - Data format trong .pt files: phải có keys `id`, `text`
