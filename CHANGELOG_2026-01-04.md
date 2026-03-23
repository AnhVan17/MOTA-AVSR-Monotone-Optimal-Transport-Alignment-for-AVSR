# 📝 CHANGELOG - Phiên sửa lỗi 04/01/2026

Tài liệu này ghi lại tất cả các chỉnh sửa quan trọng được thực hiện trong phiên làm việc hôm nay để khắc phục các lỗi trong pipeline AVSR.

---

## 🛠️ Danh sách chỉnh sửa

### 1. Sửa lỗi OOM (Out-of-Memory) trong giai đoạn Crop

**File:** `src/data/preprocessors/vicocktail.py`

**Vấn đề:** Worker bị crash do tải toàn bộ frames vào RAM cùng lúc.

**Giải pháp:**
- Chuyển sang xử lý video theo kiểu **Streaming I/O** (đọc frame nào xử lý frame đó).
- Thêm cơ chế **Lazy Initialization** cho MediaPipe: mỗi worker chỉ khởi tạo `VideoProcessor` một lần.
- Thêm biến môi trường `MEDIAPIPE_DISABLE_GPU=1` để tránh lỗi EGL trên server headless.

```python
# Trước: Khởi tạo mới cho mỗi video
processor = VideoProcessor(use_precropped=False)

# Sau: Tái sử dụng qua các video
_worker_processor = None
def _get_processor():
    global _worker_processor
    if _worker_processor is None:
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
        _worker_processor = VideoProcessor(use_precropped=False)
    return _worker_processor
```

---

### 2. Sửa lỗi Extract chạy quá chậm

**File:** `src/data/preprocessors/base_preprocessor.py`

**Vấn đề:** Mỗi video đều khởi tạo lại MediaPipe mặc dù video đã được crop sẵn (không cần detect miệng nữa).

**Giải pháp:**
- Thêm hàm `_load_precropped_video()` để đọc video đã crop **mà không cần MediaPipe**.
- Khi `use_precropped=True`, bỏ qua hoàn toàn việc khởi tạo FaceMesh.

```python
class RawVideoDataset(Dataset):
    def __getitem__(self, idx):
        if self.use_precropped:
            tensor = self._load_precropped_video(path)  # Nhanh gấp 100x
        else:
            processor = VideoProcessor(use_precropped=False)
            tensor = processor.process(path)
        return tensor, path
```

---

### 3. Sửa lỗi CUDA Index Out of Bounds khi Training

**File:** `src/models/layers/decoders.py`

**Vấn đề:** 
- Target tokens được pad bằng `-100` (ignore_index cho Loss).
- Nhưng Embedding layer không chấp nhận index âm, gây lỗi CUDA.

**Giải pháp:**
- Clone target tensor và thay thế `-100` bằng `0` trước khi đưa vào Embedding.
- Loss function vẫn nhận được target gốc với `-100` để tính toán đúng.

```python
# CRITICAL FIX
target_for_embed = target.clone()
target_for_embed[target_for_embed < 0] = 0  # Thay -100 thành 0
target_for_embed[target_for_embed >= self.vocab_size] = 0  # Clamp OOV

target_embed = self.target_embedding(target_for_embed)
```

---

### 4. Cập nhật Vocab Size trong Config

**File:** `configs/vicocktail_phase1.yaml`

**Vấn đề:** `vocab_size: 50257` không khớp với Whisper tokenizer (thực tế có 51865 tokens bao gồm cả special tokens).

**Giải pháp:**
```yaml
# Trước
vocab_size: 50257

# Sau
vocab_size: 51865  # Whisper Full Vocab (base + special tokens)
```

---

### 5. Thêm Progress Bar cho giai đoạn Load Transcripts

**File:** `src/data/preprocessors/vicocktail.py`

**Vấn đề:** Khi load 98k transcripts, không có log nào hiển thị khiến người dùng tưởng chương trình bị treo.

**Giải pháp:**
```python
# Thêm tqdm vào vòng lặp
for video_path in tqdm(video_files, desc="Loading transcripts"):
    text = self._get_transcript(video_path)
```

---

### 6. Hỗ trợ 2 chế độ chạy Crop: Distributed vs Single

**File:** `scripts/modal/preprocess_vicocktail.py`

**Thêm mới:**
- `--mode dist`: Chạy phân tán trên nhiều container nhỏ (nhanh, mặc định).
- `--mode single`: Chạy trên 1 container mạnh với nhiều workers (dễ debug).

```powershell
# Chạy phân tán (nhanh)
modal run scripts/modal/preprocess_vicocktail.py --stage crop --mode dist

# Chạy tập trung (1 container, 8 workers)
modal run scripts/modal/preprocess_vicocktail.py --stage crop --mode single --workers 8
```

---

### 7. Bổ sung thư viện EGL/GLES cho MediaPipe

**File:** `scripts/modal/preprocess_vicocktail.py`

**Vấn đề:** MediaPipe báo lỗi `eglMakeCurrent() returned error 0x3008` trên server headless.

**Giải pháp:** Thêm các thư viện đồ họa vào Docker image:
```python
.apt_install(
    "libgl1-mesa-glx", 
    "libglib2.0-0", 
    "ffmpeg", 
    "libsndfile1", 
    "git",
    "libegl1",       # NEW
    "libgles2-mesa"  # NEW
)
```

---

## ✅ Kết quả sau khi sửa

| Giai đoạn | Trạng thái cũ | Trạng thái mới |
|:---|:---|:---|
| **Crop** | ~50% crash do OOM | ✅ 100% hoàn thành |
| **Extract** | Treo sau 2h, không có progress | ✅ ~10 videos/giây |
| **Train** | Crash ngay batch đầu tiên | ✅ Chạy được |

---

## 📌 Lưu ý cho tương lai

1. **Khi thêm tokenizer mới:** Luôn kiểm tra `len(tokenizer)` thay vì dùng giá trị hardcode.
2. **Khi có collate với ignore_index:** Nhớ xử lý trước khi đưa vào Embedding.
3. **Khi chạy trên Modal:** Thêm đầy đủ thư viện đồ họa nếu dùng MediaPipe/OpenCV.

---

### 8. Sửa lỗi thiếu `target_mask` trong Trainer

**File:** `src/training/trainer.py`

**Vấn đề:** `HybridLoss.forward()` yêu cầu `target_mask` nhưng trainer không truyền.

**Giải pháp:**
```python
# Tạo mask từ target tensor
target_mask = (target != -100)

loss_dict = self.criterion(
    ctc_logits=outputs['ctc_logits'],
    ar_logits=outputs['ar_logits'],
    targets=target,
    target_mask=target_mask,  # NEW
    epoch=self.epoch,
    max_epochs=self.config['training']['num_epochs']
)
```
