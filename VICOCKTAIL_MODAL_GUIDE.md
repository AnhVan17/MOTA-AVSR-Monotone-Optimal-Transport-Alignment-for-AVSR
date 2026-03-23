# 🍹 Hướng Dẫn Xử Lý Dữ Liệu Vicocktail trên Modal (Kiến Trúc Phân Tán)

Tài liệu này hướng dẫn quy trình **End-to-End** tối ưu nhất để xử lý bộ dữ liệu Vicocktail. Quy trình đã được nâng cấp lên kiến trúc **Distributed Parallelism** (Song song hóa phân tán) và **Streaming I/O** để khắc phục triệt để lỗi **Tràn RAM (OOM)** và **Tràn ổ cứng (Disk Full)**.

---

## 🚀 Tổng Quan Quy Trình (Mới)

| Bước | Script tương ứng | Mục đích | Công nghệ mới |
| :--- | :--- | :--- | :--- |
| **1. Download** | `scripts/modal/download.py` | Tải dữ liệu từ HuggingFace. | `hf_transfer` (Siêu tốc) |
| **2. Unpack** | `scripts/modal/unpack_vicocktail_rescue.py` | Giải nén + Đổi tên + Dọn dẹp. | `Streaming Unpack` |
| **3. Debug** | `scripts/modal/preprocess_vicocktail_debug.py` | Kiểm tra lỗi trên tập nhỏ. | `Safe Testing` |
| **4. Preprocess**| `scripts/modal/preprocess_vicocktail.py` | **Mouth Crop & Feature Extract**. | **Distributed Map** |

---

## 🛠️ Chi Tiết Các Bước

### BƯỚC 1 & 2: Download & Unpack
Tương tự quy trình cũ, đảm bảo dữ liệu giải nén nằm tại `/mnt/vicocktail/raw`.

### BƯỚC 3: Kiến Trúc Preprocessing MỚI (Cực nhanh & Ổn định)
Chúng ta đã chuyển từ việc xử lý trên 1 máy (dễ chết RAM) sang **xử lý trên hàng trăm máy con** cùng lúc.

#### 4.1. Phase 1: Distributed Mouth Cropping (CPU)
Thay vì dùng 1 container 64GB RAM, Modal sẽ tự động tạo ra hàng chục container nhỏ (8GB RAM) để xử lý song song từng group video.

```powershell
# Chạy toàn bộ dataset (Tốc độ x10 so với trước)
modal run scripts/modal/preprocess_vicocktail.py --stage crop

# (Tùy chọn) Chạy thử 100 video để kiểm tra
modal run scripts/modal/preprocess_vicocktail.py --stage crop --limit 100
```

**Ưu điểm của Phase 1 mới:**
*   **Streaming Frames**: Không còn tải toàn bộ video vào RAM. Xử lý đến đâu ghi đến đó.
*   **Fault Tolerance**: Nếu 1 video bị lỗi, chỉ container đó bị ảnh hưởng, toàn bộ pipeline vẫn tiếp tục.
*   **Auto-Resume**: Nếu chạy lại, script sẽ tự động bỏ qua các video đã xử lý thành công.

#### 4.2. Phase 2: Feature Extraction (GPU A100)
Sau khi đã cắt miệng xong, bước này dùng GPU để trích xuất vector đặc trưng.

```powershell
modal run scripts/modal/preprocess_vicocktail.py --stage extract
```
*   **Đầu ra**: File `.pt` (Audio Whisper + Visual ResNet) và các file Manifest (`train.jsonl`, `val.jsonl`, `test.jsonl`).

---

## ⚠️ Giải Quyết Sự Cố (Troubleshooting)

### 1. Lỗi "Process terminated abruptly" (OOM)
*   **Đã khắc phục**: Nhờ cơ chế **Streaming I/O**, RAM sử dụng hiện tại cực thấp (< 2GB/video).
*   **Nếu vẫn bị**: Giảm số lượng concurrency trong code (mặc định đang rất cao để đạt tốc độ tối đa).

### 2. Lỗi Disk Full
*   **Xử lý**: Luôn kiểm tra dung lượng Volume trước khi chạy:
    ```powershell
    modal run scripts/modal/check_volume.py
    ```
*   Nếu đầy, dùng `scripts/modal/cleanup_volume.py` để xóa các file rác hoặc file tạm.

---

## 📂 Cấu Trúc Dữ Liệu (Trên Volume)

```
/mnt/
├── vicocktail/raw/           # Video gốc (.mp4)
├── processed/
│   ├── vicocktail_cropped/   # Video chỉ chứa miệng (Output Phase 1)
│   ├── features/vicocktail/  # Features .pt (Output Phase 2)
│   └── manifests/            # train.jsonl, val.jsonl, test.jsonl
```

