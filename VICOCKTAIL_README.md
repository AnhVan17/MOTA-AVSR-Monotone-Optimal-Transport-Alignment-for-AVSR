# Vicocktail AVSR Preprocessing Pipeline

Tài liệu này mô tả quy trình xử lý dữ liệu (preprocessing) cho tập dữ liệu **Vicocktail**. Quy trình được thiết kế thành module 2 giai đoạn (2-Phase) để tối ưu hóa việc tái sử dụng dữ liệu.

## Tổng Quan

Thay vì xử lý từ A-Z một lần (gây lãng phí tài nguyên nếu muốn thay đổi mô hình feature extraction), chúng ta chia làm 2 bước:

1.  **Phase 1: Cropping & Cleaning**
    *   **Input**: Video gốc (Raw Video, Full Face).
    *   **Process**:
        *   Dùng face-alignment (GPU-native) để phát hiện khuôn mặt và cắt vùng miệng (Mouth Cropping).
        *   Dùng FFmpeg để copy âm thanh gốc sang video mới.
    *   **Output**: Dataset mới chứa các video "sạch" (chỉ có miệng + tiếng). Dataset này **độc lập**, có thể dùng cho bất kỳ mô hình nào khác.

2.  **Phase 2: Feature Extraction**
    *   **Input**: Dataset đã crop ở Phase 1.
    *   **Process**:
        *   **Visual**: Trích xuất đặc trưng hình ảnh bằng ResNet18 (không cần chạy lại Face Detection).
        *   **Audio**: Trích xuất đặc trưng âm thanh bằng Whisper Encoder.
    *   **Output**: Các file `.pt` chứa tensor (Audio, Visual, Text) sẵn sàng để training.

---

## Hướng Dẫn Sử Dụng

### Script Quản Lý

Sử dụng script `scripts/run_vicocktail_pipeline.py` để chạy cả 2 phase.

### Bước 1: Tạo Clean Dataset (Phase 1)

Chạy lệnh này **một lần duy nhất** khi có dữ liệu thô mới.

```bash
python scripts/run_vicocktail_pipeline.py --mode crop \
    --input_dir "D:/Path/To/Raw_Vicocktail" \
    --output_dir "D:/Path/To/Cropped_Vicocktail"
```

*   `--input_dir`: Đường dẫn đến thư mục chứa video gốc.
*   `--output_dir`: Nơi lưu các video đã cắt.

### Bước 2: Tạo Features để Training (Phase 2)

Chạy lệnh này bất cứ khi nào bạn muốn tạo dữ liệu training (ví dụ: đổi model Audio mới, đổi settings feature).

```bash
python scripts/run_vicocktail_pipeline.py --mode extract \
    --input_dir "D:/Path/To/Cropped_Vicocktail" \
    --output_dir "D:/Path/To/Processed_Features" \
    --manifest_path "data/manifests/vicocktail_train.jsonl"
```

*   `--input_dir`: Đường dẫn đến thư mục `output_dir` của Bước 1.
*   `--output_dir`: Nơi lưu các file `.pt`.
*   `--manifest_path`: Nơi lưu file manifest `.jsonl` để config vào model.

---

## Cấu Trúc Thư Mục

Sau khi chạy xong, cấu trúc thư mục sẽ nối tiếp như sau:

```
D:/
├── Raw_Vicocktail/         (Video gốc, Full Face)
│   ├── video1.mp4
│   └── video1.txt
│
├── Cropped_Vicocktail/     (Kết quả Phase 1 - Video miệng + Tiếng)
│   ├── video1.mp4          (Đã crop, có tiếng)
│   └── video1.txt          (Copy sang)
│
└── Processed_Features/     (Kết quả Phase 2 - Tensor cho Training)
    ├── video1.pt           (Chứa {'audio': ..., 'visual': ..., 'text': ...})
    └── ...
```

## Tích Hợp Vào Training

Trong file config (`configs/model/config.yaml`), trỏ đường dẫn đến manifest vừa tạo:

```yaml
data:
  train_manifest: "data/manifests/vicocktail_train.jsonl"
  # ...
```

Và đảm bảo code training sử dụng `VicocktailDataset` (đã được cập nhật để đọc file `.pt`).
