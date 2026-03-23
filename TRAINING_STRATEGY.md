
# Chiến Lược Huấn Luyện Tối Ưu (Training Strategy)
*Dành cho hệ thống AVSR đa nhiệm: Grid & Vicocktail*

Tài liệu này giải thích lý do tại sao **không nên trộn lẫn** hai bộ dữ liệu này một cách bừa bãi và đề xuất lộ trình huấn luyện chuẩn SOTA.

---

## 1. Tại Sao Không Nên "Trộn Lẫn" (Joint Training)?

Mặc dù về mặt code (`loader.py`, `mota.py`) hoàn toàn có thể chạy trộn lẫn 2 dataset, nhưng ta gặp 2 vấn đề lớn:

### Vấn đề 1: Lãng Phí Tài Nguyên (Padding Waste)
-   **Grid:** Video ngắn (~3s).
-   **Vicocktail:** Video dài biến thiên (2s - 15s).
-   **Hệ quả:** Nếu một Batch (lô dữ liệu) chứa 1 video Vicocktail (15s) và 31 video Grid (3s):
    -   31 video Grid sẽ bị nhồi thêm 12s toàn số `0` (padding) để bằng độ dài video Vicocktail.
    -   GPU vẫn phải tính toán trên 12s rác này -> **Lãng phí 80% sức mạnh tính toán.**

### Vấn đề 2: Xung Đột Phân Phối (Distribution Shift)
-   **Grid:** Môi trường tĩnh, sạch, từ vựng nghèo nàn.
-   **Vicocktail:** Môi trường động, ồn, từ vựng phong phú.
-   Model sẽ bị "bối rối" khi phải tối ưu hóa cho hai mục tiêu quá khác nhau cùng lúc, dẫn đến Loss dao động mạnh và khó đạt đỉnh (local minima).

---

## 2. Quy Trình Huấn Luyện Đề Xuất (Curriculum Learning)

Chúng ta sẽ áp dụng chiến thuật **Transfer Learning** (Học chuyển tiếp):

### Giai đoạn 1: Pre-training (Môi trường sạch)
-   **Dataset:** Grid.
-   **Cấu hình:** `train_phase1.py` hoặc `train_phase2.py` (tắt MQOT hoặc bật đều được, nhưng tắt thì train nhanh hơn).
-   **Mục tiêu:** Dạy model kiến thức cơ bản về "Lip Reading" (mối tương quan giữa chuyển động môi và âm vị). Do Grid rất sạch, model sẽ học rất nhanh các feature cơ bản này.
-   **Output:** File `grid_best_model.pt`.

### Giai đoạn 2: Fine-tuning (Môi trường thực tế - Domain Adaptation)
-   **Dataset:** Vicocktail.
-   **Cách làm:** 
    -   Load weights từ `grid_best_model.pt`.
    -   Giảm Learning Rate xuống thấp (ví dụ: `1e-5` thay vì `2e-4`) để không làm hỏng kiến thức cũ quá nhanh.
    -   Chạy Training trên Vicocktail.
-   **Mục tiêu:** Model sử dụng kiến thức "đọc môi" đã học từ Grid để áp dụng vào môi trường nhiễu của Vicocktail.

---

## 3. Hướng Dẫn Thực Hiện

### Bước 1: Train Grid
Chạy lệnh trên Modal:
```bash
modal run scripts/modal/train_phase2.py --config configs/grid.yaml
```
*Sau khi xong, tải file `final_model.pt` về hoặc lưu trên Volume.*

### Bước 2: Sửa Config cho Vicocktail
Trong file `configs/vicocktail.yaml`, thêm dòng load checkpoint:
```yaml
training:
  resume_checkpoint: "/mnt/checkpoints/grid_final_model.pt" # Đường dẫn checkpoint Grid
  learning_rate: 1e-5 # Giảm LR đi 1 chút
```

### Bước 3: Train Vicocktail
```bash
modal run scripts/modal/train_phase2.py --config configs/vicocktail.yaml
```

---

## 4. Kết Luận
Việc tách riêng Grid và Vicocktail không phải vì code bị lỗi, mà là để **tối ưu hóa quy trình học của AI**. Giống như việc bạn cho trẻ con học đánh vần (Grid) trước khi cho nó đọc tiểu thuyết (Vicocktail).
