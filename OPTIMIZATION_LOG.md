
# Nhật Ký Tối Ưu Hóa & Sửa Lỗi Hệ Thống
*Ngày cập nhật: 2026-01-02*

Tài liệu này ghi lại các chỉnh sửa quan trọng nhằm đảm bảo tính toàn vẹn (integrity) và hiệu năng (performance) cho pipeline huấn luyện AVSR đa nhiệm cho cả Grid và Vicocktail.

---

## 1. Các Lỗi Nghiêm Trọng Đã Fix

### A. Missing Mask Handling (Xử lý Mask bị thiếu)
**Vấn đề:** 
Mặc dù `collate.py` đã tạo ra các boolean mask (`audio_mask`, `visual_mask`) để đánh dấu vùng dữ liệu thật so với vùng padding (giá trị 0.0), nhưng Model và Trainer đã phớt lờ chúng.
**Hậu quả:**
Mô hình Attention tính toán cả trên vùng padding, gây nhiễu tín hiệu và làm giảm đáng kể độ chính xác (đặc biệt với Vicocktail có độ dài câu rất biến thiên).

**Giải pháp:**
1.  **Update `src/models/mota.py`:** 
    -   Hàm `forward()` nhận thêm tham số `audio_len`, `visual_len`.
    -   Tự tạo mask bên trong model.
    -   Áp dụng mask nhân vào output của Encoder (`encoded * mask`).
2.  **Update `src/training/trainer.py`:**
    -   Trong vòng lặp training, unpack `audio_len` và `visual_len` từ batch dictionary.
    -   Truyền các tham số này vào hàm `self.model()`.

### B. Preprocessing Alignment
**Vấn đề:**
`QualityGate` yêu cầu Audio và Visual feature map phải cùng chiều dài thời gian (`Ta == Tv`), nhưng thực tế Audio (Whisper) và Visual (ResNet) có sample rate khác nhau.
**Giải pháp:**
Xác nhận cơ chế `F.interpolate` (Linear Interpolation) trong `src/models/fusion/quality_gate.py` là giải pháp align thô (Coarse Alignment) chấp nhận được cho Phase 1. Phase 2 sẽ có MQOT xử lý sự lệch pha phi tuyến tính.

---

## 2. Kiểm Tra Tính Tương Thích (Compatibility Check)

### Dataset: Grid
-   **Đặc điểm:** Video ngắn (3s), câu lệnh cố định.
-   **Tình trạng:** Tương thích hoàn toàn.
-   **Lợi ích:** Cơ chế Masking mới giúp Grid training ổn định hơn vì không còn bị nhiễu bởi padding (dù Grid ít padding hơn Vicocktail).

### Dataset: Vicocktail
-   **Đặc điểm:** Video dài ngắn khác nhau, nhiễu nền, mask môi thỉnh thoảng mất.
-   **Tình trạng:** Tương thích hoàn toàn.
-   **Lợi ích:**
    -   `modal_preprocess_vicocktail.py` chuyên biệt giúp xử lý transcript và audio chuẩn.
    -   Masking giúp model không học sai từ các đoạn video bị crop lỗi hoặc padding dài ngoằng.
    -   MQOT (Phase 2) sẽ phát huy tối đa tác dụng để sửa các đoạn môi bị lệch tiếng.

---

## 3. Khuyến Nghị Tiếp Theo
-   Theo dõi Log loss của CTC. Nếu CTC loss không giảm, kiểm tra lại tham số `audio_len` truyền vào loss function (Trainer đang truyền đúng, nhưng cần monitor thực tế).
-   Với Vicocktail, nên train Phase 1 (QualityGate only) khoảng 10-20 epoch cho hội tụ trước khi bật Phase 2 (MQOT) để tránh gradient nổ do MQOT quá phức tạp lúc đầu.
