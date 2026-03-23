# Kế hoạch Xây dựng Pipeline AVSR End-to-End Tích hợp MQOT (Tiếng Việt)

Tài liệu này vạch ra định hướng kiến trúc tuyến tính từ cấu hình dữ liệu đầu vào (Video), xử lý mô hình đến khi xuất ra kết quả (Text). Trọng tâm của kế hoạch là nâng cấp cơ chế dung hợp (Fusion) sang **Giai đoạn 2: Multimodal Optimal Transport (MQOT)**, nhằm giải quyết triệt để lỗi bất đồng bộ tinh (fine-grained misalignment) gây ra bởi hiện tượng đồng kết cấu (co-articulation) trong phát âm tiếng Việt, với tham vọng rớt WER xuống dưới 15%.

---

## 1. Tình trạng Hiện tại & Vấn đề

*   **Baseline (Giai đoạn 1):** Đã vận hành thành công luồng E2E sử dụng Whisper (Audio) và ResNet-18 (Visual) qua cơ sở `QualityGate`.
*   **Chỉ số:** WER hiện tại đạt **21.3%** trong môi trường nhiễu.
*   **Nút thắt:** Sự bất đồng bộ pha (misalignment) giữa hình ảnh (khẩu hình cử động trước hoặc sau do co-articulation) và âm thanh (phát âm). `QualityGate` chỉ nén dập khuôn ở mức độ từng frame (Frame-level) mà không có sự linh hoạt "chuyển dịch" thời gian.

---

## 2. Kiến trúc Luồng Xử Lý End-to-End (E2E Pipeline Flow)

Hệ thống được thiết kế theo 3 luồng nối tiếp: Tiền xử lý -> Mô hình (Dung hợp) -> Sinh văn bản.

### A. Tầng Tiền Xử Lý Dữ Liệu Tốc Độ Cao (GPU-Native Preprocessing)
Thực thi trên Cloud/Server để tạo kho features nạp cho bước Train.

1.  **Input:** Nhận chuỗi files `.mp4` (Raw Video).
2.  **Mouth Cropping (Giải quyết rào cản tốc độ):** 
    *   Loại bỏ MediaPipe, thay thế bằng **`face-alignment`** chạy trực tiếp trên GPU.
    *   Quy trình: Video -> Tensor `[B, C, H, W]` -> CUDA -> Cắt 88x88 quanh môi nhanh gấp 10x.
3.  **Feature Extraction:**
    *   **Audio (Whisper Encoder):** Đẩy băng tần Audio qua Whisper để trích xuất đặc ngữ nghĩa sâu (768-dim) không gian từ vựng.
    *   **Visual (ResNet18):** Chụp các khung hình môi (Grayscale/RGB) sinh ra tensor (512-dim).
4.  **Save/Output:** Ghép khối và nén xuống định dạng `.pt`, sẵn sàng nạp thẳng vào RAM một cách trơn tru.

### B. Tầng Trí Tuệ Mô Hình & Dung Hợp (The Brain & Fusion)
Đây là nơi chứa **Giai đoạn 2 (MQOT)** - cốt lõi của sự nâng cấp. Đặt Audio ($A$) và Video ($V$) là hai phân phối xác suất cần được vận chuyển xấp xỉ nhau.

1.  **Coarse Alignment (QualityGate):** Vẫn được giữ lại như hàng rào lọc nhiễu sơ cấp để tạo khung nền tảng.
2.  **Fine-Grained Alignment (Lớp Sinh Ma Trận MQOT):**
    *   **Tính Cost Matrix:** Xét chi phí "vận chuyển" thông tin từ frame $i$ của Tiếng sang frame $j$ của Hình dựa trên 3 độ lệch: *Cosine Distance (Đặc trưng)* + *Temporal L1 (Khoảng cách thời gian)* + *Quality Penalty (Đánh giá nhiễu của frame)*.
    *   **Thuật toán Sinkhorn-Knopp:** Giải gần đúng bài toán Vận chuyển tối ưu (OT) trên chi phí $O(T^2)$ ngay trong lúc Gradient truyền ngược. Output ra là **Ma Trận Vận Chuyển ($\Gamma \in \mathbb{R}^{T_a \times T_v}$)**.
3.  **Guided Cross-Attention:**
    *   Thay vì để mạng Attention nhìn "chùn chũn" vào mọi pixel, ta dùng $\Gamma$ để "nhấn mạnh" sự tương quan.
    *   Cơ chế: Nhân ma trận xác suất Attention nội tại với phân phối $\Gamma$. Ép mạng bắt buộc phải bắt được chuyển động môi bị lệch pha. (Rất hợp với từ đơn âm tiết tiếng Việt nhưng bị rớt nhịp).

### C. Tầng Giải Mã Ngôn Ngữ (Hybrid Decoder)
Tối ưu hóa khả năng hiểu đặc thù Ngữ pháp/Âm sắc tiếng Việt.

1.  **Sequence Modeling:** Nạp khối `Fused` cuối cùng đi qua mạng **Conformer Encoder**.
2.  **Dự đoán Thanh Điệu (Auxiliary Task):** (Đề xuất thêm) Bắn 1 nhánh Loss phụ nhận diện 6 dấu thanh tiếng Việt từ F0 audio, chống liệt khả năng mù dấu của video.
3.  **Syllable-BPE Tokenizer:** Thay vì dùng tokenizer đa ngôn ngữ của Whisper, chia subword theo cấp độ Âm tiết Tiếng Viết ($Vocab \approx 10,000$).
4.  **Hybrid Decoding (CTC / AR):** Lai ghép CTC Loss (gióng hàng tốc độ cao nhanh) và Auto-Regressive (mượt văn cảnh). Kết xuất ra Transcript trọn vẹn.

---

## 3. Lộ Trình Triển Khai (Roadmap Xây Dựng Thực Tế)

**Bước 1: Nâng cấp Tiền xử lý (Hoàn tất phần Dữ liệu)**
*   Thay class MediaPipe bằng `face-alignment` trong luồng Preprocessor.
*   Chạy Modal quét lại toàn bộ dataset (ViCocktail) tốc độ cao tạo kho `.pt`.

**Bước 2: Cập nhật Lớp Hình Phạt (Penalty) tiếng Việt cho MQOT**
*   Cải tiến `QualityEstimator` trong hệ thống (tích hợp audio làm ngữ cảnh nhằm tránh đánh giá sai các phụ âm c, k, kh, g, ng không có biểu hiện môi).
*   Chỉnh sửa hằng số biên độ (Penalty limits) trong thuật toán `Sinkhorn` phù hợp cho độ rung âm tiếng Việt.

**Bước 3: Thu hẹp Vocab & Tích hợp Tonal CTC Loss**
*   Thay Tokenizer size từ 51865 xuống ~10k âm tiết.
*   Viết thêm hàm dự đoán nhánh Thanh điệu (6 âm) tại đầu ra của Layer Conformer.

**Bước 4: Huấn Luyện Giai Đoạn 2 (Phase 2 Finetuning)**
*   Đóng băng (Freeze) lớp `QualityGate` đã hoàn chỉnh của Giai đoạn 1.
*   Mở toàn bộ learning_rate cho lớp `MQOT` và `HybridDecoder`. Giám sát Transport Map (${\Gamma}$) thông qua biểu đồ (Plot/Visualize) trên TensorBoard để xem Hình và Tiếng có chạy so le nhau không.

**Bước 5: Thử Nghiệm Thực Tế (E2E Live Inference)**
*   Bật tham số cờ `use_backbones = True` trong file mạng chính (`mota.py`).
*   Ném thẳng file Video/Webcam `.mp4` vào pipeline. Quá trình tiền cắt mồm, Whisper, ResNet18, MQOT sẽ chạy nội sinh song song trên RAM/VRAM và xuất ngay ra văn bản. Không sinh file rác trung gian.

---
*Tài liệu này đóng vai trò là kim chỉ nam toàn diện nhất cho việc hoàn thiện Dự Án MOTA hướng đến tối ưu hóa hoàn toàn cho tiếng Việt, hứa hẹn mở khóa độ chính xác cực đỉnh SOTA cho bài toán đặc thù này.*
