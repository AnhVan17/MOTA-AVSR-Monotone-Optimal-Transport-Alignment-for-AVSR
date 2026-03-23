
# Sự Tiến Hóa Kiến Trúc: Từ AURORA-XT đến MOTA

Tài liệu này phân tích sự thay đổi, tối ưu hóa và tái cấu trúc hệ thống mô hình từ phiên bản cũ (`aurora_xt.py`) sang phiên bản mới (`mota.py` và các module vệ tinh).

**Lưu ý quan trọng:** `mota.py` không phải là code viết mới hoàn toàn từ con số 0. Nó là kết quả của quá trình **Refactoring (Tái cấu trúc)** và **Evolution (Tiến hóa)** từ file `aurora_xt.py` nguyên khối. Chúng ta đã "đập nhỏ" file cũ ra, giữ lại những phần cốt lõi (như tư duy fusion), và nâng cấp từng bộ phận thành các module chuyên biệt mạnh mẽ hơn.

---

## 1. Tổng Quan Sự Thay Đổi

| Đặc điểm | AURORA-XT (Cũ) | MOTA (Mới) | Lợi ích tối ưu |
| :--- | :--- | :--- | :--- |
| **Kiến trúc File** | Monolithic (Nguyên khối). Tất cả logic (fusion, encoder, decoder) dồn vào 1-2 file. | **Modular (Module hóa).** Tách biệt rõ ràng thành `layers/`, `fusion/`, `adapters/`. | Dễ debug, dễ thay thế từng phần (VD: đổi Conformer mà không hỏng Fusion). |
| **Chiến thuật Fusion** | Single-Stage (Thường là concatenation hoặc attention đơn giản). | **Two-Stage Fusion.** Giai đoạn 1: Quality Gate. Giai đoạn 2: Optimal Transport. | Xử lý nhiễu tốt hơn. Giai đoạn 1 lọc thô, Giai đoạn 2 tinh chỉnh sự lệch pha (misalignment). |
| **Feature Space** | Audio/Visual chiếu về `d_model` (256) ngay từ đầu. | Giữ nguyên chiều sâu của Whisper (768) cho Audio và dùng **Adapter** để upscale Visual lên 768. | Giữ lại lượng thông tin ngữ nghĩa phong phú của Whisper thay vì nén sớm. |
| **Decoder** | Basic Transformer/LSTM. | **Hybrid Decoder.** Kết hợp CTC (non-autoregressive) và Attention (autoregressive). | Tăng tốc độ hội tụ và độ chính xác. |

---

## 2. Chi Tiết Tối Ưu Hóa & Tách Module

Hệ thống mới được chia nhỏ thành các file chuyên biệt trong `src/models/`, tất cả đều có nguồn gốc từ logic cũ nhưng được nâng cấp:

### A. Core Model (`mota.py`)
*Nguồn gốc: Class `AuroraXT` trong file cũ.*
Thay vì chứa hàng nghìn dòng code logic hỗn độn, `mota.py` giờ chỉ đóng vai trò **Orchestrator** (người điều phối):
-   Quản lý luồng dữ liệu (Data Flow).
-   Tích hợp toggle `use_mqot` để chuyển đổi linh hoạt giữa Phase 1 (Training cơ bản) và Phase 2 (Training nâng cao).
-   Tự động khởi tạo các module con dựa trên config.

### B. Fusion Modules (`src/models/fusion/`)

Đây là nơi chứa sự cải tiến thuật toán lớn nhất:

**1. `quality_gate.py` (Cải tiến từ logic cũ)**
-   **Chức năng:** Đánh giá độ tin cậy của Audio và Visual.
-   **Tối ưu:** Tách riêng thành một module độc lập. Sử dụng cơ chế Gating để quyết định xem frame nào bị nhiễu (noise) thì giảm trọng số ngay lập tức.

**2. `mqot.py` (Mới hoàn toàn - Trái tim của MOTA)**
-   **Chức năng:** Thực hiện **Multimodal Optimal Transport (Vận chuyển tối ưu đa phương thức)**.
-   **Tại sao tối ưu?**
    -   Attention thông thường chỉ so sánh độ tương đồng (similarity).
    -   MQOT so sánh "chi phí vận chuyển" dựa trên chất lượng (Quality-aware). Nếu một frame hình ảnh mờ/nhòe, MQOT sẽ "vận chuyển" thông tin audio đến một frame hình ảnh khác rõ hơn lân cận, thay vì cố gắng attention vào frame hỏng đó.
-   **Guided Attention:** Sử dụng bản đồ vận chuyển (Transport Map) để hướng dẫn Attention, giúp mô hình tập trung chính xác hơn.

### C. Layers & Adapters (`src/models/layers/`)

**1. `adapter.py` (VisualAdapter)**
-   **Vấn đề cũ:** ResNet ra 512 chiều, Whisper ra 768 chiều. Cũ thường ép cả 2 về 256.
-   **Tối ưu mới:** Dùng Adapter (lấy cảm hứng từ Q-Former) để biến đổi Visual (512) -> Rich Visual (768).
-   **Tác dụng:** Đồng bộ không gian vector với Whisper mà không làm mất mát thông tin ngữ nghĩa của Audio như cách nén cũ.

**2. `conformer.py`**
-   *Nguồn gốc:* Các class `ConformerBlock`, `FeedForward`, `MultiHeadedSelfAttention` nằm rải rác trong file cũ.
-   *Thay đổi:* Tách khối `ConformerBlock` ra riêng file `src/models/layers/conformer.py`. Giúp code model chính gọn gàng hơn và dễ tái sử dụng.

**3. `decoders.py`**
-   *Nguồn gốc:* Phần logic decoding (CTCLayer, TransformerDecoder) trong file cũ.
-   *Thay đổi:* Tách ra file `src/models/layers/decoders.py` và nâng cấp lên kiến trúc **Hybrid (Joint CTC/Attention)** chuẩn SOTA thay vì chỉ dùng Attention đơn thuần như trước.

---

## 3. Luồng Hoạt Động (Pipeline Comparison)

### Luồng cũ (AURORA-XT)
```mermaid
Audio -> Linear(256) \
                      (+) -> Conformer -> Decoder -> Output
Visual -> Linear(256) /
```
*Nhược điểm:* Nén dữ liệu quá sớm, fusion đơn giản, khó lọc nhiễu.

### Luồng mới (MOTA)
```mermaid
Phase 1 (Coarse):
Audio -> Linear(256) \
                      (QualityGate) -> Fused_Coarse (256)
Visual -> Linear(256) /

Phase 2 (Fine - MQOT):
Fused_Coarse -> Upsample(768) (Audio Rich)
Visual ---------------------> Adapter(768) (Visual Rich)
       |
       v
    (MQOT & Guided Attention) -> Fused_Fine (256)

Final:
Fused_Coarse + Fused_Fine -> Conformer -> Hybrid Decoder
```
*Ưu điểm:*
1.  **Dư thừa thông tin:** Giữ fused coarse để đảm bảo không mất thông tin gốc.
2.  **Tinh chỉnh:** MQOT chỉ đóng vai trò "residual" (phần dư) để sửa lỗi cho QualityGate.
3.  **Deep Features:** Làm việc trên không gian 768 chiều của Whisper giúp tận dụng pre-trained knowledge tốt hơn.

---

## 4. Kết Luận

Việc chuyển đổi sang kiến trúc MOTA mang lại nền tảng vững chắc để đạt kết quả SOTA:
1.  **Code sạch hơn:** Dễ bảo trì, dễ mở rộng thêm các module mới (với file adapter, fusion riêng).
2.  **Thuật toán mạnh hơn:** MQOT giải quyết triệt để vấn đề lệch pha thời gian và nhiễu (noise) thường gặp trong Vicocktail dataset.
3.  **Huấn luyện linh hoạt:** Có thể train Phase 1 nhanh chóng, sau đó bật Phase 2 để fine-tune, tiết kiệm tài nguyên tính toán.
