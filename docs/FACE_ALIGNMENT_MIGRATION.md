# Chuyển đổi Khung xử lý thị giác: MediaPipe sang Face-Alignment (GPU-Native)

Tài liệu này ghi chú lại quá trình và phân tích chuyên sâu về việc thay thế toàn bộ thư viện MediaPipe (CPU-only) bằng `face-alignment` (GPU-native) trong toàn hệ thống AVSR Tiếng Việt.

---

## 1. Vấn Đề (Painpoints) của MediaPipe

Trong sơ đồ kiến trúc cũ, `MediaPipe FaceMesh` được dùng để bóc tách khuôn miệng (Mouth ROI). Dù nhẹ, MediaPipe mang lại các "nút thắt cổ chai" (Bottlenecks) lớn:

1. **Gây Crash do tranh chấp môi trường (Context Conflict):** MediaPipe trong Python ưu tiên chạy trên CPU. Việc cố gắng chạy nó trên Server Headless (Modal) bắt buộc phải qua cầu nối EGL/GLES. Khi PyTorch cũng khởi tạo CUDA trên cùng môi trường, luồng OpenGL bị đụng độ gây ra lỗi `Error 0x3008`, văng tiến trình hoàn toàn.
2. **Kém ổn định với dữ liệu nhiễu (in-the-wild):** ViCocktail chứa nhiều góc mặt nghiêng, thiếu sáng. MediaPipe nhạy cảm và thường xuyên "rớt dấu" khuôn mặt (drop frames), sinh ra dữ liệu đầu vào rác cho mô hình ResNet.

---

## 2. Cách Thay thế MediaPipe & Khai thác sức mạnh GPU

Chúng ta đã chuyển đổi hoàn toàn sang `face-alignment>=1.4.0`. Thư viện này dùng mô hình dò mặt `SFD` và mạng lưới nội suy điểm `2D FAN` viết hoàn toàn trên PyTorch.

### 2.1. Dịch chuyển từ CPU sang GPU (Dịch chuyển tài nguyên)
- Thay vì để CPU tải công việc nặng nhọc cắt từng frame hình, `face-alignment` được cấu hình để khởi tạo thẳng trên `device='cuda'`.
- Toàn bộ ma trận khung hình (Video Frames) được nạp trực tiếp lên không gian VRAM của thẻ đồ họa (GPU Tensor). Thuật toán dò tìm và nội suy xử lý song song hàng nghìn điểm ảnh cùng lúc. Việc đoạt tuyệt 100% với CPU giúp loại bỏ hoàn toàn rào cản I/O Overhead (thời gian copy ngược xuôi giữa RAM máy tính và RAM card màn hình).

### 2.2. Cách thức Thay thế mã nguồn (Refactoring)
1. **Loại bỏ Tàn dư Cũ:** Xóa sạch thư mục/hành vi của MediaPipe, gỡ bỏ thư viện `mediapipe` khỏi `requirements.txt`. Gỡ bỏ mọi câu lệnh "hacker" che giấu GPU (`os.environ["CUDA_VISIBLE_DEVICES"] = ""`). Cơ chế được thanh lọc hoàn toàn.
2. **Thay đổi Hệ tọa độ (Landmarks Mapping):**
   * *MediaPipe (468 points):* Chấm điểm rải rác `[13, 14, 61, 291...]` - khó kiểm soát.
   * *face-alignment (68 points):* Dùng hệ quy chiếu 68-điểm tiêu chuẩn học thuật. Vùng môi luôn cố định khép kín từ Index `[48]` tới `[67]` (20 điểm bao quanh môi), giúp tọa độ `(cx, cy)` bắt tâm miệng chính xác tuyệt đối.

---

## 3. Cách Xây dựng & Áp dụng vào Pipeline Hệ thống

Sự thay đổi này được áp dụng thành 2 luồng hoạt động chính (Data Preparation & E2E Inference), giúp thống nhất định hướng: **"Để GPU lo tất cả"**.

### 3.1. Giai đoạn Huấn luyện: Chuẩn bị Dữ liệu (Pipeline Tiền xử lý)
Việc tiền xử lý chia làm 2 tầng nay đã được gom chung vào GPU. Quá trình xử lý song song chạy thẳng qua hệ thống Modal Server:

* **Sử dụng:** `modal run scripts/data_prep/prep_vicocktail.py --action process`
* **Workflow Tiền xử lý Mới:** 
  File nén Tar $\rightarrow$ RAM $\rightarrow$ VRAM (GPU) $\rightarrow$ `face-alignment` cắt miệng đồng loạt $\rightarrow$ `ResNet-18` ép kiểu thành vector 512-dim $\rightarrow$ Viết ra file Tensor `.pt`. Toàn bộ chuỗi khép kín trong GPU. Cuối cùng, sinh ra đặc trưng cực nhanh mà không gây Crash hệ thống.

### 3.2. Giai đoạn Thực tiễn: Sẵn sàng Suy luận Thời gian thực (Live E2E Inference)
Mục tiêu tối thượng của đề tài Khóa luận là đưa một video thô `mp4` và nhận lại Chữ. Sự thay thế sang `face-alignment` chính là **mảnh ghép còn thiếu** để luồng Inference thực sự vận hành mượt mà chuẩn Real-time.

**Cách áp dụng vào kiến trúc MOTA:**
```python
# Kích hoạt E2E trong file kiến trúc mota.py
mota_model = MOTA(use_backbones=True) 
output_text = mota_model(raw_video_frames, raw_audio_waveform)
```

**Dòng chảy luồng dữ liệu E2E mới:**
1. Khung hình (Raw Frames) đẩy trọn gói vào GPU.
2. `VideoProcessor` (gói lõi `face_alignment`) chạy chớp nhoáng trên Sub-thread GPU để bóc tách ROI miệng chuẩn xác ở mọi góc độ.
3. ROI truyền thẳng biến dưới dạng Tensor nội bộ sang `ResNet-18` (Visual Backbone).
4. `Audio Backbone` chạy song song để giải nén Audio 16kHz tới `Whisper`.
5. Đẩy đặc trưng gộp vào khối `QualityGate` + `MQOT` $\rightarrow$ sinh ký tự cuối cùng.

**Đánh giá:** Giai đoạn này cho thấy Năng lực Tối ưu hóa mô hình từ mặt Lý thuyết đến Hạ tầng Kỹ thuật. Face-alignment thay thế MediaPipe để đạt được tốc độ cao, loại bỏ rào cản phần cứng và dọn đường đưa dự án AVSR chạm ngõ cấp độ Sản phẩm Công nghiệp (Production-ready).
