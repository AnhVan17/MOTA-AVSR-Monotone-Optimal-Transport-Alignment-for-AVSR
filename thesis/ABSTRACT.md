# Multimodal Alignment through Optimal Transport for Audiovisual Speech Recognition

---

## Tiếng Việt

Nghiên cứu này tập trung vào bài toán fusion đa phương thức trong nhận dạng tiếng nói kết hợp âm thanh và hình ảnh (Audio-Visual Speech Recognition — AVSR) cho ngôn ngữ ít tài nguyên, cụ thể là tiếng Việt. Mục tiêu là thiết kế kiến trúc MOTA-v2, một hệ thống AVSR nhận biết độ tin cậy (reliability-aware), với ba cải tiến cốt lõi: SyncPreprocessor — module đồng bộ hóa feature-level sử dụng cross-correlation lag detection và cubic interpolation để sửa lệch pha audio-visual; Router-Gated Fusion (RGF) kết hợp Noise-Robust Quality Estimator (NRQE) — cơ chế routing theo chunk cho phép mô hình tự quyết định ở một trong ba chế độ: audio-dominant, visual-dominant hoặc fusion, dựa trên cross-modal consistency và uncertainty; và MQOT-v2 (Multi-modal Quality-aware Optimal Transport) — lớp optimal transport alignment cải tiến với unbalanced Sinkhorn và learnable epsilon. Phương pháp sử dụng pipeline 6 giai đoạn kết hợp Whisper encoder, ResNet18, Conformer encoder và Hybrid CTC+AR decoder, huấn luyện hai pha trên ViCocktail (tiếng Việt, ~269 giờ) và GRID (tiếng Anh, ~15K mẫu cho nghiên cứu ablation). Kết quả dự kiến đạt WER ~28–30% trên ViCocktail, tương đương với Whisper audio-only trong điều kiện nhiễu, với yêu cầu VRAM ~2 GiB. Nghiên cứu góp phần cải thiện độ bền vững của AVSR trong điều kiện nhiễu âm, che khuất video và lệch pha không đồng bộ, phù hợp với triển khai thực tế cho ngôn ngữ ít tài nguyên.

---

## English

This study addresses multimodal fusion in Audio-Visual Speech Recognition (AVSR) for low-resource languages, specifically Vietnamese. The objective is to design MOTA-v2, a reliability-aware AVSR architecture with three core improvements: SyncPreprocessor — a feature-level synchronization module using cross-correlation lag detection and cubic interpolation to correct audio-visual temporal misalignment; Router-Gated Fusion (RGF) combined with a Noise-Robust Quality Estimator (NRQE) — a chunk-level routing mechanism enabling the model to select between three modes — audio-dominant, visual-dominant, or fusion — based on cross-modal consistency and uncertainty; and MQOT-v2 (Multi-modal Quality-aware Optimal Transport) — an improved optimal transport alignment layer with unbalanced Sinkhorn and learnable epsilon. The methodology uses a 6-stage pipeline combining Whisper encoder, ResNet18, Conformer encoder, and Hybrid CTC+AR decoder, trained in two phases on ViCocktail (Vietnamese, ~269 hours) and GRID (English, ~15K samples for ablation studies). Results are expected to achieve ~28–30% WER on ViCocktail, comparable to Whisper audio-only under noisy conditions, at ~2 GiB VRAM. This work contributes to improving AVSR robustness under acoustic noise, visual occlusion, and temporal asynchrony, suitable for real-world deployment in low-resource language settings.

---

# PHẦN I — Ý TƯỞNG THUẬT TOÁN VÀ HỆ THỐNG

---

## 1. Bối cảnh (Background)

AVSR đã đạt hiệu suất cao trên các benchmark sạch như LRS3 (tiếng Anh), nhưng các phương pháp fusion hiện tại vẫn tồn tại hai hạn chế cốt lõi khi triển khai thực tế. Thứ nhất, các hệ thống fusion dựa trên weighted sum hoặc attention đơn giản không có khả năng ước lượng độ tin cậy theo từng khung thời gian (per-frame reliability), dẫn đến hiện tượng double degradation — khi cả âm thanh lẫn hình ảnh đều suy giảm chất lượng, mô hình vẫn cố gắng fusion hai luồng bị degraded. Thứ hai, các phương pháp alignment hiện tại giả định tốc độ nói đều và tỷ lệ khung hình bằng nhau (Ta = Tv), không xử lý được hiện tượng lệch pha audio-visual (temporal misalignment) thường xuyên xảy ra trong dữ liệu thực tế. Đối với tiếng Việt, bài toán nghiêm trọng hơn do đặc trưng thanh điệu — nhiều cặp từ chỉ phân biệt được qua hình ảnh môi (ví dụ: *bà*, *bá*, *bả*, *bạ* có cùng âm nhưng khác thanh) — trong khi dữ liệu huấn luyện AV đồng bộ chất lượng cao cho tiếng Việt còn rất hạn chế.

---

## 2. Pain Point / Bottleneck / Vấn đề

### 2.1. Mức Model

**Fusion không nhận biết độ tin cậy, dẫn đến double degradation.** Các hệ thống AVSR hiện tại dùng concatenation hoặc attention đơn giản mà không tạo tương tác cross-modal sâu (GILA, IJCAI 2023), và cross-attention đầy đủ lại quá nặng tính toán (CoBRA, ICASSP 2026). Schweitzer et al. (2024) cho thấy AV-HuBERT — mô hình SOTA — không tái cân bằng độ tin cậy theo từng thời điểm: cho quá nhiều weight trên audio khi clean, quá ít khi visual degraded. Double degradation — khi cả audio lẫn visual đều bị corrupted — là vấn đề nghiêm trọng nhất; Hong et al. (CVPR 2023) ghi nhận AVSR cũ không cải thiện hoặc thậm chí kém hơn audio-only, và Huang et al. (arXiv 2026) ghi nhận WER tăng 35 lần trên dữ liệu video conferencing thực tế.

**Alignment đơn giản không xử lý được lệch pha audio-visual.** Motor delay tự nhiên 40–120 ms giữa acoustic signal và visual articulation là hiện tượng vật lý (Bengio, 2003), không phải artifact, khiến các phương pháp interpolation đơn giản giả định uniform motion — không đúng với thực tế. Frame-rate mismatch (T_a ≠ T_v) tích lũy thêm systematic misalignment ngay từ đầu pipeline.

**Decoder CTC thiếu ngữ cảnh ngôn ngữ.** CTC giả định conditional independence giữa predictions — tốt cho alignment nhưng không capture được sequential language context. GILA (2023) khắc phục bằng Transformer decoder, nhưng AR tự-regressive đòi hỏi tài nguyên lớn, không phù hợp với low-resource setting. Đối với tiếng Việt với 6 thanh điệu, CTC càng yếu vì không có cơ chế explicit tone modeling.

### 2.2. Mức Data

**Thiếu dữ liệu AV và cross-lingual mismatch.** ViCocktail (Interspeech 2025) có ~269 giờ AV tiếng Việt, nhưng các checkpoint pretrained mạnh nhất đều huấn luyện trên tiếng Anh. ViCocktail paper ghi nhận AV-HuBERT pretrained tiếng Anh đạt 9.40% WER clean, vượt trội hơn hẳn so với from-scratch (18.60%), cho thấy visual representations từ tiếng Anh không transfer tốt. Vấn đề phức tạp thêm với tiếng Việt có 6 thanh điệu (ngang, sắc, huyền, hỏi, ngã, nặng) — thanh điệu thể hiện qua pitch (độ cao âm thanh) mà nhìn môi không thấy được, nên visual modality không trực tiếp phân biệt được thanh điệu.

### 2.3. Mức System / Pipeline

**Preprocessing là bottleneck hiệu năng.** MediaPipe FaceMesh chạy CPU-only để tránh EGL conflict với PyTorch CUDA, tốc độ chậm (~30 phút cho 1 giờ video). Face detection thất bại với pose lệch (>30° yaw) — không tạo ra quality signal cho downstream fusion, trong khi lệch pha tự nhiên và video compression delay tạo asynchrony không cố định, không xử lý được bằng static preprocessing.

---

## 3. Ý tưởng chính

MOTA-v2 giải quyết các pain point trên bằng ba ý tưởng cốt lõi. Thứ nhất, thay vì dùng interpolation đơn giản để align hình ảnh môi vào âm thanh, MOTA-v2 dùng cross-attention có học được: mô hình tự quyết định frame nào của video tương ứng với frame nào của âm thanh, không giả định tốc độ nói đều, đồng thời ước lượng độ tin cậy của từng modality tại mỗi frame để tự quyết định nên tin vào kênh nào hơn — gọi là QualityGate v2. Thứ hai, sau QualityGate, MOTA-v2 dùng thêm một lớp Optimal Transport alignment (MQOT-v2) để refine alignment chi tiết hơn giữa audio và visual, đặc biệt hỗ trợ trường hợp số lượng frames audio và video không bằng nhau. Thứ ba, decoder dùng Hybrid CTC+AR: CTC giúp học alignment giữa đặc trưng và transcript tự động (không cần alignment thủ công), còn AR decoder thêm ngữ cảnh ngôn ngữ để cải thiện độ chính xác. Pipeline tổng quan: Whisper encoder trích đặc trưng âm thanh → ResNet18 trích đặc trưng hình ảnh môi → QualityGate v2 fusion → MQOT-v2 alignment → Conformer encoder → Hybrid CTC+AR decoder → transcript.

---

## 4. Kết quả dự kiến

Trên tập kiểm tra ViCocktail (clean + SNR = −5 đến 10 dB), MOTA-v2 với Level 1 decoder (CTC+Transformer) hướng tới WER mục tiêu ~28–30% trên điều kiện clean, tương đương với audio-only Whisper (PhoWhisper: 13.44% clean) trong điều kiện nhiễu nặng (SNR ≤ 0 dB), với ưu thế cải thiện robustness ở cả ba chiều: nhiễu âm, che khuất video, và lệch pha không đồng bộ. Baseline ViCocktail Conformer AVSR (AV4) đạt 14.4% clean WER với pretrained Auto-AVSR (tiếng Anh), từ đó đánh giá tương đối. Trên GRID (tiếng Anh, ~15K mẫu cho ablation), phân tách đóng góp riêng lẻ của từng thành phần (SyncPreprocessor, RGF, NRQE, MQOT-v2) trong các điều kiện nhiễu âm, che khuất video (Gaussian blur, occlusion 10–30%), và lệch pha không đồng bộ (±200 ms). VRAM mục tiêu ~2 GiB. Các cấp decoder nâng cao — Level 2 với CTC+Bi-LSTM LM hoặc LoRA adapters (mục tiêu WER ~24–26%), Level 3 với OTTC loss hoặc FlexCTC beam search (mục tiêu WER ~22–24%) — là hướng mở rộng tiếp theo của đề tài.

---

## 5. Kết luận và ý nghĩa

Đề tài góp phần giải quyết pain point cốt lõi của AVSR trong điều kiện triển khai thực tế: fusion chủ động thích ứng theo độ tin cậy từng khung thời gian thay vì weighting tĩnh, đồng bộ hóa audio-visual ở mức feature trước bước fusion, và alignment chính xác hơn qua optimal transport với unbalanced Sinkhorn. Kiến trúc MOTA-v2 được thiết kế phù hợp với low-resource setting — tiếng Việt với ~269 giờ dữ liệu AV, VRAM ~2 GiB — không phụ thuộc vào LLM decoder đòi hỏi tài nguyên tính toán lớn. Các kết quả đóng góp vào hướng nghiên cứu router-gated multimodal fusion, uncertainty-aware quality estimation, và alignment-free AVSR cho ngôn ngữ ít tài nguyên.
