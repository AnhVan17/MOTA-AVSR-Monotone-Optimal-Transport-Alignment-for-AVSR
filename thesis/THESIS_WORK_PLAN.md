# Kế Hoạch Phân Công 2 Người — Luận Văn MOTA-v2 AVSR

**Ngày lập:** 2026-04-27
**Đề tài:** MOTA-v2 — Multimodal Optimal Transport Alignment for Vietnamese AVSR
**Nhóm:** 2 thành viên (A & B)
**Thời lượng còn lại:** **30 ngày (1 tháng)** — sprint cường độ cao
**Mục tiêu trang:** **80–100 trang luận văn**

> ⚠️ **Lưu ý quan trọng:**
>
> - `thesis_proposal.tex` **CHỈ là đề cương tham khảo ban đầu, KHÔNG dùng làm khung viết chính thức**. Luận văn 80-100 trang sẽ được viết hoàn toàn mới từ đầu, dựa trên [ABSTRACT.md](ABSTRACT.md) + [ALGORITHM_SYSTEM_OVERVIEW.md](ALGORITHM_SYSTEM_OVERVIEW.md) + kết quả thực nghiệm sắp chạy.
> - Plan này được nén từ 12 tuần xuống còn 30 ngày → **mỗi ngày phải có deliverable cụ thể**, không có "buffer time".

---

## Mục Lục

1. [Cơ sở phân chia](#1-cơ-sở-phân-chia)
2. [Nguyên tắc phối hợp](#2-nguyên-tắc-phối-hợp)
3. [Phân vai tổng thể](#3-phân-vai-tổng-thể)
4. [Mục lục luận văn (6 chương, 80-100 trang)](#4-mục-lục-luận-văn-6-chương-80-100-trang)
5. [Phân công chi tiết theo chương](#5-phân-công-chi-tiết-theo-chương)
6. [Phân công code & thực nghiệm](#6-phân-công-code--thực-nghiệm)
7. [Lịch trình 30 ngày](#7-lịch-trình-30-ngày)
8. [Checklist hàng ngày / hàng tuần](#8-checklist-hàng-ngày--hàng-tuần)
9. [Rủi ro và phương án cắt giảm phạm vi](#9-rủi-ro-và-phương-án-cắt-giảm-phạm-vi)

---

## 1. Cơ sở phân chia

Tài liệu `docs/Week 2 - PHan Biet Luan Van và Technical Report.docx` nhấn mạnh **luận văn ≠ technical report**. Luận văn phải trả lời:

- Bài toán nào? Vì sao đáng nghiên cứu?
- Cách hiện tại có hạn chế gì?
- Đề xuất gì khác? Vì sao chọn phương pháp này?
- Kiểm chứng thế nào? Kết quả nói lên điều gì?

→ Phân công phải đảm bảo **mỗi người vừa code vừa viết**, không chia kiểu "1 người code, 1 người viết". Vì viết luận văn cần hiểu sâu thực nghiệm đã làm; và code phải có người viết tài liệu phản ánh đúng.

### 1.1. Trạng thái hiện tại của codebase

Theo [CODE_REVIEW_2026-04-26.md](CODE_REVIEW_2026-04-26.md):
- **3 bug CRITICAL** chặn training (ChainedScheduler, MQOT.forward duplicate, transport_map shape).
- **7 bug HIGH/MEDIUM** ảnh hưởng correctness.
- **Không có test suite, không có CI**.
- Các module mới đề cập trong [ABSTRACT.md](ABSTRACT.md) (SyncPreprocessor, Router-Gated Fusion, NRQE) **chưa có trong code** — cần implement.

### 1.2. Trạng thái thesis

Hiện có (chỉ là input tham khảo, KHÔNG phải khung chính thức):

- `thesis_proposal.tex` — Đề cương ban đầu, **chơi chơi**, sẽ KHÔNG dùng để viết luận văn cuối.
- [ABSTRACT.md](ABSTRACT.md) — Abstract VN/EN + 5 sections (Background, Pain Point, Idea, Expected Results, Conclusion). **Đây là input chính cho ý tưởng**.
- [ALGORITHM_SYSTEM_OVERVIEW.md](ALGORITHM_SYSTEM_OVERVIEW.md) — Tổng quan thuật toán.
- 4 file `Week 2 - *.docx` — Hướng dẫn cách viết luận văn (mục lục, mẫu chương 1, nguyên tắc dùng hình/bảng, phân biệt LV vs technical report).

Cần viết hoàn toàn mới:

- 6 chương đầy đủ, **80-100 trang**.
- Bảng kết quả thực nghiệm.
- Phân tích lỗi, ablation study.

### 1.3. Phân bổ trang dự kiến (80-100 trang)

| Phần | Số trang | Tỷ lệ |
|---|---|---|
| Lời cam đoan, cảm ơn, mục lục, danh mục | 6-8 | ~8% |
| Tóm tắt VN + EN | 2-3 | ~3% |
| Chương 1: Giới thiệu | 8-10 | ~10% |
| Chương 2: Cơ sở lý thuyết & Related work | 18-22 | ~22% |
| Chương 3: Phương pháp đề xuất | 18-22 | ~22% |
| Chương 4: Thực nghiệm | 12-15 | ~15% |
| Chương 5: Kết quả & Thảo luận | 12-15 | ~15% |
| Chương 6: Kết luận | 3-5 | ~4% |
| Tài liệu tham khảo | 3-5 | ~4% |
| Phụ lục | 2-5 | ~3% |
| **Tổng** | **84-110** | **100%** |

---

## 2. Nguyên tắc phối hợp

### 2.1. Quy tắc vàng (đã nén thời gian)

| Nguyên tắc | Mô tả |
|---|---|
| **Mỗi người vừa code vừa viết** | Người chạy thực nghiệm A → viết phần phương pháp/kết quả của thành phần A |
| **Pair review nhanh** | PR ≤ 4 giờ phải có người còn lại review (không block lâu) |
| **Daily sync 15 phút (sáng + tối)** | Sáng: plan trong ngày. Tối: kết quả + blocker |
| **Branch riêng** | A và B làm trên branch riêng, merge vào `main` qua PR |
| **Backup checkpoint** | Mỗi best epoch → push lên Modal Volume + log WandB |
| **Viết thesis song song với code** | KHÔNG để dồn cuối tháng. Code xong section → viết section đó luôn |

### 2.2. Công cụ phối hợp

| Công cụ | Mục đích |
|---|---|
| Git + GitHub | Version control code + thesis (LaTeX) |
| Modal | Compute GPU (training, preprocessing) |
| WandB | Log metrics, alignment maps, hyperparameters |
| Overleaf hoặc Git LaTeX | Viết thesis collaboration |
| Notion/Trello | Daily task tracking |

---

## 3. Phân vai tổng thể

### Người A — "Algorithm & Modeling Lead"

**Trách nhiệm chính:**
- Sửa lỗi model + implement module mới (MQOT-v2, SyncPreprocessor)
- Chạy ablation study trên GRID
- Viết Chương 3 (Phương pháp) + 50% Chương 5 (Ablation)
- Phụ trách phần "thuật toán cốt lõi" của luận văn

**Skill yêu cầu:** PyTorch, Optimal Transport, attention mechanism, debugging gradient flow.

**Trang dự kiến viết:** ~30-35 trang (Chương 3 đầy đủ + 6-8 trang Chương 5).

### Người B — "Data, Pipeline & Evaluation Lead"

**Trách nhiệm chính:**
- Pipeline data (preprocessing, augmentation, manifest)
- Implement Router-Gated Fusion + NRQE
- Chạy main experiments trên ViCocktail (clean + noisy SNR)
- Viết Chương 1, 2, 4, 6 + 50% Chương 5 (Main results)
- Phụ trách evaluation framework + WandB dashboard

**Skill yêu cầu:** Modal cloud, audio/video processing, evaluation metrics (WER/CER/jiwer), data engineering.

**Trang dự kiến viết:** ~50-55 trang (Chương 1 + 2 + 4 + 6 + 6-8 trang Chương 5).

### Trách nhiệm chung

- Cả 2 cùng review code của nhau (theo `~/.claude/rules/`).
- Cả 2 cùng đóng góp Chương 5 (mỗi người viết phần kết quả thí nghiệm mình chạy).
- Cả 2 cùng đọc 4 file `Week 2 - *.docx` để tránh viết theo kiểu technical report.

---

## 4. Mục lục luận văn (6 chương, 80-100 trang)

> Theo hướng dẫn trong `Week 2 - Huong Dan Muc Luc.docx`, dự án này thuộc lai DS + DE (có ML model + có pipeline cloud), nên dùng khung sau. **Số trang ước lượng cho từng section** ghi trong ngoặc.

```
LỜI CAM ĐOAN                                          (1 trang)
LỜI CẢM ƠN                                            (1 trang)
TÓM TẮT (Tiếng Việt + English)                        (2-3 trang)  ← từ ABSTRACT.md
MỤC LỤC                                               (2-3 trang)
DANH MỤC HÌNH                                         (1 trang)
DANH MỤC BẢNG                                         (1 trang)
DANH MỤC TỪ VIẾT TẮT                                  (1 trang)

CHƯƠNG 1. GIỚI THIỆU                                  (8-10 trang)
   1.1. Bối cảnh nghiên cứu                           (1.5 trang)
   1.2. Phát biểu vấn đề                              (1 trang)
   1.3. Pain point của bài toán                       (2 trang)
   1.4. Mục tiêu nghiên cứu                           (0.5 trang)
   1.5. Câu hỏi nghiên cứu (5 RQs)                    (1 trang)
   1.6. Phạm vi nghiên cứu                            (0.5 trang)
   1.7. Đóng góp của đề tài                           (1 trang)
   1.8. Cấu trúc luận văn                             (0.5 trang)

CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ NGHIÊN CỨU LIÊN QUAN     (18-22 trang)
   2.1. Tổng quan AVSR                                (2 trang)
   2.2. Speech Recognition cơ bản (CTC, AR, hybrid)   (3 trang)
   2.3. Visual Speech Recognition (lipreading)        (2 trang)
   2.4. Multimodal Fusion strategies                  (4 trang)
       2.4.1. Early/Late/Hybrid Fusion
       2.4.2. Attention-based Fusion
       2.4.3. Quality-aware / Reliability-aware Fusion
   2.5. Optimal Transport trong deep learning         (3 trang)
       2.5.1. Sinkhorn algorithm
       2.5.2. Unbalanced OT
       2.5.3. OT trong sequence alignment
   2.6. Conformer architecture                        (1.5 trang)
   2.7. Whisper encoder                               (1 trang)
   2.8. Các nghiên cứu liên quan                      (3-4 trang)
       2.8.1. AV-HuBERT, Auto-AVSR (English SOTA)
       2.8.2. ViCocktail (Vietnamese baseline)
       2.8.3. PROGOT, AlignMamba (OT-based alignment)
       2.8.4. CoBRA, GILA (cross-modal interaction)
   2.9. Khoảng trống nghiên cứu                       (1 trang)

CHƯƠNG 3. PHÂN TÍCH BÀI TOÁN VÀ ĐỀ XUẤT GIẢI PHÁP     (18-22 trang)
   3.1. Phân tích pain point chi tiết                 (2 trang)
   3.2. Kiến trúc tổng thể MOTA-v2                    (2 trang)
   3.3. Các thành phần đề xuất                        (10-12 trang)
       3.3.1. SyncPreprocessor                        (2 trang)
       3.3.2. QualityGate v2                          (2 trang)
       3.3.3. Router-Gated Fusion (RGF) + NRQE        (3 trang)
       3.3.4. MQOT-v2                                 (3 trang)
       3.3.5. Conformer Encoder                       (1 trang)
       3.3.6. Hybrid Decoder (CTC+AR)                 (1 trang)
   3.4. Hybrid Loss Function                          (2 trang)
   3.5. Lý do lựa chọn (justify)                      (2-3 trang)

CHƯƠNG 4. XÂY DỰNG HỆ THỐNG VÀ THỰC NGHIỆM            (12-15 trang)
   4.1. Môi trường thực nghiệm                        (1 trang)
   4.2. Datasets                                      (2 trang)
   4.3. Preprocessing Pipeline                        (3 trang)
   4.4. Augmentation Strategy                         (1.5 trang)
   4.5. Training Setup                                (2.5 trang)
   4.6. Evaluation Protocol                           (2 trang)

CHƯƠNG 5. KẾT QUẢ VÀ THẢO LUẬN                        (12-15 trang)
   5.1. Kết quả chính trên ViCocktail                 (3-4 trang)
   5.2. Ablation Study (trên GRID)                    (3-4 trang)
   5.3. Phân tích lỗi                                 (2 trang)
   5.4. Visualization                                 (2 trang)
   5.5. Hiệu năng & chi phí                           (1.5 trang)
   5.6. Hạn chế của giải pháp                         (1 trang)

CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN                (3-5 trang)
   6.1. Kết luận                                      (1.5 trang)
   6.2. Đóng góp chính                                (1 trang)
   6.3. Hướng phát triển                              (1.5 trang)

TÀI LIỆU THAM KHẢO                                    (3-5 trang)  ← ≥ 30 refs
PHỤ LỤC                                               (2-5 trang)
   A. Hyperparameter tables
   B. Vietnamese tone confusion matrix
   C. Code listings (selected)
   D. WandB run links
```

**TỔNG: 84-110 trang** (đạt mục tiêu 80-100).

---

## 5. Phân công chi tiết theo chương

### CHƯƠNG 1. Giới thiệu (8-10 trang) — **Người B (Lead)**

| Section | Trang | Người chính | Nội dung chính | Nguồn input |
|---|---|---|---|---|
| 1.1 Bối cảnh | 1.5 | B | AVSR + low-resource Vietnamese | ABSTRACT §1 |
| 1.2 Phát biểu vấn đề | 1 | B | 3 pain points | ABSTRACT §2 |
| 1.3 Pain points chi tiết | 2 | B | Model/Data/System | ABSTRACT §2.1, 2.2, 2.3 |
| 1.4 Mục tiêu | 0.5 | B | 3 objectives | ABSTRACT §3 |
| 1.5 Câu hỏi nghiên cứu | 1 | A + B | 5 RQs (cùng brainstorm) | mới |
| 1.6 Phạm vi | 0.5 | B | datasets, decoder level | ABSTRACT §4 |
| 1.7 Đóng góp | 1 | A + B | 3 contributions | ABSTRACT §3 + Code |
| 1.8 Cấu trúc luận văn | 0.5 | B | Từ section 1 → 6 | Từ TOC |

**Câu hỏi nghiên cứu đề xuất (RQs):**
- RQ1: Cross-attention alignment có tốt hơn linear interpolation cho A/V misalignment trong AVSR tiếng Việt không?
- RQ2: Unbalanced Sinkhorn OT có cải thiện alignment khi T_a ≠ T_v so với balanced OT không?
- RQ3: Quality-aware fusion có giảm hiện tượng double degradation không?
- RQ4: Hybrid CTC+AR decoder có lợi gì cho ngôn ngữ tonal (tiếng Việt 6 thanh) so với CTC-only?
- RQ5: Mô hình MOTA-v2 có đạt WER mục tiêu (~28-30%) trên ViCocktail noisy với VRAM ≤ 2GiB?

---

### CHƯƠNG 2. Cơ sở lý thuyết (18-22 trang) — **Người B (Lead)**

| Section | Trang | Người chính | Workload | Note |
|---|---|---|---|---|
| 2.1 Tổng quan AVSR | 2 | B | 0.5 ngày | survey gọn |
| 2.2 Speech recognition (CTC, AR) | 3 | B | 1 ngày | có công thức CTC |
| 2.3 Visual speech recognition | 2 | B | 0.5 ngày | ResNet, V-TCN |
| 2.4 Multimodal fusion | 4 | A | 1.5 ngày | A am hiểu fusion |
| 2.5 Optimal Transport | 3 | A | 1.5 ngày | A làm MQOT, viết Sinkhorn |
| 2.6 Conformer | 1.5 | B | 0.5 ngày | survey |
| 2.7 Whisper | 1 | B | 0.5 ngày | survey |
| 2.8 Related work | 3-4 | A + B | 1.5 ngày | mỗi người ½ |
| 2.9 Research gap | 1 | A + B | 0.5 ngày | brainstorm chung |

**Lưu ý từ `Week 2 - Nguyên tắc tổng quát`:** Mỗi paper được nhắc tới phải trả lời: **"Nó liên quan gì đến hướng đi của ta?"**

---

### CHƯƠNG 3. Phương pháp (18-22 trang) — **Người A (Lead)**

| Section | Trang | Người chính | Module | Liên kết code |
|---|---|---|---|---|
| 3.1 Phân tích pain point | 2 | A + B | tổng hợp | ABSTRACT §2 |
| 3.2 Kiến trúc tổng thể | 2 | A | sơ đồ pipeline | [mota.py](src/models/mota.py) |
| 3.3.1 SyncPreprocessor | 2 | B | implement + viết | **MỚI** |
| 3.3.2 QualityGate v2 | 2 | A | viết + cross-attention | [quality_gate.py](src/models/fusion/quality_gate.py) |
| 3.3.3 RGF + NRQE | 3 | B | implement + viết | **MỚI** |
| 3.3.4 MQOT-v2 | 3 | A | viết + Sinkhorn | [mqot.py](src/models/fusion/mqot.py) |
| 3.3.5 Conformer | 1 | A | viết block | [conformer.py](src/models/layers/conformer.py) |
| 3.3.6 Hybrid Decoder | 1 | B | viết CTC+AR | [decoders.py](src/models/layers/decoders.py) |
| 3.4 Loss function | 2 | A | hybrid loss | [losses.py](src/training/losses.py) |
| 3.5 Justify lựa chọn | 2-3 | A + B | mỗi quyết định 1 đoạn | mới |

**Yêu cầu hình/bảng (Chương 3 cần ≥ 4 hình + 2 bảng + 2 thuật toán):**
- Hình 3.1: Sơ đồ kiến trúc tổng thể MOTA-v2
- Hình 3.2: Chi tiết QualityGate v2
- Hình 3.3: Sơ đồ MQOT-v2
- Hình 3.4: Hybrid Decoder structure
- Bảng 3.1: Hyperparameters các module
- Bảng 3.2: So sánh fusion strategies
- Thuật toán 3.1: Unbalanced Sinkhorn pseudocode
- Thuật toán 3.2: Forward pass MOTA-v2

---

### CHƯƠNG 4. Thực nghiệm (12-15 trang) — **Người B (Lead)**

| Section | Trang | Người chính | Workload |
|---|---|---|---|
| 4.1 Môi trường | 1 | B | 0.5 ngày |
| 4.2 Datasets | 2 | B | 0.5 ngày |
| 4.3 Preprocessing | 3 | B | 1 ngày |
| 4.4 Augmentation | 1.5 | A | 0.5 ngày |
| 4.5 Training setup | 2.5 | A + B | 1 ngày |
| 4.6 Evaluation protocol | 2 | B | 0.5 ngày |

---

### CHƯƠNG 5. Kết quả & Thảo luận (12-15 trang) — **A + B (chia đều)**

| Section | Trang | Người chính | Lý do |
|---|---|---|---|
| 5.1 Main results ViCocktail | 3-4 | B | B chạy main experiments |
| 5.2 Ablation study GRID | 3-4 | A | A chạy ablation |
| 5.3 Error analysis | 2 | A + B | A: tonal errors, B: occlusion errors |
| 5.4 Visualization | 2 | A | A có visualization tools |
| 5.5 Performance & cost | 1.5 | B | B đo throughput/VRAM |
| 5.6 Hạn chế | 1 | A + B | brainstorm chung |

**Bảng kết quả bắt buộc:**
- Bảng 5.1: WER/CER trên ViCocktail (clean + 4 noise levels)
- Bảng 5.2: So sánh với 3 baselines
- Bảng 5.3: Ablation từng thành phần
- Bảng 5.4: Sensitivity hyperparameters MQOT
- Bảng 5.5: Performance benchmark (VRAM, time/epoch, latency)

---

### CHƯƠNG 6. Kết luận (3-5 trang) — **Người B (Lead)**

| Section | Trang | Người chính | Note |
|---|---|---|---|
| 6.1 Kết luận | 1.5 | B | tóm tắt 3 contributions + main results |
| 6.2 Đóng góp chính | 1 | A + B | mỗi người viết phần mình làm |
| 6.3 Hướng phát triển | 1.5 | A + B | LLM (A), multi-lang (B), deployment (B) |

---

## 6. Phân công code & thực nghiệm

### 6.1. Sprint 0 — Setup (Ngày 1-2)

**Cả 2 cùng làm:**

- Đọc lại [CODE_REVIEW_2026-04-26.md](CODE_REVIEW_2026-04-26.md) + 4 file Week 2 docx.
- Setup dev environment + WandB project.
- Tạo Overleaf/Git LaTeX repo cho thesis.
- Tạo skeleton 6 chương `.tex` files với section headings.

### 6.2. Sprint 1 — Fix Critical Bugs (Ngày 3-5)

**Người A:**
| Task | File | Reference |
|---|---|---|
| Xoá `forward()` duplicate | [mqot.py:148-205](src/models/fusion/mqot.py#L148-L205) | Bug 3.1 |
| Fix shape mismatch transport_map | [losses.py:188-204](src/training/losses.py#L188-L204) | Bug 3.3 |
| Fix residual logic QualityGate | [quality_gate.py:179](src/models/fusion/quality_gate.py#L179) | Bug 3.4 |
| Add `sigmoid` cho `fine_align_gate` | [mota.py:114, 229](src/models/mota.py#L114) | Bug 4.10 |
| Tách `blank_id ≠ pad_id` | [base.yaml:13-14](configs/base.yaml#L13-L14) | Bug 3.5 |

**Người B:**
| Task | File | Reference |
|---|---|---|
| Fix `ChainedScheduler` | [trainer.py:95-111](src/training/trainer.py#L95-L111) | Bug 3.2 (CRITICAL) |
| Fix duplicate tokenizer init | [trainer.py:73, 135](src/training/trainer.py#L73) | Bug 4.2 |
| Migrate `torch.cuda.amp` → `torch.amp` | [trainer.py:122, 256](src/training/trainer.py#L122) | Bug 4.1 |
| Fix dummy data fallback | [base.py:133-142](src/data/datasets/base.py#L133-L142) | Bug 4.6 |
| Fix `count += 1` đếm 2 lần | [base.py:607, 653](src/data/preprocessors/base.py#L607-L653) | Bug 4.4 |

**Acceptance:** Phase 1 training chạy 1 epoch không crash trên Modal A10G, loss giảm.

---

### 6.3. Sprint 2 — Test suite tối thiểu + Implement modules mới (Ngày 6-12)

**Người A — MQOT-v2 + tests:**

- Unit test cho `MQOTLayer.sinkhorn_unbalanced` (row sum, gradient flow)
- Unit test cho `QualityGate`, `GuidedAttention`, `HybridLoss`
- Polish multi-head OT (num_heads ∈ {1, 4})

**Người B — Modules mới + pipeline tests:**

- `src/data/preprocessors/sync.py` — SyncPreprocessor (cross-correlation + cubic interp)
- `src/models/fusion/router_gate.py` — RGF (3 modes: audio-dominant / visual-dominant / fusion)
- `src/models/fusion/nrqe.py` — Noise-Robust Quality Estimator
- Unit test cho `CTCDecoder`, `MetricCalculator`, `Collator`
- GitHub Actions CI: lint + pytest

**Acceptance:** 4 modules integration test pass; coverage ≥ 50% (giảm từ 70% do thời gian gấp).

---

### 6.4. Sprint 3 — Main experiments (Ngày 13-20)

**Người B — Main runs trên ViCocktail (5 conditions: clean, SNR=10/5/0/-5dB):**
| Run | Config | Expected WER |
|---|---|---|
| Baseline 1 | Whisper audio-only | 13-15% (clean) |
| Baseline 2 | ViCocktail Conformer AVSR | 14-16% |
| MOTA-v2 Phase 1 | QualityGate only | ~30% |
| MOTA-v2 Phase 2 | + MQOT-v2 | ~28-30% |
| MOTA-v2 full | + Sync + RGF + NRQE | target |

**Người A — Ablation trên GRID (chạy song song với Sprint 3):**
| Ablation | Mô tả |
|---|---|
| A0 | Full model |
| A1 | -QualityGate (concat) |
| A2 | -MQOT (chỉ QualityGate) |
| A3 | -SyncPreproc |
| A4 | -RGF/NRQE |
| A5 | Balanced vs Unbalanced Sinkhorn |
| A6 | num_heads ∈ {1, 4} |
| A7 | epsilon ∈ {0.05, 0.15, 0.5} |

**Acceptance:** Có đủ data cho Bảng 5.1, 5.2, 5.3, 5.4.

---

### 6.5. Sprint 4 — Error analysis + Visualization (Ngày 21-23)

**Người A:**

- Confusion matrix tonal pairs Vietnamese (bà/bá/bả/bạ)
- Visualize transport map cho 10 sample
- Quality score timeline cho noisy samples

**Người B:**

- Failure case study: 20 samples worst WER
- Categorize errors: tonal / homophone / segmentation / occlusion
- Performance profiling (memory, throughput) trên A10G

---

## 7. Lịch trình 30 ngày

| Ngày | Sprint | Code (A + B) | Thesis (mỗi người ~3-5 trang/tuần) |
|---|---|---|---|
| 1-2 | S0: Setup | Setup env, WandB, repo. Đọc tài liệu | Skeleton .tex 6 chương |
| 3-5 | S1: Bugs | Fix 3 critical + 7 medium bugs | A: 3.2 Kiến trúc tổng thể (2tr); B: 1.1, 1.2 (2.5tr) |
| 6-8 | S2a | A: tests model; B: SyncPreproc | A: 3.3.4 MQOT-v2 (3tr); B: 1.3-1.5 (3.5tr) |
| 9-12 | S2b | A: integration tests; B: RGF + NRQE | A: 3.3.2 QG (2tr) + 3.4 Loss (2tr); B: 1.6-1.8 + 4.1-4.2 (3tr) |
| 13-16 | S3a | B: Main runs ViCocktail clean + SNR 10/5dB; A: Ablation A0-A4 | A: 2.4 Fusion (4tr) + 2.5 OT (3tr); B: 4.3 Preprocessing (3tr) |
| 17-20 | S3b | B: Main runs SNR 0/-5dB; A: Ablation A5-A7 | A: 3.3.5+3.3.6+3.5 (4tr); B: 2.1-2.3, 2.6, 2.7 (9.5tr) |
| 21-23 | S4: Analysis | Error analysis + visualization | A: 5.2 Ablation (4tr) + 5.4 Visualization (2tr); B: 5.1 Main (4tr) + 4.4-4.6 (6tr) |
| 24-26 | Polish-1 | Re-run failed experiments nếu cần | A: 2.5 OT polish + 2.8 Related work ½; B: 2.8 Related ½ + 2.9 (4tr) |
| 27-28 | Polish-2 | Final benchmarks + checkpoint | A: 5.3 Error ½ + 5.6 Hạn chế ½; B: 5.5 Performance + 6.1-6.3 (5tr) + Tóm tắt VN/EN |
| 29 | Review | Pair review code + thesis cross-check | Cả 2 review chéo full thesis |
| 30 | Submit | Buffer + bug fix cuối + finalize PDF | Format LaTeX + tài liệu tham khảo + nộp PDF |

### 7.1. Mốc deadline cứng

| Mốc | Deadline | Yêu cầu |
|---|---|---|
| M1: Code không crash | Ngày 5 | Phase 1 training 1 epoch OK |
| M2: Test cơ bản | Ngày 12 | Coverage ≥ 50%, CI xanh |
| M3: Module mới hoàn chỉnh | Ngày 12 | SyncPreproc + RGF + NRQE |
| M4: Số liệu chính | Ngày 20 | Bảng 5.1, 5.2, 5.3 đủ |
| M5: Thesis draft v1 | Ngày 26 | 6 chương đầy đủ ~80 trang |
| M6: Thesis final | Ngày 30 | PDF nộp |

---

## 8. Checklist hàng ngày / hàng tuần

### 8.1. Daily standup (15 phút sáng + 15 phút tối)

```
SÁNG:
[ ] Sprint task hôm nay là gì?
[ ] Có blocker từ hôm qua không?
[ ] Cần help/review từ đối tác không?

TỐI:
[ ] Hôm nay merged được PR nào?
[ ] Số trang thesis viết được?
[ ] Test pass / fail?
[ ] WandB run nào complete?
```

### 8.2. Trước khi commit code

```
[ ] black --check src/ tests/
[ ] isort --check src/ tests/
[ ] pytest tests/ (nếu file liên quan)
[ ] PR description rõ ràng
[ ] Người còn lại review trong 4 giờ
```

### 8.3. Trước khi commit thesis section

```
[ ] Có câu hỏi trung tâm cho section?
[ ] Có justify lý do (không liệt kê công nghệ)?
[ ] Có hình/bảng đi kèm với caption đúng + dẫn vào + giải thích?
[ ] Có baseline để so sánh?
[ ] Có thảo luận, không chỉ kết quả?
[ ] Đã đối chiếu rule trong `Week 2 - PHan Biet Luan Van và Technical Report.docx`?
[ ] Số trang đã đúng plan?
```

---

## 9. Rủi ro và phương án cắt giảm phạm vi

> Vì thời gian chỉ 30 ngày, **PHẢI có plan cắt giảm phạm vi** nếu chậm tiến độ.

### 9.1. Rủi ro chính

| Rủi ro | Xác suất | Tác động | Phương án dự phòng |
|---|---|---|---|
| Critical bugs phức tạp hơn dự kiến | Cao | Cao | Giảm scope module mới (xem 9.2) |
| Modal credit hết | Trung bình | Cao | Chỉ chạy ablation trên GRID nhỏ; ViCocktail chỉ 1-2 main runs |
| WER không đạt mục tiêu (>30%) | Cao | Trung bình | Báo cáo trung thực + phân tích lý do = contribution |
| Module mới (RGF/NRQE) không converge | Trung bình | Cao | Cắt RGF + NRQE, dồn focus vào MQOT-v2 |
| ViCocktail data download lỗi | Thấp | Cao | Cache trên Modal volume từ Sprint 0 |
| Thesis viết chậm | Cao | Cao | Cut Chapter 2 từ 22 xuống 18 trang; cut Phụ lục |
| Pair review chậm | Trung bình | Trung bình | Async review, PR ≤ 200 LOC |

### 9.2. Phương án cắt giảm phạm vi (nếu chậm)

**Option A — Chậm 5-7 ngày:**

- Bỏ RGF + NRQE, chỉ giữ MQOT-v2 + SyncPreprocessor.
- Ablation chỉ chạy 5 thay vì 8 (bỏ A5-A7 sensitivity).

**Option B — Chậm 10+ ngày:**

- Bỏ tất cả module mới ngoài MQOT-v2 (đã có sẵn).
- Chỉ chạy main experiments không ablation trên GRID.
- Chương 5 → 8-10 trang thay vì 12-15.

**Option C — Khẩn cấp (chỉ còn 5-7 ngày):**

- Chỉ fix critical bugs, không thêm module nào.
- Chạy 2 baselines + MOTA-v2 hiện tại trên 1 noise level (clean).
- Thesis 70-80 trang thay vì 80-100.
- Tập trung vào Chương 1, 2, 3 (đề cương + lý thuyết) — đã chiếm 50+ trang.

### 9.3. Tránh viết kiểu technical report

Theo `Week 2 - PHan Biet Luan Van và Technical Report.docx`:

| ❌ Không viết | ✅ Phải viết |
|---|---|
| "Hệ thống dùng PyTorch, Modal, WandB" | "PyTorch được chọn vì hỗ trợ research-friendly, Modal vì cost-efficient" |
| "Mô hình gồm 6 stages" | "Pipeline 6 stages giải quyết 3 pain points: ..." |
| "WER đạt 28.5%" | "WER 28.5%, giảm 4.2 điểm so với baseline; lợi ích chủ yếu ở SNR ≤ 0dB" |
| "Demo chạy được" | "Trên 100 mẫu test, WER trung bình 28.5%, std 3.2%; 60% lỗi là tonal confusion" |

---

## 10. Phụ lục — Danh sách deliverables (sau 30 ngày)

### 10.1. Code deliverables

- [ ] `src/models/fusion/mqot.py` — MQOT-v2 hoàn chỉnh
- [ ] `src/data/preprocessors/sync.py` — SyncPreprocessor (mới)
- [ ] `src/models/fusion/router_gate.py` — RGF (mới, optional)
- [ ] `src/models/fusion/nrqe.py` — NRQE (mới, optional)
- [ ] `tests/` — Coverage ≥ 50%
- [ ] `.github/workflows/ci.yml` — CI

### 10.2. Thesis deliverables

- [ ] `thesis/main.tex` — File chính
- [ ] 6 file `chapters/0X_*.tex`
- [ ] `thesis/figures/` — ≥ 12 hình
- [ ] `thesis/tables/` — ≥ 8 bảng
- [ ] `thesis/references.bib` — ≥ 30 refs
- [ ] **PDF cuối ≥ 80 trang, ≤ 100 trang**

### 10.3. Experiment deliverables

- [ ] WandB project link với ≥ 12 runs
- [ ] Best checkpoint Phase 1 + Phase 2
- [ ] Inference results JSONL trên test set
- [ ] Ablation table CSV
- [ ] Visualization PNG: transport maps, quality scores, gate weights

---

## End of Work Plan (30-day version, target 80-100 pages)

> Khi có thay đổi, cập nhật vào file này và commit với message `docs(plan): update work allocation`.
