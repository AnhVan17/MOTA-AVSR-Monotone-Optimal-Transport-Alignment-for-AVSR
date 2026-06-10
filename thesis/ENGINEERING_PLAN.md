# Engineering Plan — MOTA-v2 AVSR (Section 0 → C)

**Ngày lập:** 2026-06-09
**Trạng thái codebase:** Phase-1 + Phase-2 trial đã chạy thành công trên Modal (exit 0). 3 critical bug + HIGH/MEDIUM của CODE_REVIEW đã fix. mediapipe đã được thay hoàn toàn bằng face-alignment (SFD+FAN).
**Phạm vi tài liệu:** kế hoạch kỹ thuật chi tiết cho 4 khối công việc — **0 (Clean/Refactor/Legacy)**, **A (Main Experiments)**, **B (RGF + NRQE)**, **C (Visual Frontend E2E)**.

> Bổ trợ cho [THESIS_WORK_PLAN.md](THESIS_WORK_PLAN.md) (phân công + lịch 30 ngày). File này tập trung **kỹ thuật**: làm gì, sửa file nào, vì sao, tiêu chí nghiệm thu.

---

## Ràng buộc xuyên suốt (đọc trước)

| Ràng buộc                                                                                                    | Hệ quả lên plan                                                                |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **Vietnamese-first, low-resource** (~269h ViCocktail)                                                        | Ưu tiên prior mạnh + pretrained; tránh module data-hungry                      |
| **VRAM mục tiêu ~2 GiB**                                                                                     | Batch nhỏ, gradient checkpointing khi cần; cẩn trọng với raw-frame             |
| **30 ngày, cost-sensitive (Modal credit)**                                                                   | Mỗi khối phải ra deliverable; chạy clean trước, noisy sau                      |
| **Đóng góp LÕI = OT alignment (MQOT) + reliability-aware fusion (QualityGate/RGF/NRQE)**                     | Visual frontend (C) là **tangential** → chỉ làm như ablation nếu còn thời gian |
| **Rule cá nhân** (immutability, file nhỏ 200–400 dòng, TDD ≥ coverage mục tiêu, không hardcode magic number) | Mọi hằng số → config; viết test trước; tách module nhỏ                         |

**Thứ tự ưu tiên đề xuất:** `0 → A → B → (C nếu còn thời gian)`.
**Lý do:** (0) rẻ & gỡ nợ kỹ thuật trước khi xây thêm; (A) ra **số cho bảng kết quả** — thứ luận văn bắt buộc; (B) hoàn thiện **đóng góp đã hứa trong ABSTRACT**; (C) tangential.

---

# Section 0 — Clean / Refactor / Remove Legacy

**Mục tiêu:** xoá nợ kỹ thuật + dấu vết mediapipe + dựng nền test/CI trước khi xây module mới. Rẻ, nhanh, giảm rủi ro cho A/B/C.

### 0.1. Đổi tên di sản "FaceMesh" → cropper đúng nghĩa

Ruột đã là face-alignment (SFD+FAN) nhưng tên vẫn "FaceMesh" → gây hiểu nhầm.

| Hiện tại                                                    | Đổi thành                                    | File                                                                                   |
| ----------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- |
| `FaceMeshPreprocessor`                                      | `MouthCropper`                               | [facemesh.py](../src/data/preprocessors/facemesh.py) → đổi tên file `mouth_cropper.py` |
| `FaceMeshConfig`                                            | `CropConfig`                                 | nt                                                                                     |
| `CPU_FACEMESH_IMAGE`                                        | `FACE_CROP_IMAGE`                            | [modal_image.py](../src/infra/modal_image.py)                                          |
| `prep_facemesh_cpu.py`, `APP_NAME="avsr-prep-facemesh-cpu"` | `prep_face_crop.py`, `"avsr-prep-face-crop"` | scripts/modal/data_prep                                                                |

> ⚠️ **Phối hợp nhóm:** các tên này có thể bị tham chiếu ở branch của thành viên khác. Trước khi rename: `git grep` toàn bộ + thông báo nhóm. Cân nhắc giữ alias tạm (`FaceMeshPreprocessor = MouthCropper`) 1 sprint để khỏi vỡ branch đang mở.

### 0.2. Sửa comment/print lỗi thời (nhắc FaceMesh/MediaPipe nhưng đã là face-alignment)

- [prep_vicocktail.py:52,209-211](../scripts/modal/data_prep/prep_vicocktail.py#L209)
- [prep_features_gpu.py:102,129](../scripts/modal/data_prep/prep_features_gpu.py#L102)
- [preprocess.py:10-15](../scripts/modal/data_prep/preprocess.py#L10)
- [debug_data.py:28](../scripts/modal/utils/debug_data.py#L28)
- [grid.py:5](../src/data/datasets/grid.py#L5) — "Raw video loading (MediaPipe…)" → sửa thành face-alignment

### 0.3. Gỡ apt package EGL chết (mediapipe cần, face-alignment không)

- [preprocess.py:31,55](../scripts/modal/data_prep/preprocess.py#L31): bỏ `libegl1-mesa`, `libgles2-mesa`.
- Trước khi gỡ: xác nhận `preprocess.py` (monolith cũ) còn dùng không — nếu không, **deprecate/xoá hẳn** (nó là nguồn gốc EGL conflict).
- GIỮ `libgl1-mesa-glx` (OpenCV cần, không phải EGL).

### 0.4. 🔴 Sửa mâu thuẫn tài liệu thesis (ưu tiên cao — rủi ro bảo vệ)

- [ABSTRACT.md §2.3 dòng 43](ABSTRACT.md): hiện mô tả _"MediaPipe FaceMesh CPU-only…"_ như **pain point hiện tại** → mâu thuẫn với code. Sửa thành: _"pipeline trước đây dùng MediaPipe (CPU/EGL conflict, ~30ph/giờ); đề tài đã chuyển sang face-alignment GPU-native"_ → biến thành **đóng góp hệ thống** (System contribution §2.3) thay vì lỗ hổng chưa giải.
- [CODE_REVIEW_2026-04-26.md:80](CODE_REVIEW_2026-04-26.md) đã đúng — dùng làm nguồn câu chữ.
- Sửa luôn 2 mục lặp `## 5) Các module chính` trong [ALGORITHM_SYSTEM_OVERVIEW.md](ALGORITHM_SYSTEM_OVERVIEW.md) (lines ~185, ~218 trùng).

### 0.5. Quét dead-code / trùng lặp

- Chạy `ruff` + `vulture src/` để tìm import/biến/hàm chết.
- Soát `forward()` trùng, hàm legacy 2-step (`extract_features_shard` trong prep_vicocktail vs prep_features_gpu — đang chú thích "legacy").
- Mục tiêu: high cohesion, file ≤ 400 dòng.

### 0.6. Dựng nền Test + CI (đáp ứng M2 của THESIS_WORK_PLAN — hiện CHƯA có `tests/`)

- `tests/` skeleton + `pytest.ini` (markers unit/integration).
- Unit test "rẻ mà giá trị" trước: `create_loss`/HybridLoss (mask, special_ids), `get_device`/`make_grad_scaler`, collate (drop None), tokenizer special ids.
- `.github/workflows/ci.yml`: ruff + pytest (CPU-only, không cần GPU).
- Đây là **nền cho TDD của B & C**.

**Nghiệm thu Section 0:**

- [ ] `git grep -i facemesh` chỉ còn ở chỗ có chủ đích (hoặc alias) ; không còn `mediapipe`/EGL chết.
- [ ] ABSTRACT §2.3 khớp thực tế code.
- [ ] `pytest` xanh ở local + CI; coverage báo cáo được.
- [ ] Commit nhỏ theo nhóm: `refactor(preproc): rename FaceMesh→MouthCropper`, `docs(thesis): reconcile preprocessing section`, `test: add CI + core unit tests`.

---

# Section A — Main Experiments (ƯU TIÊN 1: ra số cho luận văn)

**Mục tiêu:** sinh dữ liệu cho **Bảng 5.1 (WER/CER clean+noisy)** và **Bảng 5.2 (so baseline)**. Đây là thứ luận văn **bắt buộc** phải có.

### A.1. Chặn đường: giới hạn file của Modal Volume

- Modal Volume giới hạn **262,144 files/thư mục**. ViCocktail full sinh ra hàng triệu file `.pt` rời (mỗi sample 1 file feature) → **vượt giới hạn** (đây chính là lỗi bạn từng gặp).
- Nguồn: Modal docs (Volume limits) — cần verify lại số chính xác lúc làm.

### A.2. Giải pháp: WebDataset tar shards (gom file)

- Thay vì N file `.pt` rời → ghi thành **tar shards** (mỗi shard ~1000–5000 sample) bằng `webdataset.ShardWriter`.
- File count: `N samples → N/shardsize tar` (giảm ~1000×) → dưới giới hạn.
- Train đọc bằng `webdataset` IterableDataset (streaming, không cần random-access tất cả file).
- **Tái sử dụng cho Section C** (raw-frame cũng đóng shard tương tự).

### A.3. Thay đổi code

| File                                              | Thay đổi                                                                                                                          |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `src/data/preprocessors/base.py`                  | output `.pt` rời → `ShardWriter("...-%06d.tar")`; mỗi sample = 1 sample webdataset (`__key__`, `audio.pth`, `visual.pth`, `text`) |
| `src/data/datasets/` (mới) `webdataset_loader.py` | IterableDataset đọc shards; decode `.pth` → tensor; áp augmentation                                                               |
| `src/data/loader.py`                              | nhánh chọn: manifest-jsonl (cũ) vs webdataset shards (mới), qua config `data.format: webdataset`                                  |
| configs                                           | thêm `data.shards: "/mnt/.../shard-{000000..0000NN}.tar"`, `data.shard_size`                                                      |

> Giữ **đường cũ (.pt rời)** cho smoke-test local nhỏ; webdataset cho full-scale. Không hardcode — chọn qua config.

### A.4. Ma trận thí nghiệm (map thẳng vào bảng Chương 5)

| Run          | Config                                       | Điều kiện              | Bảng |
| ------------ | -------------------------------------------- | ---------------------- | ---- |
| Baseline-1   | Whisper audio-only                           | clean + SNR{10,5,0,-5} | 5.2  |
| Baseline-2   | ViCocktail Conformer AVSR (nếu tái lập được) | clean                  | 5.2  |
| MOTA Phase-1 | QualityGate only (`use_mqot=false`)          | clean + SNR{…}         | 5.1  |
| MOTA Phase-2 | + MQOT (`use_mqot=true`)                     | clean + SNR{…}         | 5.1  |

- Thứ tự chạy: **clean trước** (chốt pipeline + có số sớm) → **noisy** (SNR giảm dần).
- WandB log: loss, WER/CER, transport map, gate weights, VRAM, time/epoch.

### A.5. Bền vững vận hành

- `volume.commit()` sau mỗi best epoch (đã có, cần verify giữ checkpoint).
- Resume từ checkpoint (`pretrained_path`/`resume`).
- Checkpoint dir tách theo run (không đè team).

### A.6. Ước lượng chi phí (điền số thật lúc chạy)

- Bảng nhỏ: GPU (A10G/A100) × giờ/epoch × số epoch × số run → tổng credit. Quyết định cắt SNR levels nếu vượt budget (theo §9.2 THESIS_WORK_PLAN).

**Nghiệm thu Section A:**

- [ ] Full ViCocktail preprocess xong **không vượt file-limit** (đóng shard OK).
- [ ] Phase-1 + Phase-2 chạy hết clean, có WER/CER.
- [ ] Đủ số cho Bảng 5.1 (≥ clean + 2 mức nhiễu) và 5.2 (≥ 1 baseline).
- [ ] Checkpoint + WandB link lưu lại.

---

# Section B — Router-Gated Fusion (RGF) + NRQE (ĐÓNG GÓP LÕI #3, hiện THIẾU)

**Mục tiêu:** hiện thực đóng góp cốt lõi #3 trong ABSTRACT. Tấn công pain point **"double degradation"** (§2.1): khi cả audio & visual đều suy giảm, fusion mù làm hỏng thêm.

### B.1. Cơ sở học thuật (đã research — bám sát, không bịa)

- **RGF**: _"Improving Noise Robust AVSR via Router-Gated Cross-Modal Feature Fusion"_ (arXiv 2508.18734, 2025) — router down-weight audio token không tin cậy theo **token-level acoustic corruption score**, gated cross-attention; **giảm WER 16.5–42.7%** so AV-HuBERT. → reference trực tiếp cho RGF.
- **MoHAVE** (2502.10447): two-layer routing (inter-/intra-modal experts).
- **Robust AVSR with MoE** (2409.12370): router chọn sparse experts theo token.
- **NRQE / uncertainty**: AUAF (MC-Dropout + cross-modal cosine consistency), QMF (Quality-aware Multimodal Fusion), survey _Multimodal Fusion on Low-quality Data_ (2404.18947), _Provable Dynamic Fusion_ (2306.02050).

### B.2. NRQE — `src/models/fusion/nrqe.py` (mới)

**Vai trò:** ước lượng độ tin cậy **theo chunk** cho từng modality.

- **Input:** `audio_feat [B,Ta,d]`, `visual_feat [B,Tv,d]` (sau projection).
- **Output:** `q_a, q_v ∈ [0,1]` theo chunk (per-chunk reliability).
- **Cơ chế (2 thành phần, theo AUAF):**
  1. _Uncertainty_: head dự đoán log-variance (hoặc MC-dropout) cho mỗi modality → độ bất định cao = tin cậy thấp.
  2. _Cross-modal consistency_: cosine similarity giữa audio/visual đã chiếu chung không gian → nhất quán cao = cả hai đáng tin.
- **Chunking:** gộp T thành chunk kích thước `nrqe.chunk_size` (config). Không hardcode.

### B.3. RGF — `src/models/fusion/router_gate.py` (mới)

**Vai trò:** router theo chunk chọn 1 trong 3 chế độ.

- **Input:** `audio_feat`, `visual_feat`, `q_a`, `q_v` (từ NRQE).
- **Router:** MLP nhỏ → logits 3 chế độ → **softmax (soft routing)** `w = [w_audio, w_visual, w_fusion]` per chunk.
  - _audio-dominant_: dùng audio_feat
  - _visual-dominant_: dùng visual_feat
  - _fusion_: dùng QualityGate/cross-attn fusion
- **Output:** `out = w_a·audio + w_v·visual + w_f·fusion` (kết hợp mềm; tránh hard-argmax lúc train để gradient mượt).
- **Regularization:** entropy reg trên `w` (tránh router sụp về 1 chế độ) — trọng số qua config.

### B.4. Tích hợp vào `src/models/mota.py`

- Vị trí: **Stage 1.5**, ngay sau QualityGate (coarse) và trước MQOT (fine). RGF nhận output QualityGate làm nhánh "fusion".
- **Cờ config `use_rgf`** (giống `use_mqot`): mặc định `false` → **không phá** Phase-1/Phase-2 hiện có.
- Additive: chỉ khởi tạo NRQE/RGF khi `use_rgf=true`.

### B.5. Loss

- Tận dụng `quality_loss_weight` đã có trong [phase2_mqot.yaml](../configs/phase2_mqot.yaml).
- Aux loss tuỳ chọn: khuyến khích `q_a` thấp khi audio nhiễu (nếu có SNR label) — hoặc self-supervised qua consistency. Mặc định tắt; bật qua config.

### B.6. TDD (viết test trước)

- `q_a,q_v ∈ [0,1]`; router weights **sum=1**, **không NaN**.
- Gradient chảy tới NRQE + router.
- Degenerate test: nhét audio = noise → router dịch weight sang visual (kiểm chứng hành vi).
- Shape: `[B,T,d]` giữ nguyên qua RGF.

### B.7. Ablation (cho Bảng 5.3)

- A0 full; **A4: −RGF/NRQE** (so với full) — chứng minh đóng góp.

**Rủi ro & giảm thiểu:** router khó hội tụ → (1) warm-start từ checkpoint Phase-2, (2) soft routing + entropy reg, (3) train router sau khi backbone ổn định. Nếu không converge → **cắt theo §9.2 Option A** (giữ MQOT, bỏ RGF) — và **báo cáo trung thực = vẫn là kết quả**.

**Nghiệm thu Section B:**

- [ ] `nrqe.py` + `router_gate.py` + unit test pass.
- [ ] `use_rgf=true` chạy 1 epoch trial trên Modal không crash, loss hữu hạn.
- [ ] Có số ablation A4.

---

# Section C — Visual Frontend End-to-End (Option 2, TANGENTIAL — chỉ làm nếu còn thời gian)

**Mục tiêu:** vá điểm yếu thật của nhánh visual: ResNet18 hiện **ImageNet + 2D + đóng băng + per-frame** → không bắt **chuyển động môi**, không thích nghi data Việt. Nâng thành frontend spatiotemporal chuẩn SOTA, train E2E → đẩy WER + 1 dòng ablation.

> ⚠️ Gate quyết định: **chỉ bắt đầu C sau khi A & B xong**. Nếu trễ → bỏ, không ảnh hưởng đóng góp lõi.

### C.1. Cơ sở (đã research — spec chuẩn Auto-AVSR)

- Frontend lipreading chuẩn = **Conv3D stem (kernel 5×7×7, stride 1×2×2, 64 filters) → BN → ReLU → ResNet18**, train E2E. Nguồn: [auto_avsr](https://github.com/mpc001/auto_avsr), Stafylakis–Tzimiropoulos 2017.
- ResNet18 ImageNet 2D đóng băng = baseline yếu cho lip motion.

### C.2. Thay đổi code

| File                                         | Thay đổi                                                                                                      |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [mota.py:59-70](../src/models/mota.py#L59)   | thêm `Conv3DStem` (5×7×7) trước ResNet18; `requires_grad_(True)` khi finetune; cờ `train_visual_backbone`     |
| `src/models/layers/visual_frontend.py` (mới) | đóng gói Conv3D stem + ResNet18 thành module nhỏ                                                              |
| dataloader                                   | cần **raw cropped frames 88×88** (không phải feature .pt) → tái dùng WebDataset shards (Section A) chứa frame |
| configs                                      | `model.use_backbones: true`, `model.train_visual_backbone: true`                                              |

### C.3. Chiến lược 2 pha (cho ablation)

1. **Baseline**: frozen ImageNet feature (hiện tại) → WER_0.
2. **E2E**: Conv3D stem + unfreeze ResNet18, finetune → WER_1.

- Ablation row "frozen vs E2E frontend" (Bảng 5.3).

### C.4. Cảnh báo VRAM/chi phí

- Raw-frame nặng hơn feature → batch nhỏ + **gradient checkpointing**; cân với mục tiêu ~2GiB.
- Pretrain Conv3D+ResNet trên **GRID (tiếng Anh)** rồi finetune VN (giảm data-hungry) — khớp plan dùng GRID cho ablation.

### C.5. TDD

- Shape `[B,T,C,88,88] → [B,T,512]`.
- Conv3D stem stride đúng (T giữ, H/W giảm 2×).
- Gradient chảy khi `train_visual_backbone=true`.

**Nghiệm thu Section C:**

- [ ] `visual_frontend.py` + test pass.
- [ ] Trial E2E 1 epoch không crash; VRAM trong ngưỡng.
- [ ] Số ablation frozen-vs-E2E.

---

## Phụ thuộc & trình tự

```
Section 0 (1-2 ngày, rẻ)
   └─> Section A  (WebDataset shards) ──┐ (shards tái dùng cho C)
   └─> Section B  (RGF/NRQE, độc lập)   │
                                        └─> Section C (optional, sau A&B)
```

- **B độc lập với A**: RGF/NRQE chạy được trên feature precomputed → có thể làm song song với A.
- **C phụ thuộc A**: cần raw-frame shards.
- Map milestone THESIS_WORK_PLAN: 0+A → M1/M4 (số liệu); B → M3 (module mới); C → ablation bổ sung.

## Rủi ro & cắt scope (theo THESIS_WORK_PLAN §9.2)

- Trễ 5-7 ngày → bỏ C, giữ A+B.
- Trễ 10+ ngày → bỏ C + rút B còn RGF (bỏ NRQE riêng, gộp reliability vào QualityGate).
- Khẩn cấp → chỉ A (số main) + báo cáo trung thực.

## Sources

- Router-Gated AVSR: https://arxiv.org/abs/2508.18734
- MoHAVE: https://arxiv.org/pdf/2502.10447 · Robust AVSR MoE: https://arxiv.org/pdf/2409.12370
- Uncertainty/Quality fusion: https://arxiv.org/pdf/2404.18947 · https://arxiv.org/pdf/2306.02050
- 3D-conv frontend: https://github.com/mpc001/auto_avsr · https://arxiv.org/pdf/1703.04105
- Auto-AVSR pipeline: https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages

---

_Cập nhật khi có thay đổi; commit `docs(plan): update engineering plan`._
