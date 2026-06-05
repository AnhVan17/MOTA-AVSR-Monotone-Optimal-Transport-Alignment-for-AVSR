# Code Review: MOTA AVSR Codebase

**Ngày review:** 2026-04-26
**Reviewer:** Claude Opus 4.7
**Branch:** main
**Phạm vi:** Toàn bộ source code (`src/`, `scripts/`, `configs/`, `docs/`)

---

## Mục Lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Điểm mạnh](#2-điểm-mạnh)
3. [Bug nghiêm trọng (Critical)](#3-bug-nghiêm-trọng-critical)
4. [Bug trung bình (Medium)](#4-bug-trung-bình-medium)
5. [Vấn đề thiết kế / phong cách](#5-vấn-đề-thiết-kế--phong-cách)
6. [Tuân thủ coding rules](#6-tuân-thủ-coding-rules)
7. [Khuyến nghị ưu tiên](#7-khuyến-nghị-ưu-tiên)
8. [Tóm tắt](#8-tóm-tắt)

---

## 1. Tổng quan dự án

Dự án xây dựng hệ thống **AVSR (Audio-Visual Speech Recognition)** cho tiếng Việt, kiến trúc 2 phase:

- **Backbone**: Whisper-small (audio, 768d) + ResNet18 (visual, 512d)
- **Fusion**:
  - Phase 1: QualityGate v2 (cross-attention + quality scoring)
  - Phase 2: + MQOT (Multi-modal Quality-aware Optimal Transport)
- **Encoder/Decoder**: Conformer + Hybrid CTC+AR
- **Training infrastructure**: Modal cloud (A10G/A100)
- **Datasets**: GRID, ViCocktail (WebDataset format)
- **Tokenizer**: Whisper Vietnamese (vocab_size 51865)

### Cấu trúc thư mục

```
├── configs/                  # YAML config với inheritance (defaults: [base])
│   ├── base.yaml
│   ├── phase1_base.yaml
│   └── phase2_mqot.yaml
├── docs/                     # Algorithm overview, MODAL guide, PR reviews
├── guides/                   # ADDING_NEW_DATASET.md
├── scripts/
│   ├── data_prep/            # Modal pipelines: download, crop, extract features
│   ├── training/             # train_phase1, train_phase2, lr_finder
│   ├── inference/
│   └── utils/                # Volume audit, vocab verify, debug tools
└── src/
    ├── data/
    │   ├── augmentations.py          # SpecAugment-like
    │   ├── collate.py                # Collator (padding)
    │   ├── loader.py                 # DataLoader factory
    │   ├── datasets/                 # base, grid, vicocktail
    │   ├── preprocessors/            # base, cropper, facemesh, grid, vicocktail
    │   └── tokenizers/whisper.py
    ├── models/
    │   ├── mota.py                   # Top-level MOTA model
    │   ├── fusion/                   # quality_gate, mqot
    │   └── layers/                   # adapters, conformer, decoders
    ├── training/                     # trainer, losses
    ├── evaluation/                   # decoding, engine, metrics, visualization
    └── utils/                        # common, config_utils, logging_utils, text_cleaning, wandb_logger
```

---

## 2. Điểm mạnh

| Aspect                       | Đánh giá                                                                                      |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| **Modular hoá**              | Tách đúng `preprocessors / datasets / models / training` — dễ mở rộng                         |
| **Config inheritance**       | [config_utils.py:5-46](../src/utils/config_utils.py#L5-L46) hỗ trợ `defaults: [base]` rất gọn |
| **MQOT v2**                  | Unbalanced Sinkhorn + learnable epsilon + multi-head OT — đúng SOTA (PROGOT/AlignMamba)       |
| **QualityGate v2**           | Cross-attention alignment + causal mask — thay nội suy cứng bằng learned alignment            |
| **Training infrastructure**  | AMP + grad accumulation + early stopping + checkpoint cleanup + defensive NaN/Inf check       |
| **Vietnamese normalization** | NFC + giữ tone marks, đúng cho tiếng Việt                                                     |
| **Documentation**            | README + ALGORITHM_SYSTEM_OVERVIEW + ADDING_NEW_DATASET viết chi tiết                         |
| **GPU-native preprocessing** | Đã migrate từ MediaPipe (CPU/EGL conflict) sang face-alignment                                |
| **Tokenizer abstraction**    | Wrapper rõ ràng quanh HF WhisperTokenizer với property tiện lợi                               |
| **Logging**                  | Color formatter + dual handler (console + file)                                               |

---

## 3. Bug nghiêm trọng (Critical)

> Các bug này **block training** hoặc làm sai lệch loss/metric. Cần fix trước khi chạy training thực sự.

### 3.1 `MQOTLayer.forward()` định nghĩa **2 lần**

**File:** [src/models/fusion/mqot.py:148-205](../src/models/fusion/mqot.py#L148-L205)

```python
def forward(self, audio, video, quality):  # method 1 (line 148)
    ...
    return P

def forward(self, audio, video, quality):  # method 2 (line 176) — OVERRIDE!
    ...
    return P
```

**Vấn đề:** Trong Python, method thứ 2 sẽ **âm thầm override** method thứ 1. Hai bản logic gần giống nhau nhưng có khác biệt nhỏ (method 1 unpack `B, Ta, D = audio.shape` rồi compute; method 2 chỉ lấy `Ta, Tv`). Người đọc code sẽ confuse và có thể edit nhầm bản không được gọi.

**Fix:**

- Xoá hẳn 1 bản (giữ lại method 2 vì đó là bản đang chạy).
- Add comment giải thích logic.

**Priority:** HIGH

---

### 3.2 `ChainedScheduler` không tương thích với `ReduceLROnPlateau`

**File:** [src/training/trainer.py:95-111](../src/training/trainer.py#L95-L111) và line 187

```python
warmup_scheduler = optim.lr_scheduler.LinearLR(...)
plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(...)
self.scheduler = optim.lr_scheduler.ChainedScheduler(
    [warmup_scheduler, plateau_scheduler]
)
...
# Line 187:
self.scheduler.step(current_metric)
```

**Vấn đề:**

1. `ChainedScheduler.step()` **không nhận argument**, nhưng `ReduceLROnPlateau.step(metric)` **bắt buộc** nhận metric.
2. `ReduceLROnPlateau` không phải subclass của `_LRScheduler`, không thể combine trong `ChainedScheduler` (PyTorch sẽ raise `TypeError` hoặc behavior không xác định).
3. Cuối epoch, `self.scheduler.step(current_metric)` sẽ crash hoặc plateau scheduler không nhận đúng metric.

**Fix:**

```python
# Tách quản lý
self.warmup_scheduler = optim.lr_scheduler.LinearLR(...)
self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(...)

# Per-step warmup
if self.step <= warmup_steps:
    self.warmup_scheduler.step()

# Per-epoch plateau
self.plateau_scheduler.step(val_metric)
```

Hoặc dùng `SequentialLR` với warmup và `CosineAnnealingLR` (không cần metric).

**Priority:** CRITICAL — Phase 1 training sẽ fail từ epoch 1.

---

### 3.3 Shape mismatch transport_map trong loss quality

**File:** [src/models/fusion/mqot.py:174](../src/models/fusion/mqot.py#L174) ↔ [src/training/losses.py:188-204](../src/training/losses.py#L188-L204)

`MQOTLayer.forward` trả về:

```python
return P  # shape [B, H, Ta, Tv] — multi-head OT
```

Nhưng `HybridLoss.quality_loss` xử lý như tensor 3-D:

```python
P_col = F.normalize(transport_map, p=1, dim=1) + 1e-8  # dim=1 = head axis cho 4-D!
entropy = -torch.sum(P_col * torch.log(P_col), dim=1)
max_ent = torch.log(torch.tensor(transport_map.size(1), ...))  # = log(H), KHÔNG phải log(Ta)
```

**Vấn đề:**

- Khi `num_heads >= 1`, `dim=1` là head axis, không phải audio axis.
- `transport_map.size(1)` = `H` (số heads), không phải `Ta` (số audio frames).
- Entropy/sharpness tính sai hoàn toàn → loss vô nghĩa.

**Fix:**

```python
# Squeeze head dim nếu H=1, hoặc mean across heads
if transport_map.dim() == 4:
    transport_map = transport_map.mean(dim=1)  # [B, Ta, Tv]
# Sau đó xử lý như 3-D
P_col = F.normalize(transport_map, p=1, dim=1)  # normalize over Ta axis
entropy = -torch.sum(P_col * torch.log(P_col), dim=1)  # [B, Tv]
max_ent = torch.log(torch.tensor(transport_map.size(1), ...))  # log(Ta)
```

**Priority:** HIGH — Aux loss (`quality_loss_weight=0.1` ở Phase 2) tính sai.

---

### 3.4 Residual connection sai trong QualityGate

**File:** [src/models/fusion/quality_gate.py:179](../src/models/fusion/quality_gate.py#L179)

```python
fused = fused + F.sigmoid(self.residual_gate) * fused
```

**Vấn đề:**

- Đây **không phải** residual connection. Tương đương `fused * (1 + sigmoid(g))`, chỉ là scaling.
- Comment "Residual gate (zero-init — identity at start)" sai: zero-init `g=0` ⇒ `sigmoid(0)=0.5` ⇒ output = `1.5 * fused`, **không phải identity**.
- Mất pathway từ input gốc.

**Fix (residual đúng):**

```python
# Option 1: Convex combination
g = F.sigmoid(self.residual_gate)
fused = (1 - g) * audio_feat + g * fused

# Option 2: Pre-residual (skip connection)
fused = audio_feat + F.sigmoid(self.residual_gate) * fused

# Option 3: Để identity-at-init thực sự
# Init residual_gate = -10 (sigmoid ≈ 0) → fused ≈ audio_feat
```

**Priority:** HIGH — Ảnh hưởng feature quality từ epoch 1.

---

### 3.5 `blank_id == pad_id == 50257` (CTC blank conflict)

**File:** [src/training/losses.py:43](../src/training/losses.py#L43), [configs/base.yaml:13-14](../configs/base.yaml#L13-L14)

```yaml
vocab_size: 51865
blank_id: 50257 # EOT token for Whisper
```

```python
def __init__(self, ..., pad_id: int = 50257, blank_id: int = 50257):
```

**Vấn đề:**

- CTC blank và padding token cùng giá trị 50257 (Whisper EOT).
- Logic filter trong loss filter cùng lúc cả blank và pad ⇒ chồng chéo:
  ```python
  valid_mask = (target_seq != self.blank_id) & (target_seq != self.pad_id) & (target_seq != self.sot_id)
  ```
- CTCDecoder `greedy_decode` filter blank cũng filter luôn EOT ⇒ không thể phân biệt "kết thúc câu" vs "blank".

**Fix:**

- Tách `blank_id` riêng:
  ```yaml
  vocab_size: 51866 # +1 cho blank
  blank_id: 51865 # slot riêng cuối vocab
  pad_id: 50257 # giữ EOT cho pad
  ```
- Cần expand embedding/output layer thêm 1 slot cho blank.

**Priority:** MEDIUM-HIGH — Hiện tại "may mắn" vẫn train được nhưng decoder behavior không clean.

---

## 4. Bug trung bình (Medium)

### 4.1 Deprecated AMP API

**File:** [src/training/trainer.py:122, 256](../src/training/trainer.py#L122)

```python
self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)  # deprecated
with torch.cuda.amp.autocast(enabled=self.use_amp):           # deprecated
```

**Fix (PyTorch ≥ 2.0):**

```python
self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
with torch.amp.autocast('cuda', enabled=self.use_amp):
```

---

### 4.2 Tokenizer init **2 lần** trong Trainer

**File:** [src/training/trainer.py:73-74, 135](../src/training/trainer.py#L73-L135)

```python
self.tokenizer = WhisperTokenizer(model="openai/whisper-small", language="vi")  # line 74
...
self.tokenizer = self.train_loader.dataset.tokenizer  # line 135 — overwrite
```

**Fix:** Xoá line 135 (đã pass `tokenizer` vào `build_dataloader` rồi).

---

### 4.3 Branch redundant trong loader

**File:** [src/data/loader.py:41](../src/data/loader.py#L41)

```python
manifest_key = f"{mode}_manifest" if mode == "train" else f"{mode}_manifest"
```

Cả 2 nhánh giống hệt.

**Fix:** `manifest_key = f"{mode}_manifest"`

---

### 4.4 `count += 1` đếm 2 lần

**File:** [src/data/preprocessors/base.py:607, 653](../src/data/preprocessors/base.py#L607-L653)

```python
for videos, paths in tqdm(loader, desc="Processing"):
    for video_tensor, video_path in zip(videos, paths):
        count += 1  # line 607
        ...
        torch.save(save_dict, save_path)
        count += 1  # line 653 — DUPLICATE
```

**Fix:** Xoá `count += 1` ở line 653.

---

### 4.5 Default tokenizer model mismatch

**File:** [src/data/tokenizers/whisper.py:14](../src/data/tokenizers/whisper.py#L14)

```python
def __init__(self, ..., model: str = "openai/whisper-tiny", ...):
```

Default `whisper-tiny` (384-dim) nhưng:

- Trainer gọi `whisper-small` (768-dim).
- Preprocessor gọi `whisper-small`.
- Inference script gọi `whisper-small`.

User dùng default sẽ gặp shape mismatch.

**Fix:** Đổi default → `"openai/whisper-small"` hoặc raise nếu không pass explicit.

---

### 4.6 Dummy data khi load .pt fail

**File:** [src/data/datasets/base.py:133-142](../src/data/datasets/base.py#L133-L142)

```python
except Exception as e:
    logger.error(f"Error loading {full_path}: {e}")
    return {
        'audio': torch.zeros(300, 768),
        'visual': torch.zeros(75, 512),
        'target': target,
        ...
    }
```

**Vấn đề:**

- Hardcoded shape có thể không match.
- Sample rác âm thầm trộn vào training, gây bias.
- Không có cảnh báo metric.

**Fix:**

```python
except Exception as e:
    logger.error(f"Error loading {full_path}: {e}")
    return None  # collator skip None
```

Cập nhật `Collator` để filter `None` và return `None` nếu batch rỗng (đã có guard ở line 18-19).

---

### 4.7 Curriculum learning đã tắt nhưng README còn ghi

**File:** [src/training/losses.py:213-217](../src/training/losses.py#L213-L217)

```python
# Fixed Weights (Disabled Curriculum for Stability)
ctc_w = self.ctc_weight
ce_w = self.ce_weight
```

Nhưng README mô tả 3-stage curriculum (1-5, 6-15, 16+).

**Fix:** Bật lại sau cờ `enable_curriculum: bool` hoặc cập nhật README.

---

### 4.8 `empty_count` UnboundLocalError nguy cơ

**File:** [scripts/data_prep/prep_features_gpu.py:140-154](../scripts/data_prep/prep_features_gpu.py#L140-L154)

```python
try:
    with open(output_manifest, 'r', encoding='utf-8') as f:
        ...
        empty_count = 0
        for line in lines:
            if '"text": ""' in line:
                empty_count += 1
        ...
except Exception as e:
    logger.warning(f"Verification failed: {e}")

return f"Subset {subset_name} completed. Labels Verified: {'FAIL' if empty_count > 0 else 'PASS'}"
# ↑ empty_count có thể chưa định nghĩa nếu exception
```

**Fix:** Init `empty_count = 0` ngoài `try`.

---

### 4.9 Causal mask cố định `max_len=512`

**File:** [src/models/layers/decoders.py:24, 99](../src/models/layers/decoders.py#L24)

```python
def __init__(self, ..., max_len: int = 512):
    ...
    causal = nn.Transformer.generate_square_subsequent_mask(max_len)
    self.register_buffer('causal_mask', causal)

# forward:
causal_mask = self.causal_mask[:L, :L]  # crash nếu L > 512
```

**Fix:** Add assertion hoặc dynamic resize:

```python
assert L <= self.causal_mask.size(0), f"Target len {L} > max_len {self.causal_mask.size(0)}"
```

---

### 4.10 `fine_align_gate` không clip range

**File:** [src/models/mota.py:114, 229](../src/models/mota.py#L114)

```python
self.fine_align_gate = nn.Parameter(torch.tensor(0.1))
...
fused = fused_coarse + self.fine_align_gate * self.downsample(fused_fine)
```

Gate chưa qua `sigmoid` → có thể âm hoặc rất lớn, không bounded `[0,1]` như comment ngụ ý.

**Fix:**

```python
fused = fused_coarse + F.sigmoid(self.fine_align_gate) * self.downsample(fused_fine)
```

---

## 5. Vấn đề thiết kế / phong cách

| File                                                                                                                                                | Vấn đề                                                                         | Đề xuất                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| [src/data/preprocessors/base.py:13-14](../src/data/preprocessors/base.py#L13-L14)                                                                   | Import `DataLoader, Dataset` lặp 2 lần                                         | Xoá dòng trùng                                             |
| [src/data/preprocessors/base.py:124-158](../src/data/preprocessors/base.py#L124-L158)                                                               | `_loop_pad_video` có 20 dòng comment confusion về shape                        | Xoá comment, viết unit test                                |
| [src/training/losses.py:117-138](../src/training/losses.py#L117-L138)                                                                               | Loop Python build CTC targets per-sample (O(B) per batch)                      | Vectorize với scatter/index                                |
| [src/data/collate.py:28-32](../src/data/collate.py#L28-L32)                                                                                         | Build mask bằng Python loop                                                    | `mask = torch.arange(max_len)[None, :] < lengths[:, None]` |
| [src/evaluation/decoding.py:48-149](../src/evaluation/decoding.py#L48-L149)                                                                         | Beam search Python pure — rất chậm                                             | Dùng `torchaudio.models.decoder` hoặc `pyctcdecode`        |
| [src/training/trainer.py](../src/training/trainer.py)                                                                                               | Trainer 370 dòng, làm quá nhiều việc                                           | Tách `_train_step`, `_validate_step`, `_setup_*`           |
| [src/models/mota.py:256](../src/models/mota.py#L256)                                                                                                | `'quality' in locals()` — anti-pattern                                         | Track explicit với `quality = None` ban đầu                |
| [scripts/data_prep/prep_vicocktail.py](../scripts/data_prep/prep_vicocktail.py) & [prep_features_gpu.py](../scripts/data_prep/prep_features_gpu.py) | Định nghĩa `FileSystemPreprocessor` cùng tên 2 lần (copy-paste)                | Move lên `src/data/preprocessors/filesystem.py`            |
| [scripts/data_prep/preprocess.py:9-19](../scripts/data_prep/preprocess.py#L9-L19)                                                                   | Đánh dấu DEPRECATED nhưng README còn hướng dẫn dùng                            | Xoá file hoặc xoá khỏi README                              |
| [src/utils/**init**.py](../src/utils/__init__.py)                                                                                                   | Re-export đầy đủ từ utils, nhưng các module đều import trực tiếp `src.utils.X` | Bỏ re-export hoặc dùng nhất quán                           |

### Type hints không nhất quán

```python
# Mixed style:
aug_cfg: dict = None              # nên: Optional[Dict]
config: Dict                      # OK
transport_map: torch.Tensor = None  # nên: Optional[torch.Tensor]
```

**Fix:** Thống nhất dùng `Optional[T]` từ `typing`.

### Docstring trộn ngôn ngữ

Nhiều file mix tiếng Việt + tiếng Anh trong cùng docstring (vd. [quality_gate.py](../src/models/fusion/quality_gate.py), [mqot.py](../src/models/fusion/mqot.py)).

**Đề xuất:** Thống nhất 1 ngôn ngữ (tiếng Anh cho code công khai, tiếng Việt cho docs/README).

---

## 6. Tuân thủ coding rules

> Theo `~/.claude/rules/python/` (PEP 8, type hints, testing, security)

| Rule                                | Trạng thái            | Ghi chú                                                                                                                                                                                                 |
| ----------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PEP 8 formatting                    | ✅ Phần lớn OK        | Black/isort đã trong requirements nhưng chưa chạy CI                                                                                                                                                    |
| Type hints trên function signatures | ⚠️ Một phần           | Nhiều `dict = None` thay vì `Optional[Dict]`                                                                                                                                                            |
| Docstring                           | ⚠️ Trộn lẫn           | VN + EN trong cùng file                                                                                                                                                                                 |
| **Test coverage 80%+**              | ❌ **Không có tests** | Không có thư mục `tests/` dù `pytest` đã có trong requirements                                                                                                                                          |
| `print()` vs `logging`              | ⚠️                    | [lr_finder.py](../scripts/training/lr_finder.py), [inference_phase1.py](../scripts/inference/inference_phase1.py), [download_vicocktail.py](../scripts/data_prep/download_vicocktail.py) dùng `print()` |
| Immutability                        | ✅                    | Tensor ops đa số create new; collator build dict mutable nhưng chấp nhận                                                                                                                                |
| Error handling                      | ⚠️                    | Quá nhiều `except Exception: pass` — silent swallow                                                                                                                                                     |
| Secret management                   | ✅                    | Dùng `modal.Secret.from_name("hf-token")`                                                                                                                                                               |
| Input validation                    | ⚠️                    | Chỉ guard ở 1 vài chỗ (collator empty batch); không có schema validation cho config                                                                                                                     |

### Đề xuất bổ sung

1. **Tạo `tests/`** ít nhất unit test cho:
   - `MQOTLayer.sinkhorn_unbalanced` — kiểm tra row sum ≈ 1, gradient flows
   - `QualityGate` — output shape, no NaN
   - `CTCDecoder.greedy_decode` — case có blank, không có blank
   - `MetricCalculator.compute_wer` — edge cases (empty, exact match, all wrong)
   - `WhisperTokenizer` — encode/decode round-trip
   - `Collator` — variable length batches

2. **CI**: GitHub Actions workflow chạy `black --check`, `isort --check`, `pytest`, `mypy src/`.

3. **Validation config**: dùng `pydantic` hoặc `dataclass` thay `dict` lỏng lẻo.

4. **Pre-commit hooks** (`pre-commit`): black, isort, ruff, end-of-file-fixer, trailing-whitespace.

---

## 7. Khuyến nghị ưu tiên

### CRITICAL (block training)

1. **Fix `ChainedScheduler`** ([3.2](#32-chainedscheduler-không-tương-thích-với-reducelronplateau)) — tách warmup/plateau hoặc dùng `SequentialLR`.
2. **Xoá `forward()` duplicate** trong `MQOTLayer` ([3.1](#31-mqotlayerforward-định-nghĩa-2-lần)).
3. **Fix shape `[B,H,Ta,Tv]`** trong loss quality ([3.3](#33-shape-mismatch-transport_map-trong-loss-quality)).

### HIGH (ảnh hưởng correctness)

4. **Sửa residual logic** ở `QualityGate` ([3.4](#34-residual-connection-sai-trong-qualitygate)).
5. **Tách `blank_id` ≠ `pad_id`** ([3.5](#35-blank_id--pad_id--50257-ctc-blank-conflict)).
6. **Đừng giả dummy data** khi load .pt fail ([4.6](#46-dummy-data-khi-load-pt-fail)).
7. **Thêm `sigmoid` cho `fine_align_gate`** ([4.10](#410-fine_align_gate-không-clip-range)).

### MEDIUM (chất lượng code)

8. **Thêm test suite** + CI (pytest, black, isort, mypy).
9. **Migrate `torch.cuda.amp` → `torch.amp`** ([4.1](#41-deprecated-amp-api)).
10. **Refactor 2 bản `FileSystemPreprocessor`** thành class chung.
11. **Fix `count += 1` đếm 2 lần** ([4.4](#44-count--1-đếm-2-lần)).
12. **Default tokenizer model** ([4.5](#45-default-tokenizer-model-mismatch)).
13. **`empty_count` init** ([4.8](#48-empty_count-unboundlocalerror-nguy-cơ)).

### LOW (cosmetic)

14. **Sửa branch redundant** ở [loader.py:41](../src/data/loader.py#L41).
15. **Sync README** curriculum với code thực tế ([4.7](#47-curriculum-learning-đã-tắt-nhưng-readme-còn-ghi)).
16. **Xoá `preprocess.py` deprecated** hoặc cập nhật README.
17. **Thống nhất type hints** (`Optional[T]`).
18. **Thay `print()` bằng `logging`**.

---

## 8. Tóm tắt

### Đánh giá chung

Codebase có **kiến trúc tốt, ý tưởng đúng SOTA** (Unbalanced OT, Cross-attention QualityGate, Hybrid CTC+AR), tài liệu kỹ lưỡng (README, ALGORITHM_SYSTEM_OVERVIEW, ADDING_NEW_DATASET).

### Vấn đề chính

- **3 bug critical** sẽ làm Phase 1 training fail hoặc tính loss sai (3.1, 3.2, 3.3) cần sửa **trước khi chạy training thực sự**.
- **Thiếu test suite** và CI là gap lớn nhất về maintainability.
- **Thiết kế residual ở QualityGate** sai về mặt toán học, ảnh hưởng feature quality.

### Roadmap khắc phục đề xuất

| Tuần   | Tasks                                                              |
| ------ | ------------------------------------------------------------------ |
| Tuần 1 | Fix 3 critical bugs (3.1, 3.2, 3.3) + smoke test Phase 1           |
| Tuần 2 | Fix 4 HIGH bugs (3.4, 3.5, 4.6, 4.10) + thêm unit tests cho models |
| Tuần 3 | CI setup + refactor preprocessor + migrate AMP API                 |
| Tuần 4 | Cosmetic + sync docs + cleanup deprecated files                    |

### Score Card

| Hạng mục         | Điểm       | Ghi chú                              |
| ---------------- | ---------- | ------------------------------------ |
| Architecture     | 8/10       | SOTA design, modular                 |
| Code correctness | 5/10       | Critical bugs cần fix                |
| Code quality     | 6/10       | Type hints không nhất quán, no tests |
| Documentation    | 9/10       | README + guides đầy đủ               |
| Infrastructure   | 8/10       | Modal pipeline tốt, CI thiếu         |
| Maintainability  | 5/10       | No tests, copy-paste code            |
| **Tổng**         | **6.8/10** | Cần fix critical bugs + add tests    |

---

**End of Review**
