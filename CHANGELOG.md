# CHANGELOG - AURORA-XT / MOTA-AVSR

Tài liệu này ghi lại các thay đổi quan trọng so với nhánh `main-dev` gốc.
Repository gốc: https://github.com/AnhVan17/MOTA-AVSR-Monotone-Optimal-Transport-Alignment-for-AVSR

---

## 📅 Phiên bản hiện tại: v2.0 (2026-01-05)

---

## 🔄 Tổng quan thay đổi

### 1. Preprocessing Pipeline

#### 1.1 ViCocktail Data Processing Pipeline (MỚI)

Pipeline xử lý ViCocktail **từ đầu đến cuối** gồm **5 bước chính**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     VICOCKTAIL FULL PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STEP 0: DOWNLOAD                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  modal run scripts/modal/download.py                                       │  │
│  │                                                                            │  │
│  │  • Source: HuggingFace (avvn-train-*.tar, avvn-test-*.tar)                 │  │
│  │  • Output: /mnt/raw_mirror/*.tar                                           │  │
│  │  • Size: ~200+ tar files (train) + test files                             │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                    ▼                                             │
│  STEP 1: UNPACK TAR FILES                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  modal run scripts/modal/unpack_vicocktail_selective.py --train-limit 50   │  │
│  │                                                                            │  │
│  │  • Input: /mnt/raw_mirror/*.tar                                            │  │
│  │  • Output: /mnt/vicocktail/raw/avvn-train-000000/                          │  │
│  │            /mnt/vicocktail/raw/avvn-train-000001/                          │  │
│  │            ...                                                             │  │
│  │  • Actions:                                                                │  │
│  │    1. Extract tar files                                                    │  │
│  │    2. Rename .video → .mp4                                                 │  │
│  │  • Options:                                                                │  │
│  │    --train-limit 50  → Chỉ unpack 50 train + ALL test                      │  │
│  │    --train-limit 0   → Unpack ALL files                                    │  │
│  │    --clean-first     → Xóa data cũ trước khi unpack                        │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                    ▼                                             │
│  STEP 2: CROP MOUTH REGION                                                       │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  modal run scripts/modal/preprocess_vicocktail.py --stage crop --mode dist │  │
│  │                                                                            │  │
│  │  • Input: /mnt/vicocktail/raw/**/*.mp4 (full face videos)                  │  │
│  │  • Output: /mnt/processed/vicocktail_cropped/**/*.mp4 (mouth only)         │  │
│  │  • Process:                                                                │  │
│  │    1. MediaPipe Face Mesh detect face landmarks                            │  │
│  │    2. Extract mouth ROI (88x88 pixels)                                     │  │
│  │    3. Save as new .mp4 file (chỉ chứa vùng miệng)                          │  │
│  │  • Modes:                                                                  │  │
│  │    --mode dist   → Distributed (50 containers parallel, nhanh)             │  │
│  │    --mode single → Single container (16 CPU, chậm hơn nhưng ổn định)       │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                    ▼                                             │
│  STEP 3: KEYFRAME EXTRACTION (Optional)                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  modal run scripts/modal/preprocess_vicocktail.py --stage keyframe         │  │
│  │                                                                            │  │
│  │  • Input: /mnt/processed/vicocktail_cropped/**/*.mp4                       │  │
│  │  • Output: /mnt/processed/vicocktail_keyframes/**/frame_*.jpg              │  │
│  │  • Process:                                                                │  │
│  │    1. Calculate frame difference (pixel intensity)                         │  │
│  │    2. Keep frames where diff > threshold (30.0)                            │  │
│  │    3. Ensure min 10, max 75 frames per video                               │  │
│  │  • Purpose: Giảm redundancy, tiết kiệm storage/compute                     │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                    ▼                                             │
│  STEP 4: FEATURE EXTRACTION (GPU - A100)                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  modal run scripts/modal/preprocess_vicocktail.py --stage extract          │  │
│  │                                                                            │  │
│  │  • Input: /mnt/processed/vicocktail_cropped/**/*.mp4                       │  │
│  │  • Output:                                                                 │  │
│  │    - Features: /mnt/processed/features/vicocktail/**/*.pt                  │  │
│  │    - Manifests: /mnt/processed/manifests/train.jsonl                       │  │
│  │                 /mnt/processed/manifests/val.jsonl                         │  │
│  │                 /mnt/processed/manifests/test.jsonl                        │  │
│  │  • Process:                                                                │  │
│  │    1. use_precropped=True → Skip MediaPipe (đã crop ở Step 2)              │  │
│  │    2. ResNet-18: Extract visual features → [T, 512]                        │  │
│  │    3. Whisper-Small: Extract audio features → [T, 768]                     │  │
│  │    4. Save combined .pt file per video                                     │  │
│  │    5. Auto-split train/val (90/10), detect test from path                  │  │
│  │                                                                            │  │
│  │  .pt file format:                                                          │  │
│  │  {                                                                         │  │
│  │      'audio': tensor [T_a, 768],   # Whisper encoder features              │  │
│  │      'visual': tensor [T_v, 512],  # ResNet-18 features                    │  │
│  │      'text': "xin chào...",        # Raw transcript                        │  │
│  │      'id': "sample_001",           # Sample ID                             │  │
│  │      'path': "/original/path.mp4"  # Original video path                   │  │
│  │  }                                                                         │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                    ▼                                             │
│  STEP 5: TRAINING                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  modal run scripts/modal/train_vicocktail.py                               │  │
│  │                                                                            │  │
│  │  • Input:                                                                  │  │
│  │    - Features: /mnt/processed/features/vicocktail/**/*.pt                  │  │
│  │    - Manifests: /mnt/processed/manifests/*.jsonl                           │  │
│  │  • Output: /mnt/processed/checkpoints/phase1/*.pt                          │  │
│  │  • Config: configs/vicocktail_phase1.yaml                                  │  │
│  │  • GPU: A100-40GB                                                          │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Quick Start Commands:**

```bash
# 1. Download data từ HuggingFace
modal run scripts/modal/download.py

# 2. Unpack 50 train files + all test
modal run scripts/modal/unpack_vicocktail_selective.py --train-limit 50

# 3. Crop mouth region (distributed mode - nhanh)
modal run scripts/modal/preprocess_vicocktail.py --stage crop --mode dist

# 4. Extract features (A100 GPU)
modal run scripts/modal/preprocess_vicocktail.py --stage extract

# 5. Train model
modal run scripts/modal/train_vicocktail.py
```

**Data Flow Summary:**

```
HuggingFace (.tar)
       ↓ download.py
/raw_mirror/*.tar
       ↓ unpack_vicocktail_selective.py  
/vicocktail/raw/**/*.mp4 (full face)
       ↓ preprocess --stage crop
/vicocktail_cropped/**/*.mp4 (mouth only, 88x88)
       ↓ preprocess --stage extract
/features/vicocktail/**/*.pt + manifests/*.jsonl
       ↓ train_vicocktail.py
/checkpoints/phase1/*.pt (trained model)
```

**Thay đổi chính so với main-dev:**

| Aspect | main-dev | v2.0 |
|--------|----------|------|
| Pipeline | Single-pass | 5-step modular |
| Crop | On-the-fly | Pre-crop & save |
| Distribution | Single container | 50 parallel containers |
| Model loading | Eager | Lazy (tiết kiệm RAM) |
| Precropped support | ❌ | ✅ (10x faster) |
| Selective unpack | ❌ | ✅ (--train-limit) |

---

### 2. Training Pipeline

#### 2.1 `src/training/trainer.py`

| Thay đổi | Before | After | Lý do |
|----------|--------|-------|-------|
| **Scheduler** | CosineAnnealingLR | LambdaLR (Warmup + Cosine) | CTC cần warmup |
| **Warmup Epochs** | Không có | 5 epochs | Tránh model collapse |
| **Vocab Size** | Dynamic | Fixed 51865 | Whisper special tokens |
| **Math Import** | Không có | `import math` | Cho lr_lambda |

**Warmup Scheduler mới:**

```python
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # Linear warmup
    else:
        # Cosine annealing sau warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr/base_lr + (1 - min_lr/base_lr) * 0.5 * (1 + cos(pi * progress))
```

#### 2.2 `src/training/losses.py`

**⚠️ CRITICAL FIX: CTC Blank ID**

| Thay đổi | Before | After | Lý do |
|----------|--------|-------|-------|
| **blank_id** | `vocab_size - 1` (51864) | `0` | Standard CTC convention |

**Vấn đề gốc:**
- Khi `blank_id = 51864`, CTC khó hội tụ vì blank ở cuối vocab
- Model collapse: chỉ predict 1 ký tự ("c") cho tất cả inputs
- WER/CER ≈ 99%

**Giải pháp:**
- Đổi `blank_id = 0` (convention chuẩn của PyTorch CTC)
- Model hội tụ nhanh hơn, WER giảm đáng kể

```python
# TRƯỚC
self.blank_id = vocab_size - 1  # = 51864 (BAD)

# SAU
self.blank_id = 0  # Standard CTC blank ID (GOOD)
```

#### 2.3 `src/evaluation/metrics.py`

| Thay đổi | Before | After | Lý do |
|----------|--------|-------|-------|
| **Decode blank_id** | `len(tokenizer) - 1` | `0` | Match với training |
| **Default blank** | 4 | 0 | Consistent |

```python
# TRƯỚC
blank_id = len(self.tokenizer) - 1
pred_ids = self.ctc_greedy_decode(logits, blank_id=blank_id)

# SAU  
pred_ids = self.ctc_greedy_decode(logits, blank_id=0)
```

#### 2.4 `configs/vicocktail_phase1.yaml`

```yaml
# TRƯỚC
training:
  learning_rate: 1e-4
  gradient_clip: 2.0

# SAU
training:
  learning_rate: 5e-5        # Giảm từ 1e-4
  warmup_epochs: 5           # MỚI
  gradient_clip: 1.0         # Giảm từ 2.0
```

---

### 3. Data Loading

#### 3.1 `src/data/datasets/base_dataset.py`

- Thêm `FeatureDataset` class cho Phase 1 training
- Hỗ trợ augmentation (FeatureAugmenter)
- Error handling với dummy data fallback

#### 3.2 `src/data/datasets/vicocktail.py`

- Kế thừa từ `FeatureDataset`
- Truncation 448 tokens (Whisper limit)
- Override `_tokenize()` với length constraints

---

### 4. Model Architecture

#### 4.1 `src/models/mota.py` (MỚI - thay thế aurora_xt.py)

| Component | Mô tả |
|-----------|-------|
| **MOTA** | Main model class với QualityGate + MQOT support |
| **Phase 1** | `use_mqot=False` - Chỉ dùng QualityGate |
| **Phase 2** | `use_mqot=True` - Thêm Optimal Transport |

**Kiến trúc:**
```
Audio [768] ──┐
              ├─→ QualityGate ─→ Conformer(6) ─→ HybridDecoder
Visual [512] ─┘                                     ├─ CTC Head
                                                    └─ AR Decoder
```

---

### 5. Scripts Modal

#### 5.1 Files MỚI trong `scripts/modal/`

| File | Mục đích |
|------|----------|
| `preprocess_vicocktail.py` | 3-stage preprocessing pipeline |
| `train_vicocktail.py` | Training với A100 GPU |
| `unpack_vicocktail_selective.py` | Chọn số lượng train files |
| `download.py` | Download từ HuggingFace |

#### 5.2 Usage Examples

```bash
# Preprocessing
modal run scripts/modal/preprocess_vicocktail.py --stage crop --mode dist
modal run scripts/modal/preprocess_vicocktail.py --stage extract

# Training
modal run scripts/modal/train_vicocktail.py

# Selective unpack (50 train + all test)
modal run scripts/modal/unpack_vicocktail_selective.py --train-limit 50
```

---

## ⚠️ Known Issues & Fixes

### Issue 1: Model Collapse (Prediction = single character)

**Triệu chứng:**
- WER/CER ≈ 99%
- Model predict "c" cho tất cả inputs
- CTC loss rất cao (>15)

**Nguyên nhân:** `blank_id = vocab_size - 1`

**Giải pháp:** Đã fix trong `losses.py` và `metrics.py`

### Issue 2: CUDA OOM

**Triệu chứng:** `OutOfMemoryError: CUDA out of memory`

**Giải pháp:**
- Giảm batch_size từ 64 → 16
- Bật Mixed Precision (`use_amp: true`)

### Issue 3: Tokenizer Mismatch

**Triệu chứng:** Token IDs > 50257

**Giải pháp:** Fixed vocab_size = 51865 trong trainer.py

---

## 📚 File Structure Changes

```
ASVR_Viet/
├── configs/
│   └── vicocktail_phase1.yaml     # UPDATED: warmup, lr
├── scripts/
│   └── modal/
│       ├── preprocess_vicocktail.py  # NEW: 3-stage pipeline
│       ├── train_vicocktail.py       # NEW: Modal training
│       └── unpack_vicocktail_selective.py  # NEW
├── src/
│   ├── data/
│   │   ├── preprocessors/
│   │   │   └── base_preprocessor.py  # UPDATED: lazy load, precrop
│   │   └── datasets/
│   │       ├── base_dataset.py       # UPDATED: FeatureDataset
│   │       └── vicocktail.py         # NEW
│   ├── models/
│   │   └── mota.py                   # NEW: replaces aurora_xt.py
│   ├── training/
│   │   ├── trainer.py                # UPDATED: warmup scheduler
│   │   └── losses.py                 # FIXED: blank_id = 0
│   └── evaluation/
│       └── metrics.py                # FIXED: blank_id = 0
└── README.md                         # UPDATED: ViCocktail docs
```

---

## 🔜 Next Steps

1. **Phase 1 Training** - Train với features đã extract
2. **Phase 2 Training** - Bật MQOT layer
3. **Hyperparameter Tuning** - Grid search lr, batch_size
4. **Beam Search Decoding** - Thay CTC greedy bằng beam search

---

## 📝 Contributing

Khi thêm changes mới:
1. Update file này với mô tả thay đổi
2. Test locally trước khi push
3. Update README.md nếu cần

---

*Last updated: 2026-01-05*
