# MOTA - Multimodal Optimal Transport Alignment

## Tổng quan

MOTA là mô hình AVSR (Audio-Visual Speech Recognition) sử dụng kỹ thuật Optimal Transport để căn chỉnh và fusion các modality audio-visual.

---

## 1. Kiến trúc mô hình

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MOTA Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐              ┌──────────────────┐                     │
│  │   Audio Input    │              │   Visual Input   │                     │
│  │   (waveform)     │              │   (video frames) │                     │
│  └────────┬─────────┘              └────────┬─────────┘                     │
│           │                                  │                               │
│           ▼                                  ▼                               │
│  ┌──────────────────┐              ┌──────────────────┐                     │
│  │  Whisper Encoder │              │   ResNet18 2D    │                     │
│  │    (frozen)      │              │   (per-frame)    │                     │
│  │   → 768dim       │              │   → 512dim       │                     │
│  └────────┬─────────┘              └────────┬─────────┘                     │
│           │                                  │                               │
│           └──────────────┬───────────────────┘                               │
│                          ▼                                                   │
│             ┌────────────────────────┐                                       │
│             │     Quality Gate       │  ← Stage 1: Coarse Fusion            │
│             │   (Adaptive weighting) │                                       │
│             └────────────┬───────────┘                                       │
│                          │                                                   │
│                          ▼                                                   │
│             ┌────────────────────────┐                                       │
│             │   M-QOT + Guided Attn  │  ← Stage 2: Fine Alignment (Optional)│
│             │  (Optimal Transport)   │                                       │
│             └────────────┬───────────┘                                       │
│                          │                                                   │
│                          ▼                                                   │
│             ┌────────────────────────┐                                       │
│             │   Conformer Encoder    │  ← 6 layers                          │
│             └────────────┬───────────┘                                       │
│                          │                                                   │
│                          ▼                                                   │
│             ┌────────────────────────┐                                       │
│             │    Hybrid Decoder      │  ← CTC + Attention                   │
│             └────────────┬───────────┘                                       │
│                          │                                                   │
│                          ▼                                                   │
│                    [Transcription]                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components:

| Component | Model | Output Dimension | Mô tả |
|-----------|-------|------------------|-------|
| **Audio Encoder** | Whisper Small (frozen) | 768 | Trích xuất audio features từ waveform |
| **Visual Encoder** | ResNet18 2D | 512 | Trích xuất visual features từ mỗi frame |
| **Fusion Stage 1** | Quality Gate | d_model (256) | Fusion coarse với adaptive weighting |
| **Fusion Stage 2** | M-QOT | d_model (256) | Optimal Transport alignment (optional) |
| **Encoder** | Conformer | d_model (256) | 6 layers với convolution + attention |
| **Decoder** | Hybrid CTC + Attention | vocab_size | Dual decoding strategy |

---

## 2. Pipeline Preprocessing (ViCocktail)

### 2.1 Quy trình tổng quan

```
[Download] → [Unpack] → [Crop] → [Extract] → [Train]
```

### 2.2 Chi tiết từng bước

#### Bước 1: Download
- Sử dụng HuggingFace hoặc mirror để download `.tar` files
- Lưu vào Modal Volume: `/mnt/dataset/`

#### Bước 2: Unpack
```bash
modal run scripts/modal/unpack_vicocktail_selective.py --train-limit 20 --clean-first
```

**Input:** `.tar` files (avvn-train-*.tar, avvn-test-*.tar)

**Output:**
```
/mnt/dataset/vicocktail/raw/
├── avvn-train-000000/          ← Train subfolder
│   └── *.mp4, *.txt
├── avvn-train-000001/
├── ...
├── avvn-test-000000/           ← Test subfolder (chứa 'test' trong tên)
├── avvn-test_snr_*-000000/
└── ...
```

**Tham số:**
- `--train-limit N`: Số lượng train tars (0 = all)
- `--clean-first`: Xóa dữ liệu cũ trước khi unpack

#### Bước 3: Crop (Mouth Region)
```bash
modal run scripts/modal/preprocess_vicocktail.py --stage crop --mode single
```

**Input:** Raw videos từ bước Unpack

**Output:**
```
/mnt/processed/vicocktail_cropped/
├── avvn-train-000000/          ← Giữ nguyên cấu trúc
│   └── *.mp4 (mouth cropped), *.txt
├── avvn-test-000000/
└── ...
```

**Xử lý:**
- MediaPipe Face Mesh detect vùng miệng
- Crop và resize về 88x88 (SOTA standard)
- Giữ nguyên audio track
- Copy transcript (.txt)

#### Bước 4: Extract (Feature Extraction)
```bash
modal run scripts/modal/preprocess_vicocktail.py --stage extract
```

**Input:** Cropped videos từ bước Crop

**Output:**
```
/mnt/processed/
├── features/vicocktail/
│   ├── avvn-train-000000/
│   │   ├── video1.pt           ← Contains: {audio: [T_a, 768], visual: [T_v, 512], text: "..."}
│   │   └── video2.pt
│   └── avvn-test-000000/
│       └── ...
│
└── manifests/
    ├── train.jsonl             ← 90% của train data
    ├── val.jsonl               ← 10% của train data
    └── test.jsonl              ← 100% test data (riêng biệt)
```

**Feature extraction:**
- **Audio:** Whisper Small encoder → [T_a, 768]
- **Visual:** ResNet18 2D per-frame → [T_v, 512]

**Split Logic:**
```python
for item in all_data:
    if 'test' in folder_name:  # e.g., avvn-test-000000
        → test.jsonl
    else:
        → train_val_candidates → shuffle → 90/10 split
```

---

## 3. Training

### 3.1 Chạy training trên Modal
```bash
modal run scripts/modal/train_vicocktail.py
```

### 3.2 Configuration (vicocktail_phase1.yaml)

```yaml
model:
  audio_dim: 768            # Whisper Small
  visual_dim: 512           # ResNet18
  d_model: 256              # Internal dimension
  num_encoder_layers: 6
  num_decoder_layers: 4
  num_heads: 4
  vocab_size: 51865         # Whisper Tokenizer
  dropout: 0.1

loss:
  ctc_weight: 0.3
  ce_weight: 0.7

data:
  train_manifest: "/mnt/processed/manifests/train.jsonl"
  val_manifest: "/mnt/processed/manifests/val.jsonl"
  data_root: "/mnt/processed/features/vicocktail"
  batch_size: 16
  num_workers: 8

training:
  num_epochs: 50
  learning_rate: 5e-5
  warmup_epochs: 5
  weight_decay: 0.01
  gradient_clip: 1.0
  use_amp: true
```

### 3.3 Training Loop

```
For each epoch:
    1. Train on train.jsonl
       - Forward pass (MOTA model)
       - Compute loss (CTC 30% + CE 70%)
       - Backward + Gradient clipping
       
    2. Validate on val.jsonl
       - Compute WER, CER
       - Save best model if WER improves
       
    3. Scheduler step (Warmup + Cosine Annealing)
```

---

## 4. Cấu trúc thư mục dự án

```
ASVR_Viet/
├── configs/
│   └── vicocktail_phase1.yaml      # Training config
│
├── scripts/
│   ├── modal/
│   │   ├── unpack_vicocktail_selective.py  # Bước 2
│   │   ├── preprocess_vicocktail.py        # Bước 3, 4
│   │   └── train_vicocktail.py             # Bước 5
│   └── training.py                          # Local training
│
├── src/
│   ├── data/
│   │   ├── preprocessors/
│   │   │   ├── base_preprocessor.py    # Feature extraction
│   │   │   └── vicocktail.py           # ViCocktail-specific
│   │   ├── loader.py                    # DataLoader
│   │   └── tokenizers/
│   │       └── whisper.py               # Whisper Tokenizer
│   │
│   ├── models/
│   │   ├── mota.py                      # Main model
│   │   ├── layers/
│   │   │   ├── conformer.py
│   │   │   ├── decoders.py
│   │   │   └── adapters.py
│   │   └── fusion/
│   │       ├── quality_gate.py
│   │       └── mqot.py
│   │
│   ├── training/
│   │   ├── trainer.py                   # Core Trainer
│   │   └── losses.py                    # CTC + CE Loss
│   │
│   └── evaluation/
│       └── evaluator.py                 # WER/CER metrics
│
└── PIPELINE.md                          # This file
```

---

## 5. Commands Summary

```bash
# 1. Unpack data (with clean)
modal run scripts/modal/unpack_vicocktail_selective.py --clean-first --train-limit 20

# 2. Crop mouth region
modal run scripts/modal/preprocess_vicocktail.py --stage crop --mode single

# 3. Extract features
modal run scripts/modal/preprocess_vicocktail.py --stage extract

# 4. Train model
modal run scripts/modal/train_vicocktail.py

# Or run in detached mode (background)
modal run --detach scripts/modal/train_vicocktail.py
```

---

## 6. Hyperparameters

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| Batch Size | 16 | Reduced để tránh OOM |
| Learning Rate | 5e-5 | Reduced cho CTC stability |
| Warmup Epochs | 5 | LR warmup phase |
| Gradient Clip | 1.0 | Prevent exploding gradients |
| CTC Weight | 0.3 | CTC loss weight |
| CE Weight | 0.7 | Cross-Entropy loss weight |
| Vocab Size | 51865 | Whisper tokenizer full vocab |

---

## 7. Notes

### Visual Encoder
- Sử dụng **ResNet18 2D** (không phải 3D)
- Xử lý từng frame độc lập → [T_v, 512]
- Input: 88x88 RGB frames

### Audio Encoder
- Sử dụng **Whisper Small encoder** (frozen)
- Output: [T_a, 768] với T_a ≈ 1500 cho 30s audio

### Split Strategy
- **Train/Val Split:** 90/10 từ train data
- **Test:** Riêng biệt (folders chứa 'test' trong tên)
