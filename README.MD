# AURORA-XT: Vietnamese Audio-Visual Speech Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AURORA-XT** (Audio-Visual Recognition OptimizeR with Adaptive eXtension Transformer) là mô hình nhận dạng giọng nói tiếng Việt kết hợp cả âm thanh và hình ảnh môi (lip-reading).

## 📋 Mục Lục

- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Pipeline Chi Tiết](#pipeline-chi-tiết)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Kết Quả](#kết-quả)

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AURORA-XT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐     ┌─────────────┐                                       │
│   │   VIDEO     │     │   AUDIO     │                                       │
│   │  (MP4/AVI)  │     │  (from MP4) │                                       │
│   └──────┬──────┘     └──────┬──────┘                                       │
│          │                   │                                              │
│          ▼                   ▼                                              │
│   ┌─────────────┐     ┌─────────────┐                                       │
│   │  MediaPipe  │     │   PyAV/     │                                       │
│   │  Face Mesh  │     │   FFmpeg    │                                       │
│   │  (Mouth ROI)│     │  (16kHz)    │                                       │
│   └──────┬──────┘     └──────┬──────┘                                       │
│          │                   │                                              │
│          ▼                   ▼                                              │
│   ┌─────────────┐     ┌─────────────┐                                       │
│   │  ResNet-18  │     │   Whisper   │                                       │
│   │  (Pretrain) │     │   Encoder   │                                       │
│   │  [T, 512]   │     │  [T, 768]   │                                       │
│   └──────┬──────┘     └──────┬──────┘                                       │
│          │                   │                                              │
│          └─────────┬─────────┘                                              │
│                    │                                                        │
│                    ▼                                                        │
│          ┌─────────────────┐                                                │
│          │  Quality Gate   │  ← Adaptive audio/visual weighting             │
│          │  (Learned)      │                                                │
│          └────────┬────────┘                                                │
│                   │                                                         │
│                   ▼                                                         │
│          ┌─────────────────┐                                                │
│          │   Conformer     │  ← 6 layers, Conv + Attention                  │
│          │    Encoder      │                                                │
│          └────────┬────────┘                                                │
│                   │                                                         │
│          ┌───────┴────────┐                                                 │
│          ▼                ▼                                                 │
│   ┌─────────────┐  ┌─────────────┐                                          │
│   │  CTC Head   │  │  AR Decoder │  ← Hybrid decoding                       │
│   │  (Align)    │  │  (Context)  │                                          │
│   └──────┬──────┘  └──────┬──────┘                                          │
│          │                │                                                 │
│          └───────┬────────┘                                                 │
│                  ▼                                                          │
│          ┌─────────────────┐                                                │
│          │   Vietnamese    │                                                │
│          │     Text        │                                                │
│          └─────────────────┘                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Chi Tiết

### 1️⃣ Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   RAW DATA (ViCocktail TAR files)                                           │
│   ├── video.mp4                                                             │
│   └── transcript.txt                                                        │
│                                                                             │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      VIDEO PROCESSING                               │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │  1. Load video với Decord                                           │   │
│   │  2. Sample frames: 25 FPS → MAX 375 frames (15s)                    │   │
│   │  3. MediaPipe Face Mesh: Extract mouth ROI (96x96)                  │   │
│   │  4. Normalize: ImageNet mean/std                                    │   │
│   │  5. Loop padding nếu < 375 frames                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       AUDIO PROCESSING                              │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │  1. Extract audio: PyAV/FFmpeg                                      │   │
│   │  2. Resample: → 16kHz mono                                          │   │
│   │  3. RMS normalize: 0.1 target                                       │   │
│   │  4. Length: 240,000 samples (15s)                                   │   │
│   │  5. Loop padding nếu audio ngắn hơn                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     FEATURE EXTRACTION                              │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │                                                                     │   │
│   │  VISUAL:                        AUDIO:                              │   │
│   │  ┌──────────────────┐          ┌──────────────────┐                 │   │
│   │  │   ResNet-18      │          │  Whisper-Small   │                 │   │
│   │  │   (Pretrained)   │          │   (Encoder)      │                 │   │
│   │  │                  │          │                  │                 │   │
│   │  │  Input:          │          │  Input:          │                 │   │
│   │  │  [3, 375, 96, 96]│          │  [1, 240000]     │                 │   │
│   │  │                  │          │                  │                 │   │
│   │  │  Output:         │          │  Output:         │                 │   │
│   │  │  [375, 512]      │          │  [~450, 768]     │                 │   │
│   │  └──────────────────┘          └──────────────────┘                 │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       TEXT TOKENIZATION                             │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │  Vietnamese Character Tokenizer (121 tokens)                        │   │
│   │  - Special: <PAD>, <BOS>, <EOS>, <UNK>, <BLANK>                     │   │
│   │  - Vietnamese vowels với dấu (á, à, ả, ã, ạ, ...)                   │   │
│   │  - Basic Latin + Numbers + Punctuation                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    ▼                                                        │
│   OUTPUT: .pt files                                                         │
│   {                                                                         │
│       'audio': [T_a, 768],   # Whisper features                             │
│       'visual': [T_v, 512],  # ResNet features                              │
│       'text': [L],           # Token IDs                                    │
│       'text_raw': "..."      # Original text                                │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2️⃣ Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DATALOADER                                                                │
│   ├── Load .pt files from manifests                                         │
│   ├── Dynamic padding per batch                                             │
│   └── Create attention masks                                                │
│                                                                             │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        AURORA-XT MODEL                              │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │                                                                     │   │
│   │  1. PROJECTION LAYERS                                               │   │
│   │     Audio: [768] → [256]                                            │   │
│   │     Visual: [512] → [256]                                           │   │
│   │                                                                     │   │
│   │  2. QUALITY GATE (Adaptive Fusion)                                  │   │
│   │     - Learn per-frame quality scores                                │   │
│   │     - Weighted combination: α*audio + β*visual                      │   │
│   │     - Temporal alignment via interpolation                          │   │
│   │                                                                     │   │
│   │  3. CONFORMER ENCODER (6 layers)                                    │   │
│   │     - FFN → MHA → Conv → FFN                                        │   │
│   │     - Captures acoustic + temporal patterns                         │   │
│   │                                                                     │   │
│   │  4. HYBRID DECODER                                                  │   │
│   │     ┌─────────────┐    ┌─────────────┐                              │   │
│   │     │  CTC Head   │    │  AR Decoder │                              │   │
│   │     │  (30% loss) │    │  (70% loss) │                              │   │
│   │     └─────────────┘    └─────────────┘                              │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         HYBRID LOSS                                 │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │                                                                     │   │
│   │  L_total = α * L_CTC + β * L_CE                                     │   │
│   │                                                                     │   │
│   │  - CTC Loss: Alignment learning (không cần alignment labels)        │   │
│   │  - CE Loss: Language modeling (next token prediction)               │   │
│   │  - Curriculum: Start CTC-heavy → shift to CE                        │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        OPTIMIZATION                                 │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │  - Optimizer: AdamW (lr=3e-4, weight_decay=0.01)                    │   │
│   │  - Scheduler: CosineAnnealing                                       │   │
│   │  - Mixed Precision: FP16 (AMP)                                      │   │
│   │  - Gradient Clipping: 5.0                                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3️⃣ Model Components Summary

| Component | Model | Input | Output | Purpose |
|-----------|-------|-------|--------|---------|
| **Audio Encoder** | Whisper-Small | Waveform [240k] | [~450, 768] | Extract acoustic features |
| **Visual Encoder** | ResNet-18 | Frames [375, 96, 96] | [375, 512] | Extract lip features |
| **Quality Gate** | Learned MLP | Audio + Visual | Weights [T, 2] | Adaptive modality fusion |
| **Conformer** | 6-layer | Fused [T, 256] | Encoded [T, 256] | Contextual encoding |
| **CTC Head** | Linear | Encoded | Logits [T, vocab] | Frame-level alignment |
| **AR Decoder** | 4-layer Transformer | Encoded + Targets | Logits [L, vocab] | Autoregressive decoding |

---

## 🚀 Cài Đặt

### Yêu Cầu
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- GPU: T4 (inference), A100 (training recommended)

### Cài Đặt Local

```bash
# Clone repository
git clone https://github.com/AnhVan17/Audio-Visual-Speech-Recognition.git
cd Audio-Visual-Speech-Recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Cài Đặt Modal (Cloud)

```bash
# Install Modal
pip install modal

# Setup Modal authentication
modal setup

# Verify
modal run --help
```

---

## 📖 Sử Dụng

### 1. Preprocessing

```bash
# Local preprocessing
python -m src.data.preprocessing \
    --input_dir ./data/raw_tars \
    --output_dir ./data/processed_features \
    --manifest_dir ./data/manifests

# Modal preprocessing (recommended for large datasets)
modal run scripts/preprocessing_modal.py
```

### 2. Training

```bash
# Local training
python scripts/training.py --config configs/model/config.yaml

# Modal training (A100 GPU)
modal run scripts/training_modal.py
```

### 3. Evaluation

```python
from src.evaluation.evaluator import Evaluator
from src.data.tokenizer import VietnameseCharTokenizer

tokenizer = VietnameseCharTokenizer()
evaluator = Evaluator(tokenizer)

# Evaluate on validation set
results = evaluator.evaluate(model, val_loader, device)
print(f"WER: {results['wer']:.2f}%")
print(f"CER: {results['cer']:.2f}%")
```

---

## 📁 Cấu Trúc Thư Mục

```
ASVR_Viet/
├── configs/
│   └── model/
│       └── config.yaml          # Training configuration
├── data/
│   ├── raw_tars/                # Raw ViCocktail TAR files
│   ├── processed_features/      # Preprocessed .pt files
│   └── manifests/               # train/val/test JSONL
├── scripts/
│   ├── preprocessing_modal.py   # Modal preprocessing
│   ├── training.py              # Local training
│   └── training_modal.py        # Modal training
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset & DataLoader
│   │   ├── preprocessing.py     # Feature extraction pipeline
│   │   └── tokenizer.py         # Vietnamese tokenizer
│   ├── models/
│   │   ├── aurora_xt.py         # Main model (re-export)
│   │   └── fusion/
│   │       └── simple.py        # AURORA-XT implementation
│   ├── training/
│   │   ├── trainer.py           # Training loop
│   │   └── losses.py            # Hybrid CTC + CE loss
│   └── evaluation/
│       ├── evaluator.py         # WER/CER computation
│       └── metrics.py           # Evaluation metrics
├── checkpoints/                 # Saved models
├── requirements.txt
├── setup.py
└── README.MD
```

---

## 📊 Kết Quả

### Performance (ViCocktail Dataset)

| Epoch | WER (%) | CER (%) | Notes |
|-------|---------|---------|-------|
| 5 | ~37% | ~25% |  |
| 10 | ~32.45% | ~14.60% | |
| 20 | ~29.49% | ~13.54% | |

### Training Tips

1. **Batch Size**: 32 for A100, 16 for smaller GPUs
2. **Learning Rate**: 3e-4 with CosineAnnealing
3. **Gradient Clipping**: 5.0 (important for CTC stability)
4. **Mixed Precision**: Always enable for speed

---

## 📚 Resources

- **Project Wiki**: [Google Docs](https://docs.google.com/document/d/1THTPq162FbKcZJhWwS3jl_neUeOkKTPHO_uduup8pwc/edit)
- **Research Notes**: [Google Docs](https://docs.google.com/document/d/1B20ZbUj4u9k7BDbxh7bLlJPX6pzc6TnnRxBtMxdU1Kg/edit)
- **GitHub Issues**: [Bug Reports](https://github.com/AnhVan17/Audio-Visual-Speech-Recognition/issues)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributors
- **Nguyễn Phạm Anh Văn** - [@github_username](https://github.com/AnhVan17)
- **Võ Minh Thịnh** - [@github_username](https://github.com/finalFlash159)
