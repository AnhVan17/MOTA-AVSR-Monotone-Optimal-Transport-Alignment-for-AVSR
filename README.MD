# AURORA-XT: Audio-Visual Speech Recognition (AVSR)

## Optimized Pipeline with Whisper Subword Tokenizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Modal](https://img.shields.io/badge/deploy-Modal-blueviolet)](https://modal.com/)

**🎯 Mục Tiêu**: Tăng độ chính xác ASR từ **55% WER (test)** lên **18-25% WER** bằng cách chuyển từ character-level sang subword tokenization.

---

## 📋 Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Kiến Trúc](#kiến-trúc)
- [Yêu Cầu](#yêu-cầu)
- [Cài Đặt](#cài-đặt)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Kết Quả](#kết-quả)
- [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

---

## 🎯 Giới Thiệu

**AURORA-XT** là hệ thống nhận dạng giọng nói đa phương thức (Audio-Visual Speech Recognition) cho tiếng Việt, sử dụng:

- **Audio Features**: Whisper Encoder (frozen) → `[T, 768]`
- **Visual Features**: ResNet18 → Mouth ROI → `[T, 512]`
- **Tokenizer**: Whisper Multilingual Subword (vocab = 51,865)
- **Decoder**: Pure Attention (NO CTC)

### **Cải Tiến Chính So Với Baseline**

| Thành Phần   | Baseline        | AURORA-XT Final           |
| -------------- | --------------- | ------------------------- |
| Tokenizer      | Character (121) | Whisper Subword (51,865)  |
| Audio Input    | Mel [80, 3000]  | Whisper Features [T, 768] |
| Decoder        | CTC + Attention | **Pure Attention**  |
| d_model        | 256             | **512**             |
| Decoder Layers | 4               | **6**               |
| Test WER       | 55% ❌          | **18-25%** ✅       |

---

## 🏗️ Kiến Trúc

### **Pipeline Tổng Quan**

```
RAW VIDEO (.mp4)
    │
    ├─→ [Audio Extraction] → Waveform [16kHz]
    │       ↓
    │   [Whisper Encoder] → Audio Features [T, 768]
    │
    ├─→ [Video Frames] → Mouth ROI Detection
    │       ↓
    │   [ResNet18] → Visual Features [T_v, 512]
    │
    └─→ [WhisperTokenizer] → Token IDs [L]
             ↓
        .pt FILE SAVED
             ↓
    ┌────────────────────────┐
    │   AURORA-XT MODEL      │
    ├────────────────────────┤
    │ 1. Audio Proj  [768→512]│
    │ 2. Visual Proj [512→512]│
    │ 3. Quality Gate         │
    │ 4. Conformer (6 layers) │
    │ 5. Attention Decoder (6)│
    └────────────────────────┘
             ↓
        PREDICTIONS
```

### **Model Architecture**

```python
AuroraXT(
  # Input Projections
  audio_proj: Linear(768 -> 512)
  visual_proj: Linear(512 -> 512)
  
  # Adaptive Fusion
  quality_gate: QualityGate()
  
  # Encoder: 6x Conformer Blocks
  encoder: ModuleList(
    ConformerBlock(d=512, heads=8, kernel=31)
    × 6
  )
  
  # Decoder: Pure Attention (NO CTC)
  decoder: AttentionOnlyDecoder(
    target_embedding: Embedding(51865, 512)
    pos_encoding: Hybrid (learned + sinusoidal)
    transformer: 6 layers
    output_proj: Linear(512 -> 51865)
  )
)

Total Parameters: ~120M
```

---

## 💻 Yêu Cầu

### **Phần Cứng**

- **Preprocessing**: GPU T4+ (16GB VRAM)
- **Training**: GPU A100-40GB (hoặc A100-80GB)
- **Disk**: ~500GB cho raw data + processed features

### **Phần Mềm**

- Python 3.10+
- CUDA 11.8+
- Modal CLI
- ffmpeg, libsndfile1

---

## 🚀 Cài Đặt

### **1. Clone Repository**

```bash
git clone https://github.com/AnhVan17/Audio-Visual-Speech-Recognition.git
cd Audio-Visual-Speech-Recognition
```

### **2. Cài Đặt Dependencies (Local)**

```bash
# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### **3. Setup Modal**

```bash
# Cài đặt Modal CLI
pip install modal

# Authenticate
modal token new

# Test connection
modal run scripts/hello_modal.py
```

### **4. Upload Data lên Modal Volume**

```bash
# Tạo volume
modal volume create avsr-dataset-volume

# Upload raw data
modal volume put avsr-dataset-volume local_data/ /raw_mirror
```

---

## 🔧 Preprocessing

### **Chức Năng**

Preprocessing pipeline thực hiện:

1. **Audio Processing**

   - Extract waveform (16kHz mono)
   - Pass qua Whisper Encoder (frozen)
   - Output: `[T, 768]` features
2. **Visual Processing**

   - Detect mouth ROI (MediaPipe Face Mesh)
   - Resize to 96x96
   - Extract ResNet18 features
   - Output: `[T, 512]`
3. **Text Tokenization**

   - WhisperProcessor multilingual
   - Subword tokenization
   - Output: `[L]` token IDs

### **Chạy Preprocessing Trên Modal**

```bash
# Full pipeline
modal run scripts/preprocessing_modal.py

# Config trong script:
# - MAX_TRAIN_TARS = 25  # Số TAR files train
# - MAX_TEST_TARS = 5    # Số TAR files test
# - SAMPLES_PER_TAR = None  # Giới hạn samples/TAR (None = all)
```

### **Output Format**

Mỗi sample được lưu thành `.pt` file:

```python
torch.load('sample_000001.pt')
# {
#     'id': 'sample_000001',
#     'audio': Tensor[T, 768],      # Whisper features
#     'visual': Tensor[T_v, 512],   # ResNet features  
#     'tokens': Tensor[L],          # Token IDs (subword)
#     'text': 'xin chào các bạn'    # Raw text
# }
```

### **Manifests**

Sau preprocessing, 3 manifest files được tạo:

```
/data/manifests/
  ├── train.jsonl   # Training set
  ├── val.jsonl     # Validation set (10%)
  └── test.jsonl    # Test set
```

Mỗi dòng trong manifest:

```json
{"id": "sample_001", "path": "processed_features/tar_name/sample_001.pt", "text": "..."}
```

---

## 🎓 Training

### **Hyperparameters**

```yaml
model:
  vocab_size: 51865        # Whisper multilingual
  d_model: 512             # Hidden dimension
  num_encoder_layers: 6    # Conformer blocks
  num_decoder_layers: 6    # Transformer decoder
  num_heads: 8
  dropout: 0.2

training:
  num_epochs: 30
  learning_rate: 1e-4      # Conservative
  batch_size: 24           # Per GPU
  accumulation_steps: 6    # Effective batch = 144
  label_smoothing: 0.15
  warmup_ratio: 0.15
  gradient_clip: 1.0
  weight_decay: 0.05

loss:
  ctc_weight: 0.0          # ❌ NO CTC!
  ce_weight: 1.0           # ✅ Pure Attention
```

### **Chạy Training Trên Modal**

```bash
# Single command
modal run scripts/training_modal.py --config-path configs/model/config.yaml

# Với WandB logging (optional)
# 1. Tạo Modal secret: modal secret create wandb-secret WANDB_API_KEY=<your-key>
# 2. Set use_wandb: true trong config.yaml
```

### **Training Output**

```
 Epoch 1: 100%|███████| 1338/1338 [45:23<00:00]
 📊 Epoch 1: WER=32.45% loss=2.3421
 💾 Saved: /checkpoints/best_model.pt

 Epoch 2: 100%|███████| 1338/1338 [45:18<00:00]
 📊 Epoch 2: WER=28.12% loss=2.1032
 💾 Saved: /checkpoints/best_model.pt
 
 ...
 
 🎉 Training Complete!
    Best WER: 18.73%
    Total Steps: 80,280
```

### **Checkpoints**

```
/checkpoints/
  ├── best_model.pt          # Model với WER thấp nhất
  ├── final_model.pt         # Model ở epoch cuối
  ├── checkpoint_epoch_5.pt  # Periodic checkpoints
  ├── checkpoint_epoch_10.pt
  └── ...
```

---

## 🔮 Inference

### **Local Inference**

```python
from src.models.fusion.aurora_xt_model import create_model
from src.data.tokenizers.whisper import WhisperProcessor
import torch

# Load model
config = {...}  # Load từ config.yaml
model = create_model(config)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
processor = WhisperProcessor(model_name="openai/whisper-small", language="vi")

# Inference
audio_feat = ...  # [T, 768] from Whisper encoder
visual_feat = ... # [T, 512] from ResNet18

with torch.no_grad():
    output = model.generate(
        audio_feat.unsqueeze(0),
        visual_feat.unsqueeze(0),
        max_len=150,
        bos_token_id=processor.bos_token_id,
        eos_token_id=processor.eos_token_id
    )

# Decode
tokens = output['tokens'][0].tolist()
text = processor.decode(tokens, skip_special_tokens=True)
print(f"Transcription: {text}")
```

---

## 📊 Kết Quả

### **Tại Sao Cải Thiện?**

1. **Subword Tokenization**

   - Xử lý tốt hơn từ vựng OOV (out-of-vocabulary)
   - Tổng quát hóa tốt hơn sang test set
2. **No CTC**

   - CTC: Alignment frame-to-character (không hoạt động với subword)
   - Pure Attention: Linh hoạt hơn, học được context tốt hơn
3. **Larger Model**

   - d_model=512, 6 decoder layers → Capacity cao hơn
   - Better language modeling
4. **Whisper Features**

   - Pre-trained trên 680k hours multilingual data
   - Rich acoustic representation

---

## 📁 Cấu Trúc Thư Mục

```
ASVR_Viet/
├── configs/
│   └── model/
│       └── config.yaml           # Training hyperparameters
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # Feature extraction pipeline
│   │   ├── dataset.py            # PyTorch Dataset & DataLoader
│   │   └── tokenizers/
│   │       └── whisper.py        # WhisperProcessor wrapper
│   ├── models/
│   │   ├── aurora_xt.py          # Model factory
│   │   └── fusion/
│   │       └── aurora_xt_model.py # Complete model architecture
│   ├── training/
│   │   ├── trainer.py            # Training loop
│   │   └── losses.py             # AttentionOnlyLoss
│   └── evaluation/
│       └── metrics.py            # WER/CER evaluator
├── scripts/
│   ├── preprocessing_modal.py    # Modal preprocessing wrapper
│   ├── training_modal.py         # Modal training wrapper
│   └── validate_pipeline.py      # Quick validation
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── OPTIMIZATION_REPORT.md        # Detailed optimization guide
└── README.md                     # This file
```

---

## 📚 Tài Liệu Tham Khảo

### **Papers**

1. Whisper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
2. Conformer: [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
3. CTC: [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

### **Datasets**

- [ViCocktail](https://github.com/nhannt315/ViCocktail): Vietnamese Audio-Visual Speech Dataset

### **Related Projects**

- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Modal](https://modal.com/)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributors

- **Nguyễn Phạm Anh Văn** - [@AnhVan17](https://github.com/AnhVan17)
- **Võ Minh Thịnh** - [@finalFlash159](https://github.com/finalFlash159)

## ⭐ Acknowledgments

- OpenAI Whisper team for the pre-trained models
- Modal team for the serverless infrastructure
- HuggingFace for the transformers library
- ViCocktail dataset creators

---
