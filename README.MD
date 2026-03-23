# MOTA: Multi-modal Optimal Transport Alignment for AVSR

An Audio-Visual Speech Recognition system with two-phase training:
- **Phase 1**: QualityGate for coarse fusion
- **Phase 2**: MQOT for fine-grained alignment

---

## Table of Contents

1.  [Architecture](#1-architecture)
2.  [Feature Extractors](#2-feature-extractors)
3.  [Preprocessing](#3-preprocessing)
4.  [Training](#4-training)
5.  [Fusion](#5-fusion)
6.  [Formulas](#6-formulas)
7.  [Project Structure](#7-project-structure)
8.  [Usage](#8-usage)

---

## 1. Architecture

### 1.1 System Overview

```mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'fontSize': '14px'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart TB
    V["&emsp;&emsp;Video Input&emsp;&emsp;"] --> CROP["&emsp;&emsp;Mouth Crop&emsp;&emsp;"]
    V --> AUDIO["&emsp;&emsp;Audio Extract&emsp;&emsp;"]
    CROP --> RES["&emsp;&emsp;ResNet18&emsp;&emsp;"]
    AUDIO --> WHIS["&emsp;&emsp;Whisper Encoder&emsp;&emsp;"]
    RES --> PT["&emsp;&emsp;.pt Feature File&emsp;&emsp;"]
    WHIS --> PT
    PT --> QG["&emsp;&emsp;QualityGate&emsp;&emsp;"]
    QG --> MQ["&emsp;&emsp;MQOT Layer&emsp;&emsp;"]
    MQ --> ENC["&emsp;&emsp;Conformer Encoder&emsp;&emsp;"]
    ENC --> DEC["&emsp;&emsp;Hybrid Decoder&emsp;&emsp;"]
    DEC --> OUT["&emsp;&emsp;Transcript Output&emsp;&emsp;"]
```

### 1.2 Data Flow

| Stage | Input | Output | Tool |
|-------|-------|--------|------|
| Crop | Video | Mouth ROI | FaceAlignment |
| Audio | Video | Waveform | FFmpeg |
| Audio Feat | Waveform | T x 768 | Whisper |
| Visual Feat | Frames | T x 512 | ResNet18 |
| Fusion | A + V | T x 256 | QualityGate |
| Decode | Features | Logits | Conformer |

---

## 2. Feature Extractors

### 2.1 Whisper (Audio)

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart LR
    W1["&emsp;Waveform&emsp;"] --> W2["&emsp;Mel Spectrogram&emsp;"]
    W2 --> W3["&emsp;Conv1D Layers&emsp;"]
    W3 --> W4["&emsp;Transformer x12&emsp;"]
    W4 --> W5["&emsp;Features 1500x768&emsp;"]
```

**Specs:**

| Item | Value |
|------|-------|
| Model | whisper-small |
| Input | 16kHz, 15s |
| Mel | 80 x 3000 |
| Output | 1500 x 768 |

### 2.2 ResNet18 (Visual)

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart LR
    R1["&emsp;Frame 88x88&emsp;"] --> R2["&emsp;Conv 7x7&emsp;"]
    R2 --> R3["&emsp;MaxPool 3x3&emsp;"]
    R3 --> R4["&emsp;ResBlocks x4&emsp;"]
    R4 --> R5["&emsp;GlobalAvgPool&emsp;"]
    R5 --> R6["&emsp;512-dim Output&emsp;"]
```

**Specs:**

| Item | Value |
|------|-------|
| Model | resnet18 |
| Input | 88 x 88 RGB |
| Output | 512-dim |

---

## 3. Preprocessing

### 3.1 Pipeline Stages

**Stage 1: CROP**

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart LR
    A1["&emsp;Read Frames&emsp;"] --> A2["&emsp;Detect Face&emsp;"]
    A2 --> A3["&emsp;Crop Mouth&emsp;"]
    A3 --> A4["&emsp;Save Video&emsp;"]
```

**Stage 2: EXTRACT**

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart LR
    B1["&emsp;Extract Audio&emsp;"] --> B2["&emsp;Whisper Encode&emsp;"]
    B3["&emsp;Load Frames&emsp;"] --> B4["&emsp;ResNet Encode&emsp;"]
    B2 --> B5["&emsp;Combine Features&emsp;"]
    B4 --> B5
    B5 --> B6["&emsp;Save .pt File&emsp;"]
```

### 3.2 Mouth Detection

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 20, 'rankSpacing': 40}}}%%
flowchart LR
    M1["&emsp;BGR Frame&emsp;"] --> M2["&emsp;To RGB&emsp;"]
    M2 --> M3["&emsp;FaceMesh&emsp;"]
    M3 --> M4["&emsp;Landmarks&emsp;"]
    M4 --> M5["&emsp;Calc Center&emsp;"]
    M5 --> M6["&emsp;Crop ROI&emsp;"]
    M6 --> M7["&emsp;88x88 Output&emsp;"]
```

**Landmark IDs:** 13, 14, 61, 291, 78, 308, 324, 17

**Formulas:**
```
cx = mean(x_coords)
cy = mean(y_coords)
r = max(width, height) * 0.9
```

**Optimization:** Detect every 5 frames, reuse bbox.

### 3.3 Audio Extraction

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart TB
    V["&emsp;Video File&emsp;"] --> P1["&emsp;PyAV Method&emsp;"]
    V --> P2["&emsp;FFmpeg Fallback&emsp;"]
    P1 --> N["&emsp;RMS Normalize&emsp;"]
    P2 --> N
    N --> PAD["&emsp;Pad to 15 seconds&emsp;"]
    PAD --> OUT["&emsp;Waveform Output&emsp;"]
```

**FFmpeg Command:**
```bash
ffmpeg -i input -ar 16000 -ac 1 -f wav pipe:1
```

---

## 4. Training

### 4.1 Training Flow

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart LR
    D1["&emsp;Dataset&emsp;"] --> D2["&emsp;Collate&emsp;"]
    D2 --> D3["&emsp;DataLoader&emsp;"]
    D3 --> M["&emsp;MOTA Model&emsp;"]
    M --> L1["&emsp;CE Loss&emsp;"]
    M --> L2["&emsp;CTC Loss&emsp;"]
    L1 --> L3["&emsp;Hybrid Loss&emsp;"]
    L2 --> L3
    L3 --> G["&emsp;Gradient Clip&emsp;"]
    G --> O["&emsp;AdamW Optimizer&emsp;"]
```

### 4.2 Two Phases

**Phase 1: Baseline Training**

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart LR
    P1A["&emsp;QualityGate&emsp;"] --> P1B["&emsp;Conformer Encoder&emsp;"]
    P1B --> P1C["&emsp;Hybrid Decoder&emsp;"]
```

**Phase 2: MQOT Refinement**

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart LR
    P2A["&emsp;Freeze QualityGate&emsp;"] --> P2B["&emsp;Train MQOT Layer&emsp;"]
    P2B --> P2C["&emsp;Fine-tune All&emsp;"]
```

**Phase 1 Config:**
- use_mqot = false
- lr = 3e-4
- batch = 32

**Phase 2 Config:**
- use_mqot = true
- lr = 1e-4
- batch = 16

---

## 5. Fusion

### 5.1 QualityGate

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart LR
    A["&emsp;Audio Input&emsp;"] --> AP["&emsp;Project to 256&emsp;"]
    V["&emsp;Visual Input&emsp;"] --> VP["&emsp;Project to 256&emsp;"]
    AP --> QA["&emsp;Quality Score&emsp;"]
    VP --> QV["&emsp;Quality Score&emsp;"]
    QA --> W["&emsp;Weighted Sum&emsp;"]
    QV --> W
    AP --> W
    VP --> W
    W --> F["&emsp;Fused Output&emsp;"]
```

**Formula:**
```
q_a = sigmoid(MLP(audio))
q_v = sigmoid(MLP(visual))
alpha = q_a / (q_a + q_v)
fused = alpha * audio + (1-alpha) * visual
```

### 5.2 MQOT Layer

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart TB
    A["&emsp;Audio Features&emsp;"] --> SIM["&emsp;Compute Similarity&emsp;"]
    V["&emsp;Visual Features&emsp;"] --> SIM
    SIM --> C["&emsp;Cost Matrix&emsp;"]
    T["&emsp;Temporal Penalty&emsp;"] --> C
    Q["&emsp;Quality Penalty&emsp;"] --> C
    C --> K["&emsp;Kernel K&emsp;"]
    K --> SK["&emsp;Sinkhorn Iterations&emsp;"]
    SK --> P["&emsp;Transport Map&emsp;"]
```

**Cost Matrix:**
```
C = -similarity + λ_time * temporal + λ_quality * quality
```

**Params:**
- λ_time = 0.5
- λ_quality = 5.0
- epsilon = 0.15
- iterations = 20

---

## 6. Formulas

### 6.1 Hybrid Loss

```
L = α * L_CTC + β * L_CE
```

| Epoch | α | β |
|-------|---|---|
| 1-5 | 0.7 | 0.3 |
| 6-15 | 0.5 | 0.5 |
| 16+ | 0.3 | 0.7 |

### 6.2 CTC Loss

```
L_CTC = -log P(Y|X)
P(Y|X) = sum over alignments
```

### 6.3 Sinkhorn Algorithm

```
K = exp(-C / ε)

repeat n times:
    u = 1 / (K @ v)
    v = 1 / (K.T @ u)

P = diag(u) @ K @ diag(v)
```

### 6.4 RMS Normalization

```
RMS = sqrt(mean(x²))
x_norm = x * (0.1 / RMS)
```

---

## 7. Project Structure

```
├── configs/
│   ├── phase1_base.yaml
│   └── phase2_mqot.yaml
├── scripts/modal/
│   ├── preprocess.py
│   ├── train_phase1.py
│   └── train_phase2.py
├── src/
│   ├── data/
│   │   ├── preprocessors/
│   │   ├── datasets/
│   │   ├── loader.py
│   │   └── collate.py
│   ├── models/
│   │   ├── layers/
│   │   ├── fusion/
│   │   └── mota.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── losses.py
│   └── evaluation/
└── guides/
    └── ADDING_NEW_DATASET.md
```

---

## 8. Usage

### 8.1 Install

```bash
pip install torch transformers timm jiwer face-alignment
pip install modal && modal token new
```

### 8.2 Preprocess

```bash
# Crop mouth regions
modal run scripts/modal/preprocess.py --stage crop --dataset grid

# Extract features
modal run scripts/modal/preprocess.py --stage extract --dataset grid
```

### 8.3 Train

```bash
# Phase 1: Baseline
modal run scripts/modal/train_phase1.py

# Phase 2: MQOT
modal run scripts/modal/train_phase2.py
```

### 8.4 Config

| Param | P1 | P2 |
|-------|----|----| 
| use_mqot | false | true |
| batch | 32 | 16 |
| lr | 3e-4 | 1e-4 |

---

## References

- [Adding New Dataset](guides/ADDING_NEW_DATASET.md)
