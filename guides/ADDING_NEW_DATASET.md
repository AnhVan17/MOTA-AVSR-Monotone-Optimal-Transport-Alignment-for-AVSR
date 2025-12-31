# Hướng Dẫn Tích Hợp Dataset Mới

Tài liệu này cung cấp **quy tắc và framework chung** để tích hợp bất kỳ dataset AVSR nào vào hệ thống. Áp dụng cho mọi bộ data, không phụ thuộc vào cấu trúc cụ thể.

---

## Mục Lục

1.  [Phân Tích Dataset Mới](#1-phân-tích-dataset-mới)
2.  [Chuẩn Bị Data Trên Modal Volume](#2-chuẩn-bị-data-trên-modal-volume)
3.  [Implement Preprocessor](#3-implement-preprocessor)
4.  [Implement Dataset Class](#4-implement-dataset-class)
5.  [Đăng Ký Vào Factory Functions](#5-đăng-ký-vào-factory-functions)
6.  [Tạo Config File](#6-tạo-config-file)
7.  [Chạy Pipeline](#7-chạy-pipeline)
8.  [Template Code](#8-template-code)

---

## 1. Phân Tích Dataset Mới

**Trước khi code, BẮT BUỘC phải trả lời các câu hỏi sau:**

### 1.1 Cấu Trúc Thư Mục

```
❓ Data được tổ chức như thế nào?
   □ Flat (tất cả files trong 1 folder)
   □ Theo speaker (speaker_01/, speaker_02/, ...)
   □ Theo session/ngày
   □ Nested phức tạp

❓ Định dạng video?
   □ .mp4  □ .avi  □ .mkv  □ .webm  □ .mpg  □ Khác: _______
```

### 1.2 Transcript (Ground Truth)

```
❓ Transcript ở đâu?
   □ File riêng cùng tên với video (video1.mp4 → video1.txt)
   □ File JSON chứa tất cả transcripts
   □ File CSV/TSV
   □ Trong tên file
   □ Trong database/API

❓ Định dạng transcript?
   □ Plain text (1 dòng = 1 transcript)
   □ JSON: {"text": "...", "start": 0, "end": 5}
   □ Align format: "0 23000 word1\n23000 45000 word2"
   □ Khác: _______

❓ Ngôn ngữ?
   □ Tiếng Anh  □ Tiếng Việt  □ Đa ngôn ngữ
```

### 1.3 Đặc Điểm Video

```
❓ Video đã crop miệng chưa?
   □ Chưa (cần chạy stage CROP)
   □ Đã crop sẵn (bỏ qua stage CROP)

❓ Độ dài video trung bình?
   □ Ngắn (< 5 giây)  □ Trung bình (5-30 giây)  □ Dài (> 30 giây)
```

### 1.4 Ví Dụ Phân Tích

| Dataset | Cấu trúc | Video | Transcript | Đã crop? |
|---------|----------|-------|------------|----------|
| GRID | `speaker/video.mpg` | .mpg | .align (word timing) | ❌ |
| LRS2 | `main/video.mp4` | .mp4 | .txt (plain) | ❌ |
| ViCocktail | `flat/*.mp4` | .mp4 | .txt (plain) | ❌ |
| Custom | ??? | ??? | ??? | ??? |

---

## 2. Chuẩn Bị Data Trên Modal Volume

### 2.1 Quy Tắc Đặt Tên Thư Mục

```
/mnt/data/
├── {dataset_name}/                    # Raw data (KHÔNG SỬA)
├── {dataset_name}_cropped/            # Output stage CROP
├── processed_features/
│   └── {dataset_name}/                # Output stage EXTRACT (.pt files)
└── manifests/
    └── {dataset_name}_manifest.jsonl  # Manifest file
```

**Ví dụ:** Dataset tên `lrs2`
```
/mnt/data/lrs2/
/mnt/data/lrs2_cropped/
/mnt/data/processed_features/lrs2/
/mnt/data/manifests/lrs2_manifest.jsonl
```

### 2.2 Upload Data

```bash
# Option 1: Từ HuggingFace
modal run scripts/modal/download.py  # (sửa repo_id trong file)

# Option 2: Upload thủ công
modal volume put avsr-dataset-volume ./local_folder/ /mnt/data/{dataset_name}/

# Option 3: Mount local folder (dev only)
modal volume create avsr-dataset-volume --from-local ./data/
```

---

## 3. Implement Preprocessor

### 3.1 Quy Tắc Bắt Buộc

| Yêu Cầu | Mô Tả |
|---------|-------|
| Kế thừa `BasePreprocessor` | Đảm bảo có sẵn `VideoProcessor`, `AudioExtractor` |
| Implement `collect_metadata()` | Trả về list of dict với 4 keys bắt buộc |
| Sử dụng `normalize_text()` | Chuẩn hóa Unicode cho transcript |

### 3.2 Output của `collect_metadata()`

```python
# BẮT BUỘC trả về list of dict với format:
[
    {
        'id': str,           # ID duy nhất (thường là tên file không đuôi)
        'full_path': str,    # Đường dẫn tuyệt đối đến video
        'rel_path': str,     # Đường dẫn tương đối từ data_root
        'text': str          # Transcript đã normalize
    },
    ...
]
```

### 3.3 Logic Tìm Transcript

Tùy vào dataset, implement logic phù hợp:

```python
# Pattern 1: File .txt cùng tên
def _get_transcript(self, video_path):
    txt_path = video_path.replace('.mp4', '.txt')
    if os.path.exists(txt_path):
        return open(txt_path).read().strip()
    return ""

# Pattern 2: File .align (GRID format)
def _get_transcript(self, video_path):
    align_path = video_path.replace('.mpg', '.align')
    words = []
    for line in open(align_path):
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] not in ['sil', 'sp']:
            words.append(parts[2])
    return " ".join(words)

# Pattern 3: JSON file chung
def _get_transcript(self, video_path):
    video_id = os.path.basename(video_path).split('.')[0]
    return self.transcript_dict.get(video_id, "")

# Pattern 4: CSV/TSV
def _get_transcript(self, video_path):
    video_id = os.path.basename(video_path).split('.')[0]
    row = self.df[self.df['id'] == video_id]
    return row['text'].values[0] if len(row) > 0 else ""
```

### 3.4 File Location

```
src/data/preprocessors/
├── base.py           # Base class (KHÔNG SỬA)
├── grid.py           # GRID dataset
├── vicocktail.py     # ViCocktail dataset
└── {your_dataset}.py # ← TẠO MỚI
```

---

## 4. Implement Dataset Class

### 4.1 Trường Hợp Đơn Giản (Dùng FeatureDataset)

Nếu format `.pt` file của bạn giống chuẩn, chỉ cần:

```python
# src/data/datasets/{your_dataset}.py
from .base import FeatureDataset

class YourDataset(FeatureDataset):
    """Kế thừa hoàn toàn, không cần thay đổi gì."""
    pass
```

### 4.2 Trường Hợp Custom `.pt` Format

Nếu `.pt` file có format khác, override `__getitem__()`:

```python
class YourDataset(FeatureDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        pt_path = os.path.join(self.data_root, item['rel_path'])
        
        data = torch.load(pt_path)
        
        # Custom mapping từ format của bạn
        return {
            'audio': data['your_audio_key'],      # Phải là [T, 768]
            'visual': data['your_visual_key'],    # Phải là [T, 512]
            'target': self._tokenize(data['transcript']),
            'text': data['transcript'],
            'rel_path': item['rel_path']
        }
```

### 4.3 Chuẩn Output (Bắt Buộc)

```python
# __getitem__ PHẢI trả về dict với các keys sau:
{
    'audio': torch.Tensor,    # Shape: [T_audio, 768]
    'visual': torch.Tensor,   # Shape: [T_visual, 512]
    'target': torch.Tensor,   # Token IDs từ tokenizer
    'text': str,              # Raw text
    'rel_path': str           # Relative path
}
```

---

## 5. Đăng Ký Vào Factory Functions

### 5.1 Cập Nhật `preprocess.py`

```python
# scripts/modal/preprocess.py

# 1. Thêm vào DATA_CONFIG
DATA_CONFIG = {
    "grid": {...},
    # ========== THÊM MỚI ==========
    "{your_dataset}": {
        "raw": "/mnt/data/{your_dataset}",
        "cropped": "/mnt/data/{your_dataset}_cropped",
        "features": "/mnt/data/processed_features/{your_dataset}",
        "manifest": "/mnt/data/manifests/{your_dataset}_manifest.jsonl"
    }
}

# 2. Thêm vào get_preprocessor()
def get_preprocessor(dataset_name, data_root, use_precropped=False):
    if dataset_name == "grid":
        from src.data.preprocessors.grid import GridPreprocessor
        return GridPreprocessor(data_root, use_precropped)
    # ========== THÊM MỚI ==========
    elif dataset_name == "{your_dataset}":
        from src.data.preprocessors.{your_dataset} import YourPreprocessor
        return YourPreprocessor(data_root, use_precropped)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
```

### 5.2 Cập Nhật `loader.py`

```python
# src/data/loader.py

from .datasets.{your_dataset} import YourDataset  # ← THÊM IMPORT

def build_dataloader(config, tokenizer, mode="train"):
    dataset_name = config.get('dataset_name', 'grid')
    
    # ========== THÊM CASE MỚI ==========
    if dataset_name == "{your_dataset}":
        DatasetClass = YourDataset
    elif dataset_name == "grid":
        DatasetClass = GridDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset = DatasetClass(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        data_root=config['data_root'],
        ...
    )
```

---

## 6. Tạo Config File

### 6.1 Template

```yaml
# configs/{your_dataset}_phase1.yaml

model:
  audio_dim: 768
  visual_dim: 512
  d_model: 256
  num_encoder_layers: 6
  num_decoder_layers: 4
  vocab_size: 51865
  use_mqot: false

data:
  dataset_name: {your_dataset}                                    # ← THAY ĐỔI
  train_manifest: /mnt/data/manifests/{your_dataset}_manifest.jsonl
  val_manifest: /mnt/data/manifests/{your_dataset}_manifest.jsonl
  data_root: /mnt/data/processed_features/{your_dataset}
  batch_size: 16
  num_workers: 4
  use_precomputed_features: true

training:
  num_epochs: 20
  learning_rate: 3.0e-4
  gradient_clip: 5.0
  checkpoint_dir: /mnt/checkpoints/{your_dataset}_phase1

loss:
  ctc_weight: 0.3
  ce_weight: 0.7
```

### 6.2 Các Tham Số Cần Điều Chỉnh

| Tham Số | Khi Nào Thay Đổi |
|---------|------------------|
| `batch_size` | Giảm nếu video dài, tăng nếu video ngắn |
| `vocab_size` | Thay đổi nếu dùng tokenizer khác |
| `learning_rate` | Giảm nếu training không ổn định |

---

## 7. Chạy Pipeline

### 7.1 Commands

```bash
# Step 1: Crop (bỏ qua nếu data đã crop)
modal run scripts/modal/preprocess.py --stage crop --dataset {your_dataset}

# Step 2: Extract features
modal run scripts/modal/preprocess.py --stage extract --dataset {your_dataset}

# Step 3: Verify
modal run scripts/modal/check_volume.py --path /mnt/data/processed_features/{your_dataset}

# Step 4: Train
modal run scripts/modal/train_phase1.py --config configs/{your_dataset}_phase1.yaml
```

### 7.2 Debug Checklist

- [ ] `collect_metadata()` trả về đúng format?
- [ ] Transcript được đọc đúng?
- [ ] Paths trong config đúng?
- [ ] Dataset được import đúng trong `loader.py`?

---

## 8. Template Code

### 8.1 Preprocessor Template

```python
# src/data/preprocessors/{your_dataset}.py
import os
import glob
from .base import BasePreprocessor
from src.utils.logging_utils import setup_logger
from src.utils.text_cleaning import normalize_text

logger = setup_logger(__name__)


class YourPreprocessor(BasePreprocessor):
    """Preprocessor cho {Your Dataset Name}."""
    
    def collect_metadata(self):
        logger.info(f"[{self.__class__.__name__}] Scanning {self.data_root}...")
        
        # 1. Tìm video files (điều chỉnh extensions theo dataset)
        video_files = glob.glob(
            os.path.join(self.data_root, "**", "*.mp4"), 
            recursive=True
        )
        logger.info(f"Found {len(video_files)} videos")
        
        # 2. Build metadata
        results = []
        for video_path in video_files:
            text = self._get_transcript(video_path)
            results.append({
                'id': os.path.splitext(os.path.basename(video_path))[0],
                'full_path': video_path,
                'rel_path': os.path.relpath(video_path, self.data_root),
                'text': normalize_text(text)
            })
        
        return results
    
    def _get_transcript(self, video_path):
        """
        TODO: Implement logic đọc transcript phù hợp với dataset.
        """
        # Ví dụ: file .txt cùng tên
        txt_path = os.path.splitext(video_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return ""
```

### 8.2 Dataset Template

```python
# src/data/datasets/{your_dataset}.py
from .base import FeatureDataset
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class YourDataset(FeatureDataset):
    """Dataset cho {Your Dataset Name}."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(f"Initialized with {len(self)} samples")
```

---

## Tham Khảo

| File | Mô Tả |
|------|-------|
| `src/data/preprocessors/base.py` | Base class với VideoProcessor, AudioExtractor |
| `src/data/preprocessors/grid.py` | Ví dụ cho GRID dataset |
| `src/data/datasets/base.py` | FeatureDataset base class |
| `scripts/modal/preprocess.py` | Pipeline orchestration |
