"""
ViCocktail AVSR Preprocessing - Modal Pipeline (Optimized Edition)
=================================================================
Đặc điểm:
1. Thông minh: Tự chọn file chưa xử lý, tự khôi phục folder cũ chưa có manifest.
2. Nhanh: Hiển thị tiến độ ngay lập tức, không đợi theo thứ tự.
3. Tiện lợi: Hỗ trợ flag --manifest-only để cập nhật dataset trong vài giây.
"""

import modal
import json
import random
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

APP_NAME = "vicocktail-preprocessing-v11"
VOLUME_NAME = "avsr-dataset-volume"
VOL_MOUNT_PATH = "/data"

# Paths
INPUT_DIR = f"{VOL_MOUNT_PATH}/raw_mirror"
OUTPUT_DIR = f"{VOL_MOUNT_PATH}/processed_features"
MANIFEST_DIR = f"{VOL_MOUNT_PATH}/manifests"

# Processing limits
MAX_TRAIN_TARS = 30
MAX_TEST_TARS = None
SAMPLES_PER_TAR = None

# Config
VAL_SPLIT_RATIO = 0.1
RANDOM_SEED = 42 
CHECKPOINT_EVERY_N = 100

# Model cache
MODEL_CACHE = modal.Volume.from_name("model-cache-avsr", create_if_missing=True)
MODEL_CACHE_PATH = "/cache"

# DOCKER IMAGE
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
        "libxext6", "libxrender1", "ffmpeg", "libsndfile1",
        "build-essential", "pkg-config",
        "libavcodec-dev", "libavformat-dev", "libavutil-dev",
        "libswscale-dev", "libavdevice-dev"
    )
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.36.2",
        "timm==0.9.12",
        "huggingface-hub==0.20.3",
        "tokenizers==0.15.0",
        "decord==0.6.0",
        "opencv-python-headless==4.9.0.80",
        "mediapipe==0.10.9",
        "soundfile==0.12.1",
        "webdataset==0.2.86",
        "av==11.0.0",
        "Pillow==10.2.0",
        "tqdm==4.66.1",
        "numpy<2"  
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Add src directory to image
image_with_src = image.add_local_dir(
    local_path=Path(__file__).parent.parent / "src",
    remote_path="/root/src"
)


@app.cls(
    image=image_with_src,
    volumes={VOL_MOUNT_PATH: volume, MODEL_CACHE_PATH: MODEL_CACHE},
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=7200,
    max_containers=40,
)
class DataProcessor:
    """Distributed preprocessing - uses shared PreprocessingPipeline"""
    
    @modal.enter()
    def initialize(self):
        import warnings
        import numpy as np
        warnings.filterwarnings('ignore')
        logging.getLogger('mediapipe').setLevel(logging.ERROR)
        
        # Add src to path
        sys.path.insert(0, "/root")
        from src.data.preprocessing import PreprocessingConfig, PreprocessingPipeline
        
        class ModalConfig(PreprocessingConfig):
            DEVICE = 'cuda'
        
        self.pipeline = PreprocessingPipeline(ModalConfig())
        print(f"✅ Worker initialized (Vocab: {self.pipeline.tokenizer.vocab_size})")
    
    @modal.method()
    def process_tar_file(self, tar_path: str, is_test: bool) -> Dict:
        """Process single TAR file"""
        import torch
        import webdataset as wds
        import gc
        
        tar_name = Path(tar_path).stem
        save_dir = Path(OUTPUT_DIR) / tar_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Check if a local manifest already exists (FASTEST)
        manifest_cache = save_dir / "manifest.jsonl"
        if manifest_cache.exists():
            metadata = []
            with open(manifest_cache, 'r', encoding='utf-8') as f:
                for line in f:
                    metadata.append(json.loads(line))
            return {"tar_name": tar_name, "is_test": is_test, "samples": metadata, "status": "cache"}

        # 2. Recovery: If .pt files exist but no manifest.jsonl
        existing = list(save_dir.glob("*.pt"))
        if len(existing) > 10:
            print(f"  📂 Recovering metadata for {tar_name} ({len(existing)} files)...")
            metadata = []
            for f in existing:
                try:
                    d = torch.load(f, map_location='cpu', weights_only=False)
                    metadata.append({
                        "id": d.get('id', f.stem),
                        "path": str(f.relative_to(Path(OUTPUT_DIR).parent)),
                        "text": d.get('text_raw', "")
                    })
                except: continue
            
            # Save the local manifest so next time it's instant
            with open(manifest_cache, 'w', encoding='utf-8') as f:
                for m in metadata:
                    f.write(json.dumps(m, ensure_ascii=False) + '\n')
            return {"tar_name": tar_name, "is_test": is_test, "samples": metadata, "status": "recovered"}
        
        # 3. Normal Processing
        print(f"  📦 Processing {tar_name}")
        dataset = wds.WebDataset(f"file://{tar_path}", shardshuffle=False).decode()
        stats = {"success": 0, "skip": {}}
        metadata = []
        
        for idx, sample in enumerate(dataset):
            if SAMPLES_PER_TAR and idx >= SAMPLES_PER_TAR: break
            sample_id = sample.get("__key__", f"{tar_name}_{idx:06d}")
            
            result, error = self.pipeline.process_sample(sample, sample_id)
            if result:
                save_data = {
                    'id': result['id'],
                    'audio': result['audio'].clone(),
                    'visual': result['visual'].clone(),
                    'text': result['text'].clone() if isinstance(result['text'], torch.Tensor) else result['text'],
                    'text_raw': result['text_raw']
                }
                out_file = save_dir / f"{sample_id}.pt"
                torch.save(save_data, out_file)
                
                item = {
                    "id": sample_id,
                    "path": str(out_file.relative_to(Path(OUTPUT_DIR).parent)),
                    "text": result['text_raw']
                }
                metadata.append(item)
                stats["success"] += 1
            else:
                stats["skip"][error] = stats["skip"].get(error, 0) + 1
            
            if idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            if idx % CHECKPOINT_EVERY_N == 0:
                volume.commit()
        
        # Save local manifest
        if metadata:
            with open(manifest_cache, 'w', encoding='utf-8') as f:
                for m in metadata:
                    f.write(json.dumps(m, ensure_ascii=False) + '\n')
        
        volume.commit()
        return {"tar_name": tar_name, "is_test": is_test, "samples": metadata, "stats": stats, "status": "processed"}


@app.function(image=image, volumes={VOL_MOUNT_PATH: volume})
def create_manifests(results_to_aggregate: List[Dict]):
    """Gom tất cả samples và tạo train/val/test.jsonl"""
    test_samples = []
    train_val_samples = []
    
    for res in results_to_aggregate:
        if res['is_test']:
            test_samples.extend(res['samples'])
        else:
            train_val_samples.extend(res['samples'])
    
    random.seed(RANDOM_SEED)
    random.shuffle(train_val_samples)
    split_idx = int(len(train_val_samples) * (1 - VAL_SPLIT_RATIO))
    train_samples = train_val_samples[:split_idx]
    val_samples = train_val_samples[split_idx:]
    
    manifest_dir = Path(MANIFEST_DIR)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    def save_manifest(samples, filename):
        with open(manifest_dir / filename, 'w', encoding='utf-8') as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  ✓ {filename}: {len(samples)}")
    
    save_manifest(train_samples, "train.jsonl")
    save_manifest(val_samples, "val.jsonl")
    save_manifest(test_samples, "test.jsonl")
    volume.commit()
    
    print(f"\n📊 Global Statistics:")
    print(f"   Train: {len(train_samples)}")
    print(f"   Val:   {len(val_samples)}")
    print(f"   Test:  {len(test_samples)}")


@app.function(image=image, volumes={VOL_MOUNT_PATH: volume})
def generate_master_manifest():
    """Quét toàn bộ volume và tạo manifest từ tất cả những gì đã làm được"""
    print("\n🔍 Scanning volume for all processed shards...")
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists(): 
        print("  ⚠️ Output directory not found.")
        return

    results_to_aggregate = []
    for shard_dir in output_path.iterdir():
        if shard_dir.is_dir():
            m_file = shard_dir / "manifest.jsonl"
            if m_file.exists():
                is_test = "test" in shard_dir.name.lower()
                samples = []
                with open(m_file, 'r', encoding='utf-8') as f:
                    for line in f: samples.append(json.loads(line))
                results_to_aggregate.append({"is_test": is_test, "samples": samples})
    
    if results_to_aggregate:
        create_manifests.local(results_to_aggregate)
    else:
        print("  ⚠️ No processed shards found.")


@app.function(image=image, volumes={VOL_MOUNT_PATH: volume})
def list_tar_files():
    """List TAR files và phân loại theo trạng thái xử lý"""
    import glob
    all_tars = sorted(glob.glob(f"{INPUT_DIR}/**/*.tar", recursive=True))
    processed = []
    unprocessed = []
    for t in all_tars:
        if (Path(OUTPUT_DIR) / Path(t).stem / "manifest.jsonl").exists():
            processed.append(t)
        else:
            unprocessed.append(t)
    return {"all": all_tars, "processed": processed, "unprocessed": unprocessed}


@app.function(image=image, volumes={VOL_MOUNT_PATH: volume, MODEL_CACHE_PATH: MODEL_CACHE})
def warmup_models():
    from transformers import WhisperModel
    import timm
    print("🔥 Caching models...")
    WhisperModel.from_pretrained("openai/whisper-small", cache_dir=MODEL_CACHE_PATH)
    timm.create_model('resnet18', pretrained=True)
    MODEL_CACHE.commit()


@app.local_entrypoint()
def main(manifest_only: bool = False):
    """
    Main Entrypoint:
    - Normal: modal run scripts/preprocessing_modal.py
    - Manifest Only: modal run scripts/preprocessing_modal.py --manifest-only
    """
    print(f"\n{'='*70}\n🚀 ViCocktail Preprocessing Pipeline\n{'='*70}")
    
    if not manifest_only:
        warmup_models.remote()
        print("\nStep 2: Scanning Files...")
        status = list_tar_files.remote()
        print(f"  Total: {len(status['all'])} | Done: {len(status['processed'])} | New: {len(status['unprocessed'])}")
        
        # Ưu tiên file chưa xử lý
        unprocessed_pool = status['unprocessed']
        train_pool = [f for f in unprocessed_pool if "train" in Path(f).name.lower()]
        test_pool = [f for f in unprocessed_pool if "test" in Path(f).name.lower()]
        
        # Nếu hết file mới, mới quay lại pool cũ
        if not train_pool: train_pool = [f for f in status['all'] if "train" in Path(f).name.lower()]
        if not test_pool: test_pool = [f for f in status['all'] if "test" in Path(f).name.lower()]
        
        train_tars = random.sample(train_pool, min(len(train_pool), MAX_TRAIN_TARS))
        test_tars = random.sample(test_pool, min(len(test_pool), MAX_TEST_TARS))
        
        tasks = [(t, False) for t in train_tars] + [(t, True) for t in test_tars]
        print(f"  Selected {len(tasks)} shards for this run.")
        
        print("\nStep 3: Processing...")
        processor = DataProcessor()
        for i, res in enumerate(processor.process_tar_file.map([t for t, _ in tasks], [it for _, it in tasks], order_outputs=False)):
            m_count = len(res.get('samples', []))
            skips = res.get('stats', {}).get('skip', {})
            skip_str = f" | Skips: {skips}" if skips else ""
            print(f"  ✨ [{i+1}/{len(tasks)}] {res['tar_name']}: {m_count} samples{skip_str} ({res['status']})")

    print("\nStep 4: Generating Master Manifest...")
    generate_master_manifest.remote()
    print(f"\n{'='*70}\n🎉 COMPLETE\n{'='*70}")
