"""
ViCocktail AVSR Preprocessing - Modal Pipeline
===============================================
Modal wrapper - uses shared PreprocessingPipeline
"""

import modal
import json
import random
import logging
import sys
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
MAX_TRAIN_TARS = 15
MAX_TEST_TARS = 3
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
        
        if np.__version__.startswith('2.'):
            raise RuntimeError(f"NumPy 2.x detected! Need 1.x.")
        
        # Add src to path
        sys.path.insert(0, "/root")
        
        # Import shared pipeline
        from src.data.preprocessing import PreprocessingConfig, PreprocessingPipeline
        
        # Config for Modal
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
        print(f"\n📦 {tar_name}")
        
        save_dir = Path(OUTPUT_DIR) / tar_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        existing = list(save_dir.glob("*.pt"))
        if len(existing) > 10:
            print(f"  ⏭️  Already processed ({len(existing)} files)")
            return {
                "tar_name": tar_name,
                "is_test": is_test,
                "samples": [{
                    "id": f.stem,
                    "path": str(f.relative_to(Path(OUTPUT_DIR).parent)),
                    "text": ""
                } for f in existing]
            }
        
        dataset = wds.WebDataset(f"file://{tar_path}", shardshuffle=False).decode()
        
        stats = {"success": 0, "skip": {}}
        metadata = []
        
        for idx, sample in enumerate(dataset):
            if SAMPLES_PER_TAR and idx >= SAMPLES_PER_TAR:
                break
            
            sample_id = sample.get("__key__", f"{tar_name}_{idx:06d}")
            
            # Use shared pipeline
            result, error = self.pipeline.process_sample(sample, sample_id)
            
            if result:
                # Clone tensors to prevent memory leaks
                save_data = {
                    'id': result['id'],
                    'audio': result['audio'].clone(),
                    'visual': result['visual'].clone(),
                    'text': result['text'].clone() if isinstance(result['text'], torch.Tensor) else result['text'],
                    'text_raw': result['text_raw']
                }
                
                out_file = save_dir / f"{sample_id}.pt"
                torch.save(save_data, out_file)
                
                metadata.append({
                    "id": sample_id,
                    "path": str(out_file.relative_to(Path(OUTPUT_DIR).parent)),
                    "text": result['text_raw']
                })
                stats["success"] += 1
            else:
                stats["skip"][error] = stats["skip"].get(error, 0) + 1
            
            if idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if idx % CHECKPOINT_EVERY_N == 0 and idx > 0:
                volume.commit()
        
        volume.commit()
        print(f"  ✅ {stats['success']} processed, {sum(stats['skip'].values())} skipped")
        
        return {"tar_name": tar_name, "is_test": is_test, "samples": metadata}


@app.function(image=image, volumes={VOL_MOUNT_PATH: volume})
def create_manifests(results: List[Dict]):
    """Generate train/val/test manifests"""
    
    test_samples = []
    train_val_samples = []
    
    for result in results:
        for sample in result['samples']:
            if result['is_test']:
                test_samples.append(sample)
            else:
                train_val_samples.append(sample)
    
    # Split train/val
    random.seed(RANDOM_SEED)
    random.shuffle(train_val_samples)
    split_idx = int(len(train_val_samples) * (1 - VAL_SPLIT_RATIO))
    train_samples = train_val_samples[:split_idx]
    val_samples = train_val_samples[split_idx:]
    
    # Save manifests
    manifest_dir = Path(MANIFEST_DIR)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    def save_manifest(samples, filename):
        with open(manifest_dir / filename, 'w', encoding='utf-8') as f:
            for item in samples:
                f.write(json.dumps({
                    "id": item['id'],
                    "path": item['path'],
                    "text": item['text']
                }, ensure_ascii=False) + '\n')
        print(f"  ✓ {filename}: {len(samples)}")
    
    save_manifest(train_samples, "train.jsonl")
    save_manifest(val_samples, "val.jsonl")
    save_manifest(test_samples, "test.jsonl")
    
    volume.commit()
    
    print(f"\n Statistics:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val:   {len(val_samples)}")
    print(f"  Test:  {len(test_samples)}")


@app.function(image=image, volumes={VOL_MOUNT_PATH: volume})
def list_tar_files():
    """List TAR files"""
    import glob
    return sorted(glob.glob(f"{INPUT_DIR}/**/*.tar", recursive=True))


@app.function(image=image, volumes={VOL_MOUNT_PATH: volume, MODEL_CACHE_PATH: MODEL_CACHE})
def warmup_models():
    """Pre-download models"""
    from transformers import WhisperModel
    import timm
    
    print("🔥 Caching models...")
    WhisperModel.from_pretrained("openai/whisper-small", cache_dir=MODEL_CACHE_PATH)
    timm.create_model('resnet18', pretrained=True)
    MODEL_CACHE.commit()
    print("✓ Done")


@app.local_entrypoint()
def main():
    """Main pipeline"""
    import time
    
    print(f"\n{'='*70}")
    print("🚀 ViCocktail Preprocessing (Modal)")
    print(f"{'='*70}\n")
    
    # Warmup
    print("Step 1: Caching models...")
    warmup_models.remote()
    
    # Scan TAR files
    print("\nStep 2: Scanning TAR files...")
    all_tars = list_tar_files.remote()
    
    if not all_tars:
        print(f"No TAR files in {INPUT_DIR}")
        return
    
    train_tars = [f for f in all_tars if "train" in Path(f).name.lower()]
    test_tars = [f for f in all_tars if "test" in Path(f).name.lower()]
    
    # Apply limits - use time-based seed for random TAR selection each run
    import time
    random.seed(int(time.time()))  # Different selection each run
    if MAX_TRAIN_TARS:
        train_tars = random.sample(train_tars, min(len(train_tars), MAX_TRAIN_TARS))
    if MAX_TEST_TARS:
        test_tars = random.sample(test_tars, min(len(test_tars), MAX_TEST_TARS))
    
    # Print selected TARs for tracking
    print(f"\n📋 Selected TARs (random seed: {int(time.time())})")
    for t in sorted(train_tars):
        print(f"   - {Path(t).name}")
    
    processing_list = [(t, False) for t in train_tars] + [(t, True) for t in test_tars]
    
    print(f"\n Processing {len(processing_list)} TARs")
    print(f"   Train: {len(train_tars)}")
    print(f"   Test:  {len(test_tars)}")
    
    # Process
    print(f"\nStep 3: Processing...")
    processor = DataProcessor()
    
    results = []
    for i, result in enumerate(processor.process_tar_file.starmap(processing_list, return_exceptions=True, wrap_returned_exceptions=False)):
        if isinstance(result, Exception):
            print(f"  Error in task {i}: {result}")
        else:
            print(f"  Finished: {result['tar_name']}")
            results.append(result)
    
    # Generate manifests
    if results:
        print("\nStep 4: Generating manifests...")
        create_manifests.remote(results)
    
    print(f"\n{'='*70}")
    print("🎉 COMPLETE")
    print(f"{'='*70}")
