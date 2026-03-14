import modal
import os
import sys
import yaml
from pathlib import Path

# --- Config ---
APP_NAME = "avsr-train-SUPERSTRICT-v7"  # Super strict vocab ~2-4k tokens
VOLUME_PROCESSED = "avsr-vicocktail-processed" 

# --- Image Definition (Robust Numpy Fix) ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1")
    # 1. Force Uninstall Numpy
    .run_commands("pip uninstall -y numpy || true")
    # 2. Install numpy<2 FIRST along with Torch
    .pip_install(
        "numpy==1.26.4",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        extra_index_url="https://download.pytorch.org/whl/cu118"
    )
    # 3. Install other deps
    .pip_install(
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "soundfile==0.12.1",
        "wandb", 
        "pyyaml",
        "jiwer"  
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"})
    # Be sure to include src and configs (copy=True needed for build steps)
    .add_local_dir("configs", remote_path="/root/configs", copy=True)
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("scripts", remote_path="/root/scripts", copy=True)
    # 🔧 NEW: Generate Pruned Vocab INSIDE image
    # Now this works because files are physically copied into the image layer
    .run_commands("python /root/scripts/prune_whisper_vocab.py")
)

app = modal.App(APP_NAME)
vol_processed = modal.Volume.from_name(VOLUME_PROCESSED, create_if_missing=True)

# Mount paths
MOUNT_PROCESSED = "/mnt/processed"
CONFIG_PATH = "/root/configs/vicocktail_phase1.yaml"


@app.function(
    image=image,
    volumes={MOUNT_PROCESSED: vol_processed},
    gpu="A100-40GB",        
    timeout=86400,      
    secrets=[modal.Secret.from_name("wandb-secret")] 
)
def train_remote():
    sys.path.append("/root")
    import torch
    import json
    from src.training.trainer import Trainer
    from src.utils.logging_utils import setup_logger
    
    logger = setup_logger("Train:ViCocktail:Phase1")
    
    # ================================================================================
    # 🔍 VERIFICATION: Check if FIXED code is deployed
    # ================================================================================
    print("\n" + "="*80)
    print("🔍 VERIFYING FIXED CODE DEPLOYMENT")
    print("="*80)
    
    # 1. Check vicocktail.py dataset file
    with open('/root/src/data/datasets/vicocktail.py', 'r') as f:
        content = f.read()
        if 'def _tokenize' in content and 'encode(' in content:
            print("❌ CRITICAL: vicocktail.py has OLD _tokenize() method!")
            print("   Modal image needs to be rebuilt with fixed files!")
            return
        else:
            print("✅ vicocktail.py is correct (no _tokenize override)")
    
    # 2. Check tokenizer has encode_for_ctc AND uses pruned vocab
    from src.data.tokenizers.whisper import WhisperTokenizer
    tok = WhisperTokenizer(use_pruned_vocab=True)  # CRITICAL: Use pruned!
    
    if hasattr(tok, 'encode_for_ctc'):
        print("✅ Tokenizer has encode_for_ctc()")
    else:
        print("❌ Tokenizer missing encode_for_ctc()!")
        return
    
    # 2b. CRITICAL: Check if pruned vocab is active
    print(f"\n🔍 VOCAB CHECK:")
    print(f"   Mode: {'PRUNED' if tok.use_pruned_vocab else 'FULL (BAD!)'}")
    print(f"   Vocab size: {tok.vocab_size}")
    
    if not tok.use_pruned_vocab or tok.vocab_size > 10000:
        print("❌ CRITICAL: Vocab pruning NOT active!")
        print("   Model will predict foreign tokens (Korean, Russian, etc.)")
        print("   Check if id_mapping.pkl exists in /root/src/data/vocab_pruned/")
        
        # Check if file exists
        mapping_path = "/root/src/data/vocab_pruned/id_mapping.pkl"
        if os.path.exists(mapping_path):
            print(f"   ✅ Mapping file exists: {mapping_path}")
        else:
            print(f"   ❌ Mapping file NOT found: {mapping_path}")
        return
    else:
        print(f"   ✅ Pruned vocab active: {tok.vocab_size} tokens")
    
    # 3. Test tokenization with Vietnamese text
    test_tokens = tok.encode_for_ctc("xin chào việt nam")
    print(f"\n✅ Test tokens for 'xin chào việt nam': {test_tokens}")
    
    if test_tokens and max(test_tokens) >= tok.vocab_size:
        print(f"❌ Token ID exceeds vocab size! Max: {max(test_tokens)}, Vocab: {tok.vocab_size}")
        return
    else:
        print(f"✅ Token range is valid: [{min(test_tokens)}, {max(test_tokens)}]")
    
    # 4. Check manifest has text
    manifest_path = f"{MOUNT_PROCESSED}/manifests/train.jsonl"
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            text = sample.get('text', '')
            print(f"✅ First manifest sample text: '{text[:100]}...'")
            
            if not text:
                print("❌ CRITICAL: Text is EMPTY in manifest!")
                print("   Need to re-run preprocessing to fix transcripts")
                return
            else:
                # Tokenize the actual sample text
                sample_tokens = tok.encode_for_ctc(text)
                print(f"✅ Sample tokenized: {sample_tokens[:20]}...")
                print(f"✅ Token count: {len(sample_tokens)}")
    else:
        print(f"❌ Manifest not found: {manifest_path}")
        return
    
    print("="*80)
    print("✅ ALL VERIFICATIONS PASSED - Starting training...")
    print("="*80 + "\n")
    # ================================================================================
    
    logger.info(" Starting ViCocktail Phase 1 Training (Features)...")
    
    # 1. Load Config
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    config['data']['train_manifest'] = f"{MOUNT_PROCESSED}/manifests/train.jsonl"
    config['data']['val_manifest'] = f"{MOUNT_PROCESSED}/manifests/val.jsonl"
    config['data']['data_root'] = f"{MOUNT_PROCESSED}/features/vicocktail"
    config['checkpoint_dir'] = f"{MOUNT_PROCESSED}/checkpoints/phase1"
    
    logger.info(f"Loaded Configuration:\n{yaml.dump(config)}")
    
    # 2. Check Data Existence
    if not os.path.exists(config['data']['train_manifest']):
        logger.error(f"Train manifest not found: {config['data']['train_manifest']}")
        return
            
    # 3. Initialize Trainer
    try:
        trainer = Trainer(config)
        logger.info("Trainer initialized successfully.")
    except Exception as e:
        logger.error(f" Failed to initialize Trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Start Training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:

        logger.info("Committing volume changes...")
        vol_processed.commit()
    

@app.local_entrypoint()
def main():
    train_remote.remote()
