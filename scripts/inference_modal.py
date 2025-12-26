"""
Inference Script for AURORA-XT - Modal Pipeline
===============================================
Calculates WER/CER on test set
"""

import modal
from pathlib import Path
import yaml
import json
import torch
from tqdm import tqdm
from typing import List, Dict

# MODAL CONFIG
APP_NAME = "avsr-inference"
VOLUME_NAME = "avsr-dataset-volume"
CHECKPOINT_VOLUME = "avsr-checkpoints"
VOL_MOUNT_PATH = "/data"
CHECKPOINT_PATH = "/checkpoints"

# DOCKER IMAGE
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "numpy<2",
        "pyyaml==6.0.1",
        "jiwer==3.0.3"
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)

# Mount src directory
image_with_src = image.add_local_dir(
    local_path=Path(__file__).parent.parent / "src",
    remote_path="/root/src"
)


@app.cls(
    image=image_with_src,
    volumes={
        VOL_MOUNT_PATH: data_volume,
        CHECKPOINT_PATH: checkpoint_volume
    },
    gpu="A10G", # A10G is enough for inference and usually cheaper/more available than A100
    cpu=4.0,
    memory=32768,
    timeout=3600,  # 1 hour
)
class ModalInferrer:
    """Modal Inferrer - calculates metrics on test set"""
    
    @modal.enter()
    def initialize(self):
        """Add src to path"""
        import sys
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        print("✅ Modal inferrer initialized")
    
    @modal.method()
    def run_inference(self, config: dict, checkpoint_name: str = "best_model.pt"):
        """Run inference on test set"""
        import torch
        from torch.utils.data import DataLoader
        from src.models.aurora_xt import create_model
        from src.data.dataset import AuroraDataset, collate_fn
        from src.data.tokenizer import VietnameseCharTokenizer
        from src.evaluation.evaluator import Evaluator
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load Tokenizer
        tokenizer = VietnameseCharTokenizer()
        config['model']['vocab_size'] = tokenizer.vocab_size
        
        # 2. Create and Load Model
        print(f"🏗️ Loading model and checkpoint: {checkpoint_name}...")
        model = create_model(config['model']).to(device)
        
        ckpt_path = Path(CHECKPOINT_PATH) / checkpoint_name
        if not ckpt_path.exists():
            # Try searching in subfolders if checkpoint_dir was nested during training
            possible_paths = list(Path(CHECKPOINT_PATH).glob(f"**/{checkpoint_name}"))
            if possible_paths:
                ckpt_path = possible_paths[0]
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        print(f"📂 Loading from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 3. Setup Dataset
        test_manifest = f"{VOL_MOUNT_PATH}/manifests/test.jsonl"
        print(f"📊 Loading test set: {test_manifest}")
        
        if not Path(test_manifest).exists():
            # Check if val.jsonl exists as fallback if test.jsonl is missing
            test_manifest = f"{VOL_MOUNT_PATH}/manifests/val.jsonl"
            print(f"⚠️ test.jsonl not found, falling back to: {test_manifest}")
            
        dataset = AuroraDataset.from_manifest(test_manifest, data_root=VOL_MOUNT_PATH)
        dataloader = DataLoader(
            dataset,
            batch_size=config['data'].get('batch_size', 32),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # 4. Run Evaluation
        evaluator = Evaluator(tokenizer)
        print(f"🚀 Running inference on {len(dataset)} samples...")
        
        all_preds = []
        all_refs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                audio = batch['audio'].to(device)
                visual = batch['visual'].to(device)
                target = batch['target'].to(device)
                
                # Forward - target=None to use greedy decoding / CTC path
                outputs = model(audio, visual, target=None)
                
                preds = evaluator.decode_predictions(outputs['ctc_logits'])
                refs = evaluator.decode_targets(target)
                
                all_preds.extend(preds)
                all_refs.extend(refs)
        
        # 5. Calculate Final Metrics
        wer_score = evaluator.compute_wer(all_preds, all_refs)
        cer_score = evaluator.compute_cer(all_preds, all_refs)
        
        print("\n" + "="*50)
        print("✨ RESULTS")
        print("="*50)
        print(f"WER: {wer_score:.2f}%")
        print(f"CER: {cer_score:.2f}%")
        print(f"Samples: {len(all_preds)}")
        print("="*50)
        
        # Show some samples
        print("\n📝 Sample Predictions:")
        for i in range(min(5, len(all_preds))):
            print(f"[{i}] Pred: {all_preds[i]}")
            print(f"    Ref:  {all_refs[i]}")
            
        return {
            'wer': wer_score,
            'cer': cer_score,
            'num_samples': len(all_preds),
            'predictions': all_preds[:100], # Return first 100 for inspection
            'references': all_refs[:100]
        }


@app.local_entrypoint()
def main(config_path: str = "configs/model/config.yaml", checkpoint: str = "best_model.pt"):
    """Launch inference"""
    
    if not Path(config_path).exists():
        print(f"❌ Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("🚀 AURORA-XT Inference (Modal)")
    print("="*70)
    print(f"📋 Config: {config_path}")
    print(f"📦 Checkpoint: {checkpoint}")
    
    inferrer = ModalInferrer()
    result = inferrer.run_inference.remote(config, checkpoint)
    
    print("\n✅ Inference complete!")
    
    # Save results to local file
    output_file = Path("results/test_metrics.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    print(f"📊 Results saved to: {output_file}")
