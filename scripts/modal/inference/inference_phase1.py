import modal
import torch
import sys
import os
import json
from pathlib import Path

# Make `src` importable. Locally: add repo root (for app-build). In Modal containers the
# script is flattened to /root and src lives at /root/src, so parents[N] would be out of range.
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import ML_TRAIN_IMAGE, get_volume

# --- Config ---
APP_NAME = "avsr-inference-phase1"

app = modal.App(APP_NAME)
volume = get_volume()

@app.function(
    image=ML_TRAIN_IMAGE,
    volumes={"/mnt": volume},
    gpu="A10G",
    timeout=600
)
def run_inference(
    checkpoint_path: str, 
    manifest_path: str,
    output_path: str = "/mnt/inference_results.jsonl",
    limit: int = 10
):
    """
    Run inference on a set of features defined in a manifest.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        manifest_path: Path to manifest.jsonl
        output_path: Where to save predictions
        limit: Max samples to process (default 10)
    """
    sys.path.append("/root")
    from src.models.mota import create_model
    from src.data.tokenizers.whisper import WhisperTokenizer
    from src.data.loader import build_dataloader
    from src.evaluation.decoding import CTCDecoder
    from src.utils.config_utils import load_config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}")
    
    # 1. Load Config & Model
    # We assume config is saved with checkpoint or we use default phase1
    # For now, load default phase1 config
    config = load_config("/root/configs/phase1.yaml")
    
    # Initialize Model
    model = create_model(config['model']).to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 2. Setup Decoding
    tokenizer = WhisperTokenizer(model="openai/whisper-small")
    decoder = CTCDecoder(tokenizer)
    
    # 3. Load Data
    # Inference usually runs on Validation/Test set
    # Temporarily override config to point to provided manifest
    config['data']['test_manifest'] = manifest_path
    config['data']['batch_size'] = 1 # Sequential inference
    
    # Auto-detect data root (same logic as training script)
    manifest_p = Path(manifest_path)
    if "_manifest" in manifest_p.stem:
         subset_name = manifest_p.stem.replace("_manifest", "")
         possible_data_root = manifest_p.parent / subset_name
         if os.path.exists(possible_data_root):
             config['data']['data_root'] = str(possible_data_root)
             print(f"Auto-detected data_root: {possible_data_root}")
    
    data_loader = build_dataloader(config, tokenizer=tokenizer, mode='test')
    
    # 4. Run Inference Loop
    results = []
    print(f"Processing {limit} samples from {manifest_path}...")
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= limit: break
            
            # Move to device
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            targets = batch['target'].to(device)
            
            # Forward
            outputs = model(audio, visual, targets)
            
            # Decode
            pred_text = decoder.decode_batch(outputs['ctc_logits'], method='greedy')[0]
            target_text = decoder.decode_targets(targets)[0]
            
            print(f" Sample {i}:")
            print(f"  Ref:  {target_text}")
            print(f"  Pred: {pred_text}")
            
            results.append({
                "id": batch['rel_paths'][0], # Simplified ID
                "reference": target_text,
                "prediction": pred_text
            })
            
    # 5. Save Results
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Saved results to {output_path}")
    volume.commit()


@app.local_entrypoint()
def main(
    checkpoint: str = "/mnt/checkpoints/phase1/epoch_10.pt",
    manifest: str = "/mnt/vicocktail_features/avvn-test_snr_0_interferer_1-000000_manifest.jsonl"
):
    run_inference.remote(checkpoint, manifest)
