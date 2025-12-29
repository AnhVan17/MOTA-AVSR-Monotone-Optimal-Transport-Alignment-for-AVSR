import modal
import os
import sys
import yaml
from pathlib import Path

# --- Config ---
APP_NAME = "avsr-train-phase1"
VOLUME_NAME = "avsr-volume"
MANIFEST_PATH = "/data/manifests/grid_manifest.jsonl"

# --- Image ---
# Same image dependencies as preprocess
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "numpy<2" # Force numpy < 2
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",         # A10G is good balance
    timeout=7200        # 2 hours
)
def train_remote():
    sys.path.append("/root")
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    
    from src.data.loader import build_dataloader
    from src.data.tokenizers.whisper import WhisperTokenizer
    from src.models.fusion.aurora_xt_model import create_model

    print(f"Starting Remote Phase 1 Training")
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest {MANIFEST_PATH} not found. Run preprocessing first.")
        return

    # Phase 1 Config
    config = {
        "data_root": "/data/data/grid", # Where .pt files are (nested)
        "train_manifest": MANIFEST_PATH,
        "val_manifest": MANIFEST_PATH, # Split later properly
        "batch_size": 32, # GPU can handle bigger batch with frozen features
        "num_workers": 2,
        "use_precomputed_features": True,
        
        "audio_dim": 768,
        "visual_dim": 512,
        "d_model": 256,
        "num_encoder_layers": 4,
        "num_decoder_layers": 2,
        "vocab_size": 51866, # Fixed size
        "epochs": 10,
        "lr": 1e-4,
        "device": "cuda"
    }
    
    # Pipeline from train_phase1.py
    tokenizer = WhisperTokenizer(language="en")
    train_loader = build_dataloader(config, tokenizer, mode='train')
    
    model = create_model(config).to(config['device'])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    ctc_loss_fn = torch.nn.CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)
    
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Steps per Epoch: {len(train_loader)}")
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            audio = batch['audio'].to(config['device'])
            visual = batch['visual'].to(config['device'])
            targets = batch['target'].to(config['device'])
            
            # Simple Forward
            target_inp = targets.clone()
            target_inp[target_inp == -100] = tokenizer.pad_token_id
            
            outputs = model(audio, visual, target_inp)
            ctc_logits = outputs['ctc_logits']
            
            log_probs = ctc_logits.log_softmax(dim=2).permute(1, 0, 2)
            input_lens = torch.full((audio.size(0),), ctc_logits.size(1), dtype=torch.long)
            target_lens = torch.sum(targets != -100, dim=1)
            
            loss = ctc_loss_fn(log_probs, targets, input_lens, target_lens)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

    print("Training Finished")

@app.local_entrypoint()
def main():
    train_remote.remote()
