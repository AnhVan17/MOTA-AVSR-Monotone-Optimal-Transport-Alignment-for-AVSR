
import os
import torch
import glob
import argparse
from src.data.datasets.base import FeatureDataset

def verify_vicocktail(data_root: str):
    """
    Verify processed ViCocktail features.
    """
    print(f"Verifying features in {data_root}...")
    
    # 1. Check for Manifest
    manifest_path = "vicocktail_manifest.jsonl" # Default output of preprocessor
    if os.path.exists(manifest_path):
        print(f"✅ Manifest found: {manifest_path}")
    else:
        print(f"⚠️ Manifest not found at default location. Checking glob...")

    # 2. Check for PT files
    pt_files = glob.glob(os.path.join(data_root, "**/*.pt"), recursive=True)
    if not pt_files:
        print("❌ No .pt files found!")
        return
        
    print(f"Found {len(pt_files)} .pt files. Checking sample...")
    
    # 3. Load Sample
    sample_path = pt_files[0]
    try:
        data = torch.load(sample_path)
        print(f"Snapshot of {sample_path}:")
        print(f"   Keys: {data.keys()}")
        
        audio = data['audio']
        visual = data['visual']
        text = data['text']
        
        print(f"   Audio Shape: {audio.shape} (Expected: [T, 768])")
        print(f"   Visual Shape: {visual.shape} (Expected: [T, 512])")
        print(f"   Text: {text}")
        
        if audio.shape[-1] == 768 and visual.shape[-1] == 512:
            print("✅ Shapes Correct.")
        else:
            print("❌ Shape Mismatch!")
            
    except Exception as e:
        print(f"❌ Failed to load sample: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str, help="Root directory containing processed .pt files")
    args = parser.parse_args()
    
    verify_vicocktail(args.data_root)
