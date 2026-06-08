import json
import random
import os
import argparse
from pathlib import Path

def split_manifest(input_path, output_dir, train_ratio=0.8, seed=42):
    """
    Split a manifest JSONL into train and val files.
    """
    random.seed(seed)
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Input manifest {input_path} not found.")
        return

    print(f"Loading manifest from {input_path}...")
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    total = len(lines)
    print(f"Total samples: {total}")
    
    # Shuffle
    random.shuffle(lines)
    
    # Split
    split_idx = int(total * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # Output paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train_manifest.jsonl"
    val_path = output_dir / "val_manifest.jsonl"
    
    # Write
    with open(train_path, 'w') as f:
        f.writelines(train_lines)
        
    with open(val_path, 'w') as f:
        f.writelines(val_lines)
        
    print(f"✅ Split Complete:")
    print(f"   Train: {len(train_lines)} ({train_path})")
    print(f"   Val:   {len(val_lines)}   ({val_path})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Manifest 80/20")
    parser.add_argument("--input", "-i", required=True, help="Input manifest path")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: input directory)")
    
    args = parser.parse_args()
    
    output_dir = args.output if args.output else os.path.dirname(args.input)
    split_manifest(args.input, output_dir)
