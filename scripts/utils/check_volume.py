"""
Volume Inspector for Modal.com
==============================
Kiểm tra chi tiết cấu trúc và dữ liệu trên Modal Volume.

Usage:
    modal run scripts/modal/check_volume.py
    modal run scripts/modal/check_volume.py --path /mnt/data/grid
    modal run scripts/modal/check_volume.py --detailed
    modal run scripts/modal/check_volume.py --sample-pt
"""

import modal
import os
import sys
from collections import defaultdict

APP_NAME = "avsr-volume-inspector"
VOLUME_NAME = "avsr-volume"
VOL_MOUNT_PATH = "/mnt"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy")
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(image=image, volumes={VOL_MOUNT_PATH: volume}, timeout=600)
def inspect_volume(
    path: str = "/mnt/data",
    detailed: bool = False,
    sample_pt: bool = False,
    max_depth: int = 3
):
    """Inspect Modal volume structure and contents."""
    sys.path.append("/root")
    import torch
    import numpy as np
    from src.utils.logging_utils import setup_logger
    
    # Use simpler logger for inspector, or same standard
    logger = setup_logger("Inspector")
    
    logger.info("=" * 60)
    logger.info(f"MODAL VOLUME INSPECTOR")
    logger.info(f"   Volume: {VOLUME_NAME}")
    logger.info(f"   Path: {path}")
    logger.info("=" * 60)
    
    if not os.path.exists(path):
        logger.error(f"Path does not exist: {path}")
        return
    
    # 1. DIRECTORY STRUCTURE
    logger.info("\nDIRECTORY STRUCTURE")
    logger.info("-" * 40)
    
    stats = {
        "total_files": 0,
        "total_dirs": 0,
        "total_size_bytes": 0,
        "file_types": defaultdict(lambda: {"count": 0, "size": 0}),
    }
    
    def human_size(size_bytes):
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"
    
    def print_tree(startpath, max_depth=3):
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            if level >= max_depth:
                continue
                
            indent = '│   ' * level + '├── '
            dirname = os.path.basename(root) or startpath
            
            # Count items in this dir
            file_count = len(files)
            dir_count = len(dirs)
            
            # Using print here for tree sructure to keep it raw, 
            # or use logger.info but it might prepend timestamp to every line making tree ugly.
            # Decision: Use print for the tree VISUALIZATION, but logger for summary.
            # Wait, the user wants NO icons and professional logging. 
            # A tree structure is arguably content, not log metadata.
            # But consistent output is better. Let's use logger.info but maybe with a special formatter?
            # Or just accept the timestamp. 
            # For simplicity and compliance, I will use logger.info.
            logger.info(f'{indent}{dirname}/ ({dir_count} dirs, {file_count} files)')
            
            # Track stats
            stats["total_dirs"] += 1
            
            # Show sample files
            subindent = '│   ' * (level + 1) + '├── '
            sample_files = files[:3] if not detailed else files[:10]
            for f in sample_files:
                fpath = os.path.join(root, f)
                try:
                    fsize = os.path.getsize(fpath)
                    stats["total_files"] += 1
                    stats["total_size_bytes"] += fsize
                    
                    ext = os.path.splitext(f)[1].lower() or "(no ext)"
                    stats["file_types"][ext]["count"] += 1
                    stats["file_types"][ext]["size"] += fsize
                    
                    if detailed:
                        logger.info(f'{subindent}{f} ({human_size(fsize)})')
                except:
                    pass
            
            if len(files) > len(sample_files):
                remaining = len(files) - len(sample_files)
                logger.info(f'{subindent}... and {remaining} more files')
                
                # Still count remaining files
                for f in files[len(sample_files):]:
                    fpath = os.path.join(root, f)
                    try:
                        fsize = os.path.getsize(fpath)
                        stats["total_files"] += 1
                        stats["total_size_bytes"] += fsize
                        ext = os.path.splitext(f)[1].lower() or "(no ext)"
                        stats["file_types"][ext]["count"] += 1
                        stats["file_types"][ext]["size"] += fsize
                    except:
                        pass
    
    print_tree(path, max_depth)
    
    # 2. STATISTICS SUMMARY
    logger.info("\nSTATISTICS")
    logger.info("-" * 40)
    logger.info(f"   Total Directories: {stats['total_dirs']}")
    logger.info(f"   Total Files: {stats['total_files']}")
    logger.info(f"   Total Size: {human_size(stats['total_size_bytes'])}")
    
    # 3. FILE TYPES BREAKDOWN
    logger.info("\nFILE TYPES")
    logger.info("-" * 40)
    logger.info(f"   {'Extension':<12} {'Count':>10} {'Size':>12}")
    logger.info(f"   {'-'*12} {'-'*10} {'-'*12}")
    
    sorted_types = sorted(stats["file_types"].items(), 
                          key=lambda x: x[1]["size"], reverse=True)
    for ext, data in sorted_types:
        logger.info(f"   {ext:<12} {data['count']:>10} {human_size(data['size']):>12}")
    
    # 4. SAMPLE .PT FILE INSPECTION
    if sample_pt:
        logger.info("\nSAMPLE .PT FILE INSPECTION")
        logger.info("-" * 40)
        
        # Find first .pt file
        pt_file = None
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.pt'):
                    pt_file = os.path.join(root, f)
                    break
            if pt_file:
                break
        
        if pt_file:
            logger.info(f"   File: {pt_file}")
            try:
                data = torch.load(pt_file, map_location='cpu')
                logger.info(f"   Type: {type(data)}")
                
                if isinstance(data, dict):
                    logger.info(f"   Keys: {list(data.keys())}")
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"   - {key}: Tensor shape={value.shape}, dtype={value.dtype}")
                        elif isinstance(value, np.ndarray):
                            logger.info(f"   - {key}: Array shape={value.shape}, dtype={value.dtype}")
                        elif isinstance(value, str):
                            preview = value[:50] + "..." if len(value) > 50 else value
                            logger.info(f"   - {key}: str = '{preview}'")
                        else:
                            logger.info(f"   - {key}: {type(value).__name__}")
                            
                elif isinstance(data, torch.Tensor):
                    logger.info(f"   Shape: {data.shape}")
                    logger.info(f"   Dtype: {data.dtype}")
                    logger.info(f"   Min/Max: {data.min():.4f} / {data.max():.4f}")
                    
            except Exception as e:
                logger.error(f"   Error loading: {e}")
        else:
            logger.info("   No .pt files found")
    
    # 5. DATA VALIDATION FOR AVSR
    logger.info("\nAVSR DATA VALIDATION")
    logger.info("-" * 40)
    
    expected_paths = [
        "/mnt/vicocktail_raw",
        "/mnt/vicocktail_cropped",
        "/mnt/vicocktail_features",
        "/mnt/_legacy_archive"
    ]
    
    for p in expected_paths:
        if os.path.exists(p):
            count = sum(len(files) for _, _, files in os.walk(p))
            logger.info(f"   {p} ({count} files)")
        else:
            logger.info(f"   {p} (NOT FOUND)")
    
    logger.info("\n" + "=" * 60)
    logger.info("Done!")


@app.local_entrypoint()
def main(
    path: str = "/mnt/vicocktail_features",
    detailed: bool = False,
    sample_pt: bool = False,
    max_depth: int = 3
):
    """
    Modal Volume Inspector
    """
    inspect_volume.remote(
        path=path,
        detailed=detailed,
        sample_pt=sample_pt,
        max_depth=max_depth
    )
