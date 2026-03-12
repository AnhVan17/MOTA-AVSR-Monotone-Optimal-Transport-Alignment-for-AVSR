
import os
import tarfile
import glob
import torch
import webdataset as wds
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

from .base import BasePreprocessor, PreprocessConfig
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class ViCocktailPreprocessor(BasePreprocessor):
    """
    Preprocessor for ViCocktail Dataset (WebDataset format).
    Reads .tar shards directly and saves .pt features.
    """
    
    def collect_metadata(self) -> List[Dict]:
        """
        Scan for .tar files in data_root.
        Returns check-list of shards to process.
        """
        # Look for tar files
        tar_pattern = os.path.join(self.data_root, "**/*.tar")
        tar_files = glob.glob(tar_pattern, recursive=True)
        
        metadata = []
        for tar_path in tar_files:
            metadata.append({
                'full_path': tar_path,
                'rel_path': os.path.relpath(tar_path, self.data_root),
                'text': "SHARD" # Text is inside the tar
            })
            
        return metadata

    def run(self, output_manifest="vicocktail_manifest.jsonl", output_dir=None, extract_features=True, limit_ratio: float = 1.0, filter_keyword: str = None):
        """
        Overridden run method to handle WebDataset logic.
        """
        logger.info("Collecting .tar shards...")
        metadata = self.collect_metadata()
        logger.info(f"   Found {len(metadata)} shards (Total).")
        
        # 1. Keyword Filter (e.g. 'train' vs 'test')
        if filter_keyword and filter_keyword != 'all':
            metadata = [m for m in metadata if filter_keyword in os.path.basename(m['full_path'])]
            logger.info(f"   [Filter '{filter_keyword}'] Keeping {len(metadata)} shards.")

        if not metadata: return
        
        # Limit Ratio Logic
        if limit_ratio < 1.0:
            import random
            original_len = len(metadata)
            keep_len = int(original_len * limit_ratio)
            random.seed(42) # Reproducibility
            random.shuffle(metadata)
            metadata = metadata[:keep_len]
            logger.info(f"   [Limit Ratio {limit_ratio}] Keeping {len(metadata)}/{original_len} shards.")
        
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # load models
        self._load_models()
        
        manifest_entries = []
        
        for shard_item in tqdm(metadata, desc="Processing Shards"):
            tar_path = shard_item['full_path']
            shard_name = os.path.basename(tar_path).replace(".tar", "")
            
            try:
                # Use WebDataset to iterate
                # ViCocktail structure: key.mp4, key.wav, key.txt
                dataset = wds.WebDataset(tar_path).decode()
                
                for i, sample in enumerate(dataset):
                    key = sample.get("__key__")
                    
                    # 1. Get Text (Check 'txt' or 'label')
                    text = ""
                    if "txt" in sample:
                        text = sample["txt"]
                    elif "label" in sample:
                        text = sample["label"]
                    
                    # 2. Get Video (MP4 bytes -> save temp -> process)
                    # Check available keys for video
                    video_key = None
                    for ext in ['mp4', 'webm', 'mkv', 'avi', 'mov', 'video']:
                        if ext in sample:
                            video_key = ext
                            break
                    
                    if not video_key:
                        # Log available keys for debugging
                        logger.warning(f"No video found for {key}. Keys: {sample.keys()}")
                        continue
                    
                    # For 'video' key, assume mp4 for saving
                    save_ext = video_key if video_key != 'video' else 'mp4'

                    temp_vid_path = f"/tmp/{key}.{save_ext}"
                    with open(temp_vid_path, "wb") as f:
                        f.write(sample[video_key])
                        
                    # 3. Process Visual
                    # visual_extractor needs [B, C, H, W] tensor? 
                    # No, BasePreprocessor uses:
                    # dataset = RawVideoDataset([path]) -> loader -> visual_extractor
                    
                    # Let's call video_processor directly
                    # We need to instance VideoProcessor locally or use self's if adapted.
                    # BasePreprocessor does not instance VideoProcessor usually, it assumes RawVideoDataset does.
                    # Here we are the loop.
                    
                    # Re-use logic from BasePreprocessor logic flow but adapted for stream
                    # We need to manually call VideoProcessor.
                    from src.data.preprocessors.base import VideoProcessor
                    vp = VideoProcessor(use_precropped=False)
                    video_tensor = vp.process(temp_vid_path) # [T, C, H, W]
                    
                    if video_tensor is None:
                        os.remove(temp_vid_path)
                        continue
                        
                    # Extract Features (Visual)
                    # VideoTensor [T, C, H, W] 
                    # We need to batch it for ResNet
                    video_tensor = video_tensor.to(PreprocessConfig.DEVICE)
                    visual_feats_list = []
                    with torch.no_grad():
                        for j in range(0, len(video_tensor), PreprocessConfig.BATCH_SIZE):
                            batch = video_tensor[j : j + PreprocessConfig.BATCH_SIZE]
                            visual_feats_list.append(self.visual_extractor(batch).cpu())
                    visual_feats = torch.cat(visual_feats_list, dim=0)
                    
                    # 4. Process Audio
                    # Extract from temp video file (since we have it)
                    # Or use sample['wav'] if available? ViCocktail usually has mp4 only or both.
                    # safest is use process_file on the mp4
                    audio_feats = self.audio_extractor.process_file(temp_vid_path)
                    
                    # Cleanup
                    os.remove(temp_vid_path)
                    
                    if audio_feats is None: 
                        audio_feats = torch.zeros((1, 768))

                    # 5. Save .pt
                    save_dict = {
                        'id': key,
                        'visual': visual_feats,
                        'audio': audio_feats,
                        'text': text,
                        'path': f"{shard_name}/{key}.mp4"
                    }
                    
                    if self.output_dir:
                        # output_dir/shard_name/key.pt
                        save_subdir = os.path.join(self.output_dir, shard_name)
                        os.makedirs(save_subdir, exist_ok=True)
                        save_path = os.path.join(save_subdir, f"{key}.pt")
                    else:
                        logger.error("Output dir required for Tar processing")
                        return

                    torch.save(save_dict, save_path)
                    
                    # Add to manifest
                    manifest_entries.append({
                        "id": key,
                        "rel_path": os.path.relpath(save_path, self.output_dir), # relative to data root
                        "text": text
                    })
                    
            except Exception as e:
                logger.error(f"Failed to process shard {shard_name}: {e}")
                
        # Save manifest
        import json
        with open(output_manifest, 'w', encoding='utf-8') as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Done. Processed {len(manifest_entries)} samples.")
