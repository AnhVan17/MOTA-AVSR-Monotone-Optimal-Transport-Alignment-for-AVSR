
import os
import glob
import torch
import webdataset as wds
from tqdm import tqdm
from typing import List, Dict

from .base import BasePreprocessor, PreprocessConfig
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

_VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov")


def _count_video_members(tar_path: str):
    """Count video clips (one per sample) in a raw .tar — gives the progress-bar a total.

    Header-only scan (seeks past data), so it's cheap (~seconds) vs ~40min processing.
    Returns None on failure → the bar degrades to a plain counter (no %/ETA).
    """
    import tarfile
    try:
        with tarfile.open(tar_path) as t:
            return sum(1 for m in t.getmembers() if m.name.lower().endswith(_VIDEO_EXTS))
    except Exception as e:
        logger.warning(f"Could not count clips in {os.path.basename(tar_path)} ({e}); progress bar will show count only.")
        return None


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

    def run(self, output_manifest="vicocktail_manifest.jsonl", output_dir=None, extract_features=True, limit_ratio: float = 1.0, filter_keyword: str = None, max_samples: int = None, shard_pattern=None, shard_maxcount: int = 2000, shard_names: List[str] = None):
        """
        Overridden run method to handle WebDataset logic.

        max_samples: nếu set, dừng sau khi xử đủ N mẫu (tiện cho preprocess thử local).
        shard_names: nếu set, CHỈ xử lý các raw shard có basename nằm trong list (chạy theo batch).
        """
        logger.info("Collecting .tar shards...")
        metadata = self.collect_metadata()
        logger.info(f"   Found {len(metadata)} shards (Total).")

        # 1. Keyword Filter (e.g. 'train' vs 'test')
        if filter_keyword and filter_keyword != 'all':
            metadata = [m for m in metadata if filter_keyword in os.path.basename(m['full_path'])]
            logger.info(f"   [Filter '{filter_keyword}'] Keeping {len(metadata)} shards.")

        # 1b. Explicit shard list (batch mode): keep only the named raw shards.
        if shard_names:
            wanted = set(shard_names)
            metadata = [m for m in metadata if os.path.basename(m['full_path']) in wanted]
            logger.info(f"   [Explicit shards] Keeping {len(metadata)}/{len(wanted)} requested: {sorted(wanted)[:3]}...")

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

        # Create the face-alignment video processor ONCE and reuse it. Creating it per sample
        # re-inits + re-compiles the model (~1 min each) → catastrophically slow on full data.
        from src.data.preprocessors.base import VideoProcessor
        vp = VideoProcessor(use_precropped=self.use_precropped)

        # Optional WebDataset sharded output (packs samples into .tar shards).
        shard_sink, n_sharded = None, 0
        if shard_pattern:
            shard_dir = os.path.dirname(shard_pattern)
            if shard_dir:
                os.makedirs(shard_dir, exist_ok=True)
            shard_sink = wds.ShardWriter(shard_pattern, maxcount=shard_maxcount)
            logger.info(f"   WebDataset shards: {shard_pattern} (maxcount={shard_maxcount})")

        manifest_entries = []
        import time
        t_crop = t_visual = t_audio = 0.0  # PROFILE: stage timing accumulators (seconds)
        
        for s_idx, shard_item in enumerate(metadata):
            tar_path = shard_item['full_path']
            shard_name = os.path.basename(tar_path).replace(".tar", "")

            # Dừng sớm nếu đã đủ max_samples (gộp qua các shard).
            if max_samples is not None and (len(manifest_entries) + n_sharded) >= max_samples:
                break

            # Count clips up-front so the progress bar has a real total (%, ETA).
            n_in_shard = _count_video_members(tar_path)
            logger.info(f"[{s_idx + 1}/{len(metadata)}] Shard {shard_name} — {n_in_shard if n_in_shard is not None else '?'} clips")

            pbar = None
            try:
                # Use WebDataset to iterate
                # ViCocktail structure: key.mp4, key.wav, key.txt
                dataset = wds.WebDataset(tar_path).decode()

                # Per-clip bar: a real bar in a TTY; in Modal (non-TTY) tqdm emits one line
                # every `mininterval`s → traceable progress with %/rate/ETA. Counts each clip
                # read (incl. skipped), which is what wall-clock tracks.
                pbar = tqdm(dataset, total=n_in_shard, desc=shard_name, unit="clip", mininterval=10.0)
                for i, sample in enumerate(pbar):
                    if max_samples is not None and (len(manifest_entries) + n_sharded) >= max_samples:
                        break
                    key = sample.get("__key__")
                    
                    # 1. Get Text (Check 'txt' or 'label')
                    text = ""
                    if "txt" in sample:
                        text = sample["txt"]
                    elif "label" in sample:
                        text = sample["label"]
                    # WebDataset .decode() leaves .txt as raw bytes → decode to str
                    # (else JSON manifest write fails + .pt stores bytes).
                    if isinstance(text, bytes):
                        text = text.decode("utf-8")
                    
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
                        
                    # 3. Process Visual — reuse hoisted `vp` (PROFILE: t_crop = decode + face-align crop)
                    _t0 = time.perf_counter()
                    video_tensor = vp.process(temp_vid_path)  # [T, C, H, W]
                    t_crop += time.perf_counter() - _t0

                    if video_tensor is None:
                        os.remove(temp_vid_path)
                        continue

                    # Extract Visual features (PROFILE: t_visual = ResNet)
                    _t0 = time.perf_counter()
                    video_tensor = video_tensor.to(PreprocessConfig.DEVICE)
                    visual_feats_list = []
                    with torch.no_grad():
                        for j in range(0, len(video_tensor), PreprocessConfig.BATCH_SIZE):
                            batch = video_tensor[j : j + PreprocessConfig.BATCH_SIZE]
                            visual_feats_list.append(self.visual_extractor(batch).cpu())
                    visual_feats = torch.cat(visual_feats_list, dim=0)
                    t_visual += time.perf_counter() - _t0

                    # 4. Process Audio (PROFILE: t_audio = decode audio + Whisper)
                    _t0 = time.perf_counter()
                    audio_feats = self.audio_extractor.process_file(temp_vid_path)
                    t_audio += time.perf_counter() - _t0
                    
                    # Cleanup
                    os.remove(temp_vid_path)
                    
                    if audio_feats is None: 
                        audio_feats = torch.zeros((1, 768))

                    # 5. Save — WebDataset shard OR loose .pt
                    if shard_sink is not None:
                        key_s = str(key).replace(".", "_").replace("/", "_")
                        shard_sink.write({
                            "__key__": key_s,
                            "audio.pth": audio_feats.cpu(),
                            "visual.pth": visual_feats.cpu(),
                            "txt": text,
                        })
                        n_sharded += 1
                    else:
                        save_dict = {
                            'id': key, 'visual': visual_feats, 'audio': audio_feats,
                            'text': text, 'path': f"{shard_name}/{key}.mp4",
                        }
                        if self.output_dir:
                            save_subdir = os.path.join(self.output_dir, shard_name)
                            os.makedirs(save_subdir, exist_ok=True)
                            save_path = os.path.join(save_subdir, f"{key}.pt")
                        else:
                            logger.error("Output dir required for Tar processing")
                            return
                        torch.save(save_dict, save_path)
                        manifest_entries.append({
                            "id": key,
                            "rel_path": os.path.relpath(save_path, self.output_dir),
                            "text": text,
                        })
                    
            except Exception as e:
                logger.error(f"Failed to process shard {shard_name}: {e}")
            finally:
                if pbar is not None:
                    pbar.close()

        # Close shards + write _meta.json (WebDataset mode), else write jsonl manifest.
        import json
        _n = max(1, n_sharded + len(manifest_entries))
        logger.info(
            f"PROFILE ({_n} samples): "
            f"crop(decode+face-align)={t_crop:.1f}s ({t_crop/_n:.2f}/sample) | "
            f"visual(ResNet)={t_visual:.1f}s ({t_visual/_n:.2f}/sample) | "
            f"audio(Whisper)={t_audio:.1f}s ({t_audio/_n:.2f}/sample)"
        )
        logger.info(
            f"PROFILE crop-split: decode(cv2)={getattr(vp, '_prof_decode', 0.0):.1f}s "
            f"({getattr(vp, '_prof_decode', 0.0)/_n:.2f}/sample) | "
            f"detect+crop(face-align)={getattr(vp, '_prof_detect', 0.0):.1f}s "
            f"({getattr(vp, '_prof_detect', 0.0)/_n:.2f}/sample)"
        )
        _fa = getattr(vp, '_prof_fa', 0.0)
        logger.info(
            f"PROFILE detail: n_frames={getattr(vp, '_n_frames', 0)} | "
            f"n_detections={getattr(vp, '_n_det', 0)} | face-align-calls={_fa:.1f}s"
        )
        if shard_sink is not None:
            shard_sink.close()
            from src.data.shards import _pattern_to_glob, _meta_path
            import glob as _glob
            num_shards = len(_glob.glob(_pattern_to_glob(shard_pattern)))
            with open(_meta_path(shard_pattern), 'w', encoding='utf-8') as f:
                json.dump({"num_samples": n_sharded, "num_shards": num_shards}, f)
            logger.info(f"Done. {n_sharded} samples → {num_shards} shards; meta={_meta_path(shard_pattern)}")
            return

        with open(output_manifest, 'w', encoding='utf-8') as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Done. Processed {len(manifest_entries)} samples.")
