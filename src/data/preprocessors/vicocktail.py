
import os
import glob
import json
import shutil
import cv2
import subprocess
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from .base_preprocessor import BasePreprocessor, VideoProcessor, PreprocessConfig
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Try to import normalize_text, fallback if not found
try:
    from src.utils.text_cleaning import normalize_text
except ImportError:
    # Fallback default
    def normalize_text(text): return text.lower().strip()


# ============================================================================
# PHASE 1: Multiprocessing Worker Function
# ============================================================================
_worker_processor = None

def _get_processor():
    """Lazy initialization of processor per worker process"""
    global _worker_processor
    if _worker_processor is None:
        # Disable GPU for MediaPipe in headless environments to avoid EGL errors
        import os
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
        from .base_preprocessor import VideoProcessor
        _worker_processor = VideoProcessor(use_precropped=False)
    return _worker_processor

def _process_single_video_wrapper(args):
    """
    Standalone function for multiprocessing mouth cropping.
    """
    video_path, data_root, save_dir = args
    
    try:
        import cv2
        import os
        import subprocess
        import shutil
        from .base_preprocessor import PreprocessConfig
        
        # 1. Path Calculation
        rel_path = os.path.relpath(video_path, data_root)
        target_path = os.path.join(save_dir, rel_path)
        base, _ = os.path.splitext(target_path)
        target_path = base + ".mp4"
        
        # 2. Skip Logic (Optimized)
        if os.path.exists(target_path):
            return True
            
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # 3. Copy Transcript
        base_path = os.path.splitext(video_path)[0]
        for ext in ['.txt', '.label']:
            src = base_path + ext
            if os.path.exists(src):
                shutil.copy2(src, os.path.splitext(target_path)[0] + ".txt")
                break
        
        # 4. Processing
        processor = _get_processor()
        output_size = (PreprocessConfig.RESNET_INPUT_SIZE, PreprocessConfig.RESNET_INPUT_SIZE)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1: fps = 25
        
        temp_silent = target_path.replace(".mp4", "_silent.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_silent, fourcc, fps, output_size)
        
        frame_idx = 0
        prev_bbox = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Use persistent processor
                if frame_idx % 5 == 0 or prev_bbox is None:
                    crop, prev_bbox = processor.extract_mouth(frame, None)
                else:
                    crop, prev_bbox = processor.extract_mouth(frame, prev_bbox)
                
                crop = cv2.resize(crop, output_size)
                writer.write(crop)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()
        
        if frame_idx == 0:
            if os.path.exists(temp_silent): os.remove(temp_silent)
            return False
            
        # 5. FFmpeg Merge (Passive check)
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', temp_silent, '-i', video_path,
            '-c:v', 'copy', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0?',
            '-shortest', target_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if os.path.exists(temp_silent): os.remove(temp_silent)
        return True
        
    except Exception:
        # Log error to file or print since it's a subprocess
        return False



# ============================================================================
# VICOCKTAIL PREPROCESSOR
# ============================================================================
class ViCocktailPreprocessor(BasePreprocessor):
    """
    Vicocktail dataset preprocessor
    
    Phase 1: Mouth cropping (multiprocessing on CPU)
    Phase 2: Feature extraction (GPU) - inherited from BasePreprocessor
    """
    
    def __init__(self, data_root, use_precropped=False):
        """
        Args:
            data_root: Root directory of dataset
            use_precropped: Set to True for Phase 2 (after Phase 1 cropping)
        """
        # FIXED: Removed whisper_model arg to match BasePreprocessor
        super().__init__(data_root, use_precropped)

    def collect_metadata(self):
        """
        Scan data_root for video files and collect metadata
        
        Returns:
            List[Dict]: Metadata for each video
        """
        logger.info(f"[ViCocktail] Scanning {self.data_root}...")
        
        # Search for video files (Phase 2 expects .mp4 from Phase 1)
        extensions = ['*.mp4', '*.mkv', '*.webm', '*.avi']
        video_files = []
        
        for ext in extensions:
            video_files.extend(
                glob.glob(os.path.join(self.data_root, "**", ext), recursive=True)
            )
        
        logger.info(f"[ViCocktail] Found {len(video_files)} video files")
        logger.info(f"[ViCocktail] Loading transcripts (this may take a few minutes)...")
        
        results = []
        missing_text = 0
        
        # Add tqdm to show progress during transcript loading
        for video_path in tqdm(video_files, desc="Loading transcripts"):
            text = self._get_transcript(video_path)
            
            # Track files without transcripts
            if not text:
                missing_text += 1
            
            rel_path = os.path.relpath(video_path, self.data_root)
            item_id = os.path.splitext(os.path.basename(video_path))[0]
            
            results.append({
                'id': item_id,
                'full_path': video_path,
                'rel_path': rel_path,
                'text': text
            })
        
        if missing_text > 0:
            logger.warning(f"{missing_text} videos without transcripts")
        
        return results

    def _get_transcript(self, video_path):
        """
        Load transcript from .txt or .label file and NORMALIZE IT.
        """
        base_path = os.path.splitext(video_path)[0]
        text = ""
        
        # Priority 1: .txt
        txt_path = base_path + ".txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to read {txt_path}: {e}")
        
        # Priority 2: .label (webdataset format)
        if not text:
            label_path = base_path + ".label"
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                except Exception as e:
                    logger.warning(f"Failed to read {label_path}: {e}")
        
        # Normalize Text (Critical for ASR)
        return normalize_text(text) if text else ""

    def phase1_crop_dataset(self, save_dir, max_workers=None):
        """
        Phase 1: Parallel mouth cropping
        
        Args:
            save_dir: Output directory for cropped videos
            max_workers: Number of parallel workers (default: NUM_WORKERS from config)
        """
        logger.info("="*60)
        logger.info("PHASE 1: MOUTH CROPPING (PARALLEL)")
        logger.info("="*60)
        logger.info(f"Source: {self.data_root}")
        logger.info(f"Target: {save_dir}")
        
        if max_workers is None:
            max_workers = PreprocessConfig.NUM_WORKERS
        
        logger.info(f"Workers: {max_workers}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Find all raw video files
        extensions = ['*.mp4', '*.mkv', '*.webm', '*.avi', '*.video']
        video_files = []
        
        for ext in extensions:
            video_files.extend(
                glob.glob(os.path.join(self.data_root, "**", ext), recursive=True)
            )
        
        logger.info(f"Found: {len(video_files)} videos")
        
        if not video_files:
            logger.warning("No videos found!")
            return
        
        # Prepare tasks
        tasks = [(v, self.data_root, save_dir) for v in video_files]
        
        success_count = 0
        failed_count = 0
        
        # Execute in parallel
        logger.info(f"Processing with {max_workers} workers...")
        
        # Use a lazy map to avoid memory overhead of many futures
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(_process_single_video_wrapper, tasks), total=len(tasks), desc="Cropping"):
                if result:
                    success_count += 1
                else:
                    failed_count += 1
        
        logger.info("="*60)
        logger.info(f"COMPLETED: {success_count}/{len(video_files)}")
        logger.info(f"FAILED: {failed_count}/{len(video_files)}")
        logger.info("="*60)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
def preprocess_vicocktail_phase1(raw_dir, clean_dir, max_workers=8):
    """
    Convenience function for Phase 1
    """
    preprocessor = ViCocktailPreprocessor(
        data_root=raw_dir,
        use_precropped=False
    )
    preprocessor.phase1_crop_dataset(clean_dir, max_workers=max_workers)


def preprocess_vicocktail_phase2(clean_dir, manifest_path):
    """
    Convenience function for Phase 2
    """
    preprocessor = ViCocktailPreprocessor(
        data_root=clean_dir,
        use_precropped=True  # Important: skip cropping in Phase 2
    )
    preprocessor.run(output_manifest=manifest_path, extract_features=True)
