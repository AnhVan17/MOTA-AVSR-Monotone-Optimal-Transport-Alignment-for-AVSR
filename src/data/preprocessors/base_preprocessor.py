import os
import json
import torch
import cv2
import numpy as np
import timm
import mediapipe as mp
import io
import soundfile as sf
import random
from tqdm import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperModel, WhisperFeatureExtractor
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

try:
    from src.utils.text_cleaning import normalize_text
except ImportError:
    # Fallback if running outside package context
    import sys
    sys.path.append(os.getcwd())
    from src.utils.text_cleaning import normalize_text
        

class PreprocessConfig:
    # Visual Params (SOTA: 88x88)
    IMAGE_SIZE = 88        # Mouth crop size (SOTA standard)
    RESNET_INPUT_SIZE = 88 # Changed from 224 to 88 (SOTA)
    USE_GRAYSCALE = False  # RGB for pretrained ResNet (existing videos are RGB)
    
    # Audio Params
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_LENGTH = 240000 # 15s * 16000
    
    # System Params
    BATCH_SIZE = 256       # Increased for A100 (40-80GB VRAM)
    NUM_WORKERS = 8 # Increase workers for parallel FaceMesh processing
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. VIDEO PROCESSOR (Mouth Crop) ---
class VideoProcessor:
    """
    Dedicated class for cropping mouth region (ROI) from video using MediaPipe.
    Or simply loading frames if use_precropped=True.
    """
    def __init__(self, use_precropped=False):
        self.use_precropped = use_precropped
        
        if not self.use_precropped:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )

    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        processed_frames = []
        prev_bbox = None
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                if self.use_precropped:
                    # If pre-cropped, the whole frame is the mouth.
                    crop = frame
                else:
                    # Optimization: Detect every 5 frames
                    if frame_idx % 5 == 0 or prev_bbox is None:
                        crop, prev_bbox = self.extract_mouth(frame, None)
                    else:
                        crop, prev_bbox = self.extract_mouth(frame, prev_bbox)
                
                # Resize to 88x88 (SOTA standard)
                crop = cv2.resize(crop, (PreprocessConfig.RESNET_INPUT_SIZE, PreprocessConfig.RESNET_INPUT_SIZE)) 
                
                # Convert to grayscale (SOTA: grayscale for lip reading)
                if PreprocessConfig.USE_GRAYSCALE:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                else:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                processed_frames.append(crop)
                frame_idx += 1
                
        finally:
            cap.release()
            
        if not processed_frames: return None

        # Convert to tensor
        frames_np = np.array(processed_frames)
        if PreprocessConfig.USE_GRAYSCALE:
            # Grayscale: (T, H, W) -> (T, 1, H, W)
            video_tensor = torch.tensor(frames_np, dtype=torch.float32).unsqueeze(1) / 255.0
        else:
            # RGB: (T, H, W, C) -> (T, C, H, W)
            video_tensor = torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            
        return video_tensor

    def _loop_pad_video(self, video_tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        Loop pad video frames instead of zero padding.
        This helps model learn better from short videos.
        
        Args:
            video_tensor: [C, T, H, W] video tensor
            target_frames: Target number of frames
            
        Returns:
            Padded video tensor [C, target_frames, H, W]
        """
        if video_tensor.dim() == 4: # [T, C, H, W] or [C, T, H, W] ?
            # Based on above process() return:
            # RGB: [T, C, H, W] (from permute(0,3,1,2) on (T,H,W,C)) -> Wait.
            # np array is (T, H, W, C).
            # permute(0, 3, 1, 2) -> (T, C, H, W). 
            # Usually PyTorch models expect (B, C, T, H, W) or (B, T, C, H, W).
            # ResNet expects (B, C, H, W). We process frame by frame.
            pass
            
        # NOTE: The user script output [C, T, H, W]. 
        # My current code outputs [T, C, H, W].
        # Let's check VisualFeatureExtractor. It takes (B, C, H, W).
        # In run() loop:
        # batch = video_tensor[i : i + BATCH] -> Slices T dimension.
        # So video_tensor MUST be [T, C, H, W].
        
        T = video_tensor.size(0)
        if T >= target_frames:
            return video_tensor[:target_frames]
        
        repeats = (target_frames // T) + 1
        looped = video_tensor.repeat(repeats, 1, 1, 1) # Repeat along T
        return looped[:target_frames]

    def extract_mouth(self, frame, prev_bbox=None):
        """Mouth cropping logic using MediaPipe"""
        h, w = frame.shape[:2]
        
        # If previous bbox exists, try fast crop
        if prev_bbox is not None:
            x1, y1, x2, y2 = prev_bbox
            if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                return frame[y1:y2, x1:x2], prev_bbox

        # If not available or need to re-detect -> Run MediaPipe
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                # Landmark points around the mouth
                mouth_idx = [13, 14, 61, 291, 78, 308, 324, 17]
                xs = [lm[i].x * w for i in mouth_idx]
                ys = [lm[i].y * h for i in mouth_idx]
                
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                
                # Calculate crop radius
                radius = max(
                    int((max(xs) - min(xs)) * 1.8) // 2,
                    int((max(ys) - min(ys)) * 1.8) // 2,
                    PreprocessConfig.IMAGE_SIZE // 2
                )
                
                y1 = max(0, cy - radius)
                y2 = min(h, cy + radius)
                x1 = max(0, cx - radius)
                x2 = min(w, cx + radius)
                
                bbox = (x1, y1, x2, y2)
                return frame[y1:y2, x1:x2], bbox
        except:
            pass
            
        # Fallback: Crop center
        cy, cx = h // 2, w // 2
        r = PreprocessConfig.IMAGE_SIZE // 2
        return frame[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)], None

# --- 2. AUDIO PROCESSOR (Whisper) ---

class AudioFeatureExtractor:
    """Extract Whisper features from audio"""
    
    def __init__(self):
        # Whisper-small encoder (feature extractor only)
        self.audio_model = WhisperModel.from_pretrained(
            "openai/whisper-small"
        ).to(PreprocessConfig.DEVICE).eval()
        
        self.whisper_processor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-small"
        )
        logger.debug(f"Loaded {self.audio_model} on {PreprocessConfig.DEVICE}")
        
    def process_file(self, video_path):
        """Extract audio waveform then features from video file"""
        waveform = self.extract_waveform(video_path)
        if waveform is None:
            return None
        return self.extract_features(waveform)

    def extract_waveform(self, video_path):
        """Use PyAV then FFmpeg to extract audio"""
        # 1. Try PyAV (Robust)
        try:
            waveform = self._try_pyav(video_path)
            if waveform is not None:
                return waveform
        except Exception:
            pass
            
        # 2. Fallback to FFmpeg (subprocess)
        try:
            import subprocess
            
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(PreprocessConfig.AUDIO_SAMPLE_RATE),
                '-ac', '1',
                '-f', 'wav', 'pipe:1'
            ]
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            
            audio_data, _ = proc.communicate(timeout=15)
            
            if len(audio_data) < 100:
                return None
            
            waveform, _ = sf.read(io.BytesIO(audio_data), dtype='float32')
            return self._normalize_pad(torch.from_numpy(waveform).unsqueeze(0))
            
        except Exception as e:
            # logger.error(f"Audio extract failed for {video_path}: {e}")
            return None

    def _try_pyav(self, video_path) -> torch.Tensor:
        """Extract audio using PyAV"""
        try:
            import av
            container = av.open(video_path)
            audio_stream = next((s for s in container.streams.audio), None)
            
            if not audio_stream:
                container.close()
                return None
            
            resampler = av.AudioResampler(
                format='s16p',
                layout='mono',
                rate=PreprocessConfig.AUDIO_SAMPLE_RATE
            )
            
            frames = []
            for frame in container.decode(audio_stream):
                for resampled in resampler.resample(frame):
                    frames.append(resampled.to_ndarray().flatten())
            
            container.close()
            
            if not frames:
                return None
            
            waveform = np.concatenate(frames)
            waveform = torch.from_numpy(waveform).float() / 32768.0
            
            return self._normalize_pad(waveform.unsqueeze(0))
        except ImportError:
            # logger.warning("PyAV not installed")
            return None
        except Exception:
            return None

    def _normalize_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """RMS normalize + loop pad/trim"""
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 1e-8:
            waveform = waveform * (0.1 / rms)
        
        cur_len = waveform.size(1)
        target_len = PreprocessConfig.AUDIO_LENGTH
        
        if cur_len < target_len:
            # Loop pad (Context preservation)
            return self._loop_pad_audio(waveform, target_len)
        else:
            return waveform[:, :target_len]

    def _loop_pad_audio(self, waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """Loop pad audio to target length"""
        cur_len = waveform.size(1)
        if cur_len == 0:
            return torch.zeros(1, target_length)
            
        repeats = (target_length // cur_len) + 1
        looped = waveform.repeat(1, repeats)
        return looped[:, :target_length]

    @torch.no_grad()
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: [1, T] waveform
        Output: [T_seq, 768] Whisper features
        """
        audio_np = waveform.squeeze(0).cpu().numpy()
        
        inputs = self.whisper_processor(
            audio_np,
            sampling_rate=PreprocessConfig.AUDIO_SAMPLE_RATE,
            return_tensors="pt"
        )
        
        # [1, 80, 3000] -> [1, 1500, 768] (encoder outputs)
        # Note: Whisper Small Encoder Output is [Batch, 1500, 768] for 30s audio.
        # Check actual output shape.
        audio_feats = self.audio_model.encoder(
            inputs.input_features.to(PreprocessConfig.DEVICE)
        ).last_hidden_state.cpu()
        
        return audio_feats.squeeze(0) 


# --- 3. HELPER CLASSES ---

class VisualFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18 (Visual Encoder)
        # in_chans=1 for grayscale input (SOTA), in_chans=3 for RGB
        in_channels = 1 if PreprocessConfig.USE_GRAYSCALE else 3
        self.backbone = timm.create_model(
            'resnet18', 
            pretrained=True, 
            num_classes=0, 
            global_pool='',
            in_chans=in_channels  # Support grayscale input
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.to(PreprocessConfig.DEVICE).eval()

    def forward(self, x):
        with torch.no_grad():
            return self.pool(self.backbone(x)).flatten(1)

# --- 4. DATASET wrapper (for Visual batching) ---

class RawVideoDataset(Dataset):
    """
    Dataset to load video frames for Visual Extraction
    """
    def __init__(self, video_paths, use_precropped=False):
        self.video_paths = video_paths
        self.use_precropped = use_precropped
        
    def __len__(self): return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        
        # OPTIMIZATION: When use_precropped=True, skip MediaPipe entirely
        # Just read frames directly - they are already mouth-cropped
        if self.use_precropped:
            tensor = self._load_precropped_video(path)
        else:
            # Only init MediaPipe if we actually need to crop
            processor = VideoProcessor(use_precropped=False) 
            tensor = processor.process(path)
        
        if tensor is None: 
            return torch.zeros((1, 3, PreprocessConfig.RESNET_INPUT_SIZE, PreprocessConfig.RESNET_INPUT_SIZE)), path
            
        return tensor, path
    
    def _load_precropped_video(self, video_path):
        """Fast loader for pre-cropped videos - NO MediaPipe needed"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Resize to standard size (videos are already mouth-only)
                frame = cv2.resize(frame, (PreprocessConfig.RESNET_INPUT_SIZE, PreprocessConfig.RESNET_INPUT_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
            
        if not frames: return None
        
        # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
        frames_np = np.array(frames)
        video_tensor = torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        return video_tensor

def collate_video_wrapper(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


# --- 5. KEY FRAME EXTRACTOR (Shared - Dataset Agnostic) ---

class KeyFrameExtractor:
    """
    Extract key frames from cropped video.
    Removes redundant similar frames to reduce data size and speed up training.
    Works with ANY dataset - only needs a cropped video as input.
    """
    
    def __init__(self, threshold: float = 30.0, max_frames: int = 75, min_frames: int = 10):
        """
        Args:
            threshold: Minimum pixel difference to consider a frame as "key" (0-255 scale)
            max_frames: Maximum number of frames to keep
            min_frames: Minimum frames to ensure (even if similar)
        """
        self.threshold = threshold
        self.max_frames = max_frames
        self.min_frames = min_frames
    
    def extract_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract key frames from video file.
        
        Returns:
            np.ndarray: Shape (N, H, W, C) where N is number of key frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()
        
        if not frames:
            return np.array([])
        
        return self._select_key_frames(frames)
    
    def _select_key_frames(self, frames: list) -> np.ndarray:
        """Select key frames based on frame difference."""
        if len(frames) <= self.min_frames:
            return np.array(frames)
        
        key_indices = [0]  # Always keep first frame
        prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        for i in range(1, len(frames)):
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.mean(np.abs(curr_frame - prev_frame))
            
            if diff > self.threshold:
                key_indices.append(i)
                prev_frame = curr_frame
        
        # Always include last frame
        if key_indices[-1] != len(frames) - 1:
            key_indices.append(len(frames) - 1)
        
        # Ensure minimum frames by sampling
        if len(key_indices) < self.min_frames:
            step = len(frames) // self.min_frames
            key_indices = list(range(0, len(frames), max(1, step)))[:self.min_frames]
        
        # Limit to max frames
        if len(key_indices) > self.max_frames:
            step = len(key_indices) // self.max_frames
            key_indices = key_indices[::max(1, step)][:self.max_frames]
        
        return np.array([frames[i] for i in sorted(set(key_indices))])
    
    def save_as_tensor(self, frames: np.ndarray, output_path: str):
        """
        Save key frames as a PyTorch tensor file (.pt).
        
        Args:
            frames: Shape (N, H, W, C) - BGR format from OpenCV
            output_path: Path to save .pt file
        """
        if len(frames) == 0:
            torch.save(torch.zeros((1, 3, PreprocessConfig.RESNET_INPUT_SIZE, PreprocessConfig.RESNET_INPUT_SIZE)), output_path)
            return
        
        # Resize and convert: (N, H, W, C) BGR -> (N, C, H, W) RGB normalized
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (PreprocessConfig.RESNET_INPUT_SIZE, PreprocessConfig.RESNET_INPUT_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            processed.append(rgb)
        
        tensor = torch.tensor(np.array(processed), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        torch.save(tensor, output_path)
    
    def save_as_images(self, frames: np.ndarray, output_dir: str):
        """
        Save key frames as individual JPEG images.
        
        Args:
            frames: Shape (N, H, W, C)
            output_dir: Directory to save images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            resized = cv2.resize(frame, (PreprocessConfig.RESNET_INPUT_SIZE, PreprocessConfig.RESNET_INPUT_SIZE))
            output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_path, resized)


# --- 6. BASE PREPROCESSOR ---

class BasePreprocessor(ABC):
    """Abstract base class for dataset-specific preprocessors."""
    
    def __init__(self, data_root, use_precropped=False):
        self.data_root = data_root
        self.use_precropped = use_precropped
        
        # Lazy Loading: Don't load models yet
        self.visual_extractor = None
        self.audio_extractor = None
        logger.info(f"Initialized {self.__class__.__name__} (Lazy Load)")

    def _load_models(self):
        """Helper to load heavy models only when needed"""
        if self.visual_extractor is None:
            logger.info(f"Loading Extractors on {PreprocessConfig.DEVICE}...")
            self.visual_extractor = VisualFeatureExtractor()
            self.audio_extractor = AudioFeatureExtractor()
            logger.info("Models Loaded.")

    @abstractmethod
    def collect_metadata(self):
        """Should return list of dict: {'full_path', 'rel_path', 'text'}"""
        pass

    def run(self, output_manifest="manifest.jsonl", output_dir=None, extract_features=True):
        """
        Run preprocessing pipeline.
        
        Args:
            output_manifest: Path to save the manifest file
            output_dir: Directory to save .pt feature files. If None, saves next to input videos.
            extract_features: Whether to extract and save features
        """
        logger.info("Collecting metadata...")
        metadata = self.collect_metadata()
        logger.info(f"   Found {len(metadata)} samples.")

        if not metadata: return
        
        # Store output_dir for use in processing
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"   Output directory: {output_dir}")

        # Optimization: Create a map for O(1) access
        metadata_map = {m['full_path']: m for m in metadata}

        # 1. Extract Features
        if extract_features:
            # LOAD MODELS NOW
            self._load_models()
            
            logger.info("Running Multimodal Extraction (Audio + Visual)...")
            video_paths = [m['full_path'] for m in metadata]
            
            # Use DataLoader for Visual (Frames are heavy, supports Batching)
            dataset = RawVideoDataset(video_paths, use_precropped=self.use_precropped)
            loader = DataLoader(
                dataset, 
                batch_size=1, 
                num_workers=PreprocessConfig.NUM_WORKERS, 
                collate_fn=collate_video_wrapper
            )
            
            # Iterate through videos
            count = 0
            total = len(dataset)
            logger.info(f"Starting processing loop for {total} items...")
            
            for videos, paths in tqdm(loader, desc="Processing"):
                for video_tensor, video_path in zip(videos, paths):
                    count += 1
                    if count % 100 == 0 or count == 1:
                        logger.info(f"   [{count}/{total}] Processing {video_path}...")
                    try:
                        # A. Visual Features
                        video_tensor = video_tensor.to(PreprocessConfig.DEVICE)
                        visual_feats_list = []
                        
                        # Process video frames in batches
                        for i in range(0, len(video_tensor), PreprocessConfig.BATCH_SIZE):
                            batch = video_tensor[i : i + PreprocessConfig.BATCH_SIZE]
                            visual_feats_list.append(self.visual_extractor(batch).cpu())
                        
                        visual_feats = torch.cat(visual_feats_list, dim=0) if visual_feats_list else torch.zeros((1, 512))

                        # B. Audio Features
                        audio_feats = self.audio_extractor.process_file(video_path)
                        if audio_feats is None:
                            # Fallback if audio fails
                            audio_feats = torch.zeros((1, 768))

                        # C. Save Combined .pt
                        item = metadata_map.get(video_path)
                        text = item['text'] if item else ""
                        item_id = item.get('id', os.path.splitext(os.path.basename(video_path))[0])

                        save_dict = {
                            'id': item_id,
                            'visual': visual_feats, # (T_v, 512)
                            'audio': audio_feats,   # (T_a, 768)
                            'text': text,
                            'path': video_path # Original path
                        }
                        
                        # Determine save path
                        if self.output_dir:
                            # Use output_dir, preserve relative structure
                            rel_path = item.get('rel_path', os.path.basename(video_path))
                            pt_filename = os.path.splitext(rel_path)[0] + ".pt"
                            save_path = os.path.join(self.output_dir, pt_filename)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        else:
                            # Save next to original video
                            save_path = video_path.replace(".mpg", ".pt").replace(".mp4", ".pt").replace(".webm", ".pt")
                        
                        torch.save(save_dict, save_path)
                        count += 1
                        
                    except Exception as e:
                        logger.error(f"FAILED {video_path}: {e}")

        # 2. Save Manifest
        logger.info(f"Saving manifest to {output_manifest}...")
        with open(output_manifest, 'w', encoding='utf-8') as f:
            for item in metadata:
                # Update output path to be the .pt file
                original_ext = os.path.splitext(item['full_path'])[-1]
                pt_rel_path = item['rel_path'].replace(original_ext, ".pt")
                
                entry = {
                    "id": item.get('id', os.path.splitext(os.path.basename(item['rel_path']))[0]),
                    "rel_path": pt_rel_path, # Point to the processed .pt
                    "text": item['text'],
                    # Optional: Include original video path if needed
                    # "video_path": item['rel_path']
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Done!")