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
from transformers import WhisperModel, WhisperFeatureExtractor
        

class PreprocessConfig:
    # Visual Params
    IMAGE_SIZE = 96        # Mouth crop size
    RESNET_INPUT_SIZE = 224 # ResNet18 input size
    
    # Audio Params
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_LENGTH = 240000 # 15s * 16000
    
    # System Params
    BATCH_SIZE = 64
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
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
        finally:
            cap.release()
            
        if not frames: return None

        processed_frames = []
        prev_bbox = None
        
        for i, frame in enumerate(frames):
            
            if self.use_precropped:
                # If pre-cropped, the whole frame is the mouth.
                crop = frame
            else:
                # Optimization: Detect every 5 frames
                if i % 5 == 0 or prev_bbox is None:
                    crop, prev_bbox = self.extract_mouth(frame, None)
                else:
                    crop, prev_bbox = self.extract_mouth(frame, prev_bbox)
            
            # Resize to 224x224 for ResNet18
            crop = cv2.resize(crop, (PreprocessConfig.RESNET_INPUT_SIZE, PreprocessConfig.RESNET_INPUT_SIZE)) 
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            processed_frames.append(crop)

        # (T, H, W, C) -> (T, C, H, W) -> Normalize [0, 1]
        return torch.tensor(np.array(processed_frames), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

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
        print(f"Loaded {self.audio_model} on {PreprocessConfig.DEVICE}")
        
    def process_file(self, video_path):
        """Extract audio waveform then features from video file"""
        waveform = self.extract_waveform(video_path)
        if waveform is None:
            return None
        return self.extract_features(waveform)

    def extract_waveform(self, video_path):
        """Use FFmpeg via subprocess to extract audio directly"""
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
            # print(f"Audio extract failed for {video_path}: {e}")
            return None

    def _normalize_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """RMS normalize + loop pad/trim"""
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 1e-8:
            waveform = waveform * (0.1 / rms)
        
        # Trim to AUDIO_LENGTH (for fixed length inputs like Whisper training needs 30s)
        # However, for FEATURE EXTRACTION of 'whisper-small', 
        # the model can handle shorter inputs if using the processor correctly with padding.
        # But for AVSR consistency, we often pad/trim. Let's keep loop pad for now.
        
        cur_len = waveform.size(1)
        if cur_len < PreprocessConfig.AUDIO_LENGTH:
            # Loop pad
            repeats = (PreprocessConfig.AUDIO_LENGTH // cur_len) + 1
            looped = waveform.repeat(1, repeats)
            waveform = looped[:, :PreprocessConfig.AUDIO_LENGTH]
        else:
            waveform = waveform[:, :PreprocessConfig.AUDIO_LENGTH]
        
        return waveform

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
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='')
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
        
        # Init processor locally to avoid Multiprocessing issues
        processor = VideoProcessor(use_precropped=self.use_precropped) 
        tensor = processor.process(path)
        
        if tensor is None: 
            return torch.zeros((1, 3, PreprocessConfig.RESNET_INPUT_SIZE, PreprocessConfig.RESNET_INPUT_SIZE)), path
            
        return tensor, path

def collate_video_wrapper(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


# --- 5. BASE PREPROCESSOR ---

class BasePreprocessor(ABC):
    def __init__(self, data_root, use_precropped=False):
        self.data_root = data_root
        self.use_precropped = use_precropped
        
        # Load Extractors
        print(f"Initializing Extractors on {PreprocessConfig.DEVICE}...")
        self.visual_extractor = VisualFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        print(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def collect_metadata(self):
        """Should return list of dict: {'full_path', 'rel_path', 'text'}"""
        pass

    def run(self, output_manifest="manifest.jsonl", extract_features=True):
        print("Collecting metadata...")
        metadata = self.collect_metadata()
        print(f"   Found {len(metadata)} samples.")

        if not metadata: return

        # Optimization: Create a map for O(1) access
        metadata_map = {m['full_path']: m for m in metadata}

        # 1. Extract Features
        if extract_features:
            print("Running Multimodal Extraction (Audio + Visual)...")
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
            print(f"Starting processing loop for {total} items...")
            
            for videos, paths in tqdm(loader, desc="Processing"):
                for video_tensor, video_path in zip(videos, paths):
                    count += 1
                    if count % 100 == 0 or count == 1:
                        print(f"   [{count}/{total}] Processing {video_path}...")
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
                        
                        save_path = video_path.replace(".mpg", ".pt").replace(".mp4", ".pt").replace(".webm", ".pt")
                        torch.save(save_dict, save_path)
                        count += 1
                        
                    except Exception as e:
                        print(f"FAILED {video_path}: {e}")

        # 2. Save Manifest
        print(f"Saving manifest to {output_manifest}...")
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
        print("Done!")