import os
import json
import torch
import cv2
import numpy as np
import timm
import mediapipe as mp
from tqdm import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset

class PreprocessConfig:
    IMAGE_SIZE = 96       # Set to 96 (standard for mouth crop) instead of 224 (full face)
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoProcessor:
    """
    Dedicated class for cropping mouth region (ROI) from video using MediaPipe.
    Ported from legacy code to ensure feature quality.
    """
    def __init__(self):
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
            # Logic: Detect every 5 frames to optimize speed
            if i % 5 == 0 or prev_bbox is None:
                crop, prev_bbox = self.extract_mouth(frame, None)
            else:
                crop, prev_bbox = self.extract_mouth(frame, prev_bbox)
            
            # Normalize to [0, 1] and convert channel
            # Resize to 224x224 (or 96x96 depending on your ResNet config)
            # Here we resize to 224 to fit pre-trained ResNet18
            crop = cv2.resize(crop, (224, 224)) 
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            processed_frames.append(crop)

        # (T, H, W, C) -> (T, C, H, W)
        return torch.tensor(np.array(processed_frames), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    def extract_mouth(self, frame, prev_bbox=None):
        """Mouth cropping logic identical to legacy code"""
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
            
        # Fallback: Crop center of image if all else fails
        cy, cx = h // 2, w // 2
        r = PreprocessConfig.IMAGE_SIZE // 2
        return frame[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)], None


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

class RawVideoDataset(Dataset):
    """
    This dataset is now smarter: It calls VideoProcessor to crop mouth
    instead of just naive resizing.
    """
    def __init__(self, video_paths):
        self.video_paths = video_paths
        # Initialize Processor (MediaPipe) every time a worker is created
        # (Note: MediaPipe is not picklable so we init in __getitem__ or use a trick,
        # but for simplicity we init every call - a bit slower but safe for multiprocessing)
        
    def __len__(self): return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        
        # Locally init processor to avoid Pickling error in multithreading
        processor = VideoProcessor() 
        tensor = processor.process(path)
        
        if tensor is None: 
            # Fallback dummy
            return torch.zeros((1, 3, 224, 224)), path
            
        return tensor, path

def collate_video_wrapper(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


class BasePreprocessor(ABC):
    def __init__(self, data_root):
        self.data_root = data_root
        self.extractor = VisualFeatureExtractor()
        print(f"Initialized {self.__class__.__name__} on {PreprocessConfig.DEVICE}")

    @abstractmethod
    def collect_metadata(self):
        pass

    def run(self, output_manifest="manifest.jsonl", extract_features=True):
        print("🔍 Collecting metadata...")
        metadata = self.collect_metadata()
        if not metadata: return

        # 1. Extract Features (Includes Mouth Crop)
        if extract_features:
            print("⚙️  Extracting Visual Features (Mouth Crop + ResNet)...")
            video_paths = [m['full_path'] for m in metadata]
            
            # NUM_WORKERS = 0 for safety with MediaPipe at first (or set low)
            # MediaPipe sometimes conflicts with PyTorch DataLoader multithreading
            dataset = RawVideoDataset(video_paths)
            loader = DataLoader(
                dataset, 
                batch_size=1, 
                num_workers=0, # Set to 0 to avoid MediaPipe issues
                collate_fn=collate_video_wrapper
            )

            for videos, paths in tqdm(loader, desc="Extraction"):
                for video_tensor, video_path in zip(videos, paths):
                    video_tensor = video_tensor.to(PreprocessConfig.DEVICE)
                    features_list = []
                    
                    for i in range(0, len(video_tensor), PreprocessConfig.BATCH_SIZE):
                        batch = video_tensor[i : i + PreprocessConfig.BATCH_SIZE]
                        features_list.append(self.extractor(batch).cpu())
                    
                    full_features = torch.cat(features_list, dim=0) if features_list else torch.zeros((1, 512))
                    
                    save_path = video_path.replace(".mpg", ".npy")
                    np.save(save_path, full_features.numpy())

        # 2. Save Manifest
        print(f"Saving manifest...")
        with open(output_manifest, 'w', encoding='utf-8') as f:
            for item in metadata:
                entry = {
                    "rel_path": item['rel_path'],
                    "text": item['text'],
                    "duration": 0.0
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print("Done!")