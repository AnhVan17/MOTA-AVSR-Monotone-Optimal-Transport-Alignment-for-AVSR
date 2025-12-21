import os
import cv2
import torch
import whisper
import numpy as np
from typing import Dict, Optional
from .base import BaseDataset

class GridDataset(BaseDataset):
    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        data_root: str,
        use_precomputed_features: bool = False,
        max_samples: Optional[int] = None
    ):
        self.data_root = data_root
        self.use_features = use_precomputed_features
        super().__init__(manifest_path, tokenizer, max_samples)
    
    def parse_sample(self, sample_data: Dict) -> Dict:
        # sample_data comes from the manifest line
        rel_path = sample_data['rel_path']
        full_video_path = os.path.join(self.data_root, rel_path)
        
        # 1. Load Audio -> Log-Mel Spectrogram (80, 3000)
        audio_mel = self._load_audio_mel(full_video_path)
        
        # 2. Load Visual -> Raw Frames OR Precomputed Features
        if self.use_features:
            visual_data = self._load_visual_features(full_video_path)
        else:
            visual_data = self._load_raw_video(full_video_path)

        # 3. Load Text -> Parse .align file
        text = self._get_text_label(full_video_path, sample_data.get('text', ''))

        return {
            'audio': audio_mel,
            'visual': visual_data,
            'text': text,
            'rel_path': rel_path 
        }

    def _load_audio_mel(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            return torch.zeros((80, 3000), dtype=torch.float32)

        audio = whisper.load_audio(path)
        
        # Pad or trim to 30 seconds (480,000 samples at 16kHz)
        target_len = 480000
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
        else:
            audio = audio[:target_len]
            
        mel = whisper.log_mel_spectrogram(audio) 
        return mel

    def _load_raw_video(self, path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
            
        if not frames:
            return torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            
        tensor = torch.tensor(np.array(frames), dtype=torch.float32)
        tensor = tensor.permute(0, 3, 1, 2) # (T, H, W, C) -> (T, C, H, W)
        return tensor / 255.0

    def _load_visual_features(self, video_path: str) -> torch.Tensor:
        # Path Logic: s1/video/xyz.mpg -> s1/features/xyz.npy
        feat_path = video_path.replace(".mpg", ".npy").replace("video", "features")
        
        if not os.path.exists(feat_path):
            feat_path = video_path.replace(".mpg", ".npy")
            
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
            
        return torch.from_numpy(np.load(feat_path)).float()

    def _get_text_label(self, video_path: str, fallback_text: str) -> str:
            try:
                parent_dir = os.path.dirname(video_path) # .../s1_processed
                filename = os.path.basename(video_path)  # bbaf2n.mpg
                align_filename = filename.replace(".mpg", ".align") # bbaf2n.align
                
                align_path = os.path.join(parent_dir, "align", align_filename)
                
                if os.path.exists(align_path):
                    words = []
                    with open(align_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                word = parts[2]
                                if word not in ['sil', 'sp']:
                                    words.append(word)
                    return " ".join(words)
            except Exception:
                pass 

            return fallback_text