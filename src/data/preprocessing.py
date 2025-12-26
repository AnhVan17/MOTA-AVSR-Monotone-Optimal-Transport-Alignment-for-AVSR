import torch
import io
import json
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import logging
import gc
from transformers import WhisperModel, WhisperFeatureExtractor
import timm
from decord import VideoReader, cpu
import cv2
import numpy as np
import mediapipe as mp
from torchvision import transforms
import unicodedata
from .tokenizer import VietnameseCharTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingConfig:
    # Video params
    TARGET_FPS = 25
    IMAGE_SIZE = 96
    MAX_FRAMES = 375
    
    # Audio params
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_LENGTH = 240000 
    
    # Paths
    INPUT_DIR = "./data/raw_tars"
    OUTPUT_DIR = "./data/processed_features"
    MANIFEST_DIR = "./data/manifests"
    
    # Processing
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8
    VAL_SPLIT = 0.15
    SEED = 42


class FeatureExtractor:
    """Extract Whisper + ResNet18 features"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.device = config.DEVICE
        
        logger.info(f"Loading models on {self.device}...")
        
        # Whisper-small encoder (audio)
        self.audio_model = WhisperModel.from_pretrained(
            "openai/whisper-small"
        ).to(self.device).eval()
        
        self.whisper_processor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-small"
        )
        
        # ResNet18 (visual)
        self.visual_model = timm.create_model(
            'resnet18',
            pretrained=True,
            num_classes=0,  
            global_pool=''   
        ).to(self.device).eval()
        
        logger.info("✅ Models loaded")
    
    @torch.no_grad()
    def extract_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract Whisper features
        
        Args:
            waveform: [1, 240000] audio waveform
            
        Returns:
            audio_feats: [T_a, 768] Whisper features
        """
        audio_np = waveform.squeeze(0).cpu().numpy()
        
        # Whisper feature extraction
        inputs = self.whisper_processor(
            audio_np,
            sampling_rate=self.config.AUDIO_SAMPLE_RATE,
            return_tensors="pt"
        )
        
        # Encode
        audio_feats = self.audio_model.encoder(
            inputs.input_features.to(self.device)
        ).last_hidden_state.cpu()
        
        return audio_feats.squeeze(0) 
    
    @torch.no_grad()
    def extract_visual(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract ResNet18 features
        
        Args:
            video_tensor: [3, T, H, W] video frames
            
        Returns:
            visual_feats: [T_v, 512] ResNet features
        """
        C, T, H, W = video_tensor.shape
        
        frames = video_tensor.permute(1, 0, 2, 3).to(self.device)
        
        visual_list = []
        for i in range(0, T, self.config.BATCH_SIZE):
            batch = frames[i:i+self.config.BATCH_SIZE]
            feats = self.visual_model(batch) 
            
            if feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])  
            
            visual_list.append(feats.cpu())
        
        visual_feats = torch.cat(visual_list, dim=0) 
        return visual_feats


class VideoProcessor:
    """Process video: extract mouth ROI"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_mouth(self, frame: np.ndarray, prev_bbox=None) -> Tuple[np.ndarray, Optional[tuple]]:
        """Extract mouth region from frame"""
        h, w = frame.shape[:2]
        if prev_bbox is not None:
            x1, y1, x2, y2 = prev_bbox
            if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
                    return crop_resized, prev_bbox
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                
                # Mouth landmarks
                mouth_idx = [13, 14, 61, 291, 78, 308, 324, 17]
                xs = [lm[i].x * w for i in mouth_idx]
                ys = [lm[i].y * h for i in mouth_idx]
                
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                radius = max(
                    int((max(xs) - min(xs)) * 1.8) // 2,
                    int((max(ys) - min(ys)) * 1.8) // 2,
                    self.config.IMAGE_SIZE // 2
                )
                
                y1 = max(0, cy - radius)
                y2 = min(h, cy + radius)
                x1 = max(0, cx - radius)
                x2 = min(w, cx + radius)
                
                crop = frame[y1:y2, x1:x2]
                bbox = (x1, y1, x2, y2)
                
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
                    return crop_resized, bbox
        except:
            pass
        
        cy, cx = h // 2, w // 2
        r = self.config.IMAGE_SIZE // 2
        crop = frame[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]
        
        if crop.size > 0:
            crop_resized = cv2.resize(crop, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            return crop_resized, None
        
        return np.zeros((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3), dtype=np.uint8), None
    
    def process(self, video_bytes: bytes) -> Optional[torch.Tensor]:
        """
        Process video bytes → tensor
        
        Args:
            video_bytes: Raw video bytes
            
        Returns:
            video_tensor: [3, T, H, W] or None
        """
        try:
            vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
            
            fps = vr.get_avg_fps()
            step = max(1, int(round(fps / self.config.TARGET_FPS)))
            indices = list(range(0, len(vr), step))[:self.config.MAX_FRAMES]
            
            if not indices:
                return None
            
            frames = vr.get_batch(indices).asnumpy()
            
            processed = []
            prev_bbox = None
            
            for i, frame in enumerate(frames):
                if i % 5 == 0:
                    crop, prev_bbox = self.extract_mouth(frame, None)
                else:
                    crop, prev_bbox = self.extract_mouth(frame, prev_bbox)
                
                processed.append(self.transform(crop))
            
            video_tensor = torch.stack(processed).permute(1, 0, 2, 3)
            if video_tensor.size(1) < self.config.MAX_FRAMES:
                video_tensor = self._loop_pad_video(video_tensor, self.config.MAX_FRAMES)
            
            return video_tensor
            
        except Exception as e:
            logger.debug(f"Video processing failed: {e}")
            return None
    
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
        C, T, H, W = video_tensor.shape
        if T >= target_frames:
            return video_tensor[:, :target_frames]
        
        repeats = (target_frames // T) + 1
        
        looped = video_tensor.repeat(1, repeats, 1, 1)
        return looped[:, :target_frames]


class AudioProcessor:
    """Process audio: extract waveform"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def process(self, video_bytes: bytes) -> Optional[torch.Tensor]:
        """
        Extract audio from video
        
        Args:
            video_bytes: Raw video bytes
            
        Returns:
            waveform: [1, 240000] or None (15s at 16kHz)
        """
        result = self._try_pyav(video_bytes)
        if result is not None:
            return result
        
        result = self._try_ffmpeg(video_bytes)
        if result is not None:
            return result
        
        return torch.zeros(1, self.config.AUDIO_LENGTH)
    
    def _try_pyav(self, video_bytes: bytes) -> Optional[torch.Tensor]:
        """Extract with PyAV"""
        try:
            import av
            
            container = av.open(io.BytesIO(video_bytes))
            audio_stream = next((s for s in container.streams.audio), None)
            
            if not audio_stream:
                container.close()
                return None
            
            resampler = av.AudioResampler(
                format='s16p',
                layout='mono',
                rate=self.config.AUDIO_SAMPLE_RATE
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
            
        except:
            return None
    
    def _try_ffmpeg(self, video_bytes: bytes) -> Optional[torch.Tensor]:
        """Extract with FFmpeg"""
        try:
            import subprocess
            import soundfile as sf
            
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', 'pipe:0',
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.config.AUDIO_SAMPLE_RATE),
                '-ac', '1',
                '-f', 'wav', 'pipe:1'
            ]
            
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            
            audio_data, _ = proc.communicate(input=video_bytes, timeout=15)
            
            if len(audio_data) < 100:
                return None
            
            waveform, _ = sf.read(io.BytesIO(audio_data), dtype='float32')
            return self._normalize_pad(torch.from_numpy(waveform).unsqueeze(0))
            
        except:
            return None
    
    def _normalize_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """RMS normalize + loop pad/trim"""
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 1e-8:
            waveform = waveform * (0.1 / rms)
        
        # Loop pad or trim to AUDIO_LENGTH
        cur_len = waveform.size(1)
        if cur_len < self.config.AUDIO_LENGTH:
            waveform = self._loop_pad_audio(waveform, self.config.AUDIO_LENGTH)
        else:
            waveform = waveform[:, :self.config.AUDIO_LENGTH]
        
        return waveform
    
    def _loop_pad_audio(self, waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Loop pad audio instead of zero padding.
        This helps model learn better from short audio clips.
        
        Args:
            waveform: [1, T] audio waveform
            target_length: Target length in samples
            
        Returns:
            Padded waveform [1, target_length]
        """
        cur_len = waveform.size(1)
        if cur_len >= target_length:
            return waveform[:, :target_length]
        
        if cur_len == 0:
            return torch.zeros(1, target_length)
        
        # Calculate how many times to repeat
        repeats = (target_length // cur_len) + 1
        
        # Repeat and trim
        looped = waveform.repeat(1, repeats)
        return looped[:, :target_length]


class PreprocessingPipeline:
    """Complete preprocessing pipeline"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # Initialize processors
        self.video_processor = VideoProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.feature_extractor = FeatureExtractor(config)
        
        # Initialize tokenizer
        self.tokenizer = VietnameseCharTokenizer()
        
        logger.info("✅ Preprocessing pipeline initialized")
    
    def process_sample(self, sample: Dict, sample_id: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Process single sample
        
        Args:
            sample: WebDataset sample
            sample_id: Sample identifier
            
        Returns:
            (result_dict, error_msg)
        """
        try:
            # Extract video bytes
            video_bytes = self._find_video(sample)
            if not video_bytes or len(video_bytes) < 1024:
                return None, "no_video"
            
            # Extract text
            text = self._find_text(sample)
            text_norm = self._normalize_text(text)
            if not text_norm or len(text_norm) < 2:
                return None, "no_text"
            
            # Process video
            video_tensor = self.video_processor.process(video_bytes)
            if video_tensor is None:
                return None, "video_fail"
            
            # Process audio
            audio_waveform = self.audio_processor.process(video_bytes)
            if audio_waveform is None:
                return None, "audio_fail"
            
            # Extract features
            audio_feats = self.feature_extractor.extract_audio(audio_waveform)
            visual_feats = self.feature_extractor.extract_visual(video_tensor)
            
            # Tokenize text
            token_ids = self.tokenizer.encode(text_norm)
            token_tensor = torch.tensor(token_ids, dtype=torch.long)
            
            # Return processed sample
            return {
                'id': sample_id,
                'audio': audio_feats,      
                'visual': visual_feats,    
                'text': token_tensor,      
                'text_raw': text_norm
            }, None
            
        except Exception as e:
            return None, f"error: {str(e)}"
    
    @staticmethod
    def _find_video(sample: Dict) -> Optional[bytes]:
        """Find video bytes in sample"""
        for key in ['mp4', 'video', 'avi', 'mov', 'webm']:
            if key in sample and isinstance(sample[key], bytes):
                return sample[key]
        
        for val in sample.values():
            if isinstance(val, bytes) and len(val) > 10240:
                return val
        
        return None
    
    @staticmethod
    def _find_text(sample: Dict) -> str:
        """
        Tìm text transcript. Ưu tiên JSON và các key rõ ràng trước .txt
        """
        # 1. Ưu tiên tìm trong file JSON hoặc key 'transcript'
        for key in ['json', 'transcript', 'label']:
            if key in sample:
                content = sample[key]
                if isinstance(content, dict):
                    # Tìm trong các field phổ biến của JSON
                    for sub_key in ['text', 'transcript', 'content', 'label', 'caption']:
                        if sub_key in content and isinstance(content[sub_key], str):
                            return content[sub_key]
                elif isinstance(content, str):
                    return content
                elif isinstance(content, bytes):
                    try:
                        return content.decode('utf-8').strip()
                    except:
                        pass

        # 2. Mới tìm đến key 'txt' hoặc 'text' (độ tin cậy thấp hơn)
        for key in ['txt', 'text']:
            if key in sample:
                content = sample[key]
                text_str = ""
                
                if isinstance(content, bytes):
                    try:
                        text_str = content.decode('utf-8').strip()
                    except:
                        continue
                elif isinstance(content, str):
                    text_str = content.strip()

                # 3. VALIDATION QUAN TRỌNG:
                # Bỏ qua nếu text trông giống đường dẫn file hoặc quá ngắn
                if not text_str: continue
                if text_str.endswith('.mp4') or text_str.endswith('.wav'): continue
                if '/' in text_str or '\\' in text_str: # Có thể là đường dẫn
                    # Kiểm tra kỹ hơn: nếu có dấu câu tiếng Việt thì giữ, nếu toàn ascii và gạch chéo thì bỏ
                    vi_chars = 'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'
                    if not any(c in text_str.lower() for c in vi_chars):
                        continue 
                
                return text_str

        return ""
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize Vietnamese text using NFC and cleanup"""
        if not text:
            return ""
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        # Lowercase & strip
        text = text.lower().strip()
        # Remove extra whitespaces
        text = " ".join(text.split())
        return text
    
    def run(self, input_dir: str, output_dir: str, manifest_dir: str):
        """
        Run preprocessing pipeline
        
        Args:
            input_dir: Directory with TAR files
            output_dir: Output directory for .pt files
            manifest_dir: Output directory for manifests
        """
        import webdataset as wds
        
        logger.info("="*70)
        logger.info("Starting Preprocessing")
        logger.info("="*70)
        input_path = Path(input_dir)
        tar_files = list(input_path.glob("*.tar"))
        
        if not tar_files:
            logger.error(f"No TAR files found in {input_dir}")
            return
        
        train_tars = [f for f in tar_files if 'train' in f.name.lower()]
        test_tars = [f for f in tar_files if 'test' in f.name.lower()]
        
        logger.info(f"Found {len(train_tars)} train TARs, {len(test_tars)} test TARs")
        
        all_metadata = []
        
        for tar_path in tqdm(train_tars + test_tars, desc="Processing TARs"):
            is_test = 'test' in tar_path.name.lower()
            tar_name = tar_path.stem
            
            save_dir = Path(output_dir) / tar_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            dataset = wds.WebDataset(f"file://{tar_path}").decode()

            for idx, sample in enumerate(tqdm(dataset, desc=f"  {tar_name}", leave=False)):
                sample_id = sample.get('__key__', f"{tar_name}_{idx:06d}")
                result, error = self.process_sample(sample, sample_id)
                if result:
                    out_file = save_dir / f"{sample_id}.pt"
                    torch.save(result, out_file)
                    all_metadata.append({
                        'id': sample_id,
                        'path': str(out_file.relative_to(Path(output_dir).parent)),
                        'text': result['text_raw'],
                        'is_test': is_test
                    })
                
                if idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        self._create_manifests(all_metadata, manifest_dir)
        
        logger.info("="*70)
        logger.info("Preprocessing Complete!")
        logger.info("="*70)
    
    def _create_manifests(self, metadata: list, manifest_dir: str):
        """Create train/val/test manifests"""
        test_samples = [m for m in metadata if m['is_test']]
        train_val = [m for m in metadata if not m['is_test']]
        
        random.seed(self.config.SEED)
        random.shuffle(train_val)
        split_idx = int(len(train_val) * (1 - self.config.VAL_SPLIT))
        train_samples = train_val[:split_idx]
        val_samples = train_val[split_idx:]
        
        manifest_path = Path(manifest_dir)
        manifest_path.mkdir(parents=True, exist_ok=True)
        
        def save_manifest(samples, filename):
            with open(manifest_path / filename, 'w', encoding='utf-8') as f:
                for item in samples:
                    f.write(json.dumps({
                        'id': item['id'],
                        'path': item['path'],
                        'text': item['text']
                    }, ensure_ascii=False) + '\n')
            logger.info(f"  {filename}: {len(samples)} samples")
        
        save_manifest(train_samples, 'train.jsonl')
        save_manifest(val_samples, 'val.jsonl')
        save_manifest(test_samples, 'test.jsonl')


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data/raw_tars')
    parser.add_argument('--output_dir', type=str, default='./data/processed_features')
    parser.add_argument('--manifest_dir', type=str, default='./data/manifests')
    args = parser.parse_args()
    
    config = PreprocessingConfig()
    pipeline = PreprocessingPipeline(config)
    pipeline.run(args.input_dir, args.output_dir, args.manifest_dir)


if __name__ == "__main__":
    main()