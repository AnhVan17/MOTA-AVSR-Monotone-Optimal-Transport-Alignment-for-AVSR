"""
Augmented Preprocessing Pipeline
=================================
Tạo NHIỀU versions của mỗi sample trong preprocessing:
- Clean version
- Noisy version (Gaussian noise)
- Augmented versions (time stretch, pitch shift, etc.)

→ Training NHANH như Pipeline cũ
→ Có thể so sánh Clean vs Noisy dễ dàng!
"""

import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torchaudio
from tqdm import tqdm


class AugmentationConfig:
    """Config cho augmentation"""
    
    # Audio augmentation
    ENABLE_NOISE = True
    ENABLE_TIME_STRETCH = True
    ENABLE_PITCH_SHIFT = True
    ENABLE_VOLUME = True
    
    # Noise levels
    NOISE_SNR_DB = [20, 15, 10]  # High, Medium, Low quality
    
    # Time stretch rates
    TIME_STRETCH_RATES = [0.9, 1.1]  # -10%, +10%
    
    # Pitch shift steps
    PITCH_SHIFT_STEPS = [-2, -1, 1, 2]  # semitones
    
    # Volume scales
    VOLUME_SCALES = [0.7, 0.8, 1.2, 1.3]
    
    # Video augmentation
    ENABLE_BRIGHTNESS = True
    ENABLE_CONTRAST = True
    ENABLE_HORIZONTAL_FLIP = True
    
    BRIGHTNESS_FACTORS = [0.8, 0.9, 1.1, 1.2]
    CONTRAST_FACTORS = [0.8, 0.9, 1.1, 1.2]


class AudioAugmenter:
    """
    Audio augmentation cho preprocessing
    Tạo nhiều versions của audio waveform
    """
    
    @staticmethod
    def add_gaussian_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add Gaussian noise with specified SNR
        
        Args:
            waveform: [1, T] audio waveform
            snr_db: Signal-to-noise ratio in dB (higher = cleaner)
                    20dB = high quality
                    15dB = medium quality  
                    10dB = low quality
        
        Returns:
            noisy_waveform: [1, T]
        """
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise
    
    @staticmethod
    def add_babble_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add babble noise (multiple speakers talking)
        More realistic than Gaussian noise
        """
        # Generate multi-tone "babble" noise
        T = waveform.size(1)
        sample_rate = 16000
        
        # Mix of different frequencies (simulate multiple speakers)
        freqs = [100, 250, 500, 1000, 2000, 4000]
        babble = torch.zeros_like(waveform)
        
        for freq in freqs:
            t = torch.arange(T).float() / sample_rate
            tone = torch.sin(2 * np.pi * freq * t)
            babble += tone * torch.randn(1) * 0.3
        
        # Normalize and scale by SNR
        babble = babble / babble.abs().max()
        
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        babble = babble * torch.sqrt(noise_power / babble.pow(2).mean())
        
        return waveform + babble
    
    @staticmethod
    def add_environmental_noise(waveform: torch.Tensor, noise_type: str, snr_db: float) -> torch.Tensor:
        """
        Add environmental noise (cafe, street, office, etc.)
        
        Args:
            noise_type: 'cafe', 'street', 'office', 'crowd'
        """
        T = waveform.size(1)
        sample_rate = 16000
        
        # Simulate different noise types with colored noise
        if noise_type == 'cafe':
            # Mid-frequency noise
            noise = torch.randn_like(waveform)
            # Apply bandpass filter ~300-3000 Hz
            noise = torchaudio.functional.highpass_biquad(noise, sample_rate, 300)
            noise = torchaudio.functional.lowpass_biquad(noise, sample_rate, 3000)
        
        elif noise_type == 'street':
            # Low-frequency rumble
            noise = torch.randn_like(waveform)
            noise = torchaudio.functional.lowpass_biquad(noise, sample_rate, 500)
        
        elif noise_type == 'office':
            # High-frequency hum
            noise = torch.randn_like(waveform)
            noise = torchaudio.functional.highpass_biquad(noise, sample_rate, 1000)
        
        else:  # 'crowd'
            # Wide-band noise
            noise = torch.randn_like(waveform)
        
        # Scale by SNR
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = noise * torch.sqrt(noise_power / noise.pow(2).mean())
        
        return waveform + noise
    
    @staticmethod
    def time_stretch(waveform: torch.Tensor, rate: float) -> torch.Tensor:
        """Stretch/compress time by rate (0.9 = faster, 1.1 = slower)"""
        if rate == 1.0:
            return waveform
        
        stretched = torchaudio.functional.resample(
            waveform,
            orig_freq=int(16000 * rate),
            new_freq=16000
        )
        
        # Pad/trim to original length
        target_len = waveform.size(1)
        if stretched.size(1) < target_len:
            padding = target_len - stretched.size(1)
            stretched = torch.nn.functional.pad(stretched, (0, padding))
        else:
            stretched = stretched[:, :target_len]
        
        return stretched
    
    @staticmethod
    def pitch_shift(waveform: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Shift pitch by n semitones"""
        if n_steps == 0:
            return waveform
        
        rate = 2 ** (n_steps / 12.0)
        return AudioAugmenter.time_stretch(waveform, rate)
    
    @staticmethod
    def volume_scale(waveform: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale volume"""
        return waveform * scale


class VideoAugmenter:
    """Video augmentation cho preprocessing"""
    
    @staticmethod
    def adjust_brightness(frames: torch.Tensor, factor: float) -> torch.Tensor:
        """Adjust brightness (factor > 1 = brighter)"""
        return torch.clamp(frames * factor, 0, 255 if frames.dtype == torch.uint8 else 1.0)
    
    @staticmethod
    def adjust_contrast(frames: torch.Tensor, factor: float) -> torch.Tensor:
        """Adjust contrast (factor > 1 = more contrast)"""
        mean = frames.mean()
        return torch.clamp((frames - mean) * factor + mean, 0, 255 if frames.dtype == torch.uint8 else 1.0)
    
    @staticmethod
    def horizontal_flip(frames: torch.Tensor) -> torch.Tensor:
        """Flip horizontally (OK for mouth region)"""
        return torch.flip(frames, dims=[3])
    
    @staticmethod
    def add_blur(frames: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Add Gaussian blur (simulate motion blur)"""
        import torch.nn.functional as F
        
        # Simple box blur for efficiency
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        kernel = kernel.to(frames.device)
        
        T, C, H, W = frames.shape
        frames_float = frames.float() / 255.0 if frames.dtype == torch.uint8 else frames
        
        blurred_list = []
        for c in range(C):
            channel = frames_float[:, c:c+1, :, :]
            blurred = F.conv2d(channel, kernel, padding=kernel_size//2)
            blurred_list.append(blurred)
        
        blurred = torch.cat(blurred_list, dim=1)
        
        if frames.dtype == torch.uint8:
            blurred = (blurred * 255).clamp(0, 255).byte()
        
        return blurred


class AugmentedPreprocessingPipeline:
    """
    Preprocessing pipeline tạo NHIỀU versions
    
    Output structure:
    processed_features/
        sample_001/
            clean.pt              ← Original
            noise_20db.pt         ← High quality noise
            noise_15db.pt         ← Medium quality noise
            noise_10db.pt         ← Low quality noise
            babble_20db.pt        ← Babble noise
            cafe_15db.pt          ← Cafe noise
            time_stretch_0.9.pt   ← Faster
            time_stretch_1.1.pt   ← Slower
            pitch_shift_-2.pt     ← Lower pitch
            bright_1.2.pt         ← Brighter video
            ...
    """
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
        # Load feature extractors
        from transformers import WhisperModel, WhisperFeatureExtractor
        import timm
        
        print("Loading Whisper...")
        self.whisper = WhisperModel.from_pretrained("openai/whisper-small")
        self.whisper.eval()
        
        self.whisper_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        
        print("Loading ResNet18...")
        self.resnet = timm.create_model('resnet18', pretrained=True, 
                                        num_classes=0, global_pool='')
        self.resnet.eval()
        
        self.audio_aug = AudioAugmenter()
        self.video_aug = VideoAugmenter()
        
        print("✅ Augmented preprocessing pipeline initialized")
    
    def create_augmented_versions(
        self,
        waveform: torch.Tensor,
        frames: torch.Tensor,
        text: torch.Tensor,
        sample_id: str,
        output_dir: Path
    ) -> List[Dict]:
        """
        Tạo NHIỀU augmented versions và extract features cho TẤT CẢ
        
        Returns:
            List of metadata dicts for each version
        """
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        versions = []
        
        # ==========================================
        # 1. CLEAN VERSION (Original)
        # ==========================================
        clean_data = self._extract_and_save(
            waveform, frames, text, sample_id,
            sample_dir / "clean.pt",
            version_name="clean"
        )
        versions.append(clean_data)
        
        # ==========================================
        # 2. NOISE VERSIONS (Gaussian)
        # ==========================================
        if self.config.ENABLE_NOISE:
            for snr_db in self.config.NOISE_SNR_DB:
                noisy_waveform = self.audio_aug.add_gaussian_noise(waveform, snr_db)
                
                noisy_data = self._extract_and_save(
                    noisy_waveform, frames, text, sample_id,
                    sample_dir / f"noise_{snr_db}db.pt",
                    version_name=f"noise_{snr_db}db"
                )
                versions.append(noisy_data)
        
        # ==========================================
        # 3. BABBLE NOISE
        # ==========================================
        babble_waveform = self.audio_aug.add_babble_noise(waveform, 15)
        babble_data = self._extract_and_save(
            babble_waveform, frames, text, sample_id,
            sample_dir / "babble_15db.pt",
            version_name="babble_15db"
        )
        versions.append(babble_data)
        
        # ==========================================
        # 4. ENVIRONMENTAL NOISE
        # ==========================================
        for noise_type in ['cafe', 'street']:
            env_waveform = self.audio_aug.add_environmental_noise(waveform, noise_type, 15)
            env_data = self._extract_and_save(
                env_waveform, frames, text, sample_id,
                sample_dir / f"{noise_type}_15db.pt",
                version_name=f"{noise_type}_15db"
            )
            versions.append(env_data)
        
        # ==========================================
        # 5. TIME STRETCH (optional, expensive)
        # ==========================================
        if self.config.ENABLE_TIME_STRETCH:
            for rate in self.config.TIME_STRETCH_RATES:
                stretched = self.audio_aug.time_stretch(waveform, rate)
                stretched_data = self._extract_and_save(
                    stretched, frames, text, sample_id,
                    sample_dir / f"stretch_{rate}.pt",
                    version_name=f"stretch_{rate}"
                )
                versions.append(stretched_data)
        
        # ==========================================
        # 6. VIDEO AUGMENTATIONS
        # ==========================================
        if self.config.ENABLE_BRIGHTNESS:
            for factor in [0.8, 1.2]:
                bright_frames = self.video_aug.adjust_brightness(frames, factor)
                bright_data = self._extract_and_save(
                    waveform, bright_frames, text, sample_id,
                    sample_dir / f"bright_{factor}.pt",
                    version_name=f"bright_{factor}"
                )
                versions.append(bright_data)
        
        return versions
    
    def _extract_and_save(
        self,
        waveform: torch.Tensor,
        frames: torch.Tensor,
        text: torch.Tensor,
        sample_id: str,
        save_path: Path,
        version_name: str
    ) -> Dict:
        """Extract features and save"""
        # Extract features
        with torch.no_grad():
            audio_features = self._extract_audio(waveform)
            visual_features = self._extract_visual(frames)
        
        # Save
        torch.save({
            'audio': audio_features,
            'visual': visual_features,
            'text': text,
            'id': sample_id,
            'version': version_name
        }, save_path)
        
        return {
            'id': sample_id,
            'path': str(save_path),
            'version': version_name,
            'text': text
        }
    
    def _extract_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract Whisper features"""
        audio_np = waveform.squeeze(0).cpu().numpy()
        
        inputs = self.whisper_processor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        feats = self.whisper.encoder(
            inputs.input_features.to(self.whisper.device)
        ).last_hidden_state.cpu()
        
        return feats.squeeze(0)
    
    def _extract_visual(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract ResNet features"""
        T, C, H, W = frames.shape
        
        # Normalize if needed
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        
        # Process in batches
        batch_size = 8
        visual_list = []
        
        for i in range(0, T, batch_size):
            batch = frames[i:i+batch_size].to(self.resnet.device)
            
            # Normalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch.device)
            batch = (batch - mean) / std
            
            feats = self.resnet(batch)
            
            if feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])
            
            visual_list.append(feats.cpu())
        
        return torch.cat(visual_list, dim=0)


class AugmentedDataset:
    """
    Dataset cho augmented training
    
    Có thể load:
    - Chỉ clean data
    - Chỉ noisy data
    - Mix clean + noisy
    """
    
    def __init__(
        self,
        base_manifest: str,
        noise_types: Optional[List[str]] = None,
        mix_ratio: float = 0.5
    ):
        """
        Args:
            base_manifest: Manifest file với clean paths
            noise_types: List of noise types to include, e.g. ['noise_20db', 'babble_15db']
                        If None, use clean only
            mix_ratio: Ratio of clean:noisy samples (0.5 = 50% clean, 50% noisy)
        """
        self.base_paths = self._load_manifest(base_manifest)
        self.noise_types = noise_types or []
        self.mix_ratio = mix_ratio
        
        # Build full paths list
        self.all_paths = self._build_paths()
        
        print(f"Dataset size: {len(self.all_paths)}")
        print(f"  Clean: {sum(1 for p in self.all_paths if 'clean' in str(p))}")
        print(f"  Noisy: {sum(1 for p in self.all_paths if 'clean' not in str(p))}")
    
    def _load_manifest(self, manifest_path: str) -> List[str]:
        """Load base manifest (clean paths)"""
        import json
        
        paths = []
        with open(manifest_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                paths.append(item['path'])
        return paths
    
    def _build_paths(self) -> List[str]:
        """Build full paths including augmented versions"""
        all_paths = []
        
        for base_path in self.base_paths:
            base_path = Path(base_path)
            sample_id = base_path.stem
            sample_dir = base_path.parent / sample_id
            
            # Add clean version
            if random.random() < self.mix_ratio or not self.noise_types:
                clean_path = sample_dir / "clean.pt"
                if clean_path.exists():
                    all_paths.append(str(clean_path))
            
            # Add noisy versions
            if self.noise_types and random.random() < (1 - self.mix_ratio):
                noise_type = random.choice(self.noise_types)
                noisy_path = sample_dir / f"{noise_type}.pt"
                if noisy_path.exists():
                    all_paths.append(str(noisy_path))
        
        return all_paths
    
    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        """Load sample (already has extracted features)"""
        data = torch.load(self.all_paths[idx], map_location='cpu')
        return {
            'audio': data['audio'],
            'visual': data['visual'],
            'target': data['text'],
            'id': data['id'],
            'version': data.get('version', 'clean')
        }


def create_comparison_configs():
    """
    Tạo configs để so sánh clean vs noisy training
    """
    
    configs = {
        # Experiment 1: Clean only
        'clean_only': {
            'name': 'Clean Only Baseline',
            'noise_types': None,
            'mix_ratio': 1.0,  # 100% clean
            'description': 'Train trên clean data only'
        },
        
        # Experiment 2: Noisy only (20dB)
        'noisy_20db': {
            'name': 'Noisy 20dB',
            'noise_types': ['noise_20db'],
            'mix_ratio': 0.0,  # 0% clean, 100% noisy
            'description': 'Train trên high-quality noisy data'
        },
        
        # Experiment 3: Noisy only (15dB)
        'noisy_15db': {
            'name': 'Noisy 15dB',
            'noise_types': ['noise_15db'],
            'mix_ratio': 0.0,
            'description': 'Train trên medium-quality noisy data'
        },
        
        # Experiment 4: Mixed clean + noisy (50-50)
        'mixed_50_50': {
            'name': 'Mixed 50-50',
            'noise_types': ['noise_20db', 'noise_15db', 'babble_15db'],
            'mix_ratio': 0.5,  # 50% clean, 50% noisy
            'description': 'Train trên mix clean + noisy'
        },
        
        # Experiment 5: Multi-condition training
        'multi_condition': {
            'name': 'Multi-Condition',
            'noise_types': ['noise_20db', 'noise_15db', 'noise_10db', 
                           'babble_15db', 'cafe_15db', 'street_15db'],
            'mix_ratio': 0.3,  # 30% clean, 70% noisy
            'description': 'Train trên nhiều điều kiện khác nhau'
        },
    }
    
    return configs


def print_usage_example():
    """Ví dụ sử dụng"""
    print("="*70)
    print("CÁCH SỬ DỤNG: So sánh Clean vs Noisy Training")
    print("="*70)
    
    print("\n📋 BƯỚC 1: Preprocessing (tạo augmented versions)")
    print("""
    config = AugmentationConfig()
    pipeline = AugmentedPreprocessingPipeline(config)
    
    # Tạo nhiều versions cho mỗi sample
    for sample in dataset:
        versions = pipeline.create_augmented_versions(
            waveform, frames, text, sample_id, output_dir
        )
    
    → Output: sample_001/clean.pt, sample_001/noise_20db.pt, ...
    """)
    
    print("\n📋 BƯỚC 2: Training với different configs")
    print("""
    # Experiment 1: Clean only
    dataset_clean = AugmentedDataset(
        manifest, 
        noise_types=None,
        mix_ratio=1.0
    )
    trainer.train(dataset_clean)
    → WER on clean test: ??%
    → WER on noisy test: ??%
    
    # Experiment 2: Noisy only (15dB)
    dataset_noisy = AugmentedDataset(
        manifest,
        noise_types=['noise_15db'],
        mix_ratio=0.0
    )
    trainer.train(dataset_noisy)
    → WER on clean test: ??%
    → WER on noisy test: ??%
    
    # Experiment 3: Mixed 50-50
    dataset_mixed = AugmentedDataset(
        manifest,
        noise_types=['noise_20db', 'noise_15db', 'babble_15db'],
        mix_ratio=0.5
    )
    trainer.train(dataset_mixed)
    → WER on clean test: ??%
    → WER on noisy test: ??%
    """)
    
    print("\n📊 BƯỚC 3: So sánh kết quả")
    print("""
    ┌─────────────────┬──────────────┬──────────────┬──────────────┐
    │ Training Data   │ Clean WER    │ Noisy WER    │ Avg WER      │
    ├─────────────────┼──────────────┼──────────────┼──────────────┤
    │ Clean Only      │ 18%          │ 35%          │ 26.5%        │
    │ Noisy Only      │ 22%          │ 25%          │ 23.5% ✅     │
    │ Mixed 50-50     │ 19%          │ 26%          │ 22.5% ✅✅   │
    │ Multi-Condition │ 20%          │ 24%          │ 22.0% ✅✅✅ │
    └─────────────────┴──────────────┴──────────────┴──────────────┘
    
    → Multi-condition training thường cho kết quả tốt nhất!
    """)


if __name__ == "__main__":
    print_usage_example()modal run scripts/modal/modal_preprocess_vicocktail.py --action process --phase 1