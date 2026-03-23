"""
ViCocktail Preprocessing with Noise Augmentation
=================================================
Modified preprocessing pipeline để tạo clean + noisy versions
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# Import noise generator
from vietnamese_noise_augmentation import (
    VietnameseNoiseGenerator,
    get_vietnamese_augmentation_configs
)

logger = logging.getLogger(__name__)


class ViCocktailPreprocessingWithNoise:
    """
    Preprocessing pipeline cho ViCocktail với noise augmentation
    
    Output structure:
    processed_features/
        sample_001/
            clean/
                audio.pt        ← Whisper features từ clean audio
                visual.pt       ← ResNet features
                text.pt         ← Token IDs
            noise_15db/
                audio.pt        ← Whisper features từ noisy audio
                visual.pt       ← Same visual features
                text.pt         ← Same text
            cafe_busy/
                audio.pt
                visual.pt
                text.pt
            ...
    """
    
    def __init__(self):
        # Initialize feature extractors
        from transformers import WhisperModel, WhisperFeatureExtractor
        import timm
        
        print("Loading Whisper encoder...")
        self.whisper = WhisperModel.from_pretrained("openai/whisper-small")
        self.whisper.eval()
        
        self.whisper_processor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-small"
        )
        
        print("Loading ResNet18...")
        self.resnet = timm.create_model(
            'resnet18',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        self.resnet.eval()
        
        # Initialize noise generator
        print("Initializing Vietnamese noise generator...")
        self.noise_generator = VietnameseNoiseGenerator()
        
        # Get augmentation configs
        self.aug_configs = get_vietnamese_augmentation_configs()
        
        print(f"✅ Pipeline initialized with {len(self.aug_configs)} augmentation types")
    
    def process_sample_with_noise(
        self,
        waveform: torch.Tensor,
        frames: torch.Tensor,
        text: torch.Tensor,
        sample_id: str,
        output_dir: Path,
        noise_subset: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Process 1 sample → Tạo NHIỀU versions (clean + noisy)
        
        Args:
            waveform: [1, 240000] clean audio waveform
            frames: [T, 3, H, W] video frames
            text: [L] token IDs
            sample_id: Sample identifier
            output_dir: Base output directory
            noise_subset: Chỉ tạo subset of noise types (để tiết kiệm storage)
                         Nếu None, tạo TẤT CẢ
        
        Returns:
            List of metadata dicts
        """
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_list = []
        
        # Filter configs if subset specified
        if noise_subset:
            configs_to_use = [c for c in self.aug_configs if c['name'] in noise_subset]
        else:
            configs_to_use = self.aug_configs
        
        print(f"  Processing {sample_id}: {len(configs_to_use)} versions")
        
        # Extract visual features ONCE (same for all versions)
        with torch.no_grad():
            visual_features = self._extract_visual(frames)
        
        # Create each version
        for config in tqdm(configs_to_use, desc=f"  {sample_id}", leave=False):
            version_name = config['name']
            version_dir = sample_dir / version_name
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # ================================================
            # 1. Apply audio augmentation
            # ================================================
            if config['type'] is None:
                # Clean version - no augmentation
                augmented_waveform = waveform
            else:
                # Apply noise
                kwargs = {k: v for k, v in config.items() 
                         if k not in ['name', 'type']}
                augmented_waveform = self.noise_generator.augment(
                    waveform,
                    config['type'],
                    **kwargs
                )
            
            # ================================================
            # 2. Extract audio features from augmented audio
            # ================================================
            with torch.no_grad():
                audio_features = self._extract_audio(augmented_waveform)
            
            # ================================================
            # 3. Save features
            # ================================================
            torch.save(audio_features, version_dir / "audio.pt")
            torch.save(visual_features, version_dir / "visual.pt")
            torch.save(text, version_dir / "text.pt")
            
            # ================================================
            # 4. Save metadata
            # ================================================
            metadata_list.append({
                'id': sample_id,
                'version': version_name,
                'audio_path': str(version_dir / "audio.pt"),
                'visual_path': str(version_dir / "visual.pt"),
                'text_path': str(version_dir / "text.pt"),
                'noise_type': config.get('type', 'clean'),
                'snr_db': config.get('snr_db', None)
            })
        
        return metadata_list
    
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
        
        return feats.squeeze(0)  # [T, 768]
    
    def _extract_visual(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract ResNet features"""
        T, C, H, W = frames.shape
        
        # Normalize if needed
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        
        # Process in batches
        batch_size = 8
        visual_list = []
        
        for i in range(0, T, batch_size):
            batch = frames[i:i+batch_size].to(self.resnet.device)
            feats = self.resnet(batch)
            
            if feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])
            
            visual_list.append(feats.cpu())
        
        return torch.cat(visual_list, dim=0)  # [T, 512]


class NoisyDatasetLoader:
    """
    Dataset loader cho noisy training
    Có thể chọn load clean/noisy/mixed
    """
    
    def __init__(
        self,
        base_dir: Path,
        noise_versions: Optional[List[str]] = None,
        sample_rate: float = 1.0
    ):
        """
        Args:
            base_dir: Base directory với structure sample_id/version/
            noise_versions: List versions to load, e.g. ['clean', 'noise_15db']
                           If None, load ALL versions
            sample_rate: Sample rate (1.0 = use all, 0.5 = use 50%)
        """
        self.base_dir = Path(base_dir)
        self.noise_versions = noise_versions
        self.sample_rate = sample_rate
        
        # Scan available samples
        self.samples = self._scan_samples()
        
        print(f"✅ Dataset loaded: {len(self.samples)} samples")
        if noise_versions:
            print(f"   Versions: {noise_versions}")
    
    def _scan_samples(self) -> List[Dict]:
        """Scan và build sample list"""
        samples = []
        
        # Iterate through sample directories
        for sample_dir in sorted(self.base_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            
            sample_id = sample_dir.name
            
            # Iterate through versions
            for version_dir in sample_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                
                version_name = version_dir.name
                
                # Filter by noise_versions if specified
                if self.noise_versions and version_name not in self.noise_versions:
                    continue
                
                # Check if all files exist
                audio_file = version_dir / "audio.pt"
                visual_file = version_dir / "visual.pt"
                text_file = version_dir / "text.pt"
                
                if all([audio_file.exists(), visual_file.exists(), text_file.exists()]):
                    # Sample rate filtering
                    import random
                    if random.random() < self.sample_rate:
                        samples.append({
                            'id': sample_id,
                            'version': version_name,
                            'audio_path': audio_file,
                            'visual_path': visual_file,
                            'text_path': text_file
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load sample"""
        sample_info = self.samples[idx]
        
        return {
            'audio': torch.load(sample_info['audio_path'], map_location='cpu'),
            'visual': torch.load(sample_info['visual_path'], map_location='cpu'),
            'target': torch.load(sample_info['text_path'], map_location='cpu'),
            'id': sample_info['id'],
            'version': sample_info['version']
        }


# ============================================================================
# EXPERIMENT CONFIGS
# ============================================================================

def get_experiment_configs() -> Dict[str, Dict]:
    """
    Configs cho các experiments so sánh
    """
    
    return {
        # Experiment 1: Clean only (baseline)
        'exp1_clean_only': {
            'name': 'Clean Only Baseline',
            'noise_versions': ['clean'],
            'description': 'Train chỉ trên clean data',
            'expected_wer': {
                'clean_test': '18-20%',
                'noisy_test': '30-35%'
            }
        },
        
        # Experiment 2: Single noise type (15dB)
        'exp2_noise_15db': {
            'name': 'Gaussian Noise 15dB',
            'noise_versions': ['noise_15db'],
            'description': 'Train chỉ trên noisy data (medium quality)',
            'expected_wer': {
                'clean_test': '20-22%',
                'noisy_test': '22-25%'
            }
        },
        
        # Experiment 3: Mixed clean + single noise
        'exp3_mixed_clean_noise': {
            'name': 'Mixed Clean + Noise 15dB',
            'noise_versions': ['clean', 'noise_15db'],
            'description': 'Train trên mix 50-50 clean + noisy',
            'expected_wer': {
                'clean_test': '19-21%',
                'noisy_test': '24-26%'
            }
        },
        
        # Experiment 4: Vietnamese-specific noise
        'exp4_vietnamese_noise': {
            'name': 'Vietnamese-Specific Noise',
            'noise_versions': ['clean', 'street_noisy', 'cafe_busy', 'motorbike_near'],
            'description': 'Train trên các noise types phổ biến ở VN',
            'expected_wer': {
                'clean_test': '19-21%',
                'noisy_test': '23-25%'
            }
        },
        
        # Experiment 5: Multi-condition (best robustness)
        'exp5_multi_condition': {
            'name': 'Multi-Condition Training',
            'noise_versions': [
                'clean',
                'noise_15db',
                'babble_medium',
                'street_noisy',
                'cafe_busy',
                'motorbike_near',
                'reverb_small',
                'phone_mobile'
            ],
            'description': 'Train trên nhiều điều kiện khác nhau',
            'expected_wer': {
                'clean_test': '20-22%',
                'noisy_test': '22-24%'  # Best overall!
            }
        },
        
        # Experiment 6: Lite version (tiết kiệm storage)
        'exp6_lite': {
            'name': 'Lite Version (3 noise types)',
            'noise_versions': ['clean', 'noise_15db', 'cafe_busy'],
            'description': 'Chỉ 3 versions để tiết kiệm storage',
            'expected_wer': {
                'clean_test': '19-21%',
                'noisy_test': '24-27%'
            }
        }
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def print_usage_example():
    """Hướng dẫn sử dụng"""
    
    print("="*70)
    print("🇻🇳 VICOCKTAIL PREPROCESSING WITH NOISE - USAGE GUIDE")
    print("="*70)
    
    print("\n📋 STEP 1: Preprocessing (tạo clean + noisy versions)")
    print("""
from vicocktail_preprocessing_with_noise import ViCocktailPreprocessingWithNoise

# Initialize pipeline
pipeline = ViCocktailPreprocessingWithNoise()

# Option A: Tạo TẤT CẢ noise types (~18 versions)
for sample in dataset:
    metadata = pipeline.process_sample_with_noise(
        waveform, frames, text, sample_id,
        output_dir=Path("./processed_features"),
        noise_subset=None  # All types
    )

# Option B: Chỉ tạo 1 vài types (tiết kiệm storage)
noise_subset = ['clean', 'noise_15db', 'cafe_busy', 'street_noisy']
for sample in dataset:
    metadata = pipeline.process_sample_with_noise(
        waveform, frames, text, sample_id,
        output_dir=Path("./processed_features"),
        noise_subset=noise_subset  # Only these 4
    )

→ Output: processed_features/sample_001/clean/audio.pt
                                        /noise_15db/audio.pt
                                        /cafe_busy/audio.pt
                                        ...
    """)
    
    print("\n📋 STEP 2: Training với different experiments")
    print("""
from vicocktail_preprocessing_with_noise import NoisyDatasetLoader

# Experiment 1: Clean only
dataset_clean = NoisyDatasetLoader(
    base_dir="./processed_features",
    noise_versions=['clean']
)
model_clean = train(dataset_clean)

# Experiment 2: Noisy only
dataset_noisy = NoisyDatasetLoader(
    base_dir="./processed_features",
    noise_versions=['noise_15db']
)
model_noisy = train(dataset_noisy)

# Experiment 3: Mixed
dataset_mixed = NoisyDatasetLoader(
    base_dir="./processed_features",
    noise_versions=['clean', 'noise_15db', 'cafe_busy']
)
model_mixed = train(dataset_mixed)
    """)
    
    print("\n📋 STEP 3: Evaluation")
    print("""
# Test trên BOTH clean and noisy test sets
results = {
    'clean_only': {
        'clean_test': evaluate(model_clean, test_clean),
        'noisy_test': evaluate(model_clean, test_noisy)
    },
    'noisy_only': {
        'clean_test': evaluate(model_noisy, test_clean),
        'noisy_test': evaluate(model_noisy, test_noisy)
    },
    'mixed': {
        'clean_test': evaluate(model_mixed, test_clean),
        'noisy_test': evaluate(model_mixed, test_noisy)
    }
}

# Print comparison table
print_comparison_table(results)
    """)
    
    print("\n📊 Expected Results:")
    print("""
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Training Data   │ Clean WER    │ Noisy WER    │ Average WER  │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Clean Only      │ 18%   ✅     │ 35%   ❌     │ 26.5%        │
│ Noisy Only      │ 22%          │ 25%   ✅     │ 23.5%        │
│ Mixed           │ 19%   ✅     │ 26%   ✅     │ 22.5%  ✅✅  │
│ Multi-Condition │ 20%          │ 24%   ✅✅   │ 22.0%  ✅✅✅│
└─────────────────┴──────────────┴──────────────┴──────────────┘

→ Multi-condition training thường cho kết quả BEST!
    """)
    
    print("\n💾 Storage Requirements:")
    print("""
1 sample × 1 version = 2.15MB

Lite (3 versions): 50k × 3 × 2.15MB = 322GB
Standard (8 versions): 50k × 8 × 2.15MB = 860GB  
Full (18 versions): 50k × 18 × 2.15MB = 1.94TB

→ Khuyến nghị: Lite hoặc Standard
    """)


if __name__ == "__main__":
    print_usage_example()