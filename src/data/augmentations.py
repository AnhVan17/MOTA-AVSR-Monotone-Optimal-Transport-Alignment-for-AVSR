import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class FeatureAugmenter:
    """
    On-the-fly Feature Augmentation for AVSR.
    
    Applies augmentation directly to precomputed feature tensors.
    - Audio: SpecAugment (Time Masking, Freq Masking), Gaussian Noise
    - Visual: Feature Dropout, Frame Masking
    """
    
    def __init__(
        self,
        audio_conf: dict = {},
        visual_conf: dict = {}
    ):
        # Audio Config
        self.audio_mask_time = audio_conf.get('time_mask_param', 10)
        self.audio_mask_freq = audio_conf.get('freq_mask_param', 20)
        self.audio_noise_std = audio_conf.get('noise_std', 0.01)
        self.audio_prob = audio_conf.get('prob', 0.5)

        # Visual Config
        self.visual_dropout = visual_conf.get('dropout_prob', 0.05)
        self.visual_mask_frames = visual_conf.get('frame_mask_param', 5)
        self.visual_prob = visual_conf.get('prob', 0.5)
        
    def augment_audio(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [T, D] Audio features (e.g. 768 dim)
        """
        # 1. Random Skip
        if torch.rand(1) > self.audio_prob:
            return features
            
        out = features.clone()
        T, D = out.shape
        
        # 2. Gaussian Noise
        if self.audio_noise_std > 0:
            noise = torch.randn_like(out) * self.audio_noise_std
            out = out + noise
            
        # 3. Time Masking (SpecAugment)
        # Randomly mask 1 chunk of time
        if self.audio_mask_time > 0:
            t = np.random.randint(0, self.audio_mask_time)
            t0 = np.random.randint(0, max(1, T - t))
            out[t0:t0+t, :] = 0
            
        # 4. Frequency/Channel Masking (SpecAugment)
        # Randomly mask 1 chunk of feature dims
        if self.audio_mask_freq > 0:
            f = np.random.randint(0, self.audio_mask_freq)
            f0 = np.random.randint(0, max(1, D - f))
            out[:, f0:f0+f] = 0
            
        return out
        
    def augment_visual(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [T, D] Visual features (e.g. 512 dim)
        """
        # 1. Random Skip
        if torch.rand(1) > self.visual_prob:
            return features
            
        out = features.clone()
        T, D = out.shape
        
        # 2. Feature Dropout (simulate blurry/noisy visual cues)
        # We manually apply dropout mask
        if self.visual_dropout > 0:
            mask = torch.rand_like(out) > self.visual_dropout
            out = out * mask  # Zero out dropped features (Note: scaling usually handled by nn.Dropout during train, but here we just corrupt)
            # To preserve magnitude, we can scale, but for augmentation "corruption", zeroing is fine.
            
        # 3. Frame Masking (simulate dropped frames / severe occlusion)
        if self.visual_mask_frames > 0:
            t = np.random.randint(0, self.visual_mask_frames)
            t0 = np.random.randint(0, max(1, T - t))
            out[t0:t0+t, :] = 0
            
        return out

    def __call__(self, audio: torch.Tensor, visual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.augment_audio(audio), self.augment_visual(visual)
