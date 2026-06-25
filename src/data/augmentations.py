import torch
import numpy as np
from typing import Tuple

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
        # Modality dropout: with this prob, zero ONE entire stream (never both) → force the model
        # to use the other. Key technique for AV noise-robustness (AV-HuBERT / u-HuBERT).
        self.modality_dropout_prob = audio_conf.get('modality_dropout_prob', 0.0)

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

    def augment_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Spatial augmentation for raw mouth-crop frames.

        Args:
            frames: [T, C, H, W] float in [0,1].

        Applies (per clip): random horizontal flip + random temporal frame masking.
        Standard for lip-reading frontends (Auto-AVSR style); richer than feature-level aug.
        """
        if torch.rand(1) > self.visual_prob:
            return frames

        out = frames
        # 1. Horizontal flip — flip the whole clip consistently (width = last dim).
        if torch.rand(1) < 0.5:
            out = torch.flip(out, dims=[-1])

        # 2. Temporal frame masking — zero a short run of frames (occlusion/dropout).
        if self.visual_mask_frames > 0:
            T = out.shape[0]
            t = int(np.random.randint(0, self.visual_mask_frames))
            if t > 0:
                t0 = int(np.random.randint(0, max(1, T - t)))
                out = out.clone()
                out[t0:t0 + t] = 0
        return out

    def apply_modality_dropout(self, audio: torch.Tensor, visual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """With prob `modality_dropout_prob`, zero ONE entire stream (never both) so the model must
        rely on the other — this teaches the reliability-aware fusion to lean on the visual stream
        when audio is corrupted/absent (the core of AV noise-robustness)."""
        if self.modality_dropout_prob > 0 and torch.rand(1).item() < self.modality_dropout_prob:
            if torch.rand(1).item() < 0.5:
                audio = torch.zeros_like(audio)
            else:
                visual = torch.zeros_like(visual)
        return audio, visual

    def _apply(self, audio: torch.Tensor, visual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # visual is either raw frames [T,C,H,W] (4D) or precomputed features [T,D] (2D).
        aug_visual = self.augment_frames(visual) if visual.ndim == 4 else self.augment_visual(visual)
        return self.apply_modality_dropout(self.augment_audio(audio), aug_visual)

    def __call__(
        self, audio: torch.Tensor, visual: torch.Tensor, seed: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment one sample.

        seed=None → use the global RNG (train: fresh random noise every epoch — correct).
        seed given → DETERMINISTIC: the same sample always gets the SAME noise across epochs/runs.
        Used to build a FIXED noisy dev set for reproducible model selection (robust-AVSR standard).
        We seed the RNG to `seed`, apply, then RESTORE the prior global RNG state so train-time
        randomness and any other consumer are left untouched (no global side-effect).
        """
        if seed is None:
            return self._apply(audio, visual)

        torch_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        try:
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32))
            return self._apply(audio, visual)
        finally:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(np_state)
