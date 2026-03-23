"""
Vietnamese Noisy Data Generator
================================
Tạo realistic noisy data cho tiếng Việt từ ViCocktail clean dataset

Các loại noise:
1. Gaussian noise (white noise)
2. Babble noise (nhiều người nói)
3. Environmental noise (cafe, street, traffic)
4. Room reverberation (echo)
5. Phone/microphone quality simulation
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import random
import scipy.signal as signal


class VietnameseNoiseConfig:
    """Config cho noise generation - Optimized cho tiếng Việt"""
    
    # SNR levels (Signal-to-Noise Ratio in dB)
    # Higher dB = cleaner audio
    SNR_LEVELS = {
        'clean': None,           # No noise
        'very_high': 25,         # Rất sạch (studio quality)
        'high': 20,              # Sạch (good recording)
        'medium': 15,            # Trung bình (typical speech)
        'low': 10,               # Nhiễu nhiều (noisy environment)
        'very_low': 5            # Rất nhiễu (very noisy)
    }
    
    # Noise types for Vietnamese scenarios
    NOISE_TYPES = [
        'gaussian',              # White noise
        'babble_vietnamese',     # Vietnamese babble (nhiều người Việt nói)
        'cafe',                  # Cafe/restaurant
        'street',                # Đường phố Việt Nam
        'motorbike',             # Tiếng xe máy (very common in VN!)
        'market',                # Chợ
        'office',                # Văn phòng
        'phone',                 # Phone call quality
        'reverb'                 # Room reverberation
    ]
    
    # Reverberation settings (for Vietnamese indoor spaces)
    REVERB_SETTINGS = {
        'small_room': {'delay': 0.02, 'decay': 0.3},     # Phòng nhỏ
        'large_room': {'delay': 0.05, 'decay': 0.5},     # Phòng lớn
        'hall': {'delay': 0.1, 'decay': 0.7}             # Hội trường
    }


class VietnameseNoiseGenerator:
    """
    Generate realistic noise for Vietnamese speech data
    Optimized cho các tình huống thực tế ở Việt Nam
    """
    
    def __init__(self, config: VietnameseNoiseConfig = None):
        self.config = config or VietnameseNoiseConfig()
        
        # Load noise samples nếu có (optional)
        self.noise_library = {}
        
        print("✅ Vietnamese Noise Generator initialized")
    
    # ========================================================================
    # BASIC NOISE TYPES
    # ========================================================================
    
    @staticmethod
    def add_gaussian_noise(
        waveform: torch.Tensor,
        snr_db: float
    ) -> torch.Tensor:
        """
        Add white Gaussian noise
        
        Args:
            waveform: [1, T] or [T] audio waveform
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            noisy_waveform: Same shape as input
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Calculate signal power
        signal_power = waveform.pow(2).mean()
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        
        return waveform + noise
    
    @staticmethod
    def add_babble_noise(
        waveform: torch.Tensor,
        snr_db: float,
        num_speakers: int = 5
    ) -> torch.Tensor:
        """
        Add babble noise (nhiều người nói)
        Simulates Vietnamese crowd/conversation
        
        Args:
            waveform: [1, T] audio waveform
            snr_db: Signal-to-noise ratio
            num_speakers: Number of background speakers
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        T = waveform.size(1)
        sample_rate = 16000  # Assuming 16kHz
        
        # Generate multi-speaker babble
        babble = torch.zeros_like(waveform)
        
        # Vietnamese speech frequency characteristics
        # Tiếng Việt có nhiều tones ở 200-500Hz và 1000-3000Hz
        freq_ranges = [
            (200, 500),    # Fundamental frequency range
            (500, 1000),   # Lower formants
            (1000, 2000),  # Mid formants
            (2000, 4000)   # High formants
        ]
        
        for _ in range(num_speakers):
            speaker_signal = torch.zeros_like(waveform)
            
            for freq_low, freq_high in freq_ranges:
                # Random frequency in range
                freq = random.uniform(freq_low, freq_high)
                
                # Generate tone with random amplitude
                t = torch.arange(T).float() / sample_rate
                amplitude = random.uniform(0.1, 0.5)
                tone = amplitude * torch.sin(2 * np.pi * freq * t)
                
                # Add random phase modulation (tones)
                modulation_freq = random.uniform(2, 8)  # Vietnamese tones
                modulation = 0.3 * torch.sin(2 * np.pi * modulation_freq * t)
                tone = tone * (1 + modulation)
                
                speaker_signal += tone.unsqueeze(0)
            
            # Add temporal envelope (speech-like)
            envelope = torch.rand(T // 100).repeat_interleave(100)[:T]
            envelope = envelope.unsqueeze(0)
            speaker_signal = speaker_signal * envelope
            
            babble += speaker_signal
        
        # Normalize babble
        babble = babble / babble.abs().max()
        
        # Scale by SNR
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        babble = babble * torch.sqrt(noise_power / babble.pow(2).mean())
        
        return waveform + babble
    
    # ========================================================================
    # VIETNAMESE-SPECIFIC NOISE
    # ========================================================================
    
    @staticmethod
    def add_motorbike_noise(
        waveform: torch.Tensor,
        snr_db: float
    ) -> torch.Tensor:
        """
        Add motorbike noise (XE MÁY - very common in Vietnam!)
        Low-frequency rumble + periodic revving
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        T = waveform.size(1)
        sample_rate = 16000
        
        # Base engine rumble (50-200 Hz)
        t = torch.arange(T).float() / sample_rate
        
        # Low frequency rumble
        rumble = 0.4 * torch.sin(2 * np.pi * 80 * t)
        rumble += 0.3 * torch.sin(2 * np.pi * 150 * t)
        rumble += 0.2 * torch.sin(2 * np.pi * 120 * t)
        
        # Add periodic revving (acceleration/deceleration)
        rev_freq = 0.5  # Rev every 2 seconds
        rev_envelope = 0.5 * (1 + torch.sin(2 * np.pi * rev_freq * t))
        rumble = rumble * rev_envelope
        
        # Add high-frequency component (exhaust)
        exhaust = 0.2 * torch.randn(T)
        
        # Combine
        noise = (rumble + exhaust).unsqueeze(0)
        
        # Apply lowpass filter (realistic)
        noise = torchaudio.functional.lowpass_biquad(noise, sample_rate, 2000)
        
        # Scale by SNR
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = noise * torch.sqrt(noise_power / noise.pow(2).mean())
        
        return waveform + noise
    
    @staticmethod
    def add_market_noise(
        waveform: torch.Tensor,
        snr_db: float
    ) -> torch.Tensor:
        """
        Add Vietnamese market noise (CHỢ)
        Mix of voices, activity, occasional loudspeaker
        """
        # Combination of babble + environmental sounds
        noisy = VietnameseNoiseGenerator.add_babble_noise(
            waveform, snr_db + 3, num_speakers=10
        )
        
        # Add random high-frequency activity (dishes, movements)
        T = waveform.size(1)
        activity = 0.1 * torch.randn(T).unsqueeze(0)
        activity = torchaudio.functional.highpass_biquad(activity, 16000, 2000)
        
        noisy = noisy + activity
        
        return noisy
    
    @staticmethod
    def add_cafe_noise(
        waveform: torch.Tensor,
        snr_db: float
    ) -> torch.Tensor:
        """
        Add Vietnamese cafe noise
        Background chatter + dishes + music
        """
        # Babble (conversations)
        noisy = VietnameseNoiseGenerator.add_babble_noise(
            waveform, snr_db + 5, num_speakers=4
        )
        
        # Add mid-frequency noise (dishes, cups)
        T = waveform.size(1)
        dishes = 0.05 * torch.randn(T).unsqueeze(0)
        dishes = torchaudio.functional.bandpass_biquad(dishes, 16000, 1000, 500)
        
        noisy = noisy + dishes
        
        return noisy
    
    @staticmethod
    def add_street_noise(
        waveform: torch.Tensor,
        snr_db: float
    ) -> torch.Tensor:
        """
        Add Vietnamese street noise
        Traffic + horns + general city ambience
        """
        # Start with motorbike noise
        noisy = VietnameseNoiseGenerator.add_motorbike_noise(waveform, snr_db + 5)
        
        # Add general traffic rumble
        T = waveform.size(1)
        sample_rate = 16000
        t = torch.arange(T).float() / sample_rate
        
        traffic = 0.2 * torch.sin(2 * np.pi * 60 * t)
        traffic += 0.15 * torch.sin(2 * np.pi * 100 * t)
        traffic = traffic.unsqueeze(0)
        
        # Add occasional horn (Vietnamese bike horn)
        if random.random() < 0.3:
            horn_start = random.randint(0, max(1, T - sample_rate))
            horn_duration = int(0.3 * sample_rate)  # 300ms
            horn_t = torch.arange(horn_duration).float() / sample_rate
            horn = 0.3 * torch.sin(2 * np.pi * 800 * horn_t)  # ~800Hz horn
            traffic[0, horn_start:horn_start+horn_duration] += horn
        
        noisy = noisy + traffic
        
        return noisy
    
    # ========================================================================
    # REVERBERATION (Echo trong phòng)
    # ========================================================================
    
    @staticmethod
    def add_reverberation(
        waveform: torch.Tensor,
        room_type: str = 'small_room'
    ) -> torch.Tensor:
        """
        Add room reverberation (echo)
        
        Args:
            waveform: [1, T] audio
            room_type: 'small_room', 'large_room', 'hall'
        """
        config = VietnameseNoiseConfig()
        settings = config.REVERB_SETTINGS.get(room_type, config.REVERB_SETTINGS['small_room'])
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        sample_rate = 16000
        delay_samples = int(settings['delay'] * sample_rate)
        decay = settings['decay']
        
        # Create impulse response
        T = waveform.size(1)
        impulse_length = min(int(0.5 * sample_rate), T // 4)  # Max 0.5s reverb
        
        impulse = torch.zeros(impulse_length)
        impulse[0] = 1.0  # Direct sound
        
        # Add reflections
        num_reflections = 10
        for i in range(num_reflections):
            delay = int(delay_samples * (i + 1) * (1 + random.uniform(-0.2, 0.2)))
            if delay < impulse_length:
                amplitude = decay ** (i + 1) * random.uniform(0.5, 1.0)
                impulse[delay] = amplitude
        
        # Apply reverb (convolution)
        reverb = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            impulse.view(1, 1, -1),
            padding=impulse_length - 1
        ).squeeze(0)[:, :T]
        
        return reverb
    
    # ========================================================================
    # PHONE/MICROPHONE QUALITY
    # ========================================================================
    
    @staticmethod
    def simulate_phone_quality(
        waveform: torch.Tensor,
        phone_type: str = 'mobile'
    ) -> torch.Tensor:
        """
        Simulate phone call quality
        
        Args:
            phone_type: 'mobile', 'landline', 'voip'
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        sample_rate = 16000
        
        if phone_type == 'mobile':
            # Mobile phone: bandpass 300-3400 Hz
            filtered = torchaudio.functional.highpass_biquad(waveform, sample_rate, 300)
            filtered = torchaudio.functional.lowpass_biquad(filtered, sample_rate, 3400)
            
            # Add slight compression artifacts
            filtered = torch.tanh(filtered * 1.5) / 1.5
            
        elif phone_type == 'landline':
            # Landline: narrower band 400-3000 Hz
            filtered = torchaudio.functional.highpass_biquad(waveform, sample_rate, 400)
            filtered = torchaudio.functional.lowpass_biquad(filtered, sample_rate, 3000)
        
        else:  # voip
            # VoIP: 200-7000 Hz but with packet loss simulation
            filtered = torchaudio.functional.highpass_biquad(waveform, sample_rate, 200)
            filtered = torchaudio.functional.lowpass_biquad(filtered, sample_rate, 7000)
            
            # Simulate packet loss (random dropouts)
            T = filtered.size(1)
            dropout_mask = torch.rand(T) > 0.02  # 2% packet loss
            filtered = filtered * dropout_mask.unsqueeze(0)
        
        return filtered
    
    # ========================================================================
    # MAIN AUGMENTATION FUNCTION
    # ========================================================================
    
    def augment(
        self,
        waveform: torch.Tensor,
        noise_type: str,
        snr_db: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Main augmentation function
        
        Args:
            waveform: [1, T] or [T] audio waveform
            noise_type: Type of noise to add
            snr_db: Signal-to-noise ratio (if applicable)
            **kwargs: Additional arguments for specific noise types
        
        Returns:
            augmented_waveform: Same shape as input
        """
        if noise_type == 'gaussian':
            return self.add_gaussian_noise(waveform, snr_db)
        
        elif noise_type == 'babble_vietnamese':
            return self.add_babble_noise(waveform, snr_db, 
                                        num_speakers=kwargs.get('num_speakers', 5))
        
        elif noise_type == 'motorbike':
            return self.add_motorbike_noise(waveform, snr_db)
        
        elif noise_type == 'market':
            return self.add_market_noise(waveform, snr_db)
        
        elif noise_type == 'cafe':
            return self.add_cafe_noise(waveform, snr_db)
        
        elif noise_type == 'street':
            return self.add_street_noise(waveform, snr_db)
        
        elif noise_type == 'reverb':
            return self.add_reverberation(waveform, 
                                         room_type=kwargs.get('room_type', 'small_room'))
        
        elif noise_type == 'phone':
            return self.simulate_phone_quality(waveform,
                                              phone_type=kwargs.get('phone_type', 'mobile'))
        
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    # ========================================================================
    # BATCH AUGMENTATION
    # ========================================================================
    
    def create_augmented_dataset(
        self,
        clean_waveform: torch.Tensor,
        augmentation_configs: List[dict]
    ) -> dict:
        """
        Create multiple augmented versions
        
        Args:
            clean_waveform: [1, T] clean audio
            augmentation_configs: List of config dicts, e.g.:
                [
                    {'name': 'noise_15db', 'type': 'gaussian', 'snr_db': 15},
                    {'name': 'cafe_noisy', 'type': 'cafe', 'snr_db': 10},
                    ...
                ]
        
        Returns:
            dict mapping version_name -> augmented_waveform
        """
        augmented = {'clean': clean_waveform}
        
        for config in augmentation_configs:
            name = config['name']
            noise_type = config['type']
            
            # Extract kwargs
            kwargs = {k: v for k, v in config.items() 
                     if k not in ['name', 'type']}
            
            # Augment
            augmented[name] = self.augment(
                clean_waveform,
                noise_type,
                **kwargs
            )
        
        return augmented


# ============================================================================
# VIETNAMESE-SPECIFIC AUGMENTATION CONFIGS
# ============================================================================

def get_vietnamese_augmentation_configs() -> List[dict]:
    """
    Predefined augmentation configs optimized for Vietnamese
    Các tình huống thực tế ở Việt Nam
    """
    configs = [
        # Clean baseline
        {'name': 'clean', 'type': None},
        
        # Gaussian noise (various quality levels)
        {'name': 'noise_20db', 'type': 'gaussian', 'snr_db': 20},
        {'name': 'noise_15db', 'type': 'gaussian', 'snr_db': 15},
        {'name': 'noise_10db', 'type': 'gaussian', 'snr_db': 10},
        
        # Vietnamese babble (conversations)
        {'name': 'babble_high', 'type': 'babble_vietnamese', 'snr_db': 15, 'num_speakers': 3},
        {'name': 'babble_medium', 'type': 'babble_vietnamese', 'snr_db': 10, 'num_speakers': 5},
        {'name': 'babble_low', 'type': 'babble_vietnamese', 'snr_db': 5, 'num_speakers': 8},
        
        # Vietnamese street (very common!)
        {'name': 'street_clean', 'type': 'street', 'snr_db': 15},
        {'name': 'street_noisy', 'type': 'street', 'snr_db': 10},
        
        # Motorbike (iconic Vietnam sound!)
        {'name': 'motorbike_far', 'type': 'motorbike', 'snr_db': 20},
        {'name': 'motorbike_near', 'type': 'motorbike', 'snr_db': 10},
        
        # Cafe (popular recording scenario)
        {'name': 'cafe_quiet', 'type': 'cafe', 'snr_db': 15},
        {'name': 'cafe_busy', 'type': 'cafe', 'snr_db': 8},
        
        # Market
        {'name': 'market', 'type': 'market', 'snr_db': 10},
        
        # Reverberation (indoor spaces)
        {'name': 'reverb_small', 'type': 'reverb', 'room_type': 'small_room'},
        {'name': 'reverb_large', 'type': 'reverb', 'room_type': 'large_room'},
        
        # Phone quality
        {'name': 'phone_mobile', 'type': 'phone', 'phone_type': 'mobile'},
        {'name': 'phone_voip', 'type': 'phone', 'phone_type': 'voip'},
    ]
    
    return configs


# ============================================================================
# DEMO & TESTING
# ============================================================================

def demo_vietnamese_noise():
    """Demo tạo noisy data cho ViCocktail"""
    
    print("="*70)
    print("🇻🇳 VIETNAMESE NOISE GENERATION DEMO")
    print("="*70)
    
    # Create dummy clean audio
    sample_rate = 16000
    duration = 5.0  # 5 seconds
    t = torch.arange(int(sample_rate * duration)).float() / sample_rate
    
    # Simulate speech (simple sine waves)
    clean_audio = 0.3 * torch.sin(2 * np.pi * 200 * t)  # F0
    clean_audio += 0.2 * torch.sin(2 * np.pi * 800 * t)  # Formant 1
    clean_audio += 0.1 * torch.sin(2 * np.pi * 2400 * t)  # Formant 2
    clean_audio = clean_audio.unsqueeze(0)
    
    # Initialize generator
    generator = VietnameseNoiseGenerator()
    
    # Get configs
    configs = get_vietnamese_augmentation_configs()
    
    print(f"\n📊 Sẽ tạo {len(configs)} versions:")
    for config in configs[:10]:  # Show first 10
        print(f"  - {config['name']}")
    print("  ...")
    
    # Create augmented versions
    print("\n🔊 Creating augmented versions...")
    augmented = generator.create_augmented_dataset(clean_audio, configs[1:])  # Skip 'clean'
    
    print(f"\n✅ Created {len(augmented)} augmented versions!")
    
    # Calculate statistics
    print("\n📈 Statistics:")
    for name, audio in list(augmented.items())[:5]:  # Show first 5
        rms = torch.sqrt(torch.mean(audio ** 2))
        snr = 10 * torch.log10(clean_audio.pow(2).mean() / (audio - clean_audio).pow(2).mean())
        print(f"  {name:20s} - RMS: {rms:.4f}, SNR: {snr:.2f}dB")
    
    print("\n💡 Usage cho ViCocktail:")
    print("""
    # 1. Load ViCocktail sample
    from datasets import load_dataset
    dataset = load_dataset("nguyenvulebinh/ViCocktail")
    sample = dataset['train'][0]
    
    # 2. Convert to waveform tensor
    waveform = torch.tensor(sample['audio']['array']).unsqueeze(0)
    
    # 3. Create augmented versions
    generator = VietnameseNoiseGenerator()
    configs = get_vietnamese_augmentation_configs()
    augmented = generator.create_augmented_dataset(waveform, configs)
    
    # 4. Save all versions
    for name, audio in augmented.items():
        save_path = f"processed/{sample['id']}/{name}.pt"
        torch.save(audio, save_path)
    """)


if __name__ == "__main__":
    demo_vietnamese_noise()