"""
Whisper-small Direct Inference (No Training!)
==============================================
Dùng openai/whisper-small trực tiếp để infer trên ViCocktail
→ Baseline để so sánh với Audio-Only model

Expected Performance:
- Whisper-small multilingual pretrained
- Vietnamese support có sẵn
- WER dự đoán: ~15-25% (tùy quality)
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from pathlib import Path
import json
from tqdm import tqdm
from jiwer import wer, cer


class WhisperBaseline:
    """Direct Whisper inference - NO training"""
    
    def __init__(self, model_name="openai/whisper-small", device=None):
        """
        Args:
            model_name: Whisper model variant
                - 'openai/whisper-tiny' (39M params, fast)
                - 'openai/whisper-small' (244M params, balanced) ← RECOMMENDED
                - 'openai/whisper-base' (74M params)
                - 'openai/whisper-medium' (769M params, best quality)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔊 Loading Whisper: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Whisper loaded on {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def transcribe_audio_file(self, audio_path: str, language="vi") -> str:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file (.mp4 or .wav)
            language: Language code (vi = Vietnamese)
            
        Returns:
            Transcribed text
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Process
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Generate
        with torch.no_grad():
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, 
                task="transcribe"
            )
            
            generated_ids = self.model.generate(
                inputs.input_features.to(self.device),
                forced_decoder_ids=forced_decoder_ids,
                max_length=225,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    
    def evaluate_on_manifest(self, manifest_path: str, data_root: str, max_samples=None):
        """
        Evaluate on ViCocktail manifest
        
        Args:
            manifest_path: Path to val.jsonl
            data_root: Root directory for data
            max_samples: Max samples to evaluate (None = all)
            
        Returns:
            dict with WER, CER, samples
        """
        print(f"\n📊 Evaluating on: {manifest_path}")
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"   Samples: {len(samples)}")
        
        all_preds = []
        all_refs = []
        
        for sample in tqdm(samples, desc="Transcribing"):
            # Get audio path
            # In ViCocktail: audio_path is like "clean/speaker_id/video_id.mp4"
            audio_path = Path(data_root) / sample['audio_path']
            
            if not audio_path.exists():
                print(f"⚠️ Audio not found: {audio_path}")
                continue
            
            # Transcribe
            try:
                pred_text = self.transcribe_audio_file(str(audio_path))
                ref_text = sample['text']
                
                all_preds.append(pred_text)
                all_refs.append(ref_text)
            except Exception as e:
                print(f"❌ Error on {audio_path}: {e}")
                continue
        
        # Compute metrics
        wer_score = wer(all_refs, all_preds) * 100
        cer_score = cer(all_refs, all_preds) * 100
        
        print(f"\n📈 Results:")
        print(f"   WER: {wer_score:.2f}%")
        print(f"   CER: {cer_score:.2f}%")
        print(f"   Samples evaluated: {len(all_preds)}")
        
        # Show examples
        print(f"\n📝 Examples:")
        for i in range(min(3, len(all_preds))):
            print(f"\n   [{i+1}] Pred: {all_preds[i]}")
            print(f"       Ref:  {all_refs[i]}")
        
        return {
            'wer': wer_score,
            'cer': cer_score,
            'predictions': all_preds,
            'references': all_refs
        }


def main():
    """Test Whisper baseline"""
    
    print("\n" + "="*70)
    print("🎤 WHISPER BASELINE EVALUATION")
    print("="*70)
    print("\n💡 Mục đích: Baseline để so sánh với Audio-Only model")
    print("   → Nếu Whisper WER < Audio-Only WER")
    print("     → Nên fine-tune Whisper thay vì train from scratch")
    
    # Initialize
    baseline = WhisperBaseline(model_name="openai/whisper-small")
    
    # Example: Transcribe single file
    print("\n" + "="*70)
    print("🧪 Test 1: Single File Transcription")
    print("="*70)
    
    # TODO: Update with actual path
    # audio_path = "data/clean/speaker_01/video_001.mp4"
    # transcription = baseline.transcribe_audio_file(audio_path)
    # print(f"Transcription: {transcription}")
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("🧪 Test 2: Validation Set Evaluation")
    print("="*70)
    
    # Update these paths:
    manifest_path = "data/manifests/val.jsonl"
    data_root = "data"
    
    if Path(manifest_path).exists():
        results = baseline.evaluate_on_manifest(
            manifest_path=manifest_path,
            data_root=data_root,
            max_samples=100  # Start with 100 samples
        )
        
        print("\n" + "="*70)
        print("✅ BASELINE RESULTS")
        print("="*70)
        print(f"\n📊 Whisper-small WER: {results['wer']:.2f}%")
        print(f"📊 Whisper-small CER: {results['cer']:.2f}%")
        print("\n💡 Next Steps:")
        print("   1. Compare với Audio-Only model WER")
        print("   2. Nếu Whisper tốt hơn → Fine-tune Whisper")
        print("   3. Nếu Audio-Only tốt hơn → Continue custom model")
        print("="*70)
    else:
        print(f"⚠️ Manifest not found: {manifest_path}")
        print("   Update the path and run again")


if __name__ == "__main__":
    main()
