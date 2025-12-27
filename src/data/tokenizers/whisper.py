from transformers import WhisperProcessor as HfWhisperProcessor
import torch
from typing import List, Union

class WhisperProcessor:
    """
    Wrapper around HuggingFace WhisperProcessor.
    Handles both audio feature extraction and text tokenization.
    """
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "vi",
        task: str = "transcribe",
    ):
        self.model_name = model_name
        self.language = language
        self.task = task

        # Processor includes feature_extractor and tokenizer
        self.processor = HfWhisperProcessor.from_pretrained(
            model_name,
            language=language,
            task=task
        )
        
        # Shortcuts
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id # SOT
        self.eos_token_id = self.tokenizer.eos_token_id # EOT
        self.unk_token_id = self.tokenizer.unk_token_id
        
        # Whisper specific special tokens
        self.sot_token_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftranscript|>")
        
        print(f"✅ WhisperProcessor initialized ({model_name})")
        print(f"   Vocab size: {self.vocab_size}")
        print(f"   Language: {language}, Task: {task}")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens
        )

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, batch_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode batch of token IDs"""
        return self.tokenizer.batch_decode(
            batch_ids,
            skip_special_tokens=skip_special_tokens
        )

    def get_features(self, audio: Union[torch.Tensor, list], sampling_rate: int = 16000) -> torch.Tensor:
        """Extract Mel-filterbank features from audio"""
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().numpy()
            
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        )
        return inputs.input_features # [1, 80, 3000]

    def __len__(self):
        return self.vocab_size
