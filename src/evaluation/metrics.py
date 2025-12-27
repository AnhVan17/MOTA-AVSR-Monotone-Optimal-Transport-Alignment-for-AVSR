import torch
from typing import List
from jiwer import wer, cer


class Evaluator:
    
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer
    
    def ctc_greedy_decode(
        self,
        logits: torch.Tensor,
        blank_id: int = 4
    ) -> List[List[int]]:

        pred_ids = logits.argmax(dim=-1)       
        decoded = []
        for seq in pred_ids:
            unique_tokens = []
            prev = None
            for token in seq:
                token_id = token.item()
                if token_id != prev:
                    unique_tokens.append(token_id)
                    prev = token_id
            
            unique_tokens = [t for t in unique_tokens if t != blank_id]
            decoded.append(unique_tokens)
        
        return decoded
    
    def decode_predictions(self, logits: torch.Tensor, mode: str = "ctc") -> List[str]:
        """
        Decode logits to text
        
        Args:
            logits: [B, T, V]
            mode: "ctc" or "greedy"
            
        Returns:
            List of decoded strings
        """
        # Determine blank_id (usually last token in vocab if not specified)
        # For Whisper, we'll use a high value, let's assume 50257 (pad)
        blank_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
        
        # CTC decode
        pred_ids = self.ctc_greedy_decode(logits, blank_id=blank_id)
        
        # Convert to text
        texts = []
        for ids in pred_ids:
            if len(ids) > 0:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
            else:
                text = ""
            texts.append(text.strip())
        
        return texts
    
    def decode_targets(self, targets: torch.Tensor) -> List[str]:
        """
        Decode target tokens to text, ignoring padding (-100)
        """
        texts = []
        for seq in targets:
            # Filter out -100 (CE Loss ignore_index)
            clean_seq = [t.item() for t in seq if t.item() != -100]
            text = self.tokenizer.decode(clean_seq, skip_special_tokens=True)
            texts.append(text.strip())
        
        return texts
    
    def compute_wer(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute Word Error Rate
        
        Returns:
            WER as percentage
        """
        try:
            error_rate = wer(references, predictions)
            return error_rate * 100
        except:
            return 100.0
    
    def compute_cer(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute Character Error Rate
        
        Returns:
            CER as percentage
        """
        try:
            error_rate = cer(references, predictions)
            return error_rate * 100
        except:
            return 100.0
    
    @torch.no_grad()
    def evaluate(
        self,
        model,
        dataloader,
        device,
        max_batches: int = None
    ):
        """
        Evaluate model on dataloader
        
        Args:
            model: AURORA-XT model
            dataloader: Validation dataloader
            device: Device
            max_batches: Limit batches
            
        Returns:
            dict with wer, cer, samples
        """
        model.eval()
        
        all_preds = []
        all_refs = []
        
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            target = batch['target'].to(device)
            
            outputs = model(audio, visual, target=None)
            
            preds = self.decode_predictions(outputs['ctc_logits'])
            refs = self.decode_targets(target)
            
            all_preds.extend(preds)
            all_refs.extend(refs)
        
        wer_score = self.compute_wer(all_preds, all_refs)
        cer_score = self.compute_cer(all_preds, all_refs)
        
        return {
            'wer': wer_score,
            'cer': cer_score,
            'samples': list(zip(all_preds[:5], all_refs[:5]))
        }