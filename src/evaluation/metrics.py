import torch
from typing import List
from jiwer import wer, cer


class Evaluator:
    
    def __init__(self, tokenizer, blank_id: int = 51865):
        """
        Args:
            tokenizer: Tokenizer instance
            blank_id: CTC blank token ID (default: vocab_size = 51865)
        """
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        print(f"[Evaluator] Initialized with blank_id={blank_id}")
    
    def ctc_greedy_decode(
        self,
        logits: torch.Tensor,
        blank_id: int = None
    ) -> List[List[int]]:
        """
        CTC greedy decoding
        
        Args:
            logits: [B, T, V+1] probabilities
            blank_id: Blank token ID to filter out
            
        Returns:
            List of decoded token sequences
        """
        if blank_id is None:
            blank_id = self.blank_id

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
            
            # Remove blank tokens
            # Keep only valid content tokens (< vocab_size, != blank)
            unique_tokens = [t for t in unique_tokens if t != blank_id and t < self.blank_id]
            decoded.append(unique_tokens)
        
        return decoded
    
    def decode_predictions(self, logits: torch.Tensor, debug: bool = False) -> List[str]:
        """
        Decode logits to text
        
        Args:
            logits: [B, T, V+1]
            debug: If True, print debug info
            
        Returns:
            List of decoded strings
        """
        pred_ids = self.ctc_greedy_decode(logits)
        
        # Debug: Print first sample's predictions
        if debug and len(pred_ids) > 0:
            raw_argmax = logits[0].argmax(dim=-1)[:30].tolist()
            print(f"[DEBUG] Logits shape: {logits.shape}")
            print(f"[DEBUG] Raw argmax (first 30): {raw_argmax}")
            print(f"[DEBUG] After CTC decode (first 30): {pred_ids[0][:30]}")
            print(f"[DEBUG] Number of non-blank tokens: {len(pred_ids[0])}")
            print(f"[DEBUG] Blank ID: {self.blank_id}")
        
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
        Decode target tokens to text
        
        Args:
            targets: [B, L]
            
        Returns:
            List of decoded strings
        """
        texts = []
        for seq in targets:
            # Filter out padding (-100)
            valid_tokens = seq[seq >= 0].tolist()
            if len(valid_tokens) > 0:
                text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
            else:
                text = ""
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
        # Filter out empty pairs
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        if not valid_pairs:
            return 100.0
            
        preds_filtered = [p for p, r in valid_pairs]
        refs_filtered = [r for p, r in valid_pairs]
        
        try:
            error_rate = wer(refs_filtered, preds_filtered)
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
        # Filter out empty pairs
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        if not valid_pairs:
            return 100.0
            
        preds_filtered = [p for p, r in valid_pairs]
        refs_filtered = [r for p, r in valid_pairs]
        
        try:
            error_rate = cer(refs_filtered, preds_filtered)
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
            model: MOTA model
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
            
            # Debug first batch
            debug_this_batch = (i == 0)
            preds = self.decode_predictions(outputs['ctc_logits'], debug=debug_this_batch)
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