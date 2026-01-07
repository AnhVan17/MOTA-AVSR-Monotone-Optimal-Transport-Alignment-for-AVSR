"""
Evaluation Metrics for AVSR
============================
🔧 FIXED: Use full logits sequence, not truncated by actual_len

The issue was that we were only evaluating the first `actual_len` frames,
which caused the model to only learn from that portion and ignore the rest.
"""

import torch
from typing import List, Optional
from jiwer import wer, cer


class Evaluator:
    """
    Evaluator for AVSR model
    
    🔧 FIXED: Evaluate on FULL sequence, not truncated
    """
    
    def __init__(self, tokenizer, blank_id: int = 51865):
        """
        Args:
            tokenizer: Tokenizer instance
            blank_id: CTC blank token ID
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
        CTC greedy decoding - use FULL sequence
        
        Args:
            logits: [B, T, V+1]
            blank_id: Blank token ID
            
        Returns:
            List of decoded token sequences
        """
        if blank_id is None:
            blank_id = self.blank_id

        # Argmax over FULL sequence
        pred_ids = logits.argmax(dim=-1)  # [B, T]
        
        decoded = []
        for seq in pred_ids:
            # CTC collapse: remove consecutive duplicates
            unique_tokens = []
            prev = None
            for token in seq:
                token_id = token.item()
                if token_id != prev:
                    unique_tokens.append(token_id)
                    prev = token_id
            
            # Remove blank tokens
            content_tokens = [t for t in unique_tokens if t != blank_id and t < blank_id]
            decoded.append(content_tokens)
        
        return decoded
    
    def decode_predictions(
        self, 
        logits: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,  # ← Not used anymore
        debug: bool = False
    ) -> List[str]:
        """
        🔧 FIXED: Decode predictions from FULL logits, not truncated
        
        Args:
            logits: [B, T, V+1]
            audio_lengths: Not used (kept for compatibility)
            debug: If True, print analysis
            
        Returns:
            List of decoded text strings
        """
        B, T, V = logits.shape
        
        # Get CTC decoded token IDs from FULL sequence
        pred_ids = self.ctc_greedy_decode(logits)
        
        # 🔧 Enhanced debugging
        if debug and B > 0:
            print("\n" + "="*80)
            print("🔍 [PREDICTION DEBUG]")
            print("="*80)
            print(f"   Logits shape: {logits.shape}")
            print(f"   Using FULL sequence length: {T}")
            
            # Analyze first sample (FULL sequence)
            sample_logits = logits[0]  # [T, V+1]
            raw_preds = sample_logits.argmax(dim=-1)  # [T]
            
            print(f"   Raw predictions (first 30): {raw_preds[:30].tolist()}")
            print(f"   After CTC decode: {pred_ids[0][:30] if len(pred_ids[0]) > 0 else []}")
            print(f"   Total non-blank tokens: {len(pred_ids[0])}")
            
            # Analyze blank vs non-blank (FULL sequence)
            blank_count = (raw_preds == self.blank_id).sum().item()
            print(f"   Blank tokens: {blank_count}/{T} ({100*blank_count/T:.1f}%)")
            
            # Analyze probabilities (FULL sequence)
            probs = torch.softmax(sample_logits, dim=-1)
            blank_prob = probs[:, self.blank_id].mean().item()
            nonblank_probs = probs[:, :self.blank_id]
            max_nonblank_prob = nonblank_probs.max(dim=-1)[0].mean().item()
            
            print(f"\n   📊 Probability Analysis (full sequence):")
            print(f"      Mean blank probability: {blank_prob*100:.2f}%")
            print(f"      Mean max non-blank probability: {max_nonblank_prob*100:.2f}%")
            
            # Logit statistics
            blank_logits = sample_logits[:, self.blank_id]
            nonblank_logits = sample_logits[:, :self.blank_id]
            
            print(f"\n    Logit Statistics:")
            print(f"      Blank logits mean: {blank_logits.mean():.2f}, std: {blank_logits.std():.2f}")
            print(f"      Non-blank logits mean: {nonblank_logits.mean():.2f}, std: {nonblank_logits.std():.2f}")
            
            # Warning
            if blank_prob > 0.95:
                print(f"\n   WARNING: Blank probability too high ({blank_prob*100:.1f}%)!")
            elif blank_prob < 0.70:
                print(f"\n   GOOD: Blank probability reasonable ({blank_prob*100:.1f}%)")
            
            print("="*80 + "\n")
        
        # Convert to text
        texts = []
        for ids in pred_ids:
            if len(ids) > 0:
                try:
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                except Exception as e:
                    print(f" Decoding error: {e}")
                    text = ""
            else:
                text = ""
            texts.append(text.strip())
        
        return texts
    
    def decode_targets(self, targets: torch.Tensor) -> List[str]:
        """Decode target tokens to text"""
        texts = []
        for seq in targets:
            valid_tokens = seq[seq >= 0].tolist()
            if len(valid_tokens) > 0:
                try:
                    text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                except:
                    text = ""
            else:
                text = ""
            texts.append(text.strip())
        
        return texts
    
    def compute_wer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Word Error Rate"""
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        if not valid_pairs:
            return 100.0
        
        preds_filtered = [p if p else " " for p, r in valid_pairs]
        refs_filtered = [r for p, r in valid_pairs]
        
        try:
            error_rate = wer(refs_filtered, preds_filtered)
            return min(error_rate * 100, 100.0)
        except:
            return 100.0
    
    def compute_cer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Character Error Rate"""
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        if not valid_pairs:
            return 100.0
        
        preds_filtered = [p if p else " " for p, r in valid_pairs]
        refs_filtered = [r for p, r in valid_pairs]
        
        try:
            error_rate = cer(refs_filtered, preds_filtered)
            return min(error_rate * 100, 100.0)
        except:
            return 100.0
    
    @torch.no_grad()
    def evaluate(self, model, dataloader, device, max_batches: int = None):
        """
        Evaluate model on dataloader
        
        🔧 FIXED: Pass full logits to decode_predictions
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
            
            # Get lengths (for informational purposes only)
            audio_len = batch.get('audio_len')
            visual_len = batch.get('visual_len')
            
            if audio_len is not None:
                audio_len = audio_len.to(device)
            if visual_len is not None:
                visual_len = visual_len.to(device)
            
            # Forward
            outputs = model(audio, visual, audio_len=audio_len, visual_len=visual_len, target=None)
            
            # 🔧 CRITICAL: Decode from FULL logits (not truncated)
            debug_this_batch = (i == 0)
            preds = self.decode_predictions(
                outputs['ctc_logits'],
                audio_lengths=None,  # Don't truncate!
                debug=debug_this_batch
            )
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