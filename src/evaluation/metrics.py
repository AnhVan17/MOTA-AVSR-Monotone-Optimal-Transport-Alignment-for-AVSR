"""
Evaluation Metrics for AVSR
============================
🔧 FIXED: Added Vietnamese text filter to clean output
"""

import torch
import re
from typing import List, Optional
from jiwer import wer, cer


def filter_vietnamese_text(text: str) -> str:
    """
    Filter text to keep ONLY Vietnamese characters
    
    This is a POST-PROCESSING step to remove any non-Vietnamese
    characters that slip through vocab pruning.
    """
    # Vietnamese character pattern
    # Includes: Latin letters + Vietnamese diacritics + basic punctuation
    vietnamese_pattern = re.compile(
        r'[a-zA-Z0-9\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ'
        r'.,!?;:\-\'"()]+',
        re.UNICODE
    )
    
    # Find all Vietnamese matches
    matches = vietnamese_pattern.findall(text)
    
    # Join matches
    result = ' '.join(matches)
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def is_mostly_garbage(text: str, threshold: float = 0.5) -> bool:
    """
    Check if text is mostly garbage (non-Vietnamese)
    
    Returns True if more than threshold% of characters are non-Vietnamese
    """
    if not text:
        return True
    
    # Count Vietnamese characters
    vietnamese_chars = set(
        'aàáảãạăắằẳẵặâấầẩẫậbcdđeèéẻẽẹêếềểễệghiìíỉĩịklmnoòóỏõọôốồổỗộơớờởỡợpqrstuùúủũụưứừửữựvxyỳýỷỹỵ'
        'AÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬBCDĐEÈÉẺẼẸÊẾỀỂỄỆGHIÌÍỈĨỊKLMNOÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢPQRSTUÙÚỦŨỤƯỨỪỬỮỰVXYỲÝỶỸỴ'
        '0123456789 .,!?;:\-\'"()'
    )
    
    viet_count = sum(1 for c in text if c in vietnamese_chars)
    total_count = len(text)
    
    ratio = viet_count / total_count if total_count > 0 else 0
    
    return ratio < threshold


class Evaluator:
    """
    Evaluator for AVSR model
    
    🔧 FIXED: Added Vietnamese text filter
    """
    
    def __init__(self, tokenizer, blank_id: int = 51865):
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        print(f"[Evaluator] Initialized with blank_id={blank_id}")
    
    def ctc_greedy_decode(
        self,
        logits: torch.Tensor,
        blank_id: int = None
    ) -> List[List[int]]:
        """CTC greedy decoding"""
        if blank_id is None:
            blank_id = self.blank_id

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
        audio_lengths: Optional[torch.Tensor] = None,
        debug: bool = False
    ) -> List[str]:
        """
        Decode predictions with Vietnamese text filtering
        """
        B, T, V = logits.shape
        
        pred_ids = self.ctc_greedy_decode(logits)
        
        if debug and B > 0:
            print("\n" + "="*80)
            print("🔍 [PREDICTION DEBUG]")
            print("="*80)
            print(f"   Logits shape: {logits.shape}")
            
            sample_logits = logits[0]
            raw_preds = sample_logits.argmax(dim=-1)
            
            print(f"   Raw predictions (first 30): {raw_preds[:30].tolist()}")
            print(f"   After CTC decode: {pred_ids[0][:30] if len(pred_ids[0]) > 0 else []}")
            print(f"   Total non-blank tokens: {len(pred_ids[0])}")
            
            blank_count = (raw_preds == self.blank_id).sum().item()
            print(f"   Blank tokens: {blank_count}/{T} ({100*blank_count/T:.1f}%)")
            
            probs = torch.softmax(sample_logits, dim=-1)
            blank_prob = probs[:, self.blank_id].mean().item()
            print(f"   Mean blank probability: {blank_prob*100:.2f}%")
            print("="*80 + "\n")
        
        # Convert to text with Vietnamese filtering
        texts = []
        for ids in pred_ids:
            if len(ids) > 0:
                try:
                    # Decode
                    raw_text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    
                    # 🔧 CRITICAL: Filter to Vietnamese only
                    filtered_text = filter_vietnamese_text(raw_text)
                    
                    # If mostly garbage, return empty
                    if is_mostly_garbage(raw_text):
                        text = ""
                    else:
                        text = filtered_text
                        
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
        """Evaluate model on dataloader"""
        model.eval()
        
        all_preds = []
        all_refs = []
        
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            target = batch['target'].to(device)
            
            audio_len = batch.get('audio_len')
            visual_len = batch.get('visual_len')
            
            if audio_len is not None:
                audio_len = audio_len.to(device)
            if visual_len is not None:
                visual_len = visual_len.to(device)
            
            outputs = model(audio, visual, audio_len=audio_len, visual_len=visual_len, target=None)
            
            debug_this_batch = (i == 0)
            preds = self.decode_predictions(
                outputs['ctc_logits'],
                audio_lengths=None,
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