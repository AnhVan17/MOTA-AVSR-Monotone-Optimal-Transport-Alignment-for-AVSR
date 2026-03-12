import torch
import math
from typing import List

class CTCDecoder:
    """
    Handles CTC Decoding strategies (Greedy, and Beam Search).
    """
    
    def __init__(self, tokenizer, blank_id: int = 4):
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        
    def greedy_decode(self, logits: torch.Tensor) -> List[str]:
        """
        Greedy decode logits to text.
        
        Args:
            logits: [B, T, V]
            
        Returns:
            List of decoded strings
        """
        pred_ids = logits.argmax(dim=-1)
        decoded_texts = []
        
        for seq in pred_ids:
            unique_tokens = []
            prev = None
            for token in seq:
                token_id = token.item()
                if token_id != prev:
                    unique_tokens.append(token_id)
                    prev = token_id
            
            # Filter blank
            unique_tokens = [t for t in unique_tokens if t != self.blank_id]
            
            # Convert to string
            if len(unique_tokens) > 0:
                text = self.tokenizer.decode(unique_tokens, skip_special_tokens=True)
            else:
                text = ""
            decoded_texts.append(text.strip())
            
        return decoded_texts

    def beam_search_decode(self, logits: torch.Tensor, beam_width: int = 10) -> List[str]:
        """
        Pure Python CTC Beam Search.
        
        Args:
            logits: [B, T, V] - Log probabilities (will apply log_softmax if not already)
            beam_width: Number of beams to keep
            
        Returns:
            List of decoded strings
        """
        # Ensure log probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        B, T, V = log_probs.shape
        
        decoded_texts = []
        
        for b in range(B):
            # Per-sample decoding matches
            # Initialize: (prefix_tuple) -> (prob_blank, prob_non_blank)
            beam = {(): (0.0, -float('inf'))} 
            
            for t in range(T):
                next_beam = {}
                
                # Pruning: Only consider top K tokens at this step
                curr_log_probs = log_probs[b, t]
                top_v, top_i = curr_log_probs.topk(beam_width) 
                
                p_blank_curr = curr_log_probs[self.blank_id].item()
                
                for prefix, (p_b, p_nb) in beam.items():
                    # Total prob of this prefix
                    p = torch.logaddexp(torch.tensor(p_b), torch.tensor(p_nb)).item()
                    
                    if p < -200: # Prune very low probability paths
                        continue
                        
                    # 1. Extend with Blank
                    n_p_b_score = p + p_blank_curr
                    
                    if prefix in next_beam:
                        s_b, s_nb = next_beam[prefix]
                        n_s_b = torch.logaddexp(torch.tensor(s_b), torch.tensor(n_p_b_score)).item()
                        next_beam[prefix] = (n_s_b, s_nb)
                    else:
                        next_beam[prefix] = (n_p_b_score, -float('inf'))
                        
                    # 2. Extend with Non-Blank
                    for s, token_id_tensor in zip(top_v, top_i):
                        token_id = token_id_tensor.item()
                        token_score = s.item()
                        
                        if token_id == self.blank_id:
                            continue
                            
                        new_prefix = prefix + (token_id,)
                        
                        if len(prefix) > 0 and prefix[-1] == token_id:
                            # 2a. Repeat character collapsed (Merge into same prefix)
                            # Only from p_nb (since p_b + repeat = new char)
                            p_repeat = p_nb + token_score
                            if prefix in next_beam:
                                s_b, s_nb = next_beam[prefix]
                                n_s_nb = torch.logaddexp(torch.tensor(s_nb), torch.tensor(p_repeat)).item()
                                next_beam[prefix] = (s_b, n_s_nb)
                            else:
                                next_beam[prefix] = (-float('inf'), p_repeat)
                            
                            # 2b. Repeat character new (Transition from blank)
                            # From p_b
                            p_new = p_b + token_score
                            if new_prefix in next_beam:
                                s_b, s_nb = next_beam[new_prefix]
                                n_s_nb = torch.logaddexp(torch.tensor(s_nb), torch.tensor(p_new)).item()
                                next_beam[new_prefix] = (s_b, n_s_nb)
                            else:
                                next_beam[new_prefix] = (-float('inf'), p_new)
                        
                        else:
                            # 2c. New character (extends prefix)
                            # From both p_b and p_nb
                            p_new = p + token_score
                            if new_prefix in next_beam:
                                s_b, s_nb = next_beam[new_prefix]
                                n_s_nb = torch.logaddexp(torch.tensor(s_nb), torch.tensor(p_new)).item()
                                next_beam[new_prefix] = (s_b, n_s_nb)
                            else:
                                next_beam[new_prefix] = (-float('inf'), p_new)

                # Check if next_beam is empty (all pruned)
                if not next_beam:
                    break

                # Keep top K
                # Score = logaddexp(p_b, p_nb)
                sorted_beam = sorted(
                    next_beam.items(),
                    key=lambda x: torch.logaddexp(torch.tensor(x[1][0]), torch.tensor(x[1][1])).item(),
                    reverse=True
                )
                beam = dict(sorted_beam[:beam_width])

            # Safety fallback if beam is somehow empty (shouldn't happen with break logic above)
            if not beam:
                beam = {(): (0.0, -float('inf'))}
            
            # Finalize best
            best_prefix = max(
                beam.items(),
                key=lambda x: torch.logaddexp(torch.tensor(x[1][0]), torch.tensor(x[1][1])).item()
            )[0]
            
            # Decode to string
            if len(best_prefix) > 0:
                text = self.tokenizer.decode(list(best_prefix), skip_special_tokens=True)
            else:
                text = ""
            decoded_texts.append(text.strip())
            
        return decoded_texts
        
    def decode_targets(self, targets: torch.Tensor) -> List[str]:
        """
        Decode target tensor to text.
        """
        decoded_texts = []
        for seq in targets:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            decoded_texts.append(text.strip())
        return decoded_texts

    def decode_batch(self, logits: torch.Tensor, method: str = 'greedy', beam_width: int = 10) -> List[str]:
        """
        Unified decode method.
        """
        if method == 'beam':
            return self.beam_search_decode(logits, beam_width=beam_width)
        else:
            return self.greedy_decode(logits)
