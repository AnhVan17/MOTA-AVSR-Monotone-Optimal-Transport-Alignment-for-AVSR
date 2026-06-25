import math
from typing import List

import torch

_NEG_INF = float('-inf')


def _logaddexp(a: float, b: float) -> float:
    """log(exp(a)+exp(b)) on Python floats — avoids per-scalar torch.tensor()/.item()
    (each .item() forced a GPU<->CPU sync, which made beam search ~100x slower)."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class CTCDecoder:
    """
    Handles CTC Decoding strategies (Greedy, and Beam Search).
    """

    def __init__(self, tokenizer, blank_id: int = 50257):
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
        Pure-Python CTC prefix beam search (top-K pruned per frame).

        Optimised: log-probs are moved to CPU once and ALL per-beam arithmetic uses Python
        floats via `_logaddexp` (math). The previous version created a torch tensor and called
        `.item()` for every scalar log-add — each `.item()` is a GPU<->CPU sync, which made this
        ~100x slower (≈100 s/batch). Algorithm/output are unchanged.

        Args:
            logits: [B, T, V] — raw logits (log_softmax applied here)
            beam_width: number of beams to keep (also the per-frame top-K)
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
        B, T, _ = log_probs.shape

        decoded_texts = []
        for b in range(B):
            lp = log_probs[b]
            beam = {(): (0.0, _NEG_INF)}  # prefix -> (p_blank, p_non_blank)

            for t in range(T):
                row = lp[t]
                p_blank_curr = row[self.blank_id].item()
                top_v, top_i = row.topk(beam_width)
                top_v = top_v.tolist()
                top_i = top_i.tolist()

                next_beam = {}
                for prefix, (p_b, p_nb) in beam.items():
                    p = _logaddexp(p_b, p_nb)
                    if p < -200:  # prune very low-probability paths
                        continue

                    # 1. extend with blank (prefix unchanged)
                    n_blank = p + p_blank_curr
                    if prefix in next_beam:
                        s_b, s_nb = next_beam[prefix]
                        next_beam[prefix] = (_logaddexp(s_b, n_blank), s_nb)
                    else:
                        next_beam[prefix] = (n_blank, _NEG_INF)

                    # 2. extend with non-blank (top-K tokens)
                    for token_score, token_id in zip(top_v, top_i):
                        if token_id == self.blank_id:
                            continue
                        new_prefix = prefix + (token_id,)

                        if prefix and prefix[-1] == token_id:
                            # 2a. repeat collapsed into same prefix (from p_nb)
                            p_repeat = p_nb + token_score
                            if prefix in next_beam:
                                s_b, s_nb = next_beam[prefix]
                                next_beam[prefix] = (s_b, _logaddexp(s_nb, p_repeat))
                            else:
                                next_beam[prefix] = (_NEG_INF, p_repeat)
                            # 2b. repeat as a NEW char (only via blank, from p_b)
                            p_new = p_b + token_score
                            if new_prefix in next_beam:
                                s_b, s_nb = next_beam[new_prefix]
                                next_beam[new_prefix] = (s_b, _logaddexp(s_nb, p_new))
                            else:
                                next_beam[new_prefix] = (_NEG_INF, p_new)
                        else:
                            # 2c. new char (extends prefix, from both p_b and p_nb)
                            p_new = p + token_score
                            if new_prefix in next_beam:
                                s_b, s_nb = next_beam[new_prefix]
                                next_beam[new_prefix] = (s_b, _logaddexp(s_nb, p_new))
                            else:
                                next_beam[new_prefix] = (_NEG_INF, p_new)

                if not next_beam:
                    break
                beam = dict(
                    sorted(next_beam.items(), key=lambda x: _logaddexp(x[1][0], x[1][1]), reverse=True)[:beam_width]
                )

            if not beam:
                beam = {(): (0.0, _NEG_INF)}
            best_prefix = max(beam.items(), key=lambda x: _logaddexp(x[1][0], x[1][1]))[0]
            text = self.tokenizer.decode(list(best_prefix), skip_special_tokens=True) if best_prefix else ""
            decoded_texts.append(text.strip())

        return decoded_texts

    def decode_targets(self, targets: torch.Tensor) -> List[str]:
        """
        Decode target tensor to text, filtering out padding tokens.
        """
        decoded_texts = []
        for seq in targets:
            # Filter padding tokens before decoding
            valid_tokens = seq[seq != self.blank_id]
            text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
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
