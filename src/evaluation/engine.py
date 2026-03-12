import torch
from typing import Dict, Optional, List
from tqdm import tqdm

from .decoding import CTCDecoder
from .metrics import MetricCalculator
from .visualization import Visualizer

class Evaluator:
    """
    Main Evaluation Engine.
    Orchestrates Decoding -> Metrics -> Visualization.
    """
    
    def __init__(self, tokenizer, device='cuda'):
        self.device = device
        self.decoder = CTCDecoder(tokenizer)
        self.calculator = MetricCalculator()
        self.visualizer = Visualizer()
        
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None,
        return_samples: bool = True,
        decode_method: str = 'greedy',
        beam_width: int = 5
    ) -> Dict:
        """
        Run full evaluation on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Validation/Test loader
            max_batches: Limit batches for speed
            return_samples: Whether to return text samples
            decode_method: 'greedy' or 'beam'
            beam_width: Beam width for beam search
        """
        model.eval()
        model.to(self.device)
        
        all_preds = []
        all_refs = []
        
        pbar = tqdm(dataloader, desc=f"Evaluating ({decode_method})")
        for i, batch in enumerate(pbar):
            if max_batches and i >= max_batches:
                break
                
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            outputs = model(audio, visual, target=None)
            
            # Decode
            if decode_method == 'beam':
                preds = self.decoder.beam_search_decode(outputs['ctc_logits'], beam_width=beam_width)
            else:
                preds = self.decoder.greedy_decode(outputs['ctc_logits'])
                
            refs = self.decoder.decode_targets(target)
            
            all_preds.extend(preds)
            all_refs.extend(refs)
            
        # Compute Metrics
        wer = self.calculator.compute_wer(all_preds, all_refs)
        cer = self.calculator.compute_cer(all_preds, all_refs)
        
        results = {
            'wer': wer,
            'cer': cer
        }
        
        if return_samples:
            # Return first 5 samples for manual inspection
            results['samples'] = list(zip(all_preds[:5], all_refs[:5]))
            
        return results
