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
        
    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None,
        return_samples: bool = True
    ) -> Dict:
        """
        Run full evaluation on a dataset.
        """
        model.eval()
        model.to(self.device)
        
        all_preds = []
        all_refs = []
        
        pbar = tqdm(dataloader, desc="Evaluating")
        for i, batch in enumerate(pbar):
            if max_batches and i >= max_batches:
                break
                
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            outputs = model(audio, visual, target=None)
            
            # Decode
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
