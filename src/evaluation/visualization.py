import torch
import matplotlib.pyplot as plt
import numpy as np
import io
from typing import Dict

class Visualizer:
    """
    Visualization tools for AVSR Model (MQOT & QualityGate).
    """
    
    @staticmethod
    def plot_transport_map(transport_map: torch.Tensor, save_path: str = None):
        """
        Plot the Optimal Transport Map (Soft Alignment).
        Args:
            transport_map: [Ta, Tv] - Alignment between Audio and Visual frames.
        """
        if transport_map is None:
            return
        
        map_np = transport_map.detach().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(map_np, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Transport Mass')
        plt.xlabel('Visual Frames (Tv)')
        plt.ylabel('Audio Frames (Ta)')
        plt.title('M-QOT Soft Alignment Map')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_quality_scores(q_audio: torch.Tensor, q_visual: torch.Tensor, save_path: str = None):
        """
        Plot frame-wise quality scores.
        Args:
            q_audio: [T] or [1] - Audio quality quality over time/epoch
            q_visual: [T] or [1] - Visual quality quality over time/epoch
        """
        qa = q_audio.detach().cpu().numpy().flatten()
        qv = q_visual.detach().cpu().numpy().flatten()
        
        plt.figure(figsize=(10, 4))
        plt.plot(qa, label='Audio Quality', alpha=0.8)
        plt.plot(qv, label='Visual Quality', alpha=0.8)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.title('Modality Quality Scores')
        plt.xlabel('Time/Batch')
        plt.ylabel('Score')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
