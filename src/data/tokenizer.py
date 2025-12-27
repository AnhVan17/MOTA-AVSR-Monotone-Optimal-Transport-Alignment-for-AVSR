"""
Bridge for Tokenizer - Redirects to WhisperProcessor
"""
from .tokenizers.whisper import WhisperProcessor

# Standard name used throughout the project
VietnameseCharTokenizer = WhisperProcessor

__all__ = ['VietnameseCharTokenizer', 'WhisperProcessor']
