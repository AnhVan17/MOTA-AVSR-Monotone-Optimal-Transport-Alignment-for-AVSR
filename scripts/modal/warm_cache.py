"""Warm shared model caches ONCE — run before fanning out preprocess / training.

    modal run scripts/modal/warm_cache.py

Does two things so every later container starts instantly (no runtime weight download):
  1. Builds PREPROC_IMAGE → bakes face-alignment SFD/FAN torch-hub weights into the image layer
     (otherwise each fanned-out container cold-downloads them from adrianbulat.com and the
     concurrent hits get throttled to ~10 kB/s).
  2. Seeds whisper-small into the shared ``hf-hub-cache`` volume (via HF_HOME), so workers READ
     it from the volume instead of each downloading concurrently (which races the cache).

The hf-hub-cache volume persists, so this is a one-time setup (re-run only to add models).
"""
import sys
from pathlib import Path

import modal

# Make `src` importable locally (app-build); containers have /root on path.
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import HF_CACHE_DIR, PREPROC_IMAGE

app = modal.App("avsr-warm-cache")
hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.function(
    image=PREPROC_IMAGE,  # building it bakes SFD/FAN into the image (run_function at build)
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def seed_whisper():
    """Download whisper-small into the hf-hub-cache volume (idempotent; HF_HOME=HF_CACHE_DIR)."""
    from transformers import WhisperFeatureExtractor, WhisperModel

    WhisperModel.from_pretrained("openai/whisper-small")
    WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    hf_cache.commit()
    return "hf-hub-cache seeded: whisper-small"


@app.local_entrypoint()
def main():
    print("Warming: build image (bakes SFD/FAN) + seed whisper into hf-hub-cache volume...")
    print(seed_whisper.remote())
