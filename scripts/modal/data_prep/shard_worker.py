"""Modal-free per-shard worker for in-container multiprocessing.

Kept SEPARATE from prep_vicocktail.py on purpose: a multiprocessing 'spawn' child
re-imports the module that defines its target function. This module imports only
src.* (no modal app/image construction), so re-importing it in a child is safe and cheap.

Each worker processes ONE raw shard with its OWN models on the shared GPU, writes its own
output shards (unique tag = raw-shard stem), and is idempotent (skips if _meta.json exists).
"""
import os
import sys


def prewarm_model_cache() -> None:
    """Download SFD/FAN + whisper-small weights ONCE (on CPU) before workers spawn.

    Avoids N workers racing the same first-time download (corrupted cache). Weights land in
    the shared on-disk cache; workers then load them with no network.
    """
    sys.path.append("/root")
    import face_alignment
    from transformers import WhisperFeatureExtractor, WhisperModel

    face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device="cpu", face_detector="sfd", compile=False,
    )
    WhisperModel.from_pretrained("openai/whisper-small")
    WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


def process_one_shard(args) -> str:
    """Process ONE raw shard. Runs in a spawned subprocess (own CUDA context + models).

    args = (data_root, output_root, shard_name, limit_ratio, max_samples, detect_batch).
    Returns a status string: 'OK <tag>' / 'SKIP <tag>' / 'FAIL <tag>: <err>'.
    """
    data_root, output_root, shard_name, limit_ratio, max_samples, detect_batch = args
    sys.path.append("/root")
    from src.data.preprocessors.base import PreprocessConfig
    from src.data.preprocessors.vicocktail import ViCocktailPreprocessor
    from src.data.shards import _meta_path

    PreprocessConfig.DETECT_BATCH = detect_batch
    out_tag = shard_name[:-4] if shard_name.endswith(".tar") else shard_name  # e.g. 'avvn-train-000005'
    shard_pattern = f"{output_root}/vicocktail-{out_tag}-%06d.tar"

    # Per-shard resume: a finished shard wrote its _meta.json → skip.
    if os.path.exists(_meta_path(shard_pattern)):
        return f"SKIP {out_tag} (already done)"

    try:
        proc = ViCocktailPreprocessor(data_root=data_root, use_precropped=False)
        proc.run(
            shard_pattern=shard_pattern,
            shard_names=[shard_name],
            limit_ratio=limit_ratio,
            max_samples=(max_samples or None),
        )
        return f"OK {out_tag}"
    except Exception as e:  # one bad shard shouldn't kill the whole batch
        return f"FAIL {out_tag}: {e}"
