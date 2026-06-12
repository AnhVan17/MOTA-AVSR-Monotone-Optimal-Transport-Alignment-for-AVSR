"""WebDataset feature sharding — write feature shards + build a streaming reader.

Packs many samples into a few ``.tar`` shards instead of millions of loose ``.pt``
files, to stay under Modal Volume's 500k-inode / 262k-files-per-dir limits.
See ``thesis/WEBDATASET_DESIGN.md``.

Write side : ``write_feature_shards`` (used by the preprocessor).
Read  side : ``build_webdataset``     (used by the DataLoader factory).
Both are kept dependency-light so they can be unit-tested with synthetic tensors.
"""
import glob
import io
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Union

import torch
import webdataset as wds

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def _pattern_to_glob(pattern: str) -> str:
    """'.../prefix-%06d.tar' -> '.../prefix-*.tar' (for counting produced shards)."""
    return re.sub(r"%0?\d*d", "*", pattern)


def _meta_path(pattern: str) -> str:
    """'.../prefix-%06d.tar' -> '.../prefix_meta.json'."""
    glob_pat = _pattern_to_glob(pattern)
    if "-*.tar" in glob_pat:
        return glob_pat.replace("-*.tar", "_meta.json")
    return glob_pat.replace("*.tar", "meta.json")


def write_feature_shards(
    samples: Iterable[Dict],
    output_pattern: str,
    maxcount: int = 2000,
) -> Dict:
    """Write samples to WebDataset ``.tar`` shards (frame-based schema).

    Args:
        samples: iterable of dicts with keys ``id`` (str), ``audio`` (Tensor[T_a, 768]
            Whisper features), ``video`` (Tensor[T, H, W, C] uint8 mouth-crop frames),
            ``text`` (str).
        output_pattern: shard path pattern with one printf field,
            e.g. ``/mnt/.../vicocktail-train-%06d.tar``.
        maxcount: max samples per shard before rotating to a new shard.

    Returns:
        ``{"num_samples": int, "num_shards": int}`` (also written to ``<prefix>_meta.json``).
    """
    out_dir = os.path.dirname(output_pattern)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n = 0
    # ShardWriter rotates to a new .tar every `maxcount` samples.
    with wds.ShardWriter(output_pattern, maxcount=maxcount) as sink:
        for s in samples:
            # '.' splits __key__ from the field name in WebDataset → sanitize the key.
            key = str(s["id"]).replace(".", "_").replace("/", "_")
            sink.write({
                "__key__": key,
                "audio.pth": s["audio"].cpu().half(),  # [T_a, 768] fp16 (Whisper, frozen)
                "video.pth": s["video"].cpu(),          # [T, H, W, C] uint8 frames
                "txt": s["text"],                       # str → utf-8
            })
            n += 1

    num_shards = len(glob.glob(_pattern_to_glob(output_pattern)))
    meta = {"num_samples": n, "num_shards": num_shards}
    with open(_meta_path(output_pattern), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    logger.info(f"Wrote {n} samples → {num_shards} shards: {_pattern_to_glob(output_pattern)}")
    return meta


def read_shard_meta(pattern_or_dir: str) -> Optional[Dict]:
    """Best-effort read of a ``*_meta.json`` to get num_samples (for steps/epoch)."""
    candidates = []
    if pattern_or_dir.endswith(".json"):
        candidates = [pattern_or_dir]
    else:
        candidates = glob.glob(os.path.join(pattern_or_dir, "*_meta.json")) if os.path.isdir(pattern_or_dir) \
            else [_meta_path(pattern_or_dir)]
    for c in candidates:
        if os.path.exists(c):
            with open(c, encoding="utf-8") as f:
                return json.load(f)
    return None


def build_webdataset(
    shards: Union[str, List[str]],
    tokenizer,
    train: bool = True,
    augment: bool = False,
    aug_cfg: Optional[Dict] = None,
    shuffle_buffer: int = 1000,
):
    """Build a streaming ``IterableDataset`` over feature shards.

    Yields dicts matching ``FeatureDataset``: ``{audio, visual, target, text, rel_path}``
    so the existing ``Collator`` can batch + pad them unchanged.

    Args:
        shards: brace/glob pattern or explicit list of ``.tar`` paths.
        tokenizer: object with ``.encode(text) -> List[int]``.
        train: if True, shuffle shards + sample buffer; if False, deterministic order.
        augment / aug_cfg: optional on-the-fly feature augmentation (train only).
        shuffle_buffer: reservoir size for sample-level shuffle (train only).
    """
    augmenter = None
    if augment:
        from src.data.augmentations import FeatureAugmenter
        augmenter = FeatureAugmenter(audio_conf=aug_cfg or {}, visual_conf=aug_cfg or {})

    def decode(sample: Dict) -> Dict:
        raw_txt = sample["txt"]
        text = raw_txt.decode("utf-8") if isinstance(raw_txt, (bytes, bytearray)) else raw_txt
        # audio: Whisper features stored fp16 → upcast to float for the model.
        audio = torch.load(io.BytesIO(sample["audio.pth"])).float()
        # visual: raw mouth-crop frames stored uint8 [T, H, W, C] → float [T, C, H, W] in [0,1].
        # ResNet runs inside the model (forward_backbones), so we hand it frames, not features.
        frames = torch.load(io.BytesIO(sample["video.pth"]))
        visual = frames.permute(0, 3, 1, 2).float() / 255.0
        if augmenter is not None:
            audio, visual = augmenter(audio, visual)
        target = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        return {
            "audio": audio,
            "visual": visual,          # [T, C, H, W] frames (model's frozen ResNet encodes them)
            "target": target,
            "text": text,
            "rel_path": sample["__key__"],
        }

    # webdataset expands brace patterns ("{000..099}") but NOT shell globs ("*"); a "*" pattern
    # would be passed through as a literal (missing) filename → silently 0 samples. Expand it here
    # and fail loud on no match (never silently train on an empty stream).
    if isinstance(shards, str) and "*" in shards:
        import glob
        matched = sorted(glob.glob(shards))
        if not matched:
            raise FileNotFoundError(f"No shards matched glob pattern: {shards}")
        shards = matched

    ds = wds.WebDataset(shards, shardshuffle=train, handler=wds.warn_and_continue)
    if train and shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer)
    return ds.map(decode)
