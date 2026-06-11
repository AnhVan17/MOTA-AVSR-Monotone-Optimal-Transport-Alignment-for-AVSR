"""Round-trip tests for WebDataset sharding (src/data/shards.py), frame-based schema.

Writes synthetic samples (uint8 mouth-crop frames + fp16 Whisper audio features) to .tar
shards, then reads them back through the streaming reader and asserts the decoded dict
matches what the Collator/model expect. No network, no heavy deps — synthetic tensors.
"""
import torch

from src.data.shards import write_feature_shards, build_webdataset


class _FakeTokenizer:
    eot_token_id = 50257

    def encode(self, text):
        return [ord(c) % 97 for c in text] or [0]


def _samples(n):
    for i in range(n):
        yield {
            "id": f"sample{i:04d}",
            "audio": torch.randn(10 + i, 768),                                # Whisper feats
            "video": torch.randint(0, 256, (5 + i, 8, 8, 3), dtype=torch.uint8),  # [T,H,W,C]
            "text": f"xin chao {i}",
        }


def test_write_creates_expected_shards_and_meta(tmp_path):
    pattern = str(tmp_path / "vicocktail-train-%06d.tar")
    meta = write_feature_shards(_samples(5), pattern, maxcount=2)

    # 5 samples, maxcount=2 → 3 shards (2 + 2 + 1)
    assert meta["num_samples"] == 5
    assert meta["num_shards"] == 3
    assert len(sorted(tmp_path.glob("vicocktail-train-*.tar"))) == 3
    assert (tmp_path / "vicocktail-train_meta.json").exists()


def test_read_yields_collator_ready_dict(tmp_path):
    pattern = str(tmp_path / "vicocktail-train-%06d.tar")
    write_feature_shards(_samples(5), pattern, maxcount=2)
    shards = sorted(str(p) for p in tmp_path.glob("vicocktail-train-*.tar"))

    ds = build_webdataset(shards, _FakeTokenizer(), train=False, augment=False)
    out = list(ds)

    assert len(out) == 5
    s = out[0]
    assert set(s.keys()) >= {"audio", "visual", "target", "text", "rel_path"}
    assert s["audio"].ndim == 2 and s["audio"].shape[1] == 768
    # visual is now raw frames [T, C, H, W], normalized to [0,1].
    assert s["visual"].ndim == 4 and s["visual"].shape[1] == 3
    assert s["visual"].dtype == torch.float32 and 0.0 <= float(s["visual"].max()) <= 1.0
    assert s["target"].dtype == torch.long
    assert isinstance(s["text"], str)


def test_roundtrip_is_exact(tmp_path):
    """Decoded tensors/text must match what was written (audio within fp16 precision)."""
    pattern = str(tmp_path / "t-%06d.tar")
    audio = torch.arange(30, dtype=torch.float32).reshape(3, 10)
    video = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)

    def one():
        yield {"id": "k0", "audio": audio, "video": video, "text": "xin chào"}

    write_feature_shards(one(), pattern, maxcount=10)
    shards = sorted(str(p) for p in tmp_path.glob("t-*.tar"))
    s = next(iter(build_webdataset(shards, _FakeTokenizer(), train=False)))

    # audio stored fp16 → compare against the fp16-rounded reference.
    assert torch.allclose(s["audio"], audio.half().float())
    # frames: uint8 [T,H,W,C] → float [T,C,H,W] / 255.
    assert torch.allclose(s["visual"], video.permute(0, 3, 1, 2).float() / 255.0)
    assert s["text"] == "xin chào"
    assert s["rel_path"] == "k0"


def test_train_augment_does_not_crash_on_frames(tmp_path):
    """Frame augmentation (flip/time-mask) must accept 4D [T,C,H,W] visual."""
    pattern = str(tmp_path / "a-%06d.tar")
    write_feature_shards(_samples(4), pattern, maxcount=2)
    shards = sorted(str(p) for p in tmp_path.glob("a-*.tar"))

    ds = build_webdataset(shards, _FakeTokenizer(), train=True, augment=True,
                          aug_cfg={"prob": 1.0}, shuffle_buffer=0)
    s = next(iter(ds))
    assert s["visual"].ndim == 4 and s["visual"].shape[1] == 3


def test_val_order_is_deterministic(tmp_path):
    """train=False must preserve write order (so WER is reproducible)."""
    pattern = str(tmp_path / "v-%06d.tar")
    write_feature_shards(_samples(6), pattern, maxcount=2)
    shards = sorted(str(p) for p in tmp_path.glob("v-*.tar"))

    order1 = [s["rel_path"] for s in build_webdataset(shards, _FakeTokenizer(), train=False)]
    order2 = [s["rel_path"] for s in build_webdataset(shards, _FakeTokenizer(), train=False)]
    assert order1 == order2 == [f"sample{i:04d}" for i in range(6)]
