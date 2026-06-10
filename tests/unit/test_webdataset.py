"""Round-trip tests for WebDataset feature sharding (src/data/shards.py).

Writes synthetic feature samples to .tar shards, then reads them back through the
streaming reader and asserts the decoded dict matches what FeatureDataset/Collator expect.
No network, no heavy deps — synthetic tensors + a fake tokenizer.
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
            "audio": torch.randn(10 + i, 768),
            "visual": torch.randn(5 + i, 512),
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
    assert s["visual"].ndim == 2 and s["visual"].shape[1] == 512
    assert s["target"].dtype == torch.long
    assert isinstance(s["text"], str)


def test_roundtrip_is_byte_exact(tmp_path):
    """The decoded tensors/text must equal what was written (no corruption)."""
    pattern = str(tmp_path / "t-%06d.tar")
    audio = torch.arange(30, dtype=torch.float32).reshape(3, 10)
    visual = torch.ones(4, 512)

    def one():
        yield {"id": "k0", "audio": audio, "visual": visual, "text": "xin chào"}

    write_feature_shards(one(), pattern, maxcount=10)
    shards = sorted(str(p) for p in tmp_path.glob("t-*.tar"))
    s = next(iter(build_webdataset(shards, _FakeTokenizer(), train=False)))

    assert torch.allclose(s["audio"], audio)
    assert torch.allclose(s["visual"], visual)
    assert s["text"] == "xin chào"
    assert s["rel_path"] == "k0"


def test_val_order_is_deterministic(tmp_path):
    """train=False must preserve write order (so WER is reproducible)."""
    pattern = str(tmp_path / "v-%06d.tar")
    write_feature_shards(_samples(6), pattern, maxcount=2)
    shards = sorted(str(p) for p in tmp_path.glob("v-*.tar"))

    order1 = [s["rel_path"] for s in build_webdataset(shards, _FakeTokenizer(), train=False)]
    order2 = [s["rel_path"] for s in build_webdataset(shards, _FakeTokenizer(), train=False)]
    assert order1 == order2 == [f"sample{i:04d}" for i in range(6)]
