"""build_webdataset must accept a LIST of glob patterns (train split across non-contiguous batches)."""
import pytest
import torch

from src.data.shards import build_webdataset, write_feature_shards


class _FakeTok:
    def encode(self, text):
        return [1, 2, 3]


def _write(prefix, n, tmp_path):
    samples = [
        {
            "id": f"{prefix}_{i}",
            "audio": torch.randn(5, 768),
            "video": (torch.rand(4, 88, 88, 3) * 255).to(torch.uint8),
            "text": "a b",
        }
        for i in range(n)
    ]
    write_feature_shards(samples, str(tmp_path / f"{prefix}-%06d.tar"), maxcount=10)


def test_list_of_globs_concatenates(tmp_path):
    _write("partA", 6, tmp_path)
    _write("partB", 4, tmp_path)
    globs = [str(tmp_path / "partA-*.tar"), str(tmp_path / "partB-*.tar")]
    ds = build_webdataset(globs, _FakeTok(), train=False, augment=False, shuffle_buffer=0)
    assert sum(1 for _ in ds) == 10  # 6 + 4 across the two patterns


def test_list_with_no_match_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(build_webdataset([str(tmp_path / "nope-*.tar")], _FakeTok(), train=False))


def test_single_glob_still_works(tmp_path):
    _write("solo", 5, tmp_path)
    ds = build_webdataset(str(tmp_path / "solo-*.tar"), _FakeTok(), train=False, shuffle_buffer=0)
    assert sum(1 for _ in ds) == 5
