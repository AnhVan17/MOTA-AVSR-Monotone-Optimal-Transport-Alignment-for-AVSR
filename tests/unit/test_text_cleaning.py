"""Unit tests for Vietnamese text helpers (src/utils/text_cleaning.py)."""
from src.utils.text_cleaning import normalize_text, find_text_in_sample


def test_normalize_text_empty_and_none():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_normalize_text_lowercase_strip_collapse_whitespace():
    assert normalize_text("  Xin   CHàO  ") == "xin chào"


def test_normalize_text_nfc_composition():
    # 'a' + combining grave accent → composed 'à'.
    assert normalize_text("à") == "à"


def test_find_text_prefers_json_dict_field():
    assert find_text_in_sample({"json": {"text": "Việt Nam"}}) == "Việt Nam"


def test_find_text_decodes_bytes_txt():
    assert find_text_in_sample({"txt": "xin chào".encode("utf-8")}) == "xin chào"


def test_find_text_filters_out_file_path():
    assert find_text_in_sample({"txt": "data/video/x.mp4"}) == ""


def test_find_text_missing_keys_returns_empty():
    assert find_text_in_sample({}) == ""
