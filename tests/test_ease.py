from pathlib import Path

from open_redactor.cli import unique_output_path
from open_redactor.presets import PRESETS, apply_preset
from open_redactor.sam_client import cache_key_for


def test_presets_exist():
    assert set(PRESETS) == {"family", "street", "screen-share"}
    fam = apply_preset("family")
    assert "face" in fam["targets"]
    assert fam["carry_frames"] >= 4


def test_unique_output_path(tmp_path: Path):
    f = tmp_path / "out.mp4"
    assert unique_output_path(f) == f
    f.write_bytes(b"x")
    second = unique_output_path(f)
    assert second.name == "out-1.mp4"


def test_cache_key_changes_with_phrase(tmp_path: Path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"fake video")
    assert cache_key_for(v, "face") != cache_key_for(v, "person")
    assert cache_key_for(v, "face") == cache_key_for(v, "face")


def test_hosted_endpoint_from_env(monkeypatch, tmp_path):
    import os
    from open_redactor.sam_client import SamApiClient
    monkeypatch.setenv("OPEN_REDACTOR_ENDPOINT", "https://example.test/v1/responses")
    client = SamApiClient.from_env("MODEL_API_KEY")
    assert client.endpoint == "https://example.test/v1/responses"
    client2 = SamApiClient.from_env("MODEL_API_KEY", endpoint="https://other.test/v1/responses")
    assert client2.endpoint == "https://other.test/v1/responses"
