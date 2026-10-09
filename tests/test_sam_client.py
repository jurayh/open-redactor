from open_redactor.sam_client import DEFAULT_TARGETS, parse_output_text, resolve_targets


def test_resolve_defaults_when_empty():
    assert resolve_targets(None, None, False) == DEFAULT_TARGETS


def test_resolve_target_replaces_defaults():
    assert resolve_targets(["dog"], None, False) == ["dog"]


def test_resolve_add_target_on_top_of_defaults():
    out = resolve_targets(None, ["whiteboard"], False)
    assert "whiteboard" in out
    assert "person" in out


def test_parse_zero_match_is_valid():
    result = parse_output_text("", phrase="person", shape=(10, 10))
    assert result.objects == []
    assert result.tracks == {}


def test_parse_json_lines_box():
    text = '{"frame": 0, "track_id": "a", "box": [1, 1, 3, 3]}'
    result = parse_output_text(text, phrase="person", shape=(10, 10))
    assert len(result.objects) == 1
    assert "a" in result.tracks
    assert result.tracks["a"][0].sum() == 4


def test_parse_image_multiple_objects_one_header():
    # Live image format: one frame header, then bare box tokens.
    text = (
        "<0f>0<|box;x1=10;y1=10;x2=60;y2=90;w=200;h=100|>"
        "<|box;x1=80;y1=20;x2=130;y2=95;w=200;h=100|>"
        "<|box;x1=150;y1=5;x2=190;y2=60;w=200;h=100|>"
    )
    result = parse_output_text(text, phrase="person", shape=(100, 200))
    assert len(result.objects) == 3
    assert [o.track_id for o in result.objects] == ["0", "1", "2"]
    assert all(o.frame_index == 0 for o in result.objects)
    assert result.objects[1].box == (80, 20, 130, 95)
    assert set(result.tracks.keys()) == {"0", "1", "2"}


def test_parse_bare_boxes_inherit_later_frame_header():
    text = (
        "<0f>0<|box;x1=0;y1=0;x2=10;y2=10;w=100;h=100|>"
        "<3f>1<|box;x1=20;y1=20;x2=30;y2=30;w=100;h=100|>"
        "<|box;x1=40;y1=40;x2=50;y2=50;w=100;h=100|>"
    )
    result = parse_output_text(text, phrase="person", shape=(100, 100))
    assert [(o.track_id, o.frame_index) for o in result.objects] == [
        ("0", 0), ("1", 3), ("2", 3),
    ]


def test_parse_explicit_video_tokens_unchanged():
    text = (
        "<0f>0<|box;x1=0;y1=0;x2=10;y2=10;w=100;h=100|>"
        "<1f>0<|box;x1=2;y1=2;x2=12;y2=12;w=100;h=100|>"
        "<1f>1<|box;x1=50;y1=50;x2=70;y2=70;w=100;h=100|>"
    )
    result = parse_output_text(text, phrase="face", shape=(100, 100))
    assert len(result.objects) == 3
    assert set(result.tracks["0"].keys()) == {0, 1}
    assert set(result.tracks["1"].keys()) == {1}


def test_parse_mask_source_box_without_mask_tokens():
    text = "<0f>0<|box;x1=10;y1=10;x2=60;y2=90;w=200;h=100|>"
    result = parse_output_text(text, phrase="person", shape=(100, 200))
    assert result.mask_source == "box"


def test_parse_mask_source_pixel_when_decode_places_rasters(monkeypatch):
    import numpy as np

    import open_redactor.mask_decode as mask_decode

    def fake_decode(items):
        return [np.ones((int(i["height"]), int(i["width"])), dtype=bool) for i in items]

    monkeypatch.setattr(mask_decode, "decode_masks_batch", fake_decode)
    text = (
        "<0f>0<|box;x1=10;y1=10;x2=60;y2=90;w=200;h=100|>"
        "<|mask;x=0;y=0;data=80,50,AAAA|>"
    )
    result = parse_output_text(text, phrase="person", shape=(100, 200))
    assert result.mask_source == "pixel"


def test_parse_mask_source_box_when_decode_unavailable(monkeypatch):
    import open_redactor.mask_decode as mask_decode

    monkeypatch.setattr(mask_decode, "decode_masks_batch", lambda items: None)
    text = (
        "<0f>0<|box;x1=10;y1=10;x2=60;y2=90;w=200;h=100|>"
        "<|mask;x=0;y=0;data=80,50,AAAA|>"
    )
    result = parse_output_text(text, phrase="person", shape=(100, 200))
    assert result.mask_source == "box"
