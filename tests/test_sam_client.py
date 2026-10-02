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
