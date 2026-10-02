from open_redactor.cli import build_parser, default_output_path
from pathlib import Path


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["in.mp4"])
    assert args.mode == "blur"
    assert args.mask_margin == 10
    assert args.carry_frames == 4
    assert args.local is False


def test_parser_target_repeat():
    parser = build_parser()
    args = parser.parse_args(["in.mp4", "--target", "face", "--target", "dog"])
    assert args.target == ["face", "dog"]


def test_default_output_path():
    p = Path("clip.mp4")
    out = default_output_path(p)
    assert out.name == "clip.redacted.mp4"
