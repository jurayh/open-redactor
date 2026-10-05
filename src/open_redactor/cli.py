"""CLI for Open Redactor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .pipeline import run_pipeline
from .sam_client import resolve_targets


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="open-redactor", description="Make video share-safe with SAM 3.1 prompt-driven redaction")
    p.add_argument("input", help="Input MP4 file or directory in batch mode")
    p.add_argument("--output", default=None, help="Output path. Defaults to input name plus a redacted suffix")
    p.add_argument("--target", action="append", default=None, help="Add one phrase. Can be repeated")
    p.add_argument("--targets-default", action="store_true", help="Start from the default set of person, face, license plate, and screen")
    p.add_argument("--add-target", action="append", default=None, help="Add one phrase on top of the defaults. Can be repeated")
    p.add_argument("--mode", choices=["blur", "pixelate"], default="blur", help="Redaction mode")
    p.add_argument("--strength", type=int, default=21, help="Blur radius or pixel block size")
    p.add_argument("--mask-margin", type=int, default=10, help="Pad every mask by this many pixels")
    p.add_argument("--carry-frames", type=int, default=4, help="Hold a track for this many frames after it vanishes")
    p.add_argument("--contact-sheet", action="store_true", help="Write a PNG grid of redacted sample frames")
    p.add_argument("--preview", action="store_true", help="Render only a 3 second sample plus contact sheet and coverage report")
    p.add_argument("--report", action="store_true", help="Write a coverage report next to the output. On by default")
    p.add_argument("--no-report", action="store_true", help="Skip the coverage report")
    p.add_argument("--local", action="store_true", help="Run open weights on device and send nothing to the API")
    p.add_argument("--api-key-env", default="MODEL_API_KEY", help="Name the environment variable that holds the API key")
    p.add_argument("--batch", action="store_true", help="Treat the input as a directory and process each MP4 inside")
    return p


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.redacted{input_path.suffix}")


def process_one(
    input_path: Path,
    output_path: Optional[Path],
    targets: List[str],
    mode: str,
    strength: int,
    mask_margin: int,
    carry_frames: int,
    contact_sheet: bool,
    local: bool,
    api_key_env: str,
    preview: bool = False,
    report: bool = True,
) -> int:
    out = output_path if output_path else default_output_path(input_path)
    try:
        run_pipeline(
            input_path=input_path,
            output_path=out,
            targets=targets,
            mode=mode,
            strength=strength,
            mask_margin=mask_margin,
            carry_frames=carry_frames,
            contact_sheet=contact_sheet,
            local=local,
            api_key_env=api_key_env,
            preview=preview,
            report=report,
        )
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    targets = resolve_targets(
        targets=args.target,
        add_targets=args.add_target,
        use_defaults=args.targets_default,
    )

    input_path = Path(args.input)

    if args.batch:
        if not input_path.is_dir():
            print(f"Error: batch mode needs a directory: {input_path}", file=sys.stderr)
            return 2
        mp4s = sorted(input_path.glob("*.mp4"))
        if not mp4s:
            print(f"No MP4 files found in {input_path}")
            return 0
        exit_code = 0
        for mp4 in mp4s:
            out = None
            if args.output:
                # In batch mode --output is treated as an output directory
                out_dir = Path(args.output)
                out = out_dir / default_output_path(mp4).name
            code = process_one(
                input_path=mp4,
                output_path=out,
                targets=targets,
                mode=args.mode,
                strength=args.strength,
                mask_margin=args.mask_margin,
                carry_frames=args.carry_frames,
                contact_sheet=args.contact_sheet,
                local=args.local,
                api_key_env=args.api_key_env,
                preview=args.preview,
                report=not args.no_report,
            )
            if code != 0:
                exit_code = code
        return exit_code

    # Single file mode
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        return 2
    if input_path.is_dir():
        print("Error: Input is a directory. Use --batch to process a folder.", file=sys.stderr)
        return 2

    output_path = Path(args.output) if args.output else None
    return process_one(
        input_path=input_path,
        output_path=output_path,
        targets=targets,
        mode=args.mode,
        strength=args.strength,
        mask_margin=args.mask_margin,
        carry_frames=args.carry_frames,
        contact_sheet=args.contact_sheet,
        local=args.local,
        api_key_env=args.api_key_env,
        preview=args.preview,
        report=not args.no_report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
