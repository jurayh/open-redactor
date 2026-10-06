"""CLI for Open Redactor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .pipeline import run_pipeline
from .presets import PRESETS, apply_preset
from .sam_client import resolve_targets


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="open-redactor", description="Make video share-safe with SAM 3.1 prompt-driven redaction")
    p.add_argument("input", nargs="?", default=None, help="Input video file, MP4, MOV, MKV, WebM, AVI, or M4V, or a directory in batch mode")
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
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None, help="Use a preset: family, street, screen-share, or documents")
    p.add_argument("--no-cache", action="store_true", help="Do not reuse cached SAM results")
    p.add_argument("--pii-text", action="store_true", help="Also OCR sampled frames for card numbers, SSNs, phones, and emails and blur them")
    p.add_argument("--codes", action="store_true", help="Also detect and cover QR codes and barcodes in sampled frames")
    p.add_argument("--sensitive", action="store_true", help="Turn on --pii-text and --codes together")
    p.add_argument("--provider", choices=["sam", "grounding-sam"], default=None, help="Model provider: SAM via API or hosted, or Grounding SAM locally")
    p.add_argument("--backend", choices=["api", "hosted", "local"], default=None, help="Where SAM runs: Meta Model API, your own hosted endpoint, or local open weights")
    p.add_argument("--endpoint", default=None, help="Responses API endpoint for the hosted backend, or set OPEN_REDACTOR_ENDPOINT")
    p.add_argument("--ui", action="store_true", help="Launch the drag and drop local web page instead of processing a file")
    p.add_argument("--batch", action="store_true", help="Treat the input as a directory and process each MP4 inside")
    return p


def default_output_path(input_path: Path) -> Path:
    return unique_output_path(input_path.with_name(f"{input_path.stem}.redacted.mp4"))


def unique_output_path(path: Path) -> Path:
    """Return a path that does not overwrite an existing file."""
    if not path.exists():
        return path
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path


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
    use_cache: bool = True,
    backend: str | None = None,
    endpoint: str | None = None,
    provider: str | None = None,
    pii_text: bool = False,
    codes: bool = False,
) -> int:
    out = output_path if output_path else default_output_path(input_path)
    if output_path is not None:
        out = unique_output_path(output_path)
        if out.resolve() == input_path.resolve():
            out = unique_output_path(input_path.with_name(f'{input_path.stem}.redacted.mp4'))
        elif out != output_path:
            print(f'Output exists, writing to {out} instead so nothing is overwritten.')
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
            use_cache=use_cache,
            backend=backend,
            endpoint=endpoint,
            provider=provider,
            pii_text=pii_text,
            codes=codes,
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

    if args.ui:
        from .webui import serve_ui
        print("Open http://127.0.0.1:8765 in your browser. Files stay on this machine.")
        serve_ui()
        return 0

    if not args.input:
        parser.error("input is required unless you pass --ui")

    # Presets fill only values the user did not set explicitly
    if args.preset:
        preset = apply_preset(args.preset)
        if not args.target and not args.add_target and not args.targets_default:
            args.target = list(preset["targets"])
        if args.mode == "blur" and preset["mode"] != "blur":
            args.mode = preset["mode"]
        # Strength, margin, and carry use CLI defaults, so only apply preset
        # values when the user left them at the default sentinels
        if args.strength == 21:
            args.strength = int(preset["strength"])
        if args.mask_margin == 10:
            args.mask_margin = int(preset["mask_margin"])
        if args.carry_frames == 4:
            args.carry_frames = int(preset["carry_frames"])
        print(f"Preset: {args.preset} - {preset['description']}")

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
        from .pipeline import SUPPORTED_INPUT_SUFFIXES
        mp4s = sorted(f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_INPUT_SUFFIXES)
        if not mp4s:
            print(f"No supported video files found in {input_path}")
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
                use_cache=not args.no_cache,
                backend=args.backend,
                endpoint=args.endpoint,
                provider=args.provider,
                pii_text=args.pii_text or args.sensitive,
                codes=args.codes or args.sensitive,
            )
            if code != 0:
                exit_code = code
        return exit_code

    # Photo mode: image inputs go to the image pipeline, output keeps its format
    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        from .image_pipeline import run_image_pipeline
        out = Path(args.output) if args.output else unique_output_path(
            input_path.with_name(f"{input_path.stem}.redacted{input_path.suffix}")
        )
        if args.output:
            out = unique_output_path(out)
        try:
            run_image_pipeline(
                input_path=input_path,
                output_path=out,
                targets=targets,
                mode=args.mode,
                strength=args.strength,
                mask_margin=args.mask_margin,
                local=args.local,
                api_key_env=args.api_key_env,
                backend=args.backend,
                endpoint=args.endpoint,
                provider=args.provider,
                pii_text=args.pii_text or args.sensitive,
                codes=args.codes or args.sensitive,
            )
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

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
        use_cache=not args.no_cache,
        backend=args.backend,
        endpoint=args.endpoint,
        provider=args.provider,
        pii_text=args.pii_text or args.sensitive,
        codes=args.codes or args.sensitive,
    )


if __name__ == "__main__":
    raise SystemExit(main())
