"""Audit gate for pre-commit hooks and CI.

The normal CLI audit reports findings and exits successfully because an
audit is information. Automation often needs the opposite contract: fail
when the configured severity is present. This entry point provides that
contract without changing the CLI.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .api import RedactionOptions, audit_media

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _meets_threshold(summary: dict, fail_on: str) -> bool:
    threshold = SEVERITY_ORDER[fail_on]
    counts = summary.get("severity_counts", {})
    return any(counts.get(level, 0) > 0 and rank >= threshold for level, rank in SEVERITY_ORDER.items())


def audit_files(
    paths: Iterable[str | Path],
    *,
    options: Optional[RedactionOptions] = None,
    fail_on: str = "medium",
    output_dir: Optional[str | Path] = None,
) -> int:
    """Audit files and return 0, 1 for findings, or 2 for audit errors."""
    if fail_on not in SEVERITY_ORDER:
        raise ValueError(f"Unknown fail_on severity '{fail_on}'")
    opts = options or RedactionOptions()
    destination = Path(output_dir).expanduser() if output_dir else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)

    saw_error = False
    saw_finding = False
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            if destination:
                base = destination / f"{path.stem}.audit.mp4"
                result = audit_media(path, base, opts)
            else:
                with tempfile.TemporaryDirectory(prefix="open-redactor-audit-") as tmp:
                    result = audit_media(path, Path(tmp) / f"{path.stem}.audit.mp4", opts)
            if result.summary is None:
                raise RuntimeError("Audit completed without a summary JSON")
            print(f"{path}: {json.dumps(result.summary.get('severity_counts', {}), sort_keys=True)}")
            if _meets_threshold(result.summary, fail_on):
                saw_finding = True
                print(f"{path}: sensitive elements at or above {fail_on} severity")
        except Exception as exc:
            saw_error = True
            print(f"{path}: audit failed: {exc}")
    if saw_error:
        return 2
    return 1 if saw_finding else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-redactor-audit",
        description="Audit videos and fail CI or a pre-commit hook when sensitive elements are found",
    )
    parser.add_argument("paths", nargs="+", help="Video files to audit")
    parser.add_argument("--fail-on", choices=sorted(SEVERITY_ORDER), default="medium")
    parser.add_argument("--output-dir", default=None, help="Keep audit reports and summaries in this directory")
    parser.add_argument("--preset", choices=["family", "street", "screen-share", "documents", "location"], default=None)
    parser.add_argument("--target", action="append", default=None, help="Add one phrase. Can be repeated")
    parser.add_argument("--backend", choices=["api", "hosted", "local"], default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--provider", choices=["sam", "grounding-sam"], default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--sensitive", action="store_true", help="Turn on text PII and code scanning")
    parser.add_argument("--pii-text", action="store_true")
    parser.add_argument("--codes", action="store_true")
    parser.add_argument("--audio", choices=["keep", "mute", "pitch"], default="keep")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    options = RedactionOptions(
        targets=args.target,
        preset=args.preset,
        backend=args.backend,
        endpoint=args.endpoint,
        provider=args.provider,
        local=args.local,
        api_key_env=args.api_key_env,
        use_cache=not args.no_cache,
        sensitive=args.sensitive,
        pii_text=args.pii_text,
        codes=args.codes,
        audio_mode=args.audio,
    )
    return audit_files(args.paths, options=options, fail_on=args.fail_on, output_dir=args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
