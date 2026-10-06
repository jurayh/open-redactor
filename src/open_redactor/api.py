"""Public Python API for Open Redactor.

The functions in this module are the supported integration surface for
applications, notebooks, workflow engines, and agent tools. They return
structured results instead of asking callers to parse terminal output or
shell out to the CLI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
REDACTION_MODES = {"blur", "pixelate", "replace"}
AUDIO_MODES = {"keep", "mute", "pitch"}
BACKENDS = {"api", "hosted", "local"}
PROVIDERS = {"sam", "grounding-sam"}


@dataclass(frozen=True)
class RedactionOptions:
    """Options shared by :func:`redact_media` and :func:`audit_media`."""

    targets: Optional[Sequence[str]] = None
    preset: Optional[str] = None
    mode: str = "blur"
    strength: int = 21
    mask_margin: int = 10
    carry_frames: int = 4
    backend: Optional[str] = None
    endpoint: Optional[str] = None
    provider: Optional[str] = None
    local: bool = False
    api_key_env: str = "MODEL_API_KEY"
    use_cache: bool = True
    pii_text: bool = False
    codes: bool = False
    sensitive: bool = False
    audio_mode: str = "keep"
    pitch_factor: float = 0.8

    def validate(self) -> None:
        if self.mode not in REDACTION_MODES:
            raise ValueError(f"Unknown mode '{self.mode}'. Choose blur, pixelate, or replace.")
        if self.audio_mode not in AUDIO_MODES:
            raise ValueError(f"Unknown audio mode '{self.audio_mode}'. Choose keep, mute, or pitch.")
        if self.backend is not None and self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend '{self.backend}'. Choose api, hosted, or local.")
        if self.provider is not None and self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{self.provider}'. Choose sam or grounding-sam.")
        if self.strength <= 0:
            raise ValueError("strength must be positive")
        if self.mask_margin < 0:
            raise ValueError("mask_margin cannot be negative")
        if self.carry_frames < 0:
            raise ValueError("carry_frames cannot be negative")
        if self.pitch_factor <= 0:
            raise ValueError("pitch_factor must be positive")


@dataclass(frozen=True)
class RedactionResult:
    """Structured outcome from a redaction or audit call."""

    input_path: str
    output_path: Optional[str] = None
    audit_path: Optional[str] = None
    summary_path: Optional[str] = None
    coverage_report_path: Optional[str] = None
    contact_sheet_path: Optional[str] = None
    summary: Optional[dict[str, Any]] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_options(
    options: Optional[RedactionOptions | Mapping[str, Any]],
    overrides: Mapping[str, Any],
) -> RedactionOptions:
    if options is None:
        values: dict[str, Any] = {}
    elif isinstance(options, RedactionOptions):
        values = asdict(options)
    elif isinstance(options, Mapping):
        values = dict(options)
    else:
        raise TypeError("options must be RedactionOptions, a mapping, or None")
    values.update({key: value for key, value in overrides.items() if value is not None})
    unknown = set(values) - set(RedactionOptions.__dataclass_fields__)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown Open Redactor option(s): {names}")
    result = RedactionOptions(**values)
    result.validate()
    return result


def _resolve_targets(options: RedactionOptions) -> list[str]:
    from .presets import PRESETS, apply_preset
    from .sam_client import resolve_targets

    targets = list(options.targets) if options.targets is not None else None
    if options.preset:
        if options.preset not in PRESETS:
            choices = ", ".join(sorted(PRESETS))
            raise ValueError(f"Unknown preset '{options.preset}'. Choose {choices}.")
        if targets is None:
            targets = list(apply_preset(options.preset)["targets"])
    return resolve_targets(targets=targets, add_targets=None, use_defaults=False)


def _unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused output path near {path}")


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() in IMAGE_SUFFIXES:
        return _unique_output_path(input_path.with_name(f"{input_path.stem}.redacted{input_path.suffix}"))
    return _unique_output_path(input_path.with_name(f"{input_path.stem}.redacted.mp4"))


def _choose_output_path(input_path: Path, output_path: Optional[str | Path]) -> Path:
    if output_path is None:
        return _default_output_path(input_path)
    requested = Path(output_path).expanduser()
    if requested.resolve() == input_path.resolve():
        return _default_output_path(input_path)
    return _unique_output_path(requested)


def _load_summary(path: Optional[str]) -> Optional[dict[str, Any]]:
    if not path:
        return None
    summary_path = Path(path)
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _result_from_details(input_path: Path, details: Mapping[str, Any]) -> RedactionResult:
    summary_path = details.get("summary") if isinstance(details.get("summary"), str) else None
    return RedactionResult(
        input_path=str(input_path),
        output_path=details.get("output") if isinstance(details.get("output"), str) else None,
        audit_path=details.get("audit") if isinstance(details.get("audit"), str) else None,
        summary_path=summary_path,
        coverage_report_path=details.get("report") if isinstance(details.get("report"), str) else None,
        contact_sheet_path=details.get("contact_sheet") if isinstance(details.get("contact_sheet"), str) else None,
        summary=_load_summary(summary_path),
        details=dict(details),
    )


def redact_media(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    options: Optional[RedactionOptions | Mapping[str, Any]] = None,
    *,
    contact_sheet: bool = False,
    preview: bool = False,
    report: bool = True,
    **option_overrides: Any,
) -> RedactionResult:
    """Redact one video or photo and return paths plus run analytics.

    ``targets`` may be supplied through ``options`` or as a keyword override.
    When neither targets nor a preset is supplied, the product default target
    set is used.
    """

    opts = _coerce_options(options, option_overrides)
    source = Path(input_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")
    if source.is_dir():
        raise ValueError("redact_media handles one file. Iterate over a directory or use the CLI batch mode.")
    suffix = source.suffix.lower()
    if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        supported = ", ".join(sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES))
        raise ValueError(f"Unsupported input format '{source.suffix}'. Supported inputs: {supported}.")

    targets = _resolve_targets(opts)
    destination = _choose_output_path(source, output_path)
    sensitive = opts.sensitive

    if suffix in IMAGE_SUFFIXES:
        if preview:
            raise ValueError("Preview mode is available for video, not photos.")
        from .image_pipeline import run_image_pipeline

        details = run_image_pipeline(
            input_path=source,
            output_path=destination,
            targets=targets,
            mode=opts.mode,
            strength=opts.strength,
            mask_margin=opts.mask_margin,
            local=opts.local,
            api_key_env=opts.api_key_env,
            backend=opts.backend,
            endpoint=opts.endpoint,
            provider=opts.provider,
            pii_text=opts.pii_text or sensitive,
            codes=opts.codes or sensitive,
        )
        return _result_from_details(source, details)

    from .pipeline import run_pipeline

    details = run_pipeline(
        input_path=source,
        output_path=destination,
        targets=targets,
        mode=opts.mode,
        strength=opts.strength,
        mask_margin=opts.mask_margin,
        carry_frames=opts.carry_frames,
        contact_sheet=contact_sheet,
        local=opts.local,
        api_key_env=opts.api_key_env,
        preview=preview,
        report=report,
        use_cache=opts.use_cache,
        backend=opts.backend,
        endpoint=opts.endpoint,
        provider=opts.provider,
        pii_text=opts.pii_text or sensitive,
        codes=opts.codes or sensitive,
        audio_mode=opts.audio_mode,
        pitch_factor=opts.pitch_factor,
        shadow=False,
    )
    return _result_from_details(source, details)


def audit_media(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    options: Optional[RedactionOptions | Mapping[str, Any]] = None,
    **option_overrides: Any,
) -> RedactionResult:
    """Audit one video without rendering a redacted copy.

    The returned result points to the text audit and JSON summary produced by
    the pipeline. Photo shadow audits are not part of the current pipeline, so
    photo inputs raise a clear error instead of returning a partial audit.
    """

    opts = _coerce_options(options, option_overrides)
    source = Path(input_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")
    if source.suffix.lower() in IMAGE_SUFFIXES:
        raise ValueError("audit_media currently supports video. Use redact_media for photos.")
    if source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video format '{source.suffix}'.")

    targets = _resolve_targets(opts)
    audit_base = Path(output_path).expanduser() if output_path else source.with_name(f"{source.stem}.audit.mp4")
    from .pipeline import run_pipeline

    details = run_pipeline(
        input_path=source,
        output_path=audit_base,
        targets=targets,
        mode=opts.mode,
        strength=opts.strength,
        mask_margin=opts.mask_margin,
        carry_frames=opts.carry_frames,
        contact_sheet=False,
        local=opts.local,
        api_key_env=opts.api_key_env,
        preview=False,
        report=False,
        use_cache=opts.use_cache,
        backend=opts.backend,
        endpoint=opts.endpoint,
        provider=opts.provider,
        pii_text=opts.pii_text or opts.sensitive,
        codes=opts.codes or opts.sensitive,
        audio_mode=opts.audio_mode,
        pitch_factor=opts.pitch_factor,
        shadow=True,
    )
    return _result_from_details(source, details)


# Short aliases are convenient in notebooks and agent tools.
redact = redact_media
audit = audit_media


__all__ = [
    "AUDIO_MODES",
    "BACKENDS",
    "IMAGE_SUFFIXES",
    "PROVIDERS",
    "REDACTION_MODES",
    "VIDEO_SUFFIXES",
    "RedactionOptions",
    "RedactionResult",
    "audit",
    "audit_media",
    "redact",
    "redact_media",
]
