"""Open Redactor."""

from typing import Any

__all__ = ["__version__", "RedactionOptions", "RedactionResult", "audit", "audit_media", "redact", "redact_media"]
__version__ = "0.2.8"


def __getattr__(name: str) -> Any:
    """Lazily expose the public API so importing the package stays light."""
    if name in {"RedactionOptions", "RedactionResult", "audit", "audit_media", "redact", "redact_media"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
