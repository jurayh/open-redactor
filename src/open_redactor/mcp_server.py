"""Model Context Protocol server for Open Redactor.

Run with ``open-redactor-mcp``. The server uses newline-delimited JSON-RPC
over stdio and exposes redaction as tools that an MCP client or agent suite
can call. Pipeline progress is redirected to stderr so stdout remains a
clean protocol channel.

Only run this server for a trusted local client. Its tools intentionally
operate on file paths supplied by that client. API keys are never accepted
as tool arguments; the configured backend reads the named environment
variable itself.
"""

from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Optional

from . import __version__
from .api import RedactionOptions, audit_media, redact_media

SERVER_NAME = "open-redactor"
DEFAULT_PROTOCOL_VERSION = "2025-03-26"

_OPTION_FIELDS = {field.name for field in fields(RedactionOptions)}
_OPTION_SCHEMA: dict[str, Any] = {
    "targets": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Short noun phrases to redact. Defaults are used when omitted.",
    },
    "preset": {
        "type": "string",
        "enum": ["family", "street", "screen-share", "documents", "location"],
        "description": "Named target and rendering preset.",
    },
    "mode": {"type": "string", "enum": ["blur", "pixelate", "replace"], "default": "blur"},
    "strength": {"type": "integer", "minimum": 1, "default": 21},
    "mask_margin": {"type": "integer", "minimum": 0, "default": 10},
    "carry_frames": {"type": "integer", "minimum": 0, "default": 4},
    "backend": {"type": "string", "enum": ["api", "hosted", "local"]},
    "endpoint": {"type": "string", "description": "Hosted Responses API endpoint."},
    "provider": {"type": "string", "enum": ["sam", "grounding-sam"]},
    "local": {"type": "boolean", "default": False},
    "api_key_env": {
        "type": "string",
        "default": "MODEL_API_KEY",
        "description": "Name of the environment variable holding the API key. Never pass the key itself.",
    },
    "use_cache": {"type": "boolean", "default": True},
    "pii_text": {"type": "boolean", "default": False},
    "codes": {"type": "boolean", "default": False},
    "sensitive": {"type": "boolean", "default": False},
    "audio_mode": {"type": "string", "enum": ["keep", "mute", "pitch"], "default": "keep"},
    "pitch_factor": {"type": "number", "exclusiveMinimum": 0, "default": 0.8},
    "keep": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Track keys to leave visible while everything else is redacted, for example person:0. Globs allowed. Keys are printed in the coverage report.",
    },
    "exclude": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Track keys to drop from redaction as false positives, for example person:1. Globs allowed.",
    },
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "redact_media",
        "description": "Redact one video or photo and return output paths plus structured analytics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string"},
                "output_path": {"type": "string"},
                "contact_sheet": {"type": "boolean", "default": False},
                "preview": {"type": "boolean", "default": False},
                "report": {"type": "boolean", "default": True},
                **_OPTION_SCHEMA,
            },
            "required": ["input_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "audit_media",
        "description": "Audit one video without rendering a redacted copy. Returns severity analytics and report paths.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string"},
                "output_path": {
                    "type": "string",
                    "description": "Base path used to name the audit, summary, and contact sheet files.",
                },
                **_OPTION_SCHEMA,
            },
            "required": ["input_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_redaction_summary",
        "description": "Read a summary JSON file produced by a prior Open Redactor run.",
        "inputSchema": {
            "type": "object",
            "properties": {"summary_path": {"type": "string"}},
            "required": ["summary_path"],
            "additionalProperties": False,
        },
    },
]


def _options_from_arguments(arguments: dict[str, Any]) -> RedactionOptions:
    values = {key: arguments[key] for key in _OPTION_FIELDS if key in arguments}
    return RedactionOptions(**values)


def call_tool(name: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Call one MCP tool directly. This function is also used by the tests."""
    args = arguments or {}
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")

    # The media pipelines print progress. Stdout belongs to MCP, so send all
    # operational logging to stderr while a tool runs.
    with contextlib.redirect_stdout(sys.stderr):
        if name == "redact_media":
            allowed = _OPTION_FIELDS | {"input_path", "output_path", "contact_sheet", "preview", "report"}
            unknown = set(args) - allowed
            if unknown:
                raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
            result = redact_media(
                args.get("input_path", ""),
                args.get("output_path"),
                _options_from_arguments(args),
                contact_sheet=bool(args.get("contact_sheet", False)),
                preview=bool(args.get("preview", False)),
                report=bool(args.get("report", True)),
            )
            return result.to_dict()
        if name == "audit_media":
            allowed = _OPTION_FIELDS | {"input_path", "output_path"}
            unknown = set(args) - allowed
            if unknown:
                raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
            result = audit_media(args.get("input_path", ""), args.get("output_path"), _options_from_arguments(args))
            return result.to_dict()
        if name == "read_redaction_summary":
            unknown = set(args) - {"summary_path"}
            if unknown:
                raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
            path = Path(args.get("summary_path", "")).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Summary not found: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Summary JSON must contain an object")
            return data
    raise ValueError(f"Unknown tool: {name}")


def _tool_result(data: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "content": [{"type": "text", "text": json.dumps(data, indent=2)}],
        "isError": is_error,
    }
    if not is_error:
        result["structuredContent"] = data
    return result


def handle_message(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Handle one JSON-RPC message and return a response, or None."""
    method = message.get("method")
    request_id = message.get("id")
    is_notification = "id" not in message

    def success(result: Any) -> Optional[dict[str, Any]]:
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def error(code: int, text: str) -> Optional[dict[str, Any]]:
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": text}}

    if method == "initialize":
        params = message.get("params") or {}
        protocol = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
        return success(
            {
                "protocolVersion": protocol,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": __version__},
                "instructions": "Use redact_media to create a share-safe copy or audit_media to inspect a video without changing it.",
            }
        )
    if method == "ping":
        return success({})
    if method == "tools/list":
        return success({"tools": TOOLS})
    if method == "tools/call":
        params = message.get("params") or {}
        try:
            data = call_tool(params.get("name", ""), params.get("arguments") or {})
            return success(_tool_result(data))
        except Exception as exc:
            return success(_tool_result({"error": str(exc)}, is_error=True))
    if method in {"resources/list", "prompts/list"}:
        key = "resources" if method.startswith("resources") else "prompts"
        return success({key: []})
    if isinstance(method, str) and method.startswith("notifications/"):
        return None
    return error(-32601, f"Method not found: {method}")


def _iter_messages(stream: BinaryIO) -> Iterator[tuple[dict[str, Any], bool]]:
    """Yield JSON messages from NDJSON or Content-Length framed input."""
    while True:
        line = stream.readline()
        if not line:
            return
        if not line.strip():
            continue
        if line.lower().startswith(b"content-length:"):
            framed = True
            length = int(line.split(b":", 1)[1].strip())
            while True:
                header = stream.readline()
                if not header or header in {b"\r\n", b"\n"}:
                    break
                if header.lower().startswith(b"content-length:"):
                    length = int(header.split(b":", 1)[1].strip())
            payload = stream.read(length)
            yield json.loads(payload.decode("utf-8")), framed
            continue
        yield json.loads(line.decode("utf-8")), False


def _write_message(stream: BinaryIO, message: dict[str, Any], *, framed: bool) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if framed:
        stream.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
        stream.write(payload)
    else:
        stream.write(payload + b"\n")
    stream.flush()


def main() -> int:
    """Serve MCP requests on stdin and stdout until EOF."""
    framed_output = False
    for message, framed in _iter_messages(sys.stdin.buffer):
        framed_output = framed_output or framed
        try:
            response = handle_message(message)
        except Exception as exc:  # Keep the server alive after a bad request.
            response = {
                "jsonrpc": "2.0",
                "id": message.get("id") if isinstance(message, dict) else None,
                "error": {"code": -32603, "message": str(exc)},
            }
        if response is not None:
            _write_message(sys.stdout.buffer, response, framed=framed_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
