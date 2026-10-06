import json
import subprocess
import sys
from pathlib import Path

import pytest

from open_redactor import RedactionOptions, RedactionResult, audit_media, redact_media
from open_redactor import api as api_module
from open_redactor import audit_gate, image_pipeline, mcp_server, pipeline


def test_public_api_video_route(monkeypatch, tmp_path):
    source = tmp_path / "clip.mov"
    source.write_bytes(b"video")
    summary = tmp_path / "clip.redacted.summary.json"
    summary.write_text(json.dumps({"elements_found": 2, "severity_counts": {"critical": 1}}))
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "input": str(source),
            "output": str(tmp_path / "clip.redacted.mp4"),
            "summary": str(summary),
            "report": str(tmp_path / "clip.redacted.coverage.txt"),
            "contact_sheet": None,
            "frames": 12,
            "objects": 2,
        }

    monkeypatch.setattr(pipeline, "run_pipeline", fake_pipeline)
    result = redact_media(source, targets=["passport"], sensitive=True, audio_mode="mute")

    assert isinstance(result, RedactionResult)
    assert captured["targets"] == ["passport"]
    assert captured["codes"] is True
    assert captured["pii_text"] is True
    assert captured["audio_mode"] == "mute"
    assert result.summary == {"elements_found": 2, "severity_counts": {"critical": 1}}
    assert result.to_dict()["summary_path"] == str(summary)


def test_public_api_photo_route(monkeypatch, tmp_path):
    source = tmp_path / "photo.png"
    source.write_bytes(b"photo")
    captured = {}

    def fake_image_pipeline(**kwargs):
        captured.update(kwargs)
        return {"input": str(source), "output": str(kwargs["output_path"]), "objects": 1, "photo": True}

    monkeypatch.setattr(image_pipeline, "run_image_pipeline", fake_image_pipeline)
    result = redact_media(source, options=RedactionOptions(targets=["face"], mode="pixelate"))

    assert captured["mode"] == "pixelate"
    assert result.output_path.endswith("photo.redacted.png")


def test_public_api_rejects_bad_options_and_photo_audit(tmp_path):
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"photo")
    with pytest.raises(ValueError, match="currently supports video"):
        audit_media(source)
    with pytest.raises(ValueError, match="Unknown mode"):
        redact_media(source, mode="smear")


def test_mcp_discovery_and_summary_tool(tmp_path):
    response = mcp_server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}
    )
    assert response["result"]["serverInfo"]["name"] == "open-redactor"

    tools = mcp_server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert {tool["name"] for tool in tools} == {"redact_media", "audit_media", "read_redaction_summary"}

    summary = tmp_path / "run.summary.json"
    summary.write_text(json.dumps({"elements_found": 3}))
    data = mcp_server.call_tool("read_redaction_summary", {"summary_path": str(summary)})
    assert data == {"elements_found": 3}


def test_mcp_stdio_keeps_stdout_for_protocol(tmp_path):
    summary = tmp_path / "run.summary.json"
    summary.write_text(json.dumps({"elements_found": 1}))
    requests = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "read_redaction_summary", "arguments": {"summary_path": str(summary)}},
                }
            ),
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-m", "open_redactor.mcp_server"],
        input=requests,
        text=True,
        capture_output=True,
        check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2]
    assert responses[1]["result"]["structuredContent"] == {"elements_found": 1}


def test_audit_gate_threshold(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    summary = {
        "elements_found": 1,
        "severity_counts": {"critical": 0, "high": 1, "medium": 0, "low": 0},
    }
    result = RedactionResult(input_path=str(source), summary=summary)
    monkeypatch.setattr(audit_gate, "audit_media", lambda *args, **kwargs: result)
    assert audit_gate.audit_files([source], fail_on="critical") == 0
    assert audit_gate.audit_files([source], fail_on="high") == 1


def test_precommit_manifest_uses_filename_matching_only():
    manifest = (Path(__file__).resolve().parent.parent / ".pre-commit-hooks.yaml").read_text()
    assert "types_or" not in manifest
    assert "open-redactor-audit" in manifest
    assert "mp4|mov|mkv|webm|avi|m4v" in manifest
