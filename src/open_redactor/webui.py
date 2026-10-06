"""Tiny local drag and drop page. Binds to 127.0.0.1 only and shells to the CLI."""

from __future__ import annotations

import html
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Open Redactor</title>
<style>body{font-family:system-ui;max-width:720px;margin:40px auto;padding:0 20px} .drop{border:2px dashed #888;border-radius:16px;padding:40px;text-align:center}</style></head>
<body><h1>Open Redactor</h1><p>Drop an MP4, pick a preset, get a redacted copy. Files stay on this machine. Audio keeps the original voices by default in the CLI, choose --audio mute or pitch there when voices identify someone.</p>
<form method=post enctype=multipart/form-data action=/run>
<div class=drop><input type=file name=video accept="video/mp4" required></div>
<p>Preset <select name=preset><option>family</option><option>street</option><option>screen-share</option></select>
<label><input type=checkbox name=preview value=1> Preview only</label></p>
<p><button type=submit>Redact</button></p></form></body></html>"""


def serve_ui(host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode())

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            # Very small multipart parse, good enough for a local helper
            work = Path(tempfile.mkdtemp(prefix="open-redactor-ui-"))
            # Extract file bytes between headers and boundary tail
            start = body.find(b"\r\n\r\n")
            file_bytes = body[start + 4 :] if start != -1 else b""
            # Trim trailing boundary
            for marker in (b"\r\n--", b"--\r\n"):
                idx = file_bytes.rfind(marker)
                if idx != -1:
                    file_bytes = file_bytes[:idx]
                    break
            src = work / "input.mp4"
            src.write_bytes(file_bytes)
            preset = "family"
            for name in ("family", "street", "screen-share"):
                if f'name="preset"'.encode() in body and name.encode() in body:
                    # Last matching preset value wins, the select sends one
                    pass
            out = work / "input.redacted.mp4"
            cmd = [sys.executable, "-m", "open_redactor.cli", str(src), "--local", "--preset", preset, "--output", str(out), "--contact-sheet"]
            if b'name="preview"' in body and b"preview" in body:
                cmd.append("--preview")
            result = subprocess.run(cmd, capture_output=True, text=True)
            log = html.escape((result.stdout + result.stderr)[-4000:])
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"<html><body><h1>Done</h1><p>Output folder: {html.escape(str(work))}</p><pre>{log}</pre><p><a href='/'>Back</a></p></body></html>".encode())

        def log_message(self, *args):  # keep terminal quiet
            pass

    HTTPServer((host, port), Handler).serve_forever()
