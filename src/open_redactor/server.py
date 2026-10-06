"""Small HTTP service wrapper for team and suite integrations.

Install with ``pip install "open-redactor[server]"`` and run
``open-redactor-server``. The service binds to 127.0.0.1 by default. Put it
behind your normal authentication and reverse proxy before exposing it to a
network. It is a convenience wrapper around the public Python API, not a
hosted multi-tenant product.
"""

from __future__ import annotations

import argparse
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .api import IMAGE_SUFFIXES, VIDEO_SUFFIXES, RedactionOptions, audit_media, redact_media

try:  # FastAPI resolves endpoint annotations from module globals.
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:  # The server extra is optional.
    FastAPI = File = Form = HTTPException = UploadFile = None  # type: ignore[assignment]
    FileResponse = JSONResponse = None  # type: ignore[assignment]

DEFAULT_WORK_DIR = Path.home() / ".cache" / "open-redactor" / "server"


def _split_targets(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    targets = [part.strip() for part in value.replace(";", ",").split(",")]
    return [target for target in targets if target] or None


def _options_from_form(values: dict[str, Any]) -> RedactionOptions:
    return RedactionOptions(
        targets=_split_targets(values.get("targets")),
        preset=values.get("preset") or None,
        mode=values.get("mode") or "blur",
        backend=values.get("backend") or None,
        endpoint=values.get("endpoint") or None,
        provider=values.get("provider") or None,
        local=bool(values.get("local", False)),
        sensitive=bool(values.get("sensitive", False)),
        pii_text=bool(values.get("pii_text", False)),
        codes=bool(values.get("codes", False)),
        audio_mode=values.get("audio_mode") or "keep",
    )


def create_app(work_dir: Optional[str | Path] = None):
    """Create the FastAPI application, importing FastAPI only when used."""
    if FastAPI is None:
        raise RuntimeError('Install the server extra first: pip install "open-redactor[server]"')

    root = Path(work_dir).expanduser() if work_dir else DEFAULT_WORK_DIR
    root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Open Redactor", version=__version__)

    def save_upload(upload: UploadFile, job_dir: Path) -> Path:
        filename = Path(upload.filename or "upload").name
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported input format '{suffix}'")
        path = job_dir / filename
        with path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        return path

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "open-redactor", "version": __version__}

    @app.post("/audit")
    async def audit(
        file: UploadFile = File(...),
        targets: Optional[str] = Form(default=None),
        preset: Optional[str] = Form(default=None),
        backend: Optional[str] = Form(default=None),
        endpoint: Optional[str] = Form(default=None),
        provider: Optional[str] = Form(default=None),
        local: bool = Form(default=False),
        sensitive: bool = Form(default=False),
        pii_text: bool = Form(default=False),
        codes: bool = Form(default=False),
    ):
        job_id = uuid.uuid4().hex
        job_dir = root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        source = save_upload(file, job_dir)
        options = _options_from_form(
            {
                "targets": targets,
                "preset": preset,
                "backend": backend,
                "endpoint": endpoint,
                "provider": provider,
                "local": local,
                "sensitive": sensitive,
                "pii_text": pii_text,
                "codes": codes,
            }
        )
        try:
            result = audit_media(source, job_dir / "audit-base.mp4", options)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        audit_text = None
        if result.audit_path and Path(result.audit_path).exists():
            audit_text = Path(result.audit_path).read_text(encoding="utf-8")
        return JSONResponse(
            {
                "job_id": job_id,
                "summary": result.summary,
                "audit_text": audit_text,
            }
        )

    @app.post("/redact", status_code=201)
    async def redact(
        file: UploadFile = File(...),
        targets: Optional[str] = Form(default=None),
        preset: Optional[str] = Form(default=None),
        mode: str = Form(default="blur"),
        backend: Optional[str] = Form(default=None),
        endpoint: Optional[str] = Form(default=None),
        provider: Optional[str] = Form(default=None),
        local: bool = Form(default=False),
        sensitive: bool = Form(default=False),
        pii_text: bool = Form(default=False),
        codes: bool = Form(default=False),
        audio_mode: str = Form(default="keep"),
        contact_sheet: bool = Form(default=False),
    ):
        job_id = uuid.uuid4().hex
        job_dir = root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        source = save_upload(file, job_dir)
        options = _options_from_form(
            {
                "targets": targets,
                "preset": preset,
                "mode": mode,
                "backend": backend,
                "endpoint": endpoint,
                "provider": provider,
                "local": local,
                "sensitive": sensitive,
                "pii_text": pii_text,
                "codes": codes,
                "audio_mode": audio_mode,
            }
        )
        try:
            result = redact_media(source, options=options, contact_sheet=contact_sheet)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Move the generated output and sidecars into the job directory so the
        # download route can serve them after the request finishes.
        served_name = None
        if result.output_path:
            output = Path(result.output_path)
            served = job_dir / output.name
            if output.resolve() != served.resolve():
                shutil.move(str(output), str(served))
            served_name = served.name
            for sidecar_key in ("summary_path", "coverage_report_path", "contact_sheet_path"):
                sidecar = getattr(result, sidecar_key)
                if sidecar and Path(sidecar).exists():
                    target = job_dir / Path(sidecar).name
                    if Path(sidecar).resolve() != target.resolve():
                        shutil.move(str(sidecar), str(target))
        return JSONResponse(
            status_code=201,
            content={
                "job_id": job_id,
                "download_path": f"/files/{job_id}/{served_name}" if served_name else None,
                "summary": result.summary,
            },
        )

    @app.get("/files/{job_id}/{filename}")
    def download(job_id: str, filename: str):
        if not job_id.isalnum() or Path(filename).name != filename:
            raise HTTPException(status_code=400, detail="Invalid file request")
        job_dir = (root / job_id).resolve()
        path = (job_dir / filename).resolve()
        if path.parent != job_dir or not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=path.name)

    app.state.work_dir = root
    return app


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="open-redactor-server", description="Run the Open Redactor HTTP integration")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Install the server extra first: pip install "open-redactor[server]"') from exc
    uvicorn.run(create_app(args.work_dir), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
