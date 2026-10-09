"""Environment check for Open Redactor.

Most first-run trouble is setup trouble: a missing ffmpeg, the pixel
mask decoder not installed, an OCR engine absent, an extra not pulled.
`open-redactor doctor` reports each capability as OK or MISSING with the
exact install line that fixes it, so a user can paste one report into a
bug report instead of describing their machine.

No secret values are ever printed. The API key check reports only
whether the named environment variable is set.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

Check = Tuple[str, bool, str, str]

CORE = "Required"
QUALITY = "Detection quality"
OPTIONAL = "Optional layers"


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _node_parser_check() -> Check:
    node = shutil.which("node")
    if not node:
        return (
            "Pixel mask decoder (@meta-sam/parser)",
            False,
            "node not found, SAM masks fall back to boxes",
            "Install Node 20+ and the parser package, or use the Docker image which bundles both",
        )
    try:
        probe = subprocess.run(
            ["node", "-e", "import('@meta-sam/parser').then(()=>process.exit(0)).catch(()=>process.exit(1))"],
            capture_output=True,
            timeout=15,
        )
        ok = probe.returncode == 0
    except Exception:
        ok = False
    if ok:
        return ("Pixel mask decoder (@meta-sam/parser)", True, "node and parser resolvable from this directory", "")
    return (
        "Pixel mask decoder (@meta-sam/parser)",
        False,
        "node found but the parser package does not resolve from this directory, SAM masks fall back to boxes",
        "npm install @meta-sam/parser in this directory, or use the Docker image which bundles it",
    )


def run_checks(api_key_env: str = "MODEL_API_KEY") -> List[Tuple[str, Check]]:
    checks: List[Tuple[str, Check]] = []

    version = sys.version_info
    checks.append((CORE, (
        "Python >= 3.10",
        version >= (3, 10),
        f"running {version.major}.{version.minor}.{version.micro}",
        "Install Python 3.10 or newer",
    )))
    checks.append((CORE, (
        "OpenCV and numpy",
        _importable("cv2") and _importable("numpy"),
        "media decode and rendering available",
        'pip install "open-redactor" again so dependencies resolve',
    )))
    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        checks.append((CORE, (
            binary,
            path is not None,
            f"found at {path}" if path else "not on PATH, video runs will fail",
            "Install ffmpeg with your package manager, for example brew install ffmpeg or apt install ffmpeg",
        )))

    checks.append((QUALITY, _node_parser_check()))
    try:
        import cv2

        barcode = hasattr(cv2, "barcode_BarcodeDetector") or hasattr(cv2, "barcode")
        checks.append((QUALITY, (
            "Barcode detector (OpenCV contrib)",
            barcode,
            "QR codes work either way, barcodes need the contrib build" if not barcode else "QR and barcode detection available",
            "pip install opencv-contrib-python-headless",
        )))
    except Exception:
        pass

    tess = shutil.which("tesseract")
    checks.append((OPTIONAL, (
        "Tesseract OCR binary (text PII layer)",
        tess is not None,
        f"found at {tess}" if tess else "not found, the text layer will use RapidOCR when installed",
        "Install tesseract with your package manager, or pip install \"open-redactor[pii]\" for RapidOCR",
    )))
    checks.append((OPTIONAL, (
        "RapidOCR (text PII layer)",
        _importable("rapidocr_onnxruntime"),
        "OCR engine available" if _importable("rapidocr_onnxruntime") else "not installed",
        'pip install "open-redactor[pii]"',
    )))
    checks.append((OPTIONAL, (
        "HEIC and HEIF photos",
        _importable("pillow_heif"),
        "pillow-heif installed" if _importable("pillow_heif") else "not installed, iPhone HEIC photos will ask for the extra",
        'pip install "open-redactor[heic]"',
    )))
    local_ok = _importable("torch") and _importable("transformers")
    checks.append((OPTIONAL, (
        "Local backend (torch and transformers)",
        local_ok,
        "local Grounding SAM stack importable" if local_ok else "not installed, the local backend will ask for the extra",
        'pip install "open-redactor[local]"',
    )))
    checks.append((OPTIONAL, (
        "Speech layer (faster-whisper)",
        _importable("faster_whisper"),
        "local transcription available for spoken PII muting" if _importable("faster_whisper") else "not installed, --speech-pii will ask for the extra or a transcript file",
        'pip install "open-redactor[speech]"',
    )))
    server_ok = _importable("fastapi") and _importable("uvicorn")
    checks.append((OPTIONAL, (
        "HTTP server",
        server_ok,
        "FastAPI stack importable" if server_ok else "not installed",
        'pip install "open-redactor[server]"',
    )))
    key_set = bool(os.environ.get(api_key_env))
    checks.append((OPTIONAL, (
        f"API key env var {api_key_env}",
        key_set,
        "set, the API and hosted backends can authenticate" if key_set else "not set, only the local backend can run until it is",
        f"Set {api_key_env} in your environment, never in a file you share",
    )))
    return checks


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="open-redactor-doctor", description="Check the Open Redactor environment and print fixes")
    parser.add_argument("--api-key-env", default="MODEL_API_KEY", help="Name of the environment variable that holds the API key")
    args = parser.parse_args(argv)

    checks = run_checks(api_key_env=args.api_key_env)
    print("Open Redactor doctor")
    current_group = None
    core_ok = True
    for group, (name, ok, detail, fix) in checks:
        if group != current_group:
            current_group = group
            print(f"\n{group}")
        status = "OK     " if ok else "MISSING"
        print(f"  [{status}] {name}: {detail}")
        if not ok and fix:
            print(f"           Fix: {fix}")
        if group == CORE and not ok:
            core_ok = False
    print()
    if core_ok:
        print("Core is ready. Missing optional items only limit the layers named above.")
        return 0
    print("Core pieces are missing, so video runs will fail until the Required fixes are applied.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
