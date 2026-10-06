"""Celery task example.

Install Celery in the worker application, import this task module, and call
``redact_video.delay(path)``. The result is JSON serializable.
"""

from __future__ import annotations

from celery import Celery

from open_redactor import RedactionOptions, redact_media

app = Celery("open_redactor_worker")


@app.task(name="open_redactor.redact_video")
def redact_video(input_path: str, preset: str = "family") -> dict:
    result = redact_media(
        input_path,
        options=RedactionOptions(preset=preset, sensitive=True),
        contact_sheet=True,
    )
    return result.to_dict()
