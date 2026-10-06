"""Direct Python API example."""

from open_redactor import RedactionOptions, audit_media, redact_media

clip = "interview.mp4"

audit = audit_media(
    clip,
    options=RedactionOptions(preset="street", sensitive=True, audio_mode="pitch"),
)
print(audit.summary)

if audit.summary and audit.summary["elements_found"]:
    result = redact_media(
        clip,
        options=RedactionOptions(preset="street", sensitive=True, audio_mode="pitch"),
        contact_sheet=True,
    )
    print(result.output_path)
    print(result.coverage_report_path)
