# Integrations

Open Redactor is built to sit inside other tools rather than become another destination. Pick the thinnest surface that fits the host.

| Surface | Best for | Entry point |
|---|---|---|
| Python API | Apps, notebooks, workers, and workflow engines | `redact_media` and `audit_media` |
| MCP server | Agent frameworks and desktop AI tools | `open-redactor-mcp` |
| HTTP server | Team services and no-code automation | `open-redactor-server` |
| GitHub Action | Repository checks and release media | `jurayh/open-redactor@v0.2.3` |
| Pre-commit hook | Stopping a risky video before it enters Git | `open-redactor-audit` |
| Docker | CI, servers, and repeatable installs | The repo `Dockerfile` |
| CLI | Shell scripts, OBS folders, ffmpeg workflows, Airflow, and cron | `open-redactor` |

## Python API

The Python API is the main integration surface. It returns structured paths and analytics, so callers do not parse terminal output.

```python
from open_redactor import RedactionOptions, audit_media, redact_media

audit = audit_media(
    "interview.mp4",
    options=RedactionOptions(preset="street", sensitive=True, audio_mode="pitch"),
)
print(audit.summary)

if audit.summary and audit.summary["elements_found"]:
    result = redact_media(
        "interview.mp4",
        options=RedactionOptions(preset="street", sensitive=True, audio_mode="pitch"),
        contact_sheet=True,
    )
    print(result.output_path)
    print(result.coverage_report_path)
```

`RedactionResult` carries `output_path`, `summary_path`, `coverage_report_path`, `contact_sheet_path`, and the parsed `summary` dict. Its `to_dict()` method is JSON ready for queues and workflow engines.

The calls are synchronous and CPU heavy for long clips. Run them in a worker, task queue, or workflow step rather than inside a web request handler. The file at `examples/integrations/python_api.py` is a runnable starting point.

### Celery and task queues

`examples/integrations/celery_task.py` wraps the API as a Celery task. The same shape works for RQ, Dramatiq, Arq, and cloud job runners. Pass paths and option names through the queue. Never pass an API key as a task argument.

### Airflow and Prefect

Use a Python operator that calls `redact_media` or `audit_media`, or a shell operator that calls the CLI. Use `open-redactor-audit` when the DAG should fail on findings. Its exit codes are 0 for no finding at the chosen severity, 1 for a finding, and 2 when the audit itself failed.

## MCP server for agent suites

The MCP server exposes three tools over stdio:

- `redact_media` creates a share-safe copy
- `audit_media` reports sensitive elements without rendering
- `read_redaction_summary` reads a prior run summary

Install and point any MCP client at the command:

```bash
pip install open-redactor
open-redactor-mcp
```

A typical client config looks like this:

```json
{
  "mcpServers": {
    "open-redactor": {
      "command": "open-redactor-mcp",
      "env": {
        "MODEL_API_KEY": "set this in the client secret store"
      }
    }
  }
}
```

Keep the key in the client secret store or the process environment. It is never a tool argument. Run the server only for a trusted local client because its tools work on file paths that the client supplies.

For frameworks without MCP support, `examples/integrations/agent_tool.py` converts the same tools to a common function-calling shape and dispatches calls locally.

## HTTP server

The optional FastAPI wrapper is useful for a team server, an internal tool, or no-code platforms such as n8n, Make, and Zapier that can call HTTP endpoints.

```bash
pip install "open-redactor[server]"
open-redactor-server --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health` returns service status and version
- `POST /audit` accepts a multipart video upload and returns summary JSON plus audit text
- `POST /redact` accepts a multipart media upload and returns a job id, a download path, and summary JSON
- `GET /files/{job_id}/{filename}` downloads a generated file

```bash
curl -F "file=@interview.mp4" -F "preset=street" -F "sensitive=true" \
  http://127.0.0.1:8000/audit
```

The server binds to 127.0.0.1 by default. Put authentication, TLS, upload limits, and retention rules in front of it before exposing it on a network. Uploaded files and outputs stay in the server work directory until the host removes them.

## GitHub Action

Use the action to audit media in a pull request or redact media during a release job.

```yaml
name: Media privacy check
on: [pull_request]
jobs:
  audit-media:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jurayh/open-redactor@v0.2.3
        env:
          MODEL_API_KEY: ${{ secrets.MODEL_API_KEY }}
        with:
          path: media
          task: audit
          preset: documents
          sensitive: "true"
          fail-on: medium
          output-dir: redaction-audit
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: redaction-audit
          path: redaction-audit
```

Set `task: redact` to write redacted MP4 files to `output-dir`. For local backend runs in CI, the runner needs the local model stack and enough compute, so the API backend is the practical default for most repositories.

## Pre-commit hook

Add this to `.pre-commit-config.yaml` to audit videos before they are committed:

```yaml
repos:
  - repo: https://github.com/jurayh/open-redactor
    rev: v0.2.3
    hooks:
      - id: open-redactor-audit
        args: [--fail-on=medium, --sensitive, --backend=api]
```

The hook returns a failure when it finds an element at the chosen severity. Use `--backend=local` only in an environment prepared with the local extra and model hardware. The hook covers video files. Photos should be checked through the Python API or CLI in a separate step.

## Docker

Build the image from the repo:

```bash
docker build -t open-redactor .
docker run --rm -v "$PWD:/work" -e MODEL_API_KEY open-redactor clip.mp4 --preset family
```

The image includes ffmpeg, Node with the SAM mask parser, Tesseract, the PII extra, and the server extra. Run the HTTP wrapper by changing the entry point:

```bash
docker run --rm -p 127.0.0.1:8000:8000 --entrypoint open-redactor-server \
  -e MODEL_API_KEY open-redactor --host 0.0.0.0 --port 8000
```

## CLI and shell pipelines

The CLI remains the easiest fit for shell scripts, watch folders, OBS output folders, and cron jobs. Inputs and outputs are files, not stdin and stdout streams. That is a deliberate safety choice for media work because every run also produces reports and summaries next to the output.

```bash
open-redactor recording.mp4 --shadow --preset screen-share --sensitive
open-redactor recording.mp4 --preset screen-share --sensitive --audio pitch
cat recording.redacted.summary.json
```

## Integration contract

Every real run produces the same evidence regardless of surface:

- A redacted MP4 or photo, unless the call is an audit or preview
- A `.summary.json` file with elements, severity counts, frames affected, and exposure score
- A `.coverage.txt` report for video runs
- An audit text file for shadow audits

Build downstream logic on the summary JSON and exit codes, not on human log lines.
