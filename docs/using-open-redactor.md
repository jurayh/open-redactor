# Using Open Redactor

A short guide to running it well, what stays private, and what it cannot do yet.

## The 30 second loop

Preview first, read the coverage result line, then run the full clip. See the README demo for the four commands. Preview costs 3 seconds of rendering and saves a bad full run.

## The local page

Run `open-redactor --ui` and open http://127.0.0.1:8765. Drop an MP4, pick a preset, submit. The server binds only to your own machine, writes output to a temp folder it prints, and runs the same CLI underneath. Nothing is hosted and no file leaves the machine from the page itself.

## Where SAM runs. Three backends

SAM 3.1 is an open weights model, so Open Redactor does not hard assume one home for it.

**API backend, the default.** `--backend api` posts your clip to the Meta Model API at api.meta.ai with your key from an environment variable. Best quality today with no GPU and no downloads. Your video does leave the machine, so do not use it for footage that must stay local.

**Hosted backend.** `--backend hosted --endpoint https://your-host.example/v1/responses` sends the same Responses API request shape to an endpoint you control, or set OPEN_REDACTOR_ENDPOINT once. Use this for a team server, a cloud GPU box, or a third party host that speaks the same shape. The API key still comes from the named environment variable and is sent only to that endpoint.

**Local backend.** `--backend local` or `--local` is meant for the open weights on your own hardware, so private footage never leaves the device. In the current build the local loader is not wired yet and returns zero matches with a clear log line, so use api or hosted for live detection today. The pipeline, masks, cache, and reports are backend agnostic, so wiring the weights in does not change the CLI. Local is the right home for journalists, medical, and family footage that should never be uploaded, and it is the top roadmap item.

## Privacy notes

API and hosted modes send the clip to the endpoint you chose. Local mode and the local page send nothing. SAM results are cached on your machine under your home cache folder so re-renders skip repeat uploads for the same clip and phrase. Use --no-cache to force a fresh pass. Output files are written next to your input with a redacted suffix and never overwrite the original.

## Honest limits

Audio is not redacted in v1. Voices, spoken names, and background speech can still identify someone, so listen before sharing.

Preview renders only the first 3 seconds. A clean preview does not prove the whole clip, the coverage report on the full run is the audit.

Pixel-perfect masks need Node and @meta-sam/parser on the machine. Without them the run falls back to box masks with padding and carry, which is safe but less tight.

Detection can miss. Padding, carry-forward, and gap flags reduce the leak risk, and the contact sheet is your final human check. Treat any NEEDS REVIEW line in a coverage report as a stop before sharing.

Formats are MP4 in and MP4 out in v1.
