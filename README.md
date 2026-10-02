# Open Redactor

Drop in a clip, name what to hide, get a redacted MP4 out.

Open Redactor is an open-source CLI that makes video share-safe. It uses SAM 3.1 detection, segmentation, and identity-preserving video tracking to blur or pixelate people, faces, plates, screens, or any short noun phrase you name.

## Thesis

**Thesis:** Consumer video redaction is either manual or enterprise-priced and a prompt-driven open-source CLI makes share-safe video a one-command default.

## Value-add

**Value-add:** Real video tracking quality without per-frame manual work plus a local mode that keeps private footage off any API.

## Quick start

```bash
pip install open-redactor
open-redactor input.mp4
```

That runs the default set of person, face, license plate, and screen and writes `input.redacted.mp4`.

Name what to hide:

```bash
open-redactor input.mp4 --target "license plate" --target "face"
open-redactor input.mp4 --mode pixelate --strength 24
open-redactor input.mp4 --mask-margin 12 --carry-frames 5 --contact-sheet
```

## Defaults

Defaults matter more than options. SAM finds pixels and does not decide what should be hidden so the default set is a product decision.

- person covers full bodies in frame
- face covers close-ups where the body crop misses the head
- license plate covers parked and moving vehicles
- screen covers phones, laptops, and monitors that may show private content

Running with no phrase runs the default set. Every run prints the resolved target list before processing starts.

## CLI

```text
open-redactor input.mp4
open-redactor input.mp4 --output share-safe.mp4
open-redactor input.mp4 --target "license plate" --target "face"
open-redactor input.mp4 --targets-default --add-target "whiteboard"
open-redactor input.mp4 --mode pixelate --strength 24
open-redactor input.mp4 --mask-margin 12 --carry-frames 5
open-redactor input.mp4 --contact-sheet
open-redactor input.mp4 --local
open-redactor input.mp4 --api-key-env MODEL_API_KEY
open-redactor ./clips --batch --contact-sheet
```

Key flags:

- --output sets the output path and defaults to the input name plus a redacted suffix
- --target adds one phrase and can be repeated
- --targets-default starts from the default set
- --add-target adds one phrase on top of the defaults
- --mode picks blur or pixelate and defaults to blur
- --strength controls blur radius or pixel block size
- --mask-margin pads every mask by this many pixels and defaults to 10
- --carry-frames holds a track for this many frames after it vanishes and defaults to 4
- --contact-sheet writes a PNG grid of redacted sample frames
- --local runs open weights on device and sends nothing to the API
- --api-key-env names the environment variable that holds the API key
- --batch treats the input as a directory and processes each MP4 inside

Exit code is 0 on success and non-zero on failure. A run that finds zero matches still exits 0 and writes a copy with a log line that says nothing matched.

## SAM 3.1 API mode

API mode posts to https://api.meta.ai/v1/responses with model sam-3.1. Set your key in an environment variable and name it with --api-key-env. The CLI streams video so frames arrive as server-sent events until response.completed and uses metadata mask_encoding one_bit. Results return in output_text as special-token lines with one line per frame. The parser decodes them into boxes and binary masks with stable object identity across frames.

Local mode uses the open SAM weights directly and skips the API entirely. The CLI hides the difference behind one flag.

## Failure mode

Missed frames leak identity. A detector that drops a face for two frames leaves those two frames fully exposed and a viewer can pause on them. Single-frame accuracy is not the bar. Continuous coverage is the bar.

Mitigations built into the pipeline:

- Pad masks by a margin so a slightly tight mask still covers hair, ears, and plate edges
- Carry tracks forward a few frames after the last detection so brief dropouts stay covered
- Smooth mask edges over time so masks do not flicker or shrink for one frame
- Interpolate across single-frame gaps inside an otherwise continuous track
- Fail loud on coverage gaps by logging any track gap longer than the carry window
- Contact sheet review so a human can spot a missed region before sharing

If SAM returns no mask for a frame inside a known track window, the renderer keeps the last known mask rather than rendering the frame clean.

## Install from source

```bash
git clone https://github.com/example/open-redactor
cd open-redactor
pip install -e .
```

You need ffmpeg and ffprobe on your PATH.

## Project layout

- src/open_redactor/cli.py holds the CLI entry point
- src/open_redactor/pipeline.py holds ingest, segment and track, pad and smooth, render, and contact sheet
- src/open_redactor/sam_client.py holds the SAM 3.1 API client and local stub
- src/open_redactor/masks.py holds mask padding, carry-forward, smoothing, and gap fill helpers
- tests holds unit tests for masks and CLI wiring

## License

Apache-2.0. See LICENSE.

## Roadmap notes

v1 is CLI first, MP4 in and MP4 out, audio preserved, blur and pixelate modes, per-class phrases, optional contact sheet, and optional local mode. Audio redaction, real-time mode, GUI, and formats beyond MP4 are out of scope for v1.
