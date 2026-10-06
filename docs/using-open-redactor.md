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

**Local backend.** `--backend local` runs Grounding SAM on your own hardware, so private footage never leaves the device. Grounding DINO finds the phrase, SAM 2 cuts the mask, and a small IoU tracker keeps identities stable across frames. Install the heavy stack once with pip install "open-redactor[local]" and run with --provider grounding-sam, which is the default for the local backend. It uses a GPU when one is present and falls back to CPU slowly. Without the optional stack it returns zero matches with a clear install message and never pretends it detected something. A native SAM video propagator and larger model options are the next local upgrades.

## Privacy notes

API and hosted modes send the clip to the endpoint you chose. Local mode and the local page send nothing. SAM results are cached on your machine under your home cache folder so re-renders skip repeat uploads for the same clip and phrase. Use --no-cache to force a fresh pass. Output files are written next to your input with a redacted suffix and never overwrite the original.

## Honest limits

Audio is not redacted in v1. Voices, spoken names, and background speech can still identify someone, so listen before sharing.

Preview renders only the first 3 seconds. A clean preview does not prove the whole clip, the coverage report on the full run is the audit.

Pixel-perfect masks need Node and @meta-sam/parser on the machine. Without them the run falls back to box masks with padding and carry, which is safe but less tight.

Detection can miss. Padding, carry-forward, and gap flags reduce the leak risk, and the contact sheet is your final human check. Treat any NEEDS REVIEW line in a coverage report as a stop before sharing.

Input formats are MP4, MOV, MKV, WebM, AVI, and M4V. Phones shoot MOV, screen tools emit WebM and MKV, and older cameras write AVI, so all of them are accepted and batch mode picks them all up. Output is always MP4 for share compatibility. Non-MP4 inputs sent to the API or a hosted endpoint are transcoded to a temp MP4 for the upload only, your original file is never changed.

## Sensitive documents

Use --preset documents --pii-text for passports, bank cards, driver licenses, and paperwork. The preset blurs the physical objects. The text layer blurs printed numbers and addresses found by OCR, with Luhn verification for card numbers so order numbers and timestamps stay untouched. A passport held to camera is covered as an object even when its number is too small to read, and a card number typed on a screen share is covered as text even with no physical card in frame. You need both layers for document work.

## Codes, location, and badges

--codes covers QR codes and barcodes, --preset location covers street signs, house numbers, plates, and mailboxes, and name badges and lanyards are part of the documents preset. --sensitive turns on the text and codes layers together and pairs well with the documents and location presets.
