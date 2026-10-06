# Open Redactor eval harness

A small, honest benchmark that ships in the repo. It does not claim to be EgoBlur or DAVIS. It gives every release a repeatable report card on the three layers that run anywhere, with exact ground truth because the clips are generated.

## The bundled set

- qr-static: a QR code in a fixed box for 20 frames, scored against the codes layer
- screen-form: a checkout form with a test card number, email, and phone in known text regions for 30 frames, scored against the text PII layer with an OCR engine installed
- badge-moving: a name badge moving across the frame for 30 frames, scored against live SAM predictions saved in badge.predictions.json. SAM found it from frame 7, so this clip also measures entry delay, the one gap carry-forward cannot fix

## Run it

```bash
pip install "open-redactor[pii]"
python eval/run_eval.py
```

Writes eval/results/report-card.md and report-card.json.

## Current reference scores (0.2.1 layers)

- qr-static: recall 1.00 at IoU 0.5, mean IoU 0.75, coverage 1.00, leakage 0.16
- screen-form: recall 1.00, mean IoU 0.69, coverage 1.00, leakage 0.07
- badge-moving: recall 0.77, mean IoU 0.75, coverage 0.77, longest gap 7 frames at entry

Leakage is the share of high frequency detail inside a ground truth box that survives redaction. Lower is better. 1.0 would mean the output is untouched.

## External benchmarks this maps to

EgoBlur on Ego4D for faces and plates in egocentric video, Ref-YouTube-VOS and DAVIS 2017 for phrase driven mask quality, MOT17 for tracking continuity, and WIDER FACE for face detection recall. The harness format, per-frame boxes in JSON, is deliberately close to those so external sets can be adapted into the same scorer. The scoring core lives in eval/scoring.py and is unit tested.
