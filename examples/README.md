# Examples

These commands show the v1 flows from the spec. Media from a live SAM 3.1 run lives in media.

## Media showcase

- media/hero.gif is an animated side by side with the original on the left and redacted output on the right
- media/hero-3panel.jpg shows one frame as original, SAM box view, and redacted output
- media/contact-sheet.jpg samples the face track across the clip with red outlines
- media/dropout-proof.png diagrams how carry-forward covers a two-frame detection drop
- media/terminal.png shows a live run summary with targets, frame count, and output paths

Source for the showcase was a short public portrait clip. Phrase was face. SAM returned one track across all 50 frames.

## Default run

```bash
open-redactor input.mp4
```

Runs person, face, license plate, and screen. Writes input.redacted.mp4.

## Named targets

```bash
open-redactor input.mp4 --target "license plate" --target "face"
```

Replaces the defaults with only the phrases you name.

## Defaults plus one more

```bash
open-redactor input.mp4 --targets-default --add-target "whiteboard"
```

Keeps the defaults and adds whiteboard.

## Pixelate mode

```bash
open-redactor input.mp4 --mode pixelate --strength 24
```

Strength is the pixel block size in pixelate mode and the blur kernel size in blur mode.

## More margin and longer carry

```bash
open-redactor input.mp4 --mask-margin 12 --carry-frames 5
```

A larger margin covers loose masks. A longer carry holds a track through short dropouts.

## Contact sheet for review

```bash
open-redactor input.mp4 --contact-sheet
```

Writes a PNG grid next to the output so you can spot a missed region before sharing.

## Local mode

```bash
open-redactor input.mp4 --local
```

Sends nothing to the API. Uses open weights when they are installed.

## Batch folder

```bash
open-redactor ./clips --batch --contact-sheet
```

Processes each MP4 in the folder.
