# Public showcase pack

Real world photos from Wikimedia Commons, redacted with the hosted SAM 3.1
backend at the default settings unless a scene says otherwise. Each scene
keeps its original, its redacted output, and a labeled panel. Licenses and
authors are in [ATTRIBUTION.md](ATTRIBUTION.md).

| Scene | What it proves | Detections | Command |
| --- | --- | --- | --- |
| Market crowd | A dense crowd with overlap, plus keeping one group visible | 28 people, 14 faces | `open-redactor market-crowd.jpg --target person --target face` |
| Car plate | One large plate, pixelate mode | 1 license plate | `open-redactor car-plate.jpg --target "license plate" --mode pixelate` |
| Trail group | A posed group at distance, faces and bodies together | 16 people, 18 faces | `open-redactor trail-group.jpg --target person --target face` |
| Trailhead | A location sign, the thing the location preset exists for | 1 street sign | `open-redactor trailhead.jpg --target "street sign"` |

The third panel of the market scene is the keep flow. A coverage report
names every track, and rerunning with `--keep person:19 --keep face:12
--keep face:13` leaves that one foreground group visible while the rest of
the crowd stays blurred.

Two honest notes from building this pack:

- In the market scene, SAM also matched the seated figure pictogram on the
  round cafe sign as a person, so it is blurred in the redacted panels.
  That is the false positive case `--exclude` exists for.
- The plate scene uses pixelate mode on purpose. At default strength, blur
  left a large high contrast plate readable, and a redaction you can still
  read is not a redaction. Pixelate makes it decisively unreadable.
