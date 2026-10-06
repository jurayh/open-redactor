# How Open Redactor compares

An honest comparison against the tools people reach for today. Open Redactor is the general purpose redaction layer. The specialists are better inside their one class, and two rows below say so plainly.

| Capability | Open Redactor | deface | EgoBlur | DeepPrivacy 2 | Brighter AI | Presidio |
|---|---|---|---|---|---|---|
| Prompt driven, redact any named class | Yes | No | No | No | No | No |
| Faces | Yes | Yes | Yes | Yes | Yes | Static images |
| License plates | Yes | No | Yes | No | Yes | No |
| Documents and printed text PII | Yes, OCR plus verified patterns | No | No | No | No | Images and text, no video |
| QR codes and barcodes | Yes | No | No | No | No | No |
| Location clues (house numbers, street signs) | Yes | No | No | No | No | No |
| Audio redaction (mute, pitch shift) | Yes | No | No | No | No | No |
| Replace mode with consistent fakes | Yes, stylized | No | No | Photoreal faces | Photoreal | No |
| Video and photos | Both | Video and images | Video | Both | Video | Images |
| Shadow audit before rendering | Yes | No | No | No | No | No |
| Coverage report with gap flags | Yes | No | No | No | No | No |
| Published leakage scores | Yes, eval harness in repo | No | No | No | No | No |
| Fully local mode | Yes | Yes | Yes | Yes | No | Yes |
| Open source | Apache-2.0 | MIT | Research code | Research code | Closed | MIT |
| Price | Free | Free | Free | Free | Paid | Free |

Where the others genuinely win:

- Specialized accuracy: a tuned face detector like deface or EgoBlur will beat a generalist model on faces alone in crowded scenes. Our own eval harness caught a 7 frame entry gap on a moving badge, and we publish numbers like that rather than hiding them.
- Photoreal replacement: DeepPrivacy and Brighter AI generate natural looking synthetic faces. Our replace mode uses a deliberately stylized avatar and honest fakes for data, which reads clearly as a stand-in instead of pretending to be footage.
- Enterprise scale: Brighter AI and the cloud platforms offer real time streams, support contracts, and certifications that a community CLI does not.

Where Open Redactor stands alone: name anything and redact it across video, photos, and voice, then get an audit, a coverage report, and a leakage score as evidence. No other tool in this table does the audit or publishes its leakage.
