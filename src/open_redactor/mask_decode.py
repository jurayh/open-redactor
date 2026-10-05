"""Decode SAM one_bit mask rasters via the official JS parser when available.

The SAM API returns mask tokens with a base85 payload. The canonical
decoder lives in @meta-sam/parser for Node. This module shells out to a
small Node helper when node and that package are present and falls back
to box masks with a clear log line when they are not.

No credential ever passes through this module. It only sees mask text
that the API already returned.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

HELPER_JS = r"""
import fs from "fs";
import { decodeMaskToRaster } from "@meta-sam/parser";
const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = [];
for (const item of input) {
  try {
    const raster = decodeMaskToRaster({encoding: "one_bit", payload: item.payload, width: item.width, height: item.height});
    out.push({ok: true, data: Array.from(raster)});
  } catch (e) {
    out.push({ok: false, error: String(e && e.message || e)});
  }
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""


def decode_masks_batch(items: list[dict]) -> Optional[list[Optional[np.ndarray]]]:
    """Decode a batch of masks. Each item has payload, width, height.

    Returns a list aligned with items where each entry is a bool array
    shaped height by width or None when decoding is unavailable.
    """
    if not items:
        return []
    # Quick availability check
    try:
        check = subprocess.run(
            ["node", "-e", "import('@meta-sam/parser').then(()=>process.exit(0)).catch(()=>process.exit(1))"],
            capture_output=True,
            timeout=15,
        )
        if check.returncode != 0:
            return None
    except Exception:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        helper = tmp_path / "decode.mjs"
        helper.write_text(HELPER_JS)
        inp = tmp_path / "in.json"
        outp = tmp_path / "out.json"
        inp.write_text(json.dumps(items))
        try:
            subprocess.check_call(
                ["node", str(helper), str(inp), str(outp)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except Exception:
            return None
        try:
            results = json.loads(outp.read_text())
        except Exception:
            return None
    decoded: list[Optional[np.ndarray]] = []
    for item, res in zip(items, results):
        if not res.get("ok"):
            decoded.append(None)
            continue
        arr = np.array(res["data"], dtype=np.uint8)
        expected = int(item["width"]) * int(item["height"])
        if arr.size != expected:
            decoded.append(None)
            continue
        decoded.append(arr.reshape((int(item["height"]), int(item["width"]))).astype(bool))
    return decoded
