"""Minimal U2Net background removal via onnxruntime.

Drop-in replacement for rembg.remove() for this pipeline. rembg pulls in
pymatting -> numba, whose DLL Windows App Control blocks; the actual
segmentation is just a 320x320 ONNX forward pass we can run directly.
Uses the u2net.onnx rembg already downloaded to ~/.u2net/.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

_MODEL = Path.home() / ".u2net" / "u2net.onnx"
_session = None


def _sess() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(_MODEL), providers=["CPUExecutionProvider"])
    return _session


def remove(im: Image.Image) -> Image.Image:
    """RGBA input/any -> RGBA with background alpha'd out."""
    rgb = im.convert("RGB")
    small = rgb.resize((320, 320), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)

    out = _sess().run(None, {_sess().get_inputs()[0].name: x})[0][0, 0]
    out = (out - out.min()) / max(out.max() - out.min(), 1e-6)
    mask = Image.fromarray((out * 255).astype(np.uint8)).resize(rgb.size, Image.LANCZOS)

    result = rgb.convert("RGBA")
    result.putalpha(mask)
    return result
