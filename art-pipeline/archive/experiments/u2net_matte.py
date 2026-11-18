"""High-resolution background removal via BiRefNet-Swin-Tiny (1024x1024) ONNX.

Drop-in replacement for u2net_matte.remove() that runs at 1024x1024 instead of
320x320, preserving fine bills, whiskers, claws, and removing wooden perches/logs
that 320px u2net merges into the subject. Falls back to u2net.onnx if BiRefNet is absent.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

_BIREFNET = Path.home() / ".u2net" / "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
_U2NET = Path.home() / ".u2net" / "u2net.onnx"
_session = None
_size = (1024, 1024)


def _sess() -> tuple[ort.InferenceSession, tuple[int, int]]:
    global _session, _size
    if _session is None:
        if _BIREFNET.exists():
            _session = ort.InferenceSession(str(_BIREFNET), providers=["CPUExecutionProvider"])
            _size = (1024, 1024)
        else:
            _session = ort.InferenceSession(str(_U2NET), providers=["CPUExecutionProvider"])
            _size = (320, 320)
    return _session, _size


def remove(im: Image.Image) -> Image.Image:
    """RGBA input/any -> RGBA with background alpha'd out via 1024px BiRefNet."""
    sess, size = _sess()
    rgb = im.convert("RGB")
    small = rgb.resize(size, Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)

    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    while out.ndim > 2:
        out = out[0]
    if out.min() < -0.1 or out.max() > 1.1:
        out = 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))
    out = (out - out.min()) / max(out.max() - out.min(), 1e-6)
    mask = Image.fromarray((out * 255).astype(np.uint8)).resize(rgb.size, Image.LANCZOS)

    result = rgb.convert("RGBA")
    result.putalpha(mask)
    return result
