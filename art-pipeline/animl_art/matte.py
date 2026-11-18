"""Matting and geometric canvas normalization utilities (BiRefNet 1024px ONNX + LANCZOS bbox scaling)."""

from pathlib import Path

import numpy as np
from PIL import Image

ALPHA_THRESHOLD = 10
_BIREFNET = Path.home() / ".u2net" / "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
_U2NET = Path.home() / ".u2net" / "u2net.onnx"
_session = None
_size = (1024, 1024)


def _get_session():
    """Lazily load BiRefNet (1024x1024) or fallback U2Net (320x320) ONNX runtime session."""
    global _session, _size
    if _session is None:
        import onnxruntime as ort

        if _BIREFNET.exists():
            _session = ort.InferenceSession(str(_BIREFNET), providers=["CPUExecutionProvider"])
            _size = (1024, 1024)
        elif _U2NET.exists():
            _session = ort.InferenceSession(str(_U2NET), providers=["CPUExecutionProvider"])
            _size = (320, 320)
        else:
            raise FileNotFoundError(
                f"No segmentation ONNX model found under {Path.home() / '.u2net'} "
                "(expected BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx or u2net.onnx)"
            )
    return _session, _size


def remove(im: Image.Image) -> Image.Image:
    """Return RGBA image with background masked via 1024px BiRefNet ONNX."""
    sess, size = _get_session()
    rgb = im.convert("RGB")
    small = rgb.resize(size, Image.Resampling.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)

    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    while out.ndim > 2:
        out = out[0]
    if out.min() < -0.1 or out.max() > 1.1:
        out = 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))
    out = (out - out.min()) / max(out.max() - out.min(), 1e-6)
    mask = Image.fromarray((out * 255).astype(np.uint8)).resize(rgb.size, Image.Resampling.LANCZOS)

    result = rgb.convert("RGBA")
    result.putalpha(mask)
    return result


def subject_bbox(im: Image.Image) -> tuple[int, int, int, int] | None:
    """Tight bounding box `(left, top, right, bottom)` around alpha > ALPHA_THRESHOLD."""
    a = np.asarray(im)[..., 3]
    ys, xs = np.where(a > ALPHA_THRESHOLD)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalize_canvas(png: Path, out_dir: Path, canvas: int = 2048, fill: float = 0.76) -> str:
    """Crop RGBA PNG to subject bbox and center on square transparent canvas at `fill` ratio."""
    im = Image.open(png).convert("RGBA")
    box = subject_bbox(im)
    if box is None:
        return f"SKIP {png.name}: fully transparent"
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    pre_fill = max(w, h) / max(im.size)

    subject = im.crop(box)
    target_long = int(round(canvas * fill))
    scale = target_long / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resample = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC
    subject = subject.resize((nw, nh), resample)

    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    out.paste(subject, ((canvas - nw) // 2, (canvas - nh) // 2))
    out.save(out_dir / png.name)
    up_note = f", upscale x{scale:.2f}" if scale > 1 else ""
    return f"OK   {png.name}: fill {pre_fill:.0%} -> {fill:.0%}{up_note}"


def normalize_repaired_cutout(im: Image.Image, size: int = 1024) -> Image.Image:
    """Normalize a Stage-2 repaired cutout to square RGB with pure `#FFFFFF` background."""
    im = im.convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(im)
    bg = np.all(arr >= 244, axis=-1)
    arr[bg] = 255
    return Image.fromarray(arr)
