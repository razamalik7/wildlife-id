"""Download BiRefNet-general-bb_swin_v1_tiny and isnet-general-use ONNX models to ~/.u2net/
and compare their cutouts against old u2net.onnx on Pond Slider, Green-winged Teal,
Mule Deer, American Pika, and Muskrat!
"""

from pathlib import Path
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image, ImageDraw, ImageFont

MODEL_DIR = Path.home() / ".u2net"
WORK = Path(__file__).parent / "batch_run_v4"
ARTIFACT_DIR = Path(r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021")

MODELS = {
    "u2net": (
        MODEL_DIR / "u2net.onnx",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        (320, 320),
    ),
    "isnet": (
        MODEL_DIR / "isnet-general-use.onnx",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        (1024, 1024),
    ),
    "birefnet": (
        MODEL_DIR / "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        (1024, 1024),
    ),
}


def ensure_model(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 10_000_000:
        return
    print(f"Downloading {path.name}...", flush=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded {path.name} ({path.stat().st_size / 1024**2:.1f} MB)", flush=True)


def run_seg(model_path: Path, size: tuple[int, int], im: Image.Image) -> Image.Image:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    rgb = im.convert("RGB")
    small = rgb.resize(size, Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    if "isnet" in model_path.name:
        x = (x - [0.5, 0.5, 0.5]) / [1.0, 1.0, 1.0]
    else:
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    while out.ndim > 2:
        out = out[0]
    # Apply sigmoid if raw logits (BiRefNet outputs logits or sigmoid depending on export)
    if out.min() < -0.1 or out.max() > 1.1:
        out = 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))
    out = (out - out.min()) / max(out.max() - out.min(), 1e-6)
    mask = Image.fromarray((out * 255).astype(np.uint8)).resize(rgb.size, Image.LANCZOS)
    res = rgb.convert("RGBA")
    res.putalpha(mask)

    # Crop to alpha > 20 and place on 1024 white at 76% fill
    a = np.asarray(res)[..., 3]
    ys, xs = np.where(a > 20)
    if len(xs) > 0:
        res = res.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = int(1024 * 0.76) / max(res.size)
    nw, nh = max(1, round(res.width * scale)), max(1, round(res.height * scale))
    res = res.resize((nw, nh), Image.LANCZOS)
    canv = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    canv.paste(res, ((1024 - nw) // 2, (1024 - nh) // 2), res)
    return canv.convert("RGB")


def main() -> None:
    for _, (p, url, _) in MODELS.items():
        ensure_model(p, url)

    species = ["pond_slider", "green-winged_teal", "mule_deer", "american_pika", "muskrat"]
    cols = ["Reference Photo", "Old u2net (320px)", "ISNet-General (1024px)", "BiRefNet-Swin-Tiny (1024px)"]
    cw, ch = 340, 340
    W = len(cols) * cw
    H = 48 + len(species) * (ch + 32)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for ci, t in enumerate(cols):
        d.text((ci * cw + 12, 12), t, fill=(20, 20, 20), font=font)

    out_dir = WORK / "seg_shootout"
    out_dir.mkdir(exist_ok=True)

    for ri, slug in enumerate(species):
        y_top = 48 + ri * (ch + 32)
        d.rectangle([0, y_top, W, y_top + 32], fill=(232, 232, 236))
        d.text((14, y_top + 5), f"{slug} — Segmentation Model Shootout", fill=(10, 10, 10), font=font)

        photo_p = WORK / f"{slug}_photo.png"
        photo = Image.open(photo_p).convert("RGB")
        panels = [photo]
        for m_name in ["u2net", "isnet", "birefnet"]:
            m_path, _, m_size = MODELS[m_name]
            cut_im = run_seg(m_path, m_size, photo)
            cut_im.save(out_dir / f"{slug}_{m_name}.png")
            panels.append(cut_im)

        for ci, im in enumerate(panels):
            cell = Image.new("RGB", (cw, ch), (255, 255, 255))
            thumb = im.copy()
            thumb.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
            cell.paste(thumb, ((cw - thumb.width) // 2, (ch - thumb.height) // 2))
            sheet.paste(cell, (ci * cw, y_top + 32))
            d.rectangle([ci * cw, y_top + 32, (ci + 1) * cw - 1, y_top + 32 + ch - 1], outline=(210, 210, 210))

    sheet.save(ARTIFACT_DIR / "segmentation_models_compare.jpg", quality=92)
    print("Saved segmentation_models_compare.jpg", flush=True)


if __name__ == "__main__":
    main()
