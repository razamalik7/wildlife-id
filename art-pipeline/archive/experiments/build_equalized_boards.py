"""
Applies `enhance_outline_skin_tight` across ALL 78 candidates (`#1..#78`) to equalize the border across the entire set,
computes Before vs. After Rebound-Trough outline metrics, saves all 78 equalized 1024x1024 PNGs to
`art-pipeline/equalized_candidates_78/`, and renders:
1. `picked_45_equalized_board.jpg` — The 45 selected illustrations (clean artwork, no diagnostic dots covering the outlines)
   with Before -> After BlackLine% / TotalLine% metrics.
2. `all_78_equalized_candidate_board.jpg` — All 78 candidates with the equalized skin-tight outline applied to every image
   (Green tile border = currently in your 45 picks; Blue/Grey tile border = the other 33 candidates, in case equalizing
   the outline makes you want to include a few more!).
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from analyze_style_consistency import load_all_items, ART
from measure_outline_trough import PICKED_45, analyze_rebound_outline
from tight_outline_post import enhance_outline_skin_tight
from verify_width_and_shells import enhance_outline_true_match, measure_shells_and_width

OUT_DIR = Path(__file__).parent / "equalized_candidates_78"


def equalize_candidate(p: Path) -> tuple[np.ndarray, Path, dict, dict]:
    """
    Measures raw outline width (`width_px`) and 3px rim luminance (`rim3_l`).
    - If an image already has a thick dark OG outline (`width_px >= 2.5` and `rim3_l <= 45`),
      uses gentle 1px halo cleanup (`enhance_outline_skin_tight(strength=0.65)`) so it is never over-thickened.
    - If an image has a thin or umber outline (`width_px < 2.5` or `rim3_l > 45`, like the squirrels `#41, #42`),
      applies `enhance_outline_true_match(strength=0.92)` to bring its stroke width up to `2.8-3.6px` and `Rim3_L ~ 24-28`.
    """
    m_before = measure_shells_and_width(p)
    if m_before["width_px"] >= 2.5 and m_before["rim3_l"] <= 45.0 and m_before["blk_pct"] >= 75.0:
        eq_rgb, tmp_p = enhance_outline_skin_tight(p, strength=0.65)
    else:
        eq_rgb, tmp_p = enhance_outline_true_match(p, strength=0.92)
    m_after = measure_shells_and_width(tmp_p)
    return eq_rgb, tmp_p, m_before, m_after


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_all_items()
    picked_set = set(PICKED_45)

    try:
        font_hdr = ImageFont.truetype("arialbd.ttf", 18)
        font_sub = ImageFont.truetype("arialbd.ttf", 13)
        font_lbl = ImageFont.truetype("arialbd.ttf", 11)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_lbl = ImageFont.load_default()

    all_records = []
    print(
        f"{'ID':>3s} {'Pick':>4s} {'Species':34s} | {'Before Blk/Tot':>15s} -> {'After Blk/Umb/NoL (Line_L)':>27s}"
    )
    print("-" * 98)

    for idx_0, (lbl, p) in enumerate(items):
        idx = idx_0 + 1
        eq_rgb, _, m_before, m_after = equalize_candidate(p)
        eq_path = OUT_DIR / f"cand_{idx:02d}_{p.stem}.png"
        Image.fromarray(eq_rgb).save(eq_path)

        is_picked = idx in picked_set
        mark = "[*]" if is_picked else "   "
        print(
            f"#{idx:2d} {mark:4s} {lbl:34s} | "
            f"W:{m_before['width_px']:3.1f}px->{m_after['width_px']:3.1f}px "
            f"RimL:{m_before['rim3_l']:4.0f}->{m_after['rim3_l']:4.0f} "
            f"Blk:{m_before['blk_pct']:4.0f}%->{m_after['blk_pct']:4.0f}%"
        )
        all_records.append((idx, lbl, eq_rgb, m_before, m_after, is_picked))

    # Summary statistics for the 45 picked images Before vs After equalization
    picked_recs = [r for r in all_records if r[5]]
    b_blk = [r[3]["blk_pct"] for r in picked_recs]
    a_blk = [r[4]["blk_pct"] for r in picked_recs]
    b_w = [r[3]["width_px"] for r in picked_recs]
    a_w = [r[4]["width_px"] for r in picked_recs]
    b_l = [r[3]["rim3_l"] for r in picked_recs]
    a_l = [r[4]["rim3_l"] for r in picked_recs]
    print("\n=== 45 PICKED SET SUMMARY (BEFORE vs AFTER 2.8px WIDTH + RIM EQUALIZATION) ===")
    print(
        f"BlackLine% : Before mean={np.mean(b_blk):.1f}% (min={np.min(b_blk):.1f}%) "
        f"-> After mean={np.mean(a_blk):.1f}% (min={np.min(a_blk):.1f}%, std={np.std(a_blk):.1f}%)"
    )
    print(
        f"StrokeWidth: Before mean={np.mean(b_w):.1f}px (min={np.min(b_w):.1f}px) "
        f"-> After mean={np.mean(a_w):.1f}px (min={np.min(a_w):.1f}px, std={np.std(a_w):.1f}px)"
    )
    print(
        f"Rim3_L     : Before mean={np.mean(b_l):.1f} (max={np.max(b_l):.1f}) "
        f"-> After mean={np.mean(a_l):.1f} (max={np.max(a_l):.1f}, std={np.std(a_l):.1f})"
    )

    # 1. Build `picked_45_equalized_board.jpg` (9x5 grid, clean equalized images)
    cols45, rows45 = 9, 5
    tw, th = 225, 240
    board45 = Image.new("RGB", (cols45 * tw, rows45 * th + 60), (255, 255, 255))
    d45 = ImageDraw.Draw(board45)
    d45.rectangle([0, 0, cols45 * tw, 56], fill=(26, 32, 40))
    d45.text(
        (14, 7),
        "Your 45 Picked Set — Equalized in Both Stroke Width (2.8–3.8px) & 3px Rim Darkness (RimL ~ 20–28)",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    d45.text(
        (14, 32),
        f"Mean Width={np.mean(a_w):.1f}px (min={np.min(a_w):.1f}px)  |  Mean 3px RimL={np.mean(a_l):.1f}  |  Mean BlackLine%={np.mean(a_blk):.1f}%",
        fill=(200, 235, 210),
        font=font_sub,
    )

    for k, (idx, lbl, eq_rgb, m_bef, m_aft, _) in enumerate(picked_recs):
        ri, ci = divmod(k, cols45)
        xc = ci * tw
        yc = 56 + ri * th
        board45.paste(
            Image.fromarray(eq_rgb).resize((tw - 12, th - 48), Image.Resampling.LANCZOS),
            (xc + 6, yc + 42),
        )
        d45.rectangle([xc + 1, yc + 1, xc + tw - 2, yc + th - 2], outline=(35, 175, 70), width=2)
        short = lbl.split("] ")[-1][:20]
        d45.text((xc + 6, yc + 5), f"#{idx} {short}", fill=(15, 15, 15), font=font_lbl)
        d45.text(
            (xc + 6, yc + 21),
            f"W:{m_aft['width_px']:.1f}px RimL:{m_aft['rim3_l']:.0f} Blk:{m_aft['blk_pct']:.0f}%",
            fill=(25, 125, 50),
            font=font_lbl,
        )

    out45 = ART / "picked_45_equalized_board.jpg"
    board45.save(out45, quality=93)
    print(f"Saved {out45}")

    # 2. Build `all_78_equalized_candidate_board.jpg` (10x8 grid of all 78 equalized images)
    cols78, rows78 = 10, 8
    tw78, th78 = 210, 225
    board78 = Image.new("RGB", (cols78 * tw78, rows78 * th78 + 60), (255, 255, 255))
    d78 = ImageDraw.Draw(board78)
    d78.rectangle([0, 0, cols78 * tw78, 56], fill=(26, 32, 40))
    d78.text(
        (14, 7),
        "All 78 Candidates (#1–#78) — Equalized in Both Stroke Width (2.8–3.8px) & 3px Rim Darkness",
        fill=(255, 255, 255),
        font=font_hdr,
    )
    d78.text(
        (14, 32),
        "GREEN BORDER = In Your 45 Picks  |  GREY BORDER = Other 33 Candidates (Now Width- & Rim-Equalized!)",
        fill=(210, 225, 245),
        font=font_sub,
    )

    for k, (idx, lbl, eq_rgb, m_bef, m_aft, is_picked) in enumerate(all_records):
        ri, ci = divmod(k, cols78)
        xc = ci * tw78
        yc = 56 + ri * th78
        if not is_picked:
            d78.rectangle([xc, yc, xc + tw78, yc + th78], fill=(249, 250, 252))
        board78.paste(
            Image.fromarray(eq_rgb).resize((tw78 - 12, th78 - 46), Image.Resampling.LANCZOS),
            (xc + 6, yc + 40),
        )
        if is_picked:
            bcol = (30, 180, 65)
            bw = 3
            tcol = (15, 115, 40)
        else:
            bcol = (195, 200, 208)
            bw = 1
            tcol = (90, 95, 105)
        d78.rectangle([xc + 1, yc + 1, xc + tw78 - 2, yc + th78 - 2], outline=bcol, width=bw)
        short = lbl.split("] ")[-1][:18]
        d78.text((xc + 6, yc + 5), f"#{idx} {short}", fill=tcol if is_picked else (40, 40, 40), font=font_lbl)
        d78.text(
            (xc + 6, yc + 20),
            f"W:{m_aft['width_px']:.1f}px RimL:{m_aft['rim3_l']:.0f} Blk:{m_aft['blk_pct']:.0f}%",
            fill=tcol,
            font=font_lbl,
        )

    out78 = ART / "all_78_equalized_candidate_board.jpg"
    board78.save(out78, quality=92)
    print(f"Saved {out78}")


if __name__ == "__main__":
    main()
