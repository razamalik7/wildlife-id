"""Build the SDXL style-LoRA animl_style_v7 training dataset from the locked 45 harmonized v3 illustrations.

Standalone CPU-only builder (0 MB VRAM, zero network/Ollama calls):
- Reads the 45 approved v3 illustrations from art-pipeline/equalized_candidates_78/cand_XX_*.png.
- Leaves art-pipeline/lora/dataset/ (v6) 100% untouched.
- Sanitizes exterior pixels outside the connected animal + whisker silhouette to pure (255, 255, 255) white.
- Preserves 1:1 sub-pixel anti-aliasing (zero double-LANCZOS blur) for fills in [0.73, 0.82],
  while rescaling outliers (e.g. #47 california_scrub-jay) cleanly to 0.77 fill.
- Writes .png + .txt pairs into art-pipeline/lora/dataset_v7/{img_6x,img_4x,img_2x}.
- Renders v7_dataset_45_contact_sheet.jpg in the conversation artifact directory.

Usage: .\art-pipeline\lora\venv\Scripts\python.exe art-pipeline\lora\build_dataset_v7.py
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, distance_transform_edt, label

REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "art-pipeline" / "equalized_candidates_78"
DATASET_DIR = Path(__file__).parent / "dataset_v7"
OUT_DIRS = {
    "6x": DATASET_DIR / "img_6x",
    "4x": DATASET_DIR / "img_4x",
    "2x": DATASET_DIR / "img_2x",
}
ARTIFACT_SHEET = Path(
    r"C:\Users\nars7\.gemini\antigravity\brain\cf192390-434f-49ae-a401-59243749f021\v7_dataset_45_contact_sheet.jpg"
)
SIZE = 1024
TARGET_FILL = 0.77
PASSTHROUGH_MIN_FILL = 0.73
PASSTHROUGH_MAX_FILL = 0.82

STYLE_CLAUSE = (
    "muted matte natural colors, flat cel-shaded color zones with stepped shading, "
    "subtle low-contrast warm brown-umber ink fur and contour lines, "
    "clean thin dark charcoal-umber outline, isolated on a plain solid white background"
)

# Exact 45-entry v7 manifest:
# (cand_idx, src_name, slug, subject_phrase, bucket, pose_clause, traits)
V7_MANIFEST: list[tuple[int, str, str, str, str, str, str]] = [
    (
        1,
        "cand_01_american_marten.png",
        "american_marten",
        "an american marten",
        "6x",
        "",
        "tawny brown matte coat, pale yellowish-buff throat bib, dark brown legs and bushy tail, light brown head, rounded ears with pale rims, dark eyes, slender body",
    ),
    (
        2,
        "cand_02_rhesus_macaque.png",
        "rhesus_macaque",
        "a rhesus macaque",
        "6x",
        "",
        "warm brown and sandy-gray matte coat, muted reddish-tan face skin, brown hands and feet, medium-length gray-brown tail, dark eyes",
    ),
    (
        3,
        "cand_03_american_robin.png",
        "american_robin",
        "an american robin",
        "4x",
        "",
        "dark slate-gray back and wings, warm reddish-orange breast, blackish head with white eye arcs, white throat streaks, yellow bill, brown legs",
    ),
    (
        4,
        "cand_04_american_toad.png",
        "american_toad",
        "an american toad",
        "4x",
        "",
        "mottled dark brown and tan warty skin, cream and brown dorsal blotches, pale warts, light tan throat, barred brown legs, dark eyes",
    ),
    (
        5,
        "cand_05_axis_deer.png",
        "axis_deer",
        "an axis deer",
        "4x",
        "",
        "golden-rufous coat covered with crisp white spots, white throat and underbelly, dark dorsal stripe along spine, brown legs with dark hooves, three-pronged brown antlers",
    ),
    (
        6,
        "cand_06_barking_treefrog.png",
        "barking_treefrog",
        "a barking treefrog",
        "4x",
        "",
        "bright green skin with dark brown-bordered olive dorsal spots, pale cream throat and belly, yellow-green limb markings, round adhesive toe pads, bronze-brown eyes",
    ),
    (
        7,
        "cand_07_caribou.png",
        "caribou",
        "a caribou",
        "4x",
        "",
        "dark brown body coat with lighter brown flanks, white neck mane and chest, pale muzzle and white tail base, dark brown legs and cloven hooves, large branching antlers",
    ),
    (
        8,
        "cand_08_desert_tortoise.png",
        "desert_tortoise",
        "a desert tortoise",
        "4x",
        "",
        "high-domed warm brown carapace with tan scute centers and concentric growth rings, wrinkled tan-brown head, stout scaly forelegs with claws, elephantine hind feet",
    ),
    (
        9,
        "cand_09_downy_woodpecker.png",
        "downy_woodpecker",
        "a downy woodpecker",
        "4x",
        "",
        "black upperparts with white dorsal back patch, black wings with white spots, white throat and belly, black-and-white striped head with small red nape patch, short dark chisel bill, gray feet",
    ),
    (
        11,
        "cand_11_eastern_red-backed_salamander.png",
        "eastern_red-backed_salamander",
        "an eastern red-backed salamander",
        "4x",
        "",
        "reddish-orange dorsal stripe from head down tail, dark slate-gray mottled sides, dark brown head with dark eyes, four small dark legs, long slender tail",
    ),
    (
        12,
        "cand_12_elk.png",
        "elk",
        "an elk",
        "4x",
        "",
        "rich tan-brown body coat, dark chocolate-brown neck mane, head, and slender legs, pale buff rump patch, large branching brown antlers, dark eyes",
    ),
    (
        14,
        "cand_14_harbor_seal.png",
        "harbor_seal",
        "a harbor seal",
        "4x",
        "",
        "mottled gray-brown and silvery-tan coat covered with dark and pale spots, rounded head with dark eyes and pale whiskers, short foreflippers and hind flippers",
    ),
    (
        15,
        "cand_15_mountain_goat.png",
        "mountain_goat",
        "a mountain goat",
        "4x",
        "",
        "thick off-white shaggy coat with cream shading, white beard and pantaloons, short sharp black horns, black nose and hooves, dark eyes",
    ),
    (
        16,
        "cand_16_nine-banded_armadillo.png",
        "nine-banded_armadillo",
        "a nine-banded armadillo",
        "4x",
        "",
        "grayish-brown armored carapace with movable transverse bands and tan scutes, long pointed gray-tan snout, prominent upright ears, armored ringed tail, clawed feet",
    ),
    (
        17,
        "cand_17_snowy_owl.png",
        "snowy_owl",
        "a snowy owl",
        "4x",
        "",
        "white plumage with dark brown bars and speckles across wings, back, and flanks, white facial disc with bright golden-yellow eyes and black bill, feathered white legs and dark talons",
    ),
    (
        18,
        "cand_18_spring_peeper.png",
        "spring_peeper",
        "a spring peeper",
        "4x",
        "",
        "warm tan-brown skin with dark brown X-shaped dorsal cross marking, dark band between eyes, dark crossbands on slender hind legs, small toe pads, dark eyes",
    ),
    (
        19,
        "cand_19_western_fence_lizard.png",
        "western_fence_lizard",
        "a western fence lizard",
        "4x",
        "",
        "mottled brown and gray-tan keeled scales with dark wavy dorsal blotches, speckled brown head, tan clawed legs, long banded tail, dark eyes",
    ),
    (
        20,
        "cand_20_white-tailed_deer.png",
        "white-tailed_deer",
        "a white-tailed deer",
        "4x",
        "",
        "warm reddish-brown coat, crisp white throat patch, pale muzzle ring and white inner ears, white belly and tail underside, slender tan legs with dark hooves, forward-curving brown antlers",
    ),
    (
        21,
        "cand_21_wood_duck.png",
        "wood_duck",
        "a wood duck",
        "4x",
        "",
        "crested dark green and purple head with crisp white facial stripes, red eye and red-based bill, rich chestnut breast with white flecks, bronze-tan flanks, dark blue-and-black back, orange webbed feet",
    ),
    (
        22,
        "cand_22_common_box_turtle_3.png",
        "common_box_turtle_3",
        "a common box turtle",
        "2x",
        "",
        "high-domed dark brown carapace with bright orange-yellow radiating markings, mottled brown and orange head, reddish-orange eye, sturdy scaly brown legs",
    ),
    (
        23,
        "cand_23_common_box_turtle_4.png",
        "common_box_turtle_4",
        "a common box turtle",
        "2x",
        "",
        "high-domed dark brown carapace with golden-yellow radiating lines and blotches, reddish-brown speckled head and neck, yellowish-brown eye, stout clawed brown legs",
    ),
    (
        24,
        "cand_24_gopher_snake_1.png",
        "gopher_snake_1",
        "a gopher snake",
        "2x",
        "",
        "coiled sandy-tan body with dark chocolate-brown dorsal blotches and smaller dark lateral spots, pale cream lower scales, tan head with dark postorbital stripe",
    ),
    (
        25,
        "cand_25_gopher_snake_2.png",
        "gopher_snake_2",
        "a gopher snake",
        "2x",
        "",
        "sinuous sandy-tan body with dark brown dorsal saddles and alternating lateral spots, cream throat, tan head with brown facial band, banded tail",
    ),
    (
        26,
        "cand_26_00_northern_shoveler.png",
        "northern_shoveler",
        "a northern shoveler",
        "4x",
        ", in flight with wings raised",
        "dark green head with yellow eye and long dark spatulate bill, crisp white breast and neck, rich chestnut belly and flanks, raised wings showing pale blue-gray forewing patch and white bar, orange webbed feet",
    ),
    (
        28,
        "cand_28_02_american_wigeon.png",
        "american_wigeon",
        "an american wigeon",
        "4x",
        "",
        "pale cream crown and speckled white cheek with dark green postorbital band, pale gray bill with black tip, warm brownish-taupe breast and flanks, black-and-white edged wing coverts with green patch, white belly, dark gray webbed feet",
    ),
    (
        29,
        "cand_29_03_green-winged_teal.png",
        "green-winged_teal",
        "a green-winged teal",
        "4x",
        "",
        "chestnut-brown head with green ear-patch extending down nape, dark slate bill, black-spotted buff-white breast, teal-green and brown patterned wings and flanks, yellowish-tan webbed feet",
    ),
    (
        30,
        "cand_30_04_mule_deer.png",
        "mule_deer",
        "a mule deer",
        "4x",
        "",
        "warm grayish-brown coat, pale cream throat patch and pale muzzle with dark nose, dark forehead patch, large dark-rimmed ears, tall bifurcated brown antlers, cream belly, four slender brown legs with dark hooves",
    ),
    (
        31,
        "cand_31_05_muskrat.png",
        "muskrat",
        "a muskrat",
        "4x",
        "",
        "plump crouching body with dense warm brown coat, lighter buff-tan cheeks and fine pale whiskers, small rounded ears, dark eye, clawed brown forepaws and hind paw",
    ),
    (
        34,
        "cand_34_08_pond_slider.png",
        "pond_slider",
        "a red-eared pond slider",
        "4x",
        "",
        "olive-green and muted olive-brown oval carapace with dark scute seams and pale rim, dark olive head and neck with bold pale yellow stripes, bright crimson-red postorbital stripe behind red-rimmed eye, clawed olive webbed feet",
    ),
    (
        36,
        "cand_36_00_opt3_13608867.png",
        "common_watersnake",
        "a common watersnake",
        "4x",
        "",
        "coiled tan and warm brown patterned body with dark brown crossbands and alternating blotches, pale cream-yellow labial scales with dark bars, brown head with dark postorbital stripe",
    ),
    (
        41,
        "cand_41_fresh20_v2_03_golden_mantled_squirrel.png",
        "golden-mantled_ground_squirrel",
        "a golden-mantled ground squirrel",
        "4x",
        "",
        "warm russet-golden head and shoulder mantle, unstriped buff-brown cheek with pale eye ring, prominent black-bordered white lateral stripe along side, cream-buff underbelly, bushy brown-and-buff tail",
    ),
    (
        42,
        "cand_42_fresh20_v2_04_western_gray_squirrel.png",
        "western_gray_squirrel",
        "a western gray squirrel",
        "4x",
        "",
        "frosted silver-gray coat, crisp white throat and underbelly, large bushy silver-gray plumed banner tail, rounded gray-brown ears, pale eye ring and dark eye",
    ),
    (
        43,
        "cand_43_fresh20_v2_05_yellow_bellied_marmot.png",
        "yellow-bellied_marmot",
        "a yellow-bellied marmot",
        "6x",
        "",
        "grizzled reddish-brown and gray-tan coat, warm golden-yellow chest and underbelly, dark brown head with pale buff patch between eyes and pale muzzle, dark brown bushy tail, sturdy clawed paws",
    ),
    (
        47,
        "cand_47_fresh20_v2_09_california_scrub_jay.png",
        "california_scrub-jay",
        "a california scrub-jay",
        "4x",
        "",
        "bright blue crown, nape, wings, and long tail, dark charcoal-brown cheek mask with thin white eyebrow, whitish throat framed by blue necklace band, pale brownish-gray back and belly, slender black bill and dark legs",
    ),
    (
        51,
        "cand_51_fresh20_v2_13_side_blotched_lizard.png",
        "common_side-blotched_lizard",
        "a common side-blotched lizard",
        "4x",
        "",
        "brown and tan dorsal scales, black-and-white speckled lateral stripe above bright warm orange-yellow lower flank, pale bluish-gray speckled throat, four clawed legs and long tapering tail",
    ),
    (
        52,
        "cand_52_fresh20_v2_14_southern_alligator_lizard.png",
        "southern_alligator_lizard",
        "a southern alligator lizard (Elgaria multicarinata)",
        "4x",
        ", curved body",
        "warm tan-brown keeled dorsal scales with faint crossbands, greenish-gray flanks with dark vertical bars, smooth brown head plates with pale lip and dark eye, four small clawed legs and long thick tail",
    ),
    (
        53,
        "cand_53_fresh20_v2_15_spiny_softshell.png",
        "spiny_softshell",
        "a spiny softshell turtle",
        "4x",
        "",
        "flat leathery olive-gray and warm brown oval carapace scattered with dark circular spots and pale tan rim, pointed tubular snorkel snout, olive-tan head and scaly webbed forelimb",
    ),
    (
        55,
        "cand_55_fresh20_v2_17_wood_frog.png",
        "wood_frog",
        "a wood frog",
        "4x",
        "",
        "warm peachy-tan and light brown skin with dark brown dorsal spots and dark dorsolateral ridge, prominent dark brown mask across eye and brown circular tympanum, pale cream belly with dark mottling, barred hind legs",
    ),
    (
        56,
        "cand_56_fresh20_v2_18_california_newt.png",
        "california_newt",
        "a california newt",
        "4x",
        "",
        "granular dark slate-gray and warm gray-brown dorsal skin, bright warm orange lower flanks, jaw, and underside, large dark eye with pale ring, four sturdy clawed legs and upturned tail",
    ),
    (
        57,
        "cand_57_fresh20_v2_19_leopard_shark.png",
        "leopard_shark",
        "a leopard shark",
        "6x",
        "",
        "sleek silvery-gray fusiform body patterned with prominent dark charcoal-brown dorsal saddles and large oval spots across back and fins, pale white underside, five gill slits, two dorsal fins, spotted pectoral fin, notched caudal fin",
    ),
    (
        62,
        "cand_62_04_merriams_kangaroo_rat.png",
        "merriams_kangaroo_rat",
        "a merriam's kangaroo rat (Dipodomys merriami)",
        "4x",
        "",
        "sandy buff-tan coat with darker grizzled dorsal streaks, cream-white cheek, throat, and underbelly, large dark eye, rounded brownish-pink ear, long hind foot, very long pale bi-colored tail with tufted tip",
    ),
    (
        66,
        "cand_66_08_belted_kingfisher.png",
        "belted_kingfisher",
        "a belted kingfisher",
        "4x",
        ", perched with both feet gripping a short natural wooden twig stub strictly under its feet",
        "shaggy slate-blue crested head, crisp white collar, slate-blue breast band above warm rufous-chestnut flanks and white belly, slate-blue back and wings with tiny white wing-tip spots, long stout dark bill",
    ),
    (
        68,
        "cand_68_10_hooded_merganser.png",
        "hooded_merganser",
        "a hooded merganser",
        "4x",
        ", swimming with webbed feet trailing behind",
        "black-and-brown crested head with large fan-shaped white patch, bright golden-orange eye, slender dark bill, white breast with two vertical black bars, rich cinnamon-brown flanks, black-and-white striped tertials, orange-brown webbed feet",
    ),
    (
        69,
        "cand_69_11_rough_greensnake.png",
        "rough_greensnake",
        "a rough greensnake",
        "4x",
        "",
        "slender coiled bright leaf-green body with smooth scales, pale yellowish-green lower sides and upper lip, green head with dark eye, long tapering tail",
    ),
    (
        76,
        "cand_76_18_western_tiger_salamander.png",
        "western_tiger_salamander",
        "a western tiger salamander",
        "4x",
        "",
        "dark charcoal-black body and tail patterned with bold irregular yellow bars and blotches, broad rounded dark snout with dark eyes, four sturdy dark legs with yellow spots, laterally flattened blotched tail",
    ),
]


def sanitize_and_frame(src_png: Path) -> tuple[Image.Image, float, float, bool]:
    """Remove exterior off-white borders/haze while keeping whiskers intact, and center at 1024x1024."""
    im = Image.open(src_png).convert("RGB")
    arr = np.array(im)
    if arr.shape != (SIZE, SIZE, 3):
        raise ValueError(f"Expected {(SIZE, SIZE, 3)} image, got {arr.shape} for {src_png}")

    core = arr.min(axis=-1) < 235
    core[:16, :] = core[-16:, :] = core[:, :16] = core[:, -16:] = False

    lbl, _ = label(binary_dilation(core, np.ones((15, 15), dtype=bool)))
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    if sizes.max() == 0:
        raise ValueError(f"No foreground component found in {src_png}")
    main_lbl = int(sizes.argmax())
    animal_core = (lbl == main_lbl) & core

    dist = distance_transform_edt(~animal_core)
    keep = dist <= 48.0
    keep[:20, :] = keep[-20:, :] = keep[:, :20] = keep[:, -20:] = False

    cleaned = arr.copy()
    cleaned[~keep] = 255

    ys, xs = np.where(np.any(cleaned < 255, axis=-1))
    if ys.size == 0 or xs.size == 0:
        raise ValueError(f"Empty cleaned subject for {src_png}")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    crop = Image.fromarray(cleaned[y0:y1, x0:x1])
    orig_fill = max(crop.size) / SIZE

    resized = False
    if PASSTHROUGH_MIN_FILL <= orig_fill <= PASSTHROUGH_MAX_FILL:
        subj = crop
    else:
        target = round(SIZE * TARGET_FILL)
        scale = target / max(crop.size)
        nw = max(1, round(crop.width * scale))
        nh = max(1, round(crop.height * scale))
        subj = crop.resize((nw, nh), Image.Resampling.LANCZOS)
        resized = True

    canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    paste_x = (SIZE - subj.width) // 2
    paste_y = (SIZE - subj.height) // 2
    canvas.paste(subj, (paste_x, paste_y))

    out_arr = np.array(canvas)
    oys, oxs = np.where(np.any(out_arr < 255, axis=-1))
    final_fill = max(int(oys.max() - oys.min() + 1), int(oxs.max() - oxs.min() + 1)) / SIZE
    return canvas, orig_fill, final_fill, resized


def build_caption(subject_phrase: str, pose_clause: str, traits: str) -> str:
    """Build caption matching exact inference prompt format."""
    return (
        f"animl_gouache_style, {subject_phrase}, full body wildlife illustration{pose_clause}, "
        f"single animal, {traits}, {STYLE_CLAUSE}"
    )


def verify_output_image(dst_png: Path, final_fill: float) -> None:
    """Verify dimensions, clean 20px border, and valid subject fill."""
    im = Image.open(dst_png)
    if im.size != (SIZE, SIZE) or im.mode != "RGB":
        raise AssertionError(f"{dst_png}: bad size/mode {im.size}/{im.mode}")
    arr = np.array(im)
    border_nonwhite = (
        int(np.any(arr[:20, :] < 255, axis=-1).sum())
        + int(np.any(arr[-20:, :] < 255, axis=-1).sum())
        + int(np.any(arr[20:-20, :20] < 255, axis=-1).sum())
        + int(np.any(arr[20:-20, -20:] < 255, axis=-1).sum())
    )
    if border_nonwhite != 0:
        raise AssertionError(f"{dst_png}: found {border_nonwhite} non-white border pixels")
    if not (PASSTHROUGH_MIN_FILL <= final_fill <= PASSTHROUGH_MAX_FILL):
        raise AssertionError(f"{dst_png}: final_fill={final_fill:.3f} out of [{PASSTHROUGH_MIN_FILL}, {PASSTHROUGH_MAX_FILL}]")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\consola.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render_contact_sheet(records: list[dict], out_path: Path) -> None:
    """Render a 9x5 verification contact sheet for all 45 dataset_v7 items."""
    cols, rows = 9, 5
    cell_w, cell_h = 360, 410
    header_h = 70
    sheet = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), (244, 245, 247))
    draw = ImageDraw.Draw(sheet)
    font_title = load_font(26)
    font_sub = load_font(16)
    font_badge = load_font(13)

    draw.text(
        (20, 14),
        "animl_style_v7 Training Dataset (45 Harmonized v3 Illustrations | 180 Samples/Epoch | 20px Clean Border Verified)",
        fill=(25, 28, 34),
        font=font_title,
    )
    draw.text(
        (20, 44),
        "Green = 6x Anchor (4)   Blue = 4x Standard (37)   Amber = 2x Multi-Pose (4)   [pose-tag] = Pose/Perch Disentangled",
        fill=(85, 92, 102),
        font=font_sub,
    )

    badge_colors = {
        "6x": (34, 128, 72),
        "4x": (48, 98, 168),
        "2x": (186, 118, 22),
    }

    for i, rec in enumerate(records):
        r = i // cols
        c = i % cols
        x0 = c * cell_w + 10
        y0 = header_h + r * cell_h + 8

        tile = Image.open(rec["dst_png"]).convert("RGB").resize((340, 340), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x0, y0))
        draw.rectangle([x0, y0, x0 + 340, y0 + 340], outline=(212, 216, 222), width=1)

        badge_col = badge_colors[rec["bucket"]]
        draw.rectangle([x0 + 6, y0 + 6, x0 + 44, y0 + 26], fill=badge_col)
        draw.text((x0 + 13, y0 + 8), rec["bucket"], fill=(255, 255, 255), font=font_badge)

        if rec["pose_clause"]:
            draw.rectangle([x0 + 258, y0 + 6, x0 + 334, y0 + 26], fill=(122, 62, 148))
            draw.text((x0 + 263, y0 + 8), "pose-tag", fill=(255, 255, 255), font=font_badge)

        resize_note = " *up" if rec["resized"] else ""
        label_line1 = f"#{rec['cand_idx']:02d} {rec['slug']}"
        label_line2 = f"fill={rec['final_fill']:.2f}{resize_note} | {rec['word_count']}w"
        draw.text((x0 + 2, y0 + 346), label_line1[:34], fill=(22, 24, 28), font=font_sub)
        draw.text((x0 + 2, y0 + 368), label_line2, fill=(90, 96, 108), font=font_badge)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=92)


def main() -> None:
    if len(V7_MANIFEST) != 45:
        raise AssertionError(f"Expected 45 manifest entries, got {len(V7_MANIFEST)}")

    for d in OUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
        for old_file in d.iterdir():
            if old_file.is_file() and old_file.suffix.lower() in {".png", ".txt", ".npz"}:
                old_file.unlink()

    tokenizer = None
    try:
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    except Exception:
        tokenizer = None

    records: list[dict] = []
    bucket_counts = {"6x": 0, "4x": 0, "2x": 0}
    max_words = 0
    max_clip_tokens = 0

    for cand_idx, src_name, slug, subject_phrase, bucket, pose_clause, traits in V7_MANIFEST:
        src_png = SRC_DIR / src_name
        if not src_png.exists():
            raise FileNotFoundError(f"Missing source illustration: {src_png}")

        out_dir = OUT_DIRS[bucket]
        dst_png = out_dir / f"{slug}.png"
        dst_txt = out_dir / f"{slug}.txt"

        canvas, orig_fill, final_fill, resized = sanitize_and_frame(src_png)
        canvas.save(dst_png)
        verify_output_image(dst_png, final_fill)

        caption = build_caption(subject_phrase, pose_clause, traits)
        word_count = len(caption.split())
        if word_count >= 110:
            raise AssertionError(f"{slug}: caption too long ({word_count} words): {caption}")
        max_words = max(max_words, word_count)

        clip_tokens = None
        if tokenizer is not None:
            clip_tokens = len(tokenizer(caption, truncation=False)["input_ids"])
            max_clip_tokens = max(max_clip_tokens, clip_tokens)
            if clip_tokens > 225:
                raise AssertionError(f"{slug}: CLIP token count {clip_tokens} exceeds 225")

        dst_txt.write_text(caption, encoding="utf-8")
        bucket_counts[bucket] += 1

        records.append(
            {
                "cand_idx": cand_idx,
                "src_name": src_name,
                "slug": slug,
                "subject_phrase": subject_phrase,
                "bucket": bucket,
                "pose_clause": pose_clause,
                "orig_fill": round(orig_fill, 4),
                "final_fill": round(final_fill, 4),
                "resized": resized,
                "word_count": word_count,
                "clip_tokens": clip_tokens,
                "dst_png": str(dst_png),
                "dst_txt": str(dst_txt),
                "caption": caption,
            }
        )
        res_flag = " [RESIZED->0.77]" if resized else " [1:1]"
        tok_str = f" | {clip_tokens} tok" if clip_tokens is not None else ""
        print(
            f"  [{len(records):02d}/45] #{cand_idx:02d} ({bucket}) {slug:32s} fill={final_fill:.3f}{res_flag} | {word_count}w{tok_str}",
            flush=True,
        )

    samples_per_epoch = bucket_counts["6x"] * 6 + bucket_counts["4x"] * 4 + bucket_counts["2x"] * 2
    summary = {
        "total_images": len(records),
        "bucket_counts": bucket_counts,
        "samples_per_epoch": samples_per_epoch,
        "max_words": max_words,
        "max_clip_tokens": max_clip_tokens if tokenizer is not None else None,
        "style_clause": STYLE_CLAUSE,
        "items": records,
    }
    (DATASET_DIR / "v7_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_contact_sheet(records, ARTIFACT_SHEET)

    print(
        f"\nv7 dataset ready: {len(records)} pairs "
        f"(6x={bucket_counts['6x']}, 4x={bucket_counts['4x']}, 2x={bucket_counts['2x']}, "
        f"{samples_per_epoch} samples/epoch, max_words={max_words}"
        + (f", max_clip_tokens={max_clip_tokens}" if tokenizer is not None else "")
        + ")",
        flush=True,
    )
    print(f"Contact sheet written to: {ARTIFACT_SHEET}", flush=True)


if __name__ == "__main__":
    main()
