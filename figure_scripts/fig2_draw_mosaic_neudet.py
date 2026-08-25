#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Draw a three-panel illustration for NEU-DET:

Left:
    Original single-defect image with O2O-style boxes
    (1 target, 1 positive sample, several negative samples).

Middle:
    Same-image perturbation Mosaic:
    the same image copied 4 times, light perturbation, 2x2 stitch.

Right:
    Standard data augmentation applied to the original image:
    4 independent augmented versions arranged as 2x2
    (photometric distortion + zoom-out + horizontal flip + resize).

Output:
    figures/standard_mosaic_disadvantage_neudet.png

Input data note:
This script selects several images from the public NEU-DET dataset as input; the
exact number of images is determined by the program logic. It uses 4 images in
total: 1 main single-defect image (for the left and middle panels) and 3 random
auxiliary images (for the right standard-augmentation panel). The main image is
picked automatically from the dataset, or can be specified via a command-line argument.
"""

import os
import sys
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


# =========================
# Basic settings
# =========================

THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name == "tools":
    PROJECT_ROOT = THIS_FILE.parents[1]
else:
    PROJECT_ROOT = THIS_FILE.parent

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

DATASET_CANDIDATES = [
    "NEU-DET-split",
    "NEU-DET",
    "NEUDET",
    "neu-det",
    "datasets/NEU-DET",
    "data/NEU-DET",
    "data/NEUDET",
    "dataset/NEU-DET",
    "datasets/neu-det",
]

SEED = 42

PANEL_SIZE = 480
GAP = 8
CAPTION_H = 42

COLOR_GT = (255, 215, 0)
COLOR_POS = (255, 0, 0)
COLOR_NEG = (60, 180, 75)
COLOR_TEXT = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

try:
    RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE = Image.BILINEAR


# =========================
# Dataset utilities
# =========================

def find_dataset_root():
    searched = []
    for candidate in DATASET_CANDIDATES:
        root = PROJECT_ROOT / candidate
        searched.append(root)
        if not root.is_dir():
            continue
        possible_pairs = [
            (root / "train" / "images", root / "train" / "annotations"),
            (root / "IMAGES", root / "ANNOTATIONS"),
            (root / "images", root / "annotations"),
            (root / "JPEGImages", root / "Annotations"),
        ]
        for img_dir, ann_dir in possible_pairs:
            if img_dir.is_dir() and ann_dir.is_dir():
                return root, img_dir, ann_dir, searched
    return None, None, None, searched


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_node = root.find("size")
    if size_node is None:
        return None
    img_w = int(float(size_node.find("width").text))
    img_h = int(float(size_node.find("height").text))
    boxes, names = [], []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bbox_node = obj.find("bndbox")
        if name_node is None or bbox_node is None:
            continue
        name = name_node.text.strip()
        if name not in CLASS_NAMES:
            continue
        try:
            x1 = float(bbox_node.find("xmin").text)
            y1 = float(bbox_node.find("ymin").text)
            x2 = float(bbox_node.find("xmax").text)
            y2 = float(bbox_node.find("ymax").text)
        except Exception:
            continue
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 > x1 + 2 and y2 > y1 + 2:
            boxes.append([x1, y1, x2, y2])
            names.append(name)
    return img_w, img_h, boxes, names


def find_image_path(img_dir, stem):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def collect_single_box_samples(img_dir, ann_dir, min_area_pct=0.3, max_area_pct=20.0):
    samples = []
    xml_files = sorted([p for p in ann_dir.iterdir() if p.suffix.lower() == ".xml"])
    for xml_path in xml_files:
        parsed = parse_voc_xml(xml_path)
        if parsed is None:
            continue
        img_w, img_h, boxes, names = parsed
        if len(boxes) != 1:
            continue
        img_path = find_image_path(img_dir, xml_path.stem)
        if img_path is None:
            continue
        x1, y1, x2, y2 = boxes[0]
        area_pct = (x2 - x1) * (y2 - y1) / (img_w * img_h) * 100.0
        if min_area_pct <= area_pct <= max_area_pct:
            samples.append({
                "xml_path": xml_path,
                "img_path": img_path,
                "class": names[0],
                "box": np.array(boxes[0], dtype=np.float32),
                "boxes": [np.array(b, dtype=np.float32) for b in boxes],
                "area_pct": area_pct,
            })
    return samples


def load_sample(sample):
    img = Image.open(sample["img_path"]).convert("RGB")
    if "boxes" in sample:
        boxes = [b.copy() for b in sample["boxes"]]
    else:
        boxes = [sample["box"].copy()]
    return img, boxes, sample["class"]


# =========================
# Image and box utilities
# =========================

def _light_augment(img, seed=None):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if rng.random() < 0.9:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.5:
        img = ImageEnhance.Sharpness(img).enhance(rng.uniform(0.85, 1.25))
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.45)))
    if rng.random() < 0.8:
        arr = np.array(img).astype(np.float32)
        noise_std = rng.uniform(2.0, 6.0)
        noise = np_rng.normal(0, noise_std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def _standard_augment(img, box, panel_size, seed=None):
    """
    Standard augmentation: photometric distort + zoomout + flip + random
    crop + resize.  All steps remap the bbox accordingly.
    Returns (augmented_img, remapped_box) -- box may be None if cropped out.
    """
    rng = random.Random(seed)

    orig_w, orig_h = img.size
    b = box.copy()

    # 1. RandomPhotometricDistort
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.5:
        img = ImageEnhance.Sharpness(img).enhance(rng.uniform(0.5, 2.0))

    # 2. RandomZoomOut (shrink and paste onto gray canvas, no black)
    if rng.random() < 0.6:
        zoom_scale = rng.uniform(0.5, 0.9)
        new_w = max(int(orig_w * zoom_scale), 32)
        new_h = max(int(orig_h * zoom_scale), 32)
        shrunk = img.resize((new_w, new_h), RESAMPLE)
        offset_x = rng.randint(0, orig_w - new_w)
        offset_y = rng.randint(0, orig_h - new_h)
        canvas_bg = Image.new("RGB", (orig_w, orig_h), (114, 114, 114))
        canvas_bg.paste(shrunk, (offset_x, offset_y))
        img = canvas_bg

        b[0] = b[0] * (new_w / orig_w) + offset_x
        b[1] = b[1] * (new_h / orig_h) + offset_y
        b[2] = b[2] * (new_w / orig_w) + offset_x
        b[3] = b[3] * (new_h / orig_h) + offset_y
        b[0::2] = np.clip(b[0::2], 0, orig_w - 1)
        b[1::2] = np.clip(b[1::2], 0, orig_h - 1)

    # 3. RandomHorizontalFlip
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        cur_w = img.size[0]
        x1, x2 = b[0], b[2]
        b[0] = cur_w - x2
        b[2] = cur_w - x1

    # 4. Random crop-to-square
    cur_w, cur_h = img.size
    crop_size = min(cur_w, cur_h)
    if cur_w > cur_h:
        offset = cur_w - crop_size
        crop_x = rng.randint(0, offset) if offset > 0 else 0
        crop_y = 0
    else:
        offset = cur_h - crop_size
        crop_x = 0
        crop_y = rng.randint(0, offset) if offset > 0 else 0

    img = img.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
    b[0] -= crop_x
    b[1] -= crop_y
    b[2] -= crop_x
    b[3] -= crop_y
    b[0::2] = np.clip(b[0::2], 0, crop_size - 1)
    b[1::2] = np.clip(b[1::2], 0, crop_size - 1)

    # 5. Resize to target
    target_size = PANEL_SIZE
    sx = target_size / crop_size
    sy = target_size / crop_size
    img = img.resize((target_size, target_size), RESAMPLE)
    b[0] *= sx
    b[1] *= sy
    b[2] *= sx
    b[3] *= sy

    if b[2] <= b[0] + 2 or b[3] <= b[1] + 2:
        return img, None

    return img, b


def resize_to_square(img, boxes, size):
    w, h = img.size
    resized = img.resize((size, size), RESAMPLE)
    out = [b.copy() for b in boxes]
    for b in out:
        b[0::2] = b[0::2] * size / w
        b[1::2] = b[1::2] * size / h
    return resized, out


def scale_box(box, sx, sy, dx=0, dy=0):
    b = box.copy()
    b[0::2] = b[0::2] * sx + dx
    b[1::2] = b[1::2] * sy + dy
    return b


def jitter_box(box, img_size, scale_min=0.9, scale_max=1.22, shift_ratio=0.06, seed=None):
    rng = random.Random(seed)
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    scale = rng.uniform(scale_min, scale_max)
    nw = w * scale
    nh = h * scale
    dx = rng.uniform(-shift_ratio, shift_ratio) * img_size
    dy = rng.uniform(-shift_ratio, shift_ratio) * img_size
    ncx = cx + dx
    ncy = cy + dy
    nx1 = max(0, ncx - nw / 2)
    ny1 = max(0, ncy - nh / 2)
    nx2 = min(img_size - 1, ncx + nw / 2)
    ny2 = min(img_size - 1, ncy + nh / 2)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def sample_negative_boxes(gt_box, img_size, count=3, seed=None):
    rng = random.Random(seed)
    negs = []
    gx1, gy1, gx2, gy2 = gt_box
    for _ in range(300):
        if len(negs) >= count:
            break
        w = rng.uniform(0.20, 0.38) * img_size
        h = rng.uniform(0.18, 0.36) * img_size
        x1 = rng.uniform(0, img_size - w)
        y1 = rng.uniform(0, img_size - h)
        x2 = x1 + w
        y2 = y1 + h
        ix1 = max(x1, gx1)
        iy1 = max(y1, gy1)
        ix2 = min(x2, gx2)
        iy2 = min(y2, gy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = w * h
        if inter / (area + 1e-6) < 0.08:
            negs.append(np.array([x1, y1, x2, y2], dtype=np.float32))
    return negs


def draw_rect(draw, box, color, width=3):
    x1, y1, x2, y2 = box
    draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=color, width=width)


def draw_boxes_like_paper(img, gt_boxes, pos_boxes, neg_boxes):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for b in neg_boxes:
        draw_rect(draw, b, COLOR_NEG, width=3)
    for b in gt_boxes:
        draw_rect(draw, b, COLOR_GT, width=3)
    for b in pos_boxes:
        draw_rect(draw, b, COLOR_POS, width=3)
    return out


# =========================
# Panel generation
# =========================

def make_left_original_panel(img, boxes, panel_size, seed):
    """
    Left panel: original single-defect image, GT + positive + negatives.
    """
    panel_img, gt_boxes = resize_to_square(img, boxes, panel_size)
    all_pos, all_neg = [], []
    for i, gt_box in enumerate(gt_boxes):
        pos_box = jitter_box(gt_box, panel_size, scale_min=0.95, scale_max=1.18,
                              shift_ratio=0.035, seed=seed + 10 + i)
        all_pos.append(pos_box)
        all_neg.extend(sample_negative_boxes(gt_box, panel_size, count=2,
                                              seed=seed + 20 + i))
    return draw_boxes_like_paper(panel_img, gt_boxes=gt_boxes,
                                  pos_boxes=all_pos, neg_boxes=all_neg)


def make_middle_mosaic_panel(img, boxes, panel_size, seed):
    """
    Middle panel: same-image perturbation Mosaic (4 copies, 2x2).
    """
    cell = panel_size // 2
    canvas = Image.new("RGB", (panel_size, panel_size), COLOR_WHITE)
    all_gt, all_pos, all_neg = [], [], []
    src_w, src_h = img.size

    for i in range(4):
        row, col = i // 2, i % 2
        x0, y0 = col * cell, row * cell

        aug_img = _light_augment(img.copy(), seed=seed + i)
        resized = aug_img.resize((cell, cell), RESAMPLE)
        canvas.paste(resized, (x0, y0))

        for j, box in enumerate(boxes):
            gt_box = scale_box(box, sx=cell / src_w, sy=cell / src_h, dx=x0, dy=y0)
            local_gt = scale_box(box, sx=cell / src_w, sy=cell / src_h, dx=0, dy=0)

            pos_local = jitter_box(local_gt, cell, scale_min=0.95, scale_max=1.18,
                                    shift_ratio=0.035, seed=seed + 100 + i * 10 + j)
            pos_box = pos_local.copy()
            pos_box[0::2] += x0
            pos_box[1::2] += y0

            for nb in sample_negative_boxes(local_gt, cell, count=1,
                                            seed=seed + 200 + i * 10 + j):
                nb2 = nb.copy()
                nb2[0::2] += x0
                nb2[1::2] += y0
                all_neg.append(nb2)

            all_gt.append(gt_box)
            all_pos.append(pos_box)

    return draw_boxes_like_paper(canvas, gt_boxes=all_gt,
                                  pos_boxes=all_pos, neg_boxes=all_neg)


def make_right_standard_augmentation_panel(orig_img, orig_boxes, orig_xml_path,
                                           all_samples, panel_size, seed):
    """
    Right panel: standard data augmentation with 1 original image + 3 random
    images. Each undergoes photometric + zoomout + flip + crop + resize,
    arranged as 2x2.
    """
    rng = random.Random(seed)
    cell = panel_size // 2
    canvas = Image.new("RGB", (panel_size, panel_size), COLOR_WHITE)
    all_gt, all_pos, all_neg = [], [], []

    main_quadrant = rng.randint(0, 3)

    other_samples = [s for s in all_samples if s["xml_path"] != orig_xml_path]
    rng.shuffle(other_samples)
    random_samples = other_samples[:3]

    rand_idx = 0
    for i in range(4):
        row, col = i // 2, i % 2
        x0, y0 = col * cell, row * cell

        if i == main_quadrant:
            img, boxes = orig_img.copy(), [b.copy() for b in orig_boxes]
        else:
            rs = random_samples[rand_idx]
            rand_idx += 1
            img = Image.open(rs["img_path"]).convert("RGB")
            boxes = [rs["box"].copy()]

        img_pasted = False
        for j, box in enumerate(boxes):
            aug_img, aug_box = _standard_augment(img.copy(), box.copy(),
                                                  panel_size=panel_size,
                                                  seed=seed + 300 + i * 10 + j)
            if not img_pasted:
                aug_img = aug_img.resize((cell, cell), RESAMPLE)
                canvas.paste(aug_img, (x0, y0))
                img_pasted = True

            cell_sx = cell / PANEL_SIZE
            cell_sy = cell / PANEL_SIZE

            if aug_box is not None:
                gt_box = scale_box(aug_box, sx=cell_sx, sy=cell_sy, dx=x0, dy=y0)
                local_gt = scale_box(aug_box, sx=cell_sx, sy=cell_sy, dx=0, dy=0)
                all_gt.append(gt_box)

                pos_local = jitter_box(local_gt, cell, scale_min=0.95, scale_max=1.18,
                                        shift_ratio=0.035, seed=seed + 400 + i * 10 + j)
                pos_box = pos_local.copy()
                pos_box[0::2] += x0
                pos_box[1::2] += y0
                all_pos.append(pos_box)

                for nb in sample_negative_boxes(local_gt, cell, count=1,
                                                seed=seed + 500 + i * 10 + j):
                    nb2 = nb.copy()
                    nb2[0::2] += x0
                    nb2[1::2] += y0
                    all_neg.append(nb2)

    return draw_boxes_like_paper(canvas, gt_boxes=all_gt,
                                  pos_boxes=all_pos, neg_boxes=all_neg)


# =========================
# Caption utilities
# =========================

def get_font(size=22, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def draw_centered_text(draw, xy, text, font, fill):
    x, y, w, h = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, fill=fill, font=font)


def compose_three_panels(left_panel, middle_panel, right_panel, output_path):
    fig_w = PANEL_SIZE * 3 + GAP * 2
    fig_h = PANEL_SIZE + CAPTION_H

    canvas = Image.new("RGB", (fig_w, fig_h), COLOR_WHITE)

    canvas.paste(left_panel, (0, 0))
    canvas.paste(middle_panel, (PANEL_SIZE + GAP, 0))
    canvas.paste(right_panel, (PANEL_SIZE * 2 + GAP * 2, 0))

    draw = ImageDraw.Draw(canvas)
    font = get_font(size=22, bold=True)

    captions = [
        "(a) Original one-object image",
        "(b) Same-image Mosaic",
        "(c) Standard augmentation with random images",
    ]
    offsets = [0, PANEL_SIZE + GAP, PANEL_SIZE * 2 + GAP * 2]

    for i, (caption, ox) in enumerate(zip(captions, offsets)):
        draw_centered_text(draw, (ox, PANEL_SIZE, PANEL_SIZE, CAPTION_H),
                           caption, font, COLOR_TEXT)

    canvas.save(output_path, dpi=(300, 300))


# =========================
# Main
# =========================

def main(image_file=None):
    print("=" * 72)
    print("Drawing three-panel figure for NEU-DET")
    print("=" * 72)

    random.seed(SEED)
    np.random.seed(SEED)

    ds_root, img_dir, ann_dir, searched = find_dataset_root()

    if ds_root is None:
        print("\n[ERROR] NEU-DET dataset was not found.")
        print("[INFO] Searched paths:")
        for p in searched:
            print(f"  - {p}")
        sys.exit(1)

    print(f"[INFO] Dataset root: {ds_root}")
    print(f"[INFO] Image dir:    {img_dir}")
    print(f"[INFO] Annotation:   {ann_dir}")

    if image_file is not None:
        img_fname = os.path.basename(image_file)
        stem = os.path.splitext(img_fname)[0]
        ann_path = ann_dir / (stem + ".xml")
        img_path = img_dir / img_fname
        if not img_path.exists():
            img_path = Path(image_file)
        if not ann_path.exists():
            print(f"[ERROR] Annotation not found: {ann_path}")
            sys.exit(1)
        parsed = parse_voc_xml(ann_path)
        if parsed is None or len(parsed[2]) == 0:
            print(f"[ERROR] Image must have at least 1 bbox")
            sys.exit(1)
        img_w, img_h, boxes, names = parsed
        x1, y1, x2, y2 = boxes[0]
        area_pct = (x2-x1)*(y2-y1)/(img_w*img_h)*100
        sample = {
            "xml_path": ann_path,
            "img_path": img_path,
            "class": names[0],
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "boxes": [np.array(b, dtype=np.float32) for b in boxes],
            "area_pct": area_pct,
        }
        all_samples = collect_single_box_samples(
            img_dir=img_dir, ann_dir=ann_dir,
            min_area_pct=0.3, max_area_pct=20.0,
        )
    else:
        all_samples = collect_single_box_samples(
            img_dir=img_dir, ann_dir=ann_dir,
            min_area_pct=0.3, max_area_pct=20.0,
        )
        if len(all_samples) == 0:
            print("\n[ERROR] No valid single-box samples were found.")
            sys.exit(1)
        sorted_samples = sorted(all_samples, key=lambda s: abs(s["area_pct"] - 3.0))
        sample = sorted_samples[0]

    img, boxes, cls_name = load_sample(sample)

    print("[INFO] Selected single-defect sample:")
    print(f"  class: {cls_name}")
    print(f"  image: {sample['img_path']}")
    print(f"  ann:   {sample['xml_path']}")
    print(f"  bboxes: {len(boxes)}")
    print(f"  area:  {sample['area_pct']:.2f}%")

    left_panel = make_left_original_panel(img=img, boxes=boxes,
                                           panel_size=PANEL_SIZE, seed=SEED)

    middle_panel = make_middle_mosaic_panel(img=img, boxes=boxes,
                                             panel_size=PANEL_SIZE, seed=SEED)

    right_panel = make_right_standard_augmentation_panel(
        orig_img=img, orig_boxes=boxes,
        orig_xml_path=sample["xml_path"],
        all_samples=all_samples,
        panel_size=PANEL_SIZE,
        seed=SEED,
    )

    out_dir = PROJECT_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if image_file is not None:
        png_name = f"{cls_name}.png"
    else:
        png_name = "standard_mosaic_disadvantage_neudet.png"
    png_path = out_dir / png_name

    compose_three_panels(
        left_panel=left_panel,
        middle_panel=middle_panel,
        right_panel=right_panel,
        output_path=png_path,
    )

    print(f"[INFO] Saved PNG: {png_path}")
    print("[INFO] Done.")
    print("=" * 72)


if __name__ == "__main__":
    img_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(image_file=img_file)
