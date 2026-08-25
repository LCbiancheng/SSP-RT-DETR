#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate NEU-DET comparison figure as SVG and PDF using saved box annotations.
Reads JSON files from box_annotations/ -- no model loading required.

Key fixes:
1) Output SVG and PDF only.
2) Remove the bottom legend and its reserved blank area.
3) Use Times New Roman style fonts for English text.
4) Move row labels closer to image panels and enlarge row-label font.
5) Slightly reduce text labels drawn inside image panels.
6) Add an inner padding inside each image panel so boxes near the left/top edge
   are not hidden by panel boundaries or grid lines.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
ANNOTATIONS_DIR = SCRIPT_DIR / "box_annotations"
IMG_DIR = SCRIPT_DIR / "images"

# =========================================================
# 1. Style
# =========================================================
PANEL_SIZE = 520

# Text fine-tuning in typographic points.
# The output PDF is saved at 300 dpi, so 1 pt = 300 / 72 px.
EXPORT_DPI = 300.0
PT_TO_PX = EXPORT_DPI / 72.0
TITLE_Y_SHIFT_PX = -int(round(1 * PT_TO_PX))   # move column titles upward by 1 pt
ROW_LABEL_X_SHIFT_PX = int(round(2 * PT_TO_PX)) # move left row labels rightward by 2 pt

# Increase label column width so row labels will never touch the image panels.
LABEL_W = 160
HEADER_H = 54

# Inner white margin inside each image panel.
# This is the real fix for the problem where boxes close to the left image edge
# are partly covered by the grid/border lines.
PANEL_INNER_PAD = 10
PANEL_IMG_SIZE = PANEL_SIZE - 2 * PANEL_INNER_PAD

GRID_LINE_W = 6

# Left-side class label position. Keep it close to image panels,
# while leaving enough white space for larger rotated text.
ROW_LABEL_X = 112

COLOR_GT = (45, 130, 230)          # Blue
COLOR_CORRECT = (70, 180, 100)       # Green (moderate deep)
COLOR_WRONG = (255, 230, 50)       # Yellow
COLOR_MISS = (220, 45, 45)         # Red
COLOR_WHITE = (255, 255, 255)
COLOR_BG = (255, 255, 255)

BOX_INNER_W = 8

# The original code used a yellow text color for false-detection labels.
COLOR_DARK_YELLOW = (180, 155, 15)


# =========================================================
# 2. Font helpers
# =========================================================
def _fc_match(query: str) -> Optional[str]:
    """Find a font path with fontconfig on Linux, if available."""
    try:
        out = subprocess.check_output(
            ["fc-match", "-f", "%{file}\n", query],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out and os.path.exists(out):
            return out
    except Exception:
        return None
    return None


def _first_existing(paths: Iterable[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _find_times_font(size: int = 18, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    Prefer real Times New Roman. If it is unavailable on Linux, use Tinos,
    Nimbus Roman, Liberation Serif, or DejaVu Serif as Times-compatible fallbacks.
    """
    regular_candidates = [
        # Windows
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\Times.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        # Linux common substitutes
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/tinos/Tinos-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    bold_candidates = [
        # Windows
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\Timesbd.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/Library/Fonts/Times New Roman Bold.ttf",
        # Linux common substitutes
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
        "/usr/share/fonts/truetype/tinos/Tinos-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ]

    candidates = bold_candidates if bold else regular_candidates
    font_path = _first_existing(candidates)

    if font_path is None:
        query_list = [
            "Times New Roman:style=Bold" if bold else "Times New Roman",
            "Tinos:style=Bold" if bold else "Tinos",
            "Nimbus Roman:style=Bold" if bold else "Nimbus Roman",
            "Liberation Serif:style=Bold" if bold else "Liberation Serif",
            "DejaVu Serif:style=Bold" if bold else "DejaVu Serif",
        ]
        for q in query_list:
            font_path = _fc_match(q)
            if font_path:
                break

    if font_path:
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def _find_cjk_font(size: int = 18, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Fallback font for occasional Chinese labels such as '误检'."""
    candidates = [
        # Windows
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
    ]
    font_path = _first_existing(candidates)
    if font_path is None:
        for q in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimSun", "PingFang SC"]:
            font_path = _fc_match(q)
            if font_path:
                break
    if font_path:
        return ImageFont.truetype(font_path, size)
    return _find_times_font(size, bold=bold)


def _has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


# All English text uses Times New Roman or a Times-compatible fallback.
FONT_TITLE = _find_times_font(40, bold=True)
FONT_ROW = _find_times_font(41, bold=True)
FONT_LABEL_EN = _find_times_font(22, bold=True)
FONT_LABEL_CJK = _find_cjk_font(22, bold=True)


def _label_font(text: str):
    return FONT_LABEL_CJK if _has_non_ascii(text) else FONT_LABEL_EN


# =========================================================
# 3. Drawing helpers
# =========================================================
def draw_box(panel, box, color, sx, sy, ox=0, oy=0):
    """Draw a bounding box with an offset inside the panel."""
    x1 = int(round(box[0] * sx)) + ox
    y1 = int(round(box[1] * sy)) + oy
    x2 = int(round(box[2] * sx)) + ox
    y2 = int(round(box[3] * sy)) + oy

    # Keep the full box inside the panel.
    min_xy = BOX_INNER_W + 1
    max_xy = PANEL_SIZE - BOX_INNER_W - 2
    x1 = max(min_xy, min(max_xy, x1))
    y1 = max(min_xy, min(max_xy, y1))
    x2 = max(min_xy, min(max_xy, x2))
    y2 = max(min_xy, min(max_xy, y2))

    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return

    draw = ImageDraw.Draw(panel)
    for i in range(BOX_INNER_W):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)


def draw_rotated_text(canvas, xy, text, font, fill=(0, 0, 0), angle=90):
    lines = text.split("\n")
    lw_list, lh_list = [], []
    dummy = Image.new("RGB", (10, 10), "white")
    d = ImageDraw.Draw(dummy)
    for line in lines:
        b = d.textbbox((0, 0), line, font=font)
        lw_list.append(b[2] - b[0])
        lh_list.append(b[3] - b[1])

    tw = max(lw_list)
    th = sum(lh_list) + (len(lines) - 1) * 6
    txt = Image.new("RGBA", (tw + 16, th + 16), (255, 255, 255, 0))
    td = ImageDraw.Draw(txt)
    yp = 8
    for i, line in enumerate(lines):
        b = td.textbbox((0, 0), line, font=font)
        lw = b[2] - b[0]
        td.text(((tw - lw) / 2 + 8, yp), line, font=font, fill=fill)
        yp += lh_list[i] + 6

    rot = txt.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    x, y = xy
    canvas.paste(rot, (int(x - rot.width / 2), int(y - rot.height / 2)), rot)


def draw_centered_text(draw, box, text, font, fill=(0, 0, 0)):
    x1, y1, x2, y2 = box
    b = draw.textbbox((0, 0), text, font=font)
    tw, th = b[2] - b[0], b[3] - b[1]
    draw.text(
        (x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2),
        text,
        font=font,
        fill=fill,
    )


# =========================================================
# 4. Panel drawing from annotation dicts
# =========================================================
def draw_panel_from_ann(img, annotations):
    """Draw image, boxes and optional text labels from annotations list."""
    w, h = img.size

    # White panel with an inner image area. This prevents edge boxes from being
    # hidden by grid lines or by the panel boundary.
    panel = Image.new("RGBA", (PANEL_SIZE, PANEL_SIZE), (255, 255, 255, 255))
    resized = img.resize((PANEL_IMG_SIZE, PANEL_IMG_SIZE), Image.Resampling.LANCZOS).convert("RGBA")
    panel.paste(resized, (PANEL_INNER_PAD, PANEL_INNER_PAD))

    sx, sy = PANEL_IMG_SIZE / w, PANEL_IMG_SIZE / h

    yellow_labels = []
    for ann in annotations:
        if ann.get("_commented"):
            continue
        color_name = ann.get("color", "")
        if color_name == "blue":
            color_rgb = COLOR_GT
        elif color_name == "green":
            color_rgb = COLOR_CORRECT
        elif color_name == "yellow":
            color_rgb = COLOR_WRONG
        elif color_name == "red":
            color_rgb = COLOR_MISS
        else:
            color_rgb = tuple(ann["color_rgb"])
        draw_box(panel, ann["box"], color_rgb, sx, sy, PANEL_INNER_PAD, PANEL_INNER_PAD)

        if ann.get("color") == "yellow":
            box = ann["box"]
            cx = (box[0] + box[2]) / 2 * sx + PANEL_INNER_PAD
            ty = box[1] * sy + PANEL_INNER_PAD + 1 + ann.get("text_offset", 0)
            label_text = ann.get("label", "False Detection")
            yellow_labels.append((cx, ty, label_text))

    draw = ImageDraw.Draw(panel)
    for cx, ty, label_text in yellow_labels:
        font = _label_font(label_text)
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx - tw / 2

        if ty + bbox[1] < PANEL_INNER_PAD + 2:
            ty = PANEL_INNER_PAD + 2 - bbox[1]
        if tx < PANEL_INNER_PAD + 2:
            tx = PANEL_INNER_PAD + 2
        if tx + tw > PANEL_SIZE - PANEL_INNER_PAD - 2:
            tx = PANEL_SIZE - PANEL_INNER_PAD - tw - 2

        pad = 2
        draw.rectangle(
            [tx + bbox[0] - pad, ty + bbox[1] - pad, tx + bbox[2] + pad, ty + bbox[3] + pad],
            fill=(255, 255, 255),
        )
        draw.text((tx, ty), label_text, font=font, fill=COLOR_DARK_YELLOW)

    return panel.convert("RGB")


# =========================================================
# 5. Main figure -> SVG + PDF
# =========================================================
def make_figure_svg_pdf(anno_files, col_headers, model_keys, pdf_name):
    # Layout: columns = models, rows = classes
    row_labels = [class_display for class_display, _, _ in anno_files]

    n_cols = len(col_headers)
    n_rows = len(row_labels)
    grid_w = n_cols * PANEL_SIZE
    grid_h = n_rows * PANEL_SIZE
    canvas_w = LABEL_W + grid_w
    canvas_h = HEADER_H + grid_h

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (*COLOR_BG, 255))
    draw = ImageDraw.Draw(canvas)

    # Column titles
    for c, header in enumerate(col_headers):
        x1 = LABEL_W + c * PANEL_SIZE
        x2 = x1 + PANEL_SIZE
        draw_centered_text(
            draw,
            (x1, -2 + TITLE_Y_SHIFT_PX, x2, HEADER_H - 2 + TITLE_Y_SHIFT_PX),
            header,
            FONT_TITLE,
            fill=(0, 0, 0),
        )

    # Grid panels: rows = classes, cols = models
    for r, (class_display, img_name, rows_data) in enumerate(anno_files):
        img_path = IMG_DIR / img_name
        img = Image.open(img_path).convert("RGB")
        for c, key in enumerate(model_keys):
            anns = rows_data.get(key, [])
            panel = draw_panel_from_ann(img, anns)
            x = LABEL_W + c * PANEL_SIZE
            y = HEADER_H + r * PANEL_SIZE
            canvas.paste(panel, (x, y))

    # Row labels. Draw after panels so labels are always on top, and place them
    # far from the image boundary.
    for r, label in enumerate(row_labels):
        y1 = HEADER_H + r * PANEL_SIZE
        draw_rotated_text(
            canvas,
            (ROW_LABEL_X + ROW_LABEL_X_SHIFT_PX, y1 + PANEL_SIZE // 2),
            label,
            FONT_ROW,
            fill=(0, 0, 0),
            angle=90,
        )

    # Grid lines and outer border. They no longer cover boxes because each panel
    # has PANEL_INNER_PAD.
    for c in range(n_cols + 1):
        x = LABEL_W + c * PANEL_SIZE
        draw.line([(x, HEADER_H), (x, HEADER_H + grid_h)], fill=COLOR_WHITE, width=GRID_LINE_W)
    for r in range(n_rows + 1):
        y = HEADER_H + r * PANEL_SIZE
        draw.line([(LABEL_W, y), (LABEL_W + grid_w, y)], fill=COLOR_WHITE, width=GRID_LINE_W)
    draw.rectangle([LABEL_W, HEADER_H, LABEL_W + grid_w, HEADER_H + grid_h], outline=(40, 40, 40), width=3)

    # Ensure white RGB background
    if canvas.mode == "RGBA":
        bg = Image.new("RGB", canvas.size, (255, 255, 255))
        bg.paste(canvas, mask=canvas.split()[-1])
        canvas = bg
    elif canvas.mode != "RGB":
        bg = Image.new("RGB", canvas.size, (255, 255, 255))
        bg.paste(canvas)
        canvas = bg

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / pdf_name

    canvas.save(pdf_path, format="PDF", resolution=EXPORT_DPI)

    print(f"[INFO] PDF saved to: {pdf_path}")


# =========================================================
# 6. Main
# =========================================================
def main():
    print("=" * 80)
    print("Generate NEU-DET comparison SVG/PDF from saved annotations")
    print("=" * 80)

    json_files = sorted(ANNOTATIONS_DIR.glob("*_boxes.json"))
    if not json_files:
        print(f"[ERROR] No annotation files found in {ANNOTATIONS_DIR}")
        return

    print(f"[INFO] Found {len(json_files)} annotation files")

    anno_data = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Keep all classes for flexible layout selection
        anno_data.append((
            data["class"],
            data["image"],
            data["rows"],
        ))
        print(f"  {jf.name}: {data['class']} -- {data['image']}")

    # Reorder to match CLASS_NAMES order if possible
    CLASS_NAMES_ORDER = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]
    CLASS_DISPLAY_MAP = {
        "Crazing": "crazing",
        "Inclusion": "inclusion",
        "Patches": "patches",
        "Pitted": "pitted_surface",
        "Pitted Surface": "pitted_surface",
        "Rolled-in": "rolled-in_scale",
        "Scratches": "scratches",
    }
    order_map = {cn: i for i, cn in enumerate(CLASS_NAMES_ORDER)}
    anno_data.sort(key=lambda x: order_map.get(CLASS_DISPLAY_MAP.get(x[0], ""), 99))

    # Generate 6-column version (3 classes, horizontal layout)
    col_headers_6 = ["Ground Truth", "YOLOv10n", "YOLOv11n", "YOLOv26x", "RT-DETR", "SSP-RT-DETR"]
    model_keys_6 = ["Ground Truth", "YOLO10n", "YOLO11n", "YOLO26l", "RT-DETR", "SSP-RT-DETR"]
    anno_data_3cls = [x for x in anno_data if x[0] in ("Inclusion", "Patches", "Pitted")]
    make_figure_svg_pdf(anno_data_3cls, col_headers_6, model_keys_6,
                        "1neudet_comparison_all_models.pdf")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
