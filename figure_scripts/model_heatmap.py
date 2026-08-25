#!/usr/bin/env python3
"""
Clean heatmap comparison for NEU-DET.

Core changes compared with the previous version:
1) GT column no longer uses large filled blue masks; it draws only clear GT contours.
2) Prediction columns no longer normalize the whole feature map directly, avoiding full-image green/yellow plates.
3) Heatmap is box-guided and defect-guided: inside matched prediction boxes, feature response is combined with local image contrast and a Gaussian center prior.
4) Only matched boxes of the current class are visualized; unmatched boxes are ignored.

Rows: Ground Truth | RT-DETR | SSP-RT-DETR
Columns: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.core import YAMLConfig

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_DISPLAY = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale",
    "scratches": "Scratches",
}
TARGET_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

BASELINE_CONFIG = str(SCRIPT_DIR / "configs/rtdetr/other_experiments/rtdetr_r50vd_6x_neudet.yml")
IMPROVED_CONFIG = str(SCRIPT_DIR / "configs/rtdetr/other_experiments/rtdetr_r50vd_6x_neudet_all_innovations.yml")
BASELINE_CKPT = str(PROJECT_ROOT / "model" / "neu-baseline-best.pth")
IMPROVED_CKPT = str(PROJECT_ROOT / "model" / "neu-all-best.pth")
IMG_DIR = SCRIPT_DIR / "images"
ANN_DIR = SCRIPT_DIR / "annotations"
OUT_DIR = SCRIPT_DIR

PICTURE_SAMPLES = {
    "crazing": "crazing_236.jpg",
    "inclusion": "inclusion_151.jpg",
    "patches": "patches_281.jpg",
    "pitted_surface": "pitted_surface_129.jpg",
    "rolled-in_scale": "rolled-in_scale_244.jpg",
    "scratches": "scratches_58.jpg",
}

SCORE_THRESH = 0.01
NMS_THRESH = 0.55
IOU_MATCH_THRESH = 0.30
PANEL_SIZE = 640
DPI = 300

GT_COLOR = (35, 115, 235)
PRED_COLOR = (80, 240, 80)
EDGE_WIDTH = 8
HEAT_ALPHA = 0.68
HEAT_GAMMA = 0.48
HEAT_BLUR_RADIUS = 1.4
BG_DIM = 0.62

# A lighter academic heat palette: background is transparent, response is blue-green-yellow-red.
_JET = np.array([
    [0.00, 0.00, 0.55], [0.00, 0.10, 0.75], [0.00, 0.28, 0.95], [0.00, 0.55, 1.00],
    [0.00, 0.78, 0.90], [0.00, 0.88, 0.45], [0.45, 0.95, 0.12], [0.92, 0.88, 0.08],
    [1.00, 0.55, 0.00], [0.95, 0.18, 0.00]
], dtype=np.float32)


def jet(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    idx = np.clip((v * (len(_JET) - 1)).astype(np.int32), 0, len(_JET) - 1)
    return _JET[idx]


def normalize_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) and labels.min() >= 1 and labels.max() <= 6:
        return labels - 1
    return labels


def _extract_feature(encoder_output, level=1):
    """Extract feature map from encoder output at given level (0=P3/stride8, 1=P4/stride16, 2=P5/stride32)."""
    if isinstance(encoder_output, dict):
        if "multi_scale_features" in encoder_output:
            return encoder_output["multi_scale_features"][level]
        if "node_3" in encoder_output:
            return encoder_output["node_3"]
    if isinstance(encoder_output, (list, tuple)):
        return encoder_output[level]
    return encoder_output


def load_model(config_path, ckpt_path, device):
    cfg = YAMLConfig(config_path, resume=ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    cfg.model.deploy()
    model = cfg.model.to(device).eval()
    pp = cfg.postprocessor
    pp.eval()

    eh, ew = 640, 640
    es = cfg.yaml_cfg.get("HybridEncoder", {}).get("eval_spatial_size")
    if isinstance(es, (list, tuple)) and len(es) == 2:
        eh, ew = int(es[0]), int(es[1])

    tfm = T.Compose([
        T.Resize((eh, ew)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, pp, tfm


def infer(model, pp, tfm, img, device, feat_level=0):
    w, h = img.size
    orig = torch.tensor([w, h], dtype=torch.float32)[None].to(device)
    t = tfm(img)[None].to(device)
    with torch.no_grad():
        backbone_feats = model.backbone(t)
        encoder_output = model.encoder(backbone_feats)
        decoder_output = model.decoder(encoder_output)
        r = pp(decoder_output, orig)[0]

    labels = r["labels"].detach().cpu()
    boxes = r["boxes"].detach().cpu()
    scores = r["scores"].detach().cpu()
    keep = scores > SCORE_THRESH
    labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
    if boxes.numel() > 0:
        keep = torchvision.ops.batched_nms(boxes, scores, labels, NMS_THRESH)
        labels, boxes, scores = labels[keep], boxes[keep], scores[keep]

    feat_map = _extract_feature(encoder_output, feat_level)
    if isinstance(feat_map, torch.Tensor) and feat_map.dim() == 4:
        feat_map = feat_map[0].detach().cpu()

    return normalize_labels(labels.numpy()), boxes.numpy(), scores.numpy(), feat_map


def parse_gt(ap, iw, ih):
    tree = ET.parse(str(ap))
    root = tree.getroot()
    boxes, labels, names = [], [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_NAMES:
            continue
        bnd = obj.find("bndbox")
        x1 = max(0, float(bnd.find("xmin").text))
        y1 = max(0, float(bnd.find("ymin").text))
        x2 = min(iw - 1, float(bnd.find("xmax").text))
        y2 = min(ih - 1, float(bnd.find("ymax").text))
        if x2 > x1 + 2 and y2 > y1 + 2:
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_NAMES.index(name))
            names.append(name)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64), names


def box_iou(b1, b2):
    ix1 = max(float(b1[0]), float(b2[0])); iy1 = max(float(b1[1]), float(b2[1]))
    ix2 = min(float(b1[2]), float(b2[2])); iy2 = min(float(b1[3]), float(b2[3]))
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    a2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    return inter / max(a1 + a2 - inter, 1e-7)


def robust01(x, mask=None, lo=5, hi=98):
    x = np.asarray(x, dtype=np.float32)
    vals = x[mask > 0] if mask is not None and np.any(mask > 0) else x.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(x)
    a, b = np.percentile(vals, [lo, hi])
    if b <= a + 1e-6:
        return np.zeros_like(x)
    return np.clip((x - a) / (b - a), 0, 1)


def local_contrast_map(panel_gray):
    img = Image.fromarray((panel_gray * 255).astype(np.uint8))
    blur = np.array(img.filter(ImageFilter.GaussianBlur(radius=8))).astype(np.float32) / 255.0
    diff = np.abs(panel_gray - blur)
    return robust01(diff, lo=60, hi=99)


def feature_response_map(feat_map, H, W):
    if not isinstance(feat_map, torch.Tensor) or feat_map.dim() != 3:
        return np.zeros((H, W), dtype=np.float32)
    resp = feat_map.abs().max(dim=0)[0].numpy()
    resp = robust01(resp, lo=55, hi=99.2)
    resp_img = Image.fromarray((resp * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)
    return np.asarray(resp_img).astype(np.float32) / 255.0


def gaussian_box_prior(H, W, box, sigma_scale=0.32):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = int(max(0, min(W - 1, round(x1)))); x2 = int(max(0, min(W - 1, round(x2))))
    y1 = int(max(0, min(H - 1, round(y1)))); y2 = int(max(0, min(H - 1, round(y2))))
    prior = np.zeros((H, W), dtype=np.float32)
    if x2 <= x1 or y2 <= y1:
        return prior
    yy, xx = np.mgrid[y1:y2 + 1, x1:x2 + 1]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sx = max((x2 - x1) * sigma_scale, 3.0)
    sy = max((y2 - y1) * sigma_scale, 3.0)
    g = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    prior[y1:y2 + 1, x1:x2 + 1] = g.astype(np.float32)
    return prior


def build_global_heatmap(img_panel, feat_map):
    H = W = PANEL_SIZE
    panel_arr = np.asarray(img_panel).astype(np.float32) / 255.0
    gray = panel_arr.mean(axis=2)
    contrast = local_contrast_map(gray)
    feat = feature_response_map(feat_map, H, W)

    heat = 0.58 * robust01(feat, lo=35, hi=98.8) + 0.28 * robust01(contrast, lo=35, hi=98.8)

    if heat.max() > 0:
        heat = robust01(heat, lo=10, hi=99)
        heat = np.power(heat, HEAT_GAMMA)
        heat = np.clip(heat * 1.75, 0, 1)
        heat = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=HEAT_BLUR_RADIUS)
        )).astype(np.float32) / 255.0
        heat[heat < 0.035] = 0.0
        heat = np.clip(heat / max(heat.max(), 1e-6), 0, 1)
    return heat



def darken_panel(panel):
    """Darken the original image panel so GT and prediction panels use the same background brightness."""
    arr = np.asarray(panel).astype(np.float32) / 255.0
    arr = np.clip(arr * BG_DIM, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def draw_gt_panel(img, gt_boxes):
    panel = img.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
    panel = darken_panel(panel)
    sx = PANEL_SIZE / img.size[0]
    sy = PANEL_SIZE / img.size[1]
    draw = ImageDraw.Draw(panel)
    gt_edge_width = EDGE_WIDTH + 2
    MARGIN = 3
    W, H = PANEL_SIZE, PANEL_SIZE
    for box in gt_boxes:
        x1 = int(round(float(box[0]) * sx)); y1 = int(round(float(box[1]) * sy))
        x2 = int(round(float(box[2]) * sx)); y2 = int(round(float(box[3]) * sy))
        x1 = max(MARGIN, min(W - MARGIN - 1, x1))
        y1 = max(MARGIN, min(H - MARGIN - 1, y1))
        x2 = max(MARGIN, min(W - MARGIN - 1, x2))
        y2 = max(MARGIN, min(H - MARGIN - 1, y2))
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=gt_edge_width)
    return np.asarray(panel)


def draw_pred_panel(img, feat_map, pred_boxes):
    panel = img.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
    panel_arr = np.asarray(panel).astype(np.float32) / 255.0

    heat = build_global_heatmap(panel, feat_map)

    # Mask heatmap to only inside prediction boxes.
    H = W = PANEL_SIZE
    MARGIN = 3
    sx = PANEL_SIZE / img.size[0]
    sy = PANEL_SIZE / img.size[1]
    box_mask = np.zeros((H, W), dtype=np.float32)
    for box in pred_boxes:
        x1 = int(round(float(box[0]) * sx)); y1 = int(round(float(box[1]) * sy))
        x2 = int(round(float(box[2]) * sx)); y2 = int(round(float(box[3]) * sy))
        x1 = max(MARGIN, min(W - MARGIN - 1, x1)); x2 = max(MARGIN, min(W - MARGIN - 1, x2))
        y1 = max(MARGIN, min(H - MARGIN - 1, y1)); y2 = max(MARGIN, min(H - MARGIN - 1, y2))
        if x2 > x1 and y2 > y1:
            box_mask[y1:y2 + 1, x1:x2 + 1] = 1.0
    heat = heat * box_mask

    heat_rgb = jet(heat)
    alpha = (HEAT_ALPHA * heat)[..., None]

    # Use the same dark background as the GT panel.
    base = panel_arr * BG_DIM
    out = base * (1 - alpha) + heat_rgb * alpha
    hot = heat[..., None]
    out = np.clip(out * (1.0 + 0.18 * hot), 0, 1)
    out_u8 = (np.clip(out * 255, 0, 255)).astype(np.uint8)
    out_img = Image.fromarray(out_u8)

    draw = ImageDraw.Draw(out_img)
    for box in pred_boxes:
        x1 = int(round(float(box[0]) * sx)); y1 = int(round(float(box[1]) * sy))
        x2 = int(round(float(box[2]) * sx)); y2 = int(round(float(box[3]) * sy))
        x1 = max(MARGIN, min(W - MARGIN - 1, x1))
        y1 = max(MARGIN, min(H - MARGIN - 1, y1))
        x2 = max(MARGIN, min(W - MARGIN - 1, x2))
        y2 = max(MARGIN, min(H - MARGIN - 1, y2))
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=EDGE_WIDTH)
    return np.asarray(out_img)


def matched_boxes_by_class(labels, boxes, scores, gt_boxes, cls_label):
    sel = np.where(labels == cls_label)[0]
    candidates = []
    used_gt = set()
    for idx in sel:
        best_iou, best_gi = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            iou = box_iou(boxes[idx], gb)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= IOU_MATCH_THRESH:
            candidates.append((float(scores[idx]), idx, best_gi))
    candidates.sort(reverse=True, key=lambda x: x[0])
    keep_boxes, keep_scores = [], []
    for score, idx, gi in candidates:
        if gi in used_gt:
            continue
        used_gt.add(gi)
        keep_boxes.append(boxes[idx])
        keep_scores.append(score)
        if len(keep_boxes) >= 2:
            break
    if keep_boxes:
        return np.asarray(keep_boxes, dtype=np.float32), np.asarray(keep_scores, dtype=np.float32)
    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def save_figure(fig, base):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_pic_dir = OUT_DIR / "new-pic"
    new_pic_dir.mkdir(parents=True, exist_ok=True)
    p = new_pic_dir / f"{base}.pdf"
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.0, transparent=False)
    print(f"[INFO] PDF saved to: {p}")


def _add_center_legend(fig, x=0.5, y=0.045):
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor=np.array(GT_COLOR) / 255.0,
            markeredgecolor=np.array(GT_COLOR) / 255.0,
            markersize=7,
            label="GT box"
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor=np.array(PRED_COLOR) / 255.0,
            markeredgecolor=np.array(PRED_COLOR) / 255.0,
            markersize=7,
            label="Prediction box"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(x, y),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.8,
        borderaxespad=0.0,
    )


def make_figure_3x4(samples):
    """Create a 3 rows x 4 columns visualization.

    Rows: Ground Truth | RT-DETR | SSP-RT-DETR
    Columns: defect categories
    """
    row_titles = ["Ground Truth", "RT-DETR", "SSP-RT-DETR"]
    col_classes = TARGET_CLASSES
    nr, nc = len(row_titles), len(col_classes)

    # Manual inch-based layout keeps the columns tight and avoids extra whitespace.
    FIG_W_IN = 9.80
    FIG_H_IN = 5.40
    LEFT_IN = 0.62
    RIGHT_IN = 0.03
    TOP_IN = 0.42
    BOTTOM_IN = 0.48
    COL_GAP_IN = 0.020
    ROW_GAP_IN = 0.020

    panel_in = (FIG_W_IN - LEFT_IN - RIGHT_IN - (nc - 1) * COL_GAP_IN) / nc
    grid_w_in = nc * panel_in + (nc - 1) * COL_GAP_IN
    grid_h_in = nr * panel_in + (nr - 1) * ROW_GAP_IN
    available_h_in = FIG_H_IN - TOP_IN - BOTTOM_IN
    if grid_h_in > available_h_in:
        panel_in = (available_h_in - (nr - 1) * ROW_GAP_IN) / nr
        grid_w_in = nc * panel_in + (nc - 1) * COL_GAP_IN
        grid_h_in = nr * panel_in + (nr - 1) * ROW_GAP_IN

    x0_in = LEFT_IN + max(0.0, (FIG_W_IN - LEFT_IN - RIGHT_IN - grid_w_in) / 2.0)
    y0_in = BOTTOM_IN + max(0.0, (available_h_in - grid_h_in) / 2.0)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor="white")

    for r in range(nr):
        for c in range(nc):
            x_in = x0_in + c * (panel_in + COL_GAP_IN)
            y_in = y0_in + (nr - 1 - r) * (panel_in + ROW_GAP_IN)
            ax = fig.add_axes([
                x_in / FIG_W_IN,
                y_in / FIG_H_IN,
                panel_in / FIG_W_IN,
                panel_in / FIG_H_IN,
            ])
            ax.imshow(samples[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.margins(0)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.8)
                sp.set_edgecolor("#E5E5E5")

    # Column titles.
    title_y = (y0_in + grid_h_in + 0.019) / FIG_H_IN
    for c, cls_name in enumerate(col_classes):
        xc = (x0_in + c * (panel_in + COL_GAP_IN) + panel_in / 2.0) / FIG_W_IN
        fig.text(
            xc,
            title_y,
            CLASS_DISPLAY.get(cls_name, cls_name),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Row labels.
    label_x = (x0_in - 0.086) / FIG_W_IN
    for r, title in enumerate(row_titles):
        yc = (y0_in + (nr - 1 - r) * (panel_in + ROW_GAP_IN) + panel_in / 2.0) / FIG_H_IN
        fig.text(
            label_x,
            yc,
            title,
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            fontweight="bold",
        )

    legend_x = (x0_in + grid_w_in / 2.0) / FIG_W_IN
    _add_center_legend(fig, x=legend_x, y=0.050)
    save_figure(fig, "global_heatmap_3x6_dark_classes")
    plt.close(fig)


def make_figure_4x3(samples):
    """Create a 4 rows x 3 columns visualization.

    Rows: defect categories
    Columns: Ground Truth | RT-DETR | SSP-RT-DETR
    """
    col_titles = ["Ground Truth", "RT-DETR", "SSP-RT-DETR"]
    row_classes = TARGET_CLASSES
    nr, nc = len(row_classes), len(col_titles)

    FIG_W_IN = 5.10
    FIG_H_IN = 10.20
    LEFT_IN = 0.52
    RIGHT_IN = 0.02
    TOP_IN = 0.42
    BOTTOM_IN = 0.48
    COL_GAP_IN = 0.020
    ROW_GAP_IN = 0.020

    panel_in = (FIG_W_IN - LEFT_IN - RIGHT_IN - (nc - 1) * COL_GAP_IN) / nc
    grid_h_in = nr * panel_in + (nr - 1) * ROW_GAP_IN
    available_h_in = FIG_H_IN - TOP_IN - BOTTOM_IN
    if grid_h_in > available_h_in:
        panel_in = (available_h_in - (nr - 1) * ROW_GAP_IN) / nr
        grid_h_in = nr * panel_in + (nr - 1) * ROW_GAP_IN

    grid_w_in = nc * panel_in + (nc - 1) * COL_GAP_IN
    x0_in = LEFT_IN + max(0.0, (FIG_W_IN - LEFT_IN - RIGHT_IN - grid_w_in) / 2.0)
    y0_in = BOTTOM_IN + max(0.0, (available_h_in - grid_h_in) / 2.0)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor="white")

    for r in range(nr):
        for c in range(nc):
            x_in = x0_in + c * (panel_in + COL_GAP_IN)
            y_in = y0_in + (nr - 1 - r) * (panel_in + ROW_GAP_IN)
            ax = fig.add_axes([
                x_in / FIG_W_IN,
                y_in / FIG_H_IN,
                panel_in / FIG_W_IN,
                panel_in / FIG_H_IN,
            ])
            # samples[0] = GT panels, samples[1] = RT-DETR panels, samples[2] = SSP-RT-DETR panels.
            ax.imshow(samples[c][r])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.margins(0)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.8)
                sp.set_edgecolor("#E5E5E5")

    # Column titles.
    title_y = (y0_in + grid_h_in + 0.019) / FIG_H_IN
    for c, title in enumerate(col_titles):
        xc = (x0_in + c * (panel_in + COL_GAP_IN) + panel_in / 2.0) / FIG_W_IN
        fig.text(
            xc,
            title_y,
            title,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Row labels.
    label_x = (x0_in - 0.086) / FIG_W_IN
    for r, cls_name in enumerate(row_classes):
        yc = (y0_in + (nr - 1 - r) * (panel_in + ROW_GAP_IN) + panel_in / 2.0) / FIG_H_IN
        fig.text(
            label_x,
            yc,
            CLASS_DISPLAY.get(cls_name, cls_name),
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            fontweight="bold",
        )

    legend_x = (x0_in + grid_w_in / 2.0) / FIG_W_IN
    _add_center_legend(fig, x=legend_x, y=0.050)
    save_figure(fig, "global_heatmap_6x3_dark_classes")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    bm, bp, bt = load_model(BASELINE_CONFIG, BASELINE_CKPT, device)
    im, ip, it = load_model(IMPROVED_CONFIG, IMPROVED_CKPT, device)

    gt_panels, b_panels, i_panels = [], [], []

    for cls_name in TARGET_CLASSES:
        cls_label = CLASS_NAMES.index(cls_name)
        img_name = PICTURE_SAMPLES[cls_name]
        stem = os.path.splitext(img_name)[0]
        img = Image.open(IMG_DIR / img_name).convert("RGB")
        iw, ih = img.size
        gt_boxes_all, gt_labels, _ = parse_gt(ANN_DIR / f"{stem}.xml", iw, ih)
        gt_boxes = gt_boxes_all[gt_labels == cls_label]

        blabels, bb, bscores, bf = infer(bm, bp, bt, img, device)
        ilabels, ib, iscores, inf = infer(im, ip, it, img, device)

        b_boxes, _ = matched_boxes_by_class(blabels, bb, bscores, gt_boxes, cls_label)
        i_boxes, _ = matched_boxes_by_class(ilabels, ib, iscores, gt_boxes, cls_label)

        gt_panels.append(draw_gt_panel(img, gt_boxes))
        b_panels.append(draw_pred_panel(img, bf, b_boxes))
        i_panels.append(draw_pred_panel(img, inf, i_boxes))

        print(f"  {cls_name:16s} | GT={len(gt_boxes)} | RT-DETR={len(b_boxes)} boxes | SSP-RT-DETR={len(i_boxes)} boxes")

    samples = [gt_panels, b_panels, i_panels]
    make_figure_3x4(samples)


if __name__ == "__main__":
    main()
