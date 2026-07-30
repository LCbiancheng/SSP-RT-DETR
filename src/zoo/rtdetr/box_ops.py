'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
'''

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    w = w.clamp(min=1e-4)
    h = h.clamp(min=1e-4)

    x0 = (x_c - 0.5 * w).clamp(min=1e-4)
    y0 = (y_c - 0.5 * h).clamp(min=1e-4)
    x1 = (x_c + 0.5 * w).clamp(max=1 - 1e-4)
    y1 = (y_c + 0.5 * h).clamp(max=1 - 1e-4)

    return torch.stack((x0, y0, x1, y1), dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        bad = (boxes1[:, 2:] < boxes1[:, :2]).any(dim=1)
        print("BAD boxes1 idx:", bad.nonzero().flatten()[:20])
        print("BAD boxes1 sample:", boxes1[bad][:5])
        print("All boxes1 stats:", boxes1.min(), boxes1.max(), boxes1.mean())
        raise RuntimeError("Invalid boxes1: x2<x1 or y2<y1")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        bad = (boxes2[:, 2:] < boxes2[:, :2]).any(dim=1)
        print("BAD boxes2 idx:", bad.nonzero().flatten()[:20])
        print("BAD boxes2 sample:", boxes2[bad][:5])
        print("All boxes2 stats:", boxes2.min(), boxes2.max(), boxes2.mean())
        raise RuntimeError("Invalid boxes2: x2<x1 or y2<y1")
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def border_giou_loss(boxes1, boxes2, lambda_border=0.05, eps=1e-7, reduction='none'):
    """Border-GIoU loss for aligned xyxy box pairs.

    This keeps the original GIoU term and adds a lightweight boundary
    deviation penalty normalized by the GT box perimeter proxy.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        empty = boxes1.new_zeros((0,))
        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(f"Unsupported reduction: {reduction!r}. Use 'none', 'mean', or 'sum'.")
        if reduction == 'mean':
            return boxes1.sum() * 0.0
        if reduction == 'sum':
            return boxes1.sum() * 0.0
        return empty
    if boxes1.shape != boxes2.shape:
        raise ValueError(
            'border_giou_loss expects aligned boxes with the same shape, '
            f'got {boxes1.shape} and {boxes2.shape}.'
        )

    giou_loss = 1.0 - torch.diag(generalized_box_iou(boxes1, boxes2))

    gt_wh = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0.0)
    # Use a larger eps and clamp to prevent loss explosion from tiny boxes (e.g., dividing by 1e-7 amplifies by 10 million times)
    border_norm = (gt_wh[:, 0] + gt_wh[:, 1]).clamp(min=1e-2)
    border_loss = (boxes1 - boxes2).abs().sum(dim=-1) / border_norm
    loss = giou_loss + float(lambda_border) * border_loss

    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    if reduction != 'none':
        raise ValueError(f"Unsupported reduction: {reduction!r}. Use 'none', 'mean', or 'sum'.")
    return loss


def aligned_box_iou(boxes1, boxes2, eps=1e-7):
    """Aligned IoU for matched box pairs with shape [N, 4]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        empty = boxes1.new_zeros((0,))
        return empty, empty, empty

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = box_area(boxes1).clamp(min=eps)
    area2 = box_area(boxes2).clamp(min=eps)
    union = (area1 + area2 - inter).clamp(min=eps)
    iou = inter / union

    center1 = (boxes1[:, :2] + boxes1[:, 2:]) * 0.5
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) * 0.5
    center_dist = ((center1 - center2) ** 2).sum(dim=-1)

    enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=eps)
    enclose_diag = (enclose_wh ** 2).sum(dim=-1).clamp(min=eps)

    return iou, center_dist, enclose_diag


def _prepare_image_wh(image_wh, num_boxes, device, dtype, eps=1e-7):
    if image_wh is None:
        image_wh = torch.ones((num_boxes, 2), device=device, dtype=dtype)
    elif image_wh.dim() == 1:
        image_wh = image_wh.unsqueeze(0).expand(num_boxes, -1)
    else:
        image_wh = image_wh.to(device=device, dtype=dtype)

    return image_wh.clamp(min=eps)


def _prepare_pairwise_image_wh(image_wh, num_targets, device, dtype, eps=1e-7):
    if image_wh is None:
        image_wh = torch.ones((num_targets, 2), device=device, dtype=dtype)
    elif image_wh.dim() == 1:
        image_wh = image_wh.unsqueeze(0).expand(num_targets, -1)
    else:
        image_wh = image_wh.to(device=device, dtype=dtype)

    return image_wh.clamp(min=eps)


def _pairwise_mpdiou_similarity(boxes1, boxes2, image_wh=None, eps=1e-7):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        shape = (boxes1.shape[0], boxes2.shape[0])
        empty = boxes1.new_zeros(shape)
        return empty, empty

    pair_iou, _ = box_iou(boxes1, boxes2)
    image_wh = _prepare_pairwise_image_wh(image_wh, boxes2.shape[0], boxes1.device, boxes1.dtype, eps=eps)

    tl_delta = (boxes1[:, None, :2] - boxes2[None, :, :2]) * image_wh[None, :, :]
    br_delta = (boxes1[:, None, 2:] - boxes2[None, :, 2:]) * image_wh[None, :, :]
    corner_dist = (tl_delta ** 2).sum(dim=-1) + (br_delta ** 2).sum(dim=-1)
    norm = (image_wh[:, 0] ** 2 + image_wh[:, 1] ** 2).clamp(min=eps).unsqueeze(0)

    pair_mpdiou = pair_iou - corner_dist / norm
    return pair_mpdiou, pair_iou


def pairwise_mpdiou_cost(boxes1, boxes2, image_wh=None, eps=1e-7):
    """Pairwise MPDIoU-style localization cost for Hungarian matching.

    `boxes1` and `boxes2` are expected in normalized `xyxy` format. `image_wh`
    provides the width/height used to convert corner deltas into absolute scale
    for each target column. Only same-image columns are consumed after the
    batch-wise split in Hungarian matching, so per-target widths/heights are
    sufficient here.
    """
    pair_mpdiou, _ = _pairwise_mpdiou_similarity(boxes1, boxes2, image_wh=image_wh, eps=eps)
    return 1.0 - pair_mpdiou


def pairwise_focal_mpdiou_cost(boxes1, boxes2, image_wh=None, gamma=0.5, eps=1e-7):
    """Pairwise Focal-MPDIoU localization cost for Hungarian matching."""
    pair_mpdiou, pair_iou = _pairwise_mpdiou_similarity(boxes1, boxes2, image_wh=image_wh, eps=eps)
    if pair_mpdiou.numel() == 0:
        return pair_mpdiou

    focal_weight = pair_iou.clamp(min=eps).pow(gamma)
    return focal_weight * (1.0 - pair_mpdiou)


def mpdiou_similarity(boxes1, boxes2, image_wh=None, eps=1e-7):
    """Minimum Point Distance IoU (MPDIoU) for aligned box pairs.

    Reference:
        Siliang Ma, Yong Xu, "MPDIoU: A Loss for Efficient and Accurate
        Bounding Box Regression" (arXiv:2307.07662).

    Formula:
        MPDIoU = IoU - (d_tl^2 + d_br^2) / (W^2 + H^2)

    where d_tl / d_br are the squared distances between the matched top-left
    and bottom-right corners, and W / H denote the image width and height.

    This metric is useful for industrial defects, elongated boxes, and
    boundary-sensitive localization because it penalizes corner deviations
    directly instead of relying only on overlap geometry.
    """
    iou, _, _ = aligned_box_iou(boxes1, boxes2, eps=eps)
    if iou.numel() == 0:
        return iou, iou

    image_wh = _prepare_image_wh(image_wh, boxes1.shape[0], boxes1.device, boxes1.dtype, eps=eps)

    corner_tl_dist = ((boxes1[:, :2] - boxes2[:, :2]) ** 2).sum(dim=-1)
    corner_br_dist = ((boxes1[:, 2:] - boxes2[:, 2:]) ** 2).sum(dim=-1)
    norm = (image_wh[:, 0] ** 2 + image_wh[:, 1] ** 2).clamp(min=eps)

    mpdiou = iou - (corner_tl_dist + corner_br_dist) / norm
    return mpdiou, iou


def mpdiou_loss(boxes1, boxes2, image_wh=None, eps=1e-7):
    """Loss form of MPDIoU for aligned box pairs.

    L_MPDIoU = 1 - MPDIoU
    """
    mpdiou, iou = mpdiou_similarity(boxes1, boxes2, image_wh=image_wh, eps=eps)
    if mpdiou.numel() == 0:
        return mpdiou, mpdiou, iou
    return 1.0 - mpdiou, mpdiou, iou


def focal_mpdiou_loss(boxes1, boxes2, image_wh=None, gamma=0.5, eps=1e-7):
    """Focal-MPDIoU for aligned box pairs.

    This is a pragmatic focal extension of MPDIoU that follows the
    Focal-EIoU-style quality reweighting:

        L_Focal-MPDIoU = IoU^gamma * L_MPDIoU

    It is useful when the task values high-quality boundary refinement more
    than aggressively fitting every hard positive, which often matches
    industrial defect detection with thin or edge-sensitive boxes.
    """
    base_loss, mpdiou, iou = mpdiou_loss(boxes1, boxes2, image_wh=image_wh, eps=eps)
    if base_loss.numel() == 0:
        return base_loss, mpdiou, iou

    focal_weight = iou.clamp(min=eps).pow(gamma)
    return focal_weight * base_loss, mpdiou, iou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
