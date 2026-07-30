"""ATSS and TaskAligned assigners ported from RT-DETRv3 (Paddle) to PyTorch.

Used by PPYOLOEHead for dense label assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_iou

from src.core import register


__all__ = ['ATSSAssigner', 'TaskAlignedAssigner']


def batch_iou_similarity(boxes1, boxes2):
    iou, _ = box_iou(boxes1.flatten(0, 1),
                     boxes2.flatten(0, 1))
    return iou


def box_center(boxes):
    cx = (boxes[..., 0] + boxes[..., 2]) / 2
    cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([cx, cy], dim=-1)


def batch_distance2bbox(points, distance):
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generate_anchors_for_grid_cell(feats, fpn_strides, grid_cell_scale=5.0,
                                   grid_cell_offset=0.5):
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor_list = []

    dtype = feats[0].dtype
    device = feats[0].device

    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        shift_x = torch.arange(w, dtype=dtype, device=device) + grid_cell_offset
        shift_y = torch.arange(h, dtype=dtype, device=device) + grid_cell_offset
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        anchor_point = torch.stack([shift_x, shift_y], dim=-1)
        anchor_points.append(anchor_point)

        cell_half = stride * grid_cell_scale / 2.0
        anchors_per_level = torch.stack([
            shift_x * stride - cell_half,
            shift_y * stride - cell_half,
            shift_x * stride + cell_half,
            shift_y * stride + cell_half,
        ], dim=-1)
        anchors.append(anchors_per_level)

        num_anchors_list.append(h * w)
        stride_tensor_list.append(stride * torch.ones(h * w, 1, dtype=dtype, device=device))

    anchors = torch.cat(anchors, dim=0)
    anchor_points = torch.cat(anchor_points, dim=0)
    stride_tensor = torch.cat(stride_tensor_list, dim=0)
    return anchors, anchor_points, num_anchors_list, stride_tensor


def check_points_inside_bboxes(points, boxes):
    eps = 1e-9
    points_3d = points.unsqueeze(0).unsqueeze(0)
    boxes_4d = boxes.unsqueeze(2)

    lt = boxes_4d[..., :2]
    rb = boxes_4d[..., 2:]

    in_lt = (points_3d - lt) > eps
    in_rb = (rb - points_3d) > eps
    in_box = torch.cat([in_lt, in_rb], dim=-1).all(dim=-1)
    return in_box


@register
class ATSSAssigner(nn.Module):
    __shared__ = ['num_classes']

    def __init__(self, topk=9, num_classes=80, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.eps = eps

    @torch.no_grad()
    def forward(self, anchor_bboxes, num_anchors_list, gt_labels, gt_bboxes,
                pad_gt_mask, bg_index, gt_scores=None, pred_bboxes=None):
        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index,
                                         dtype=torch.long, device=gt_bboxes.device)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4], device=gt_bboxes.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, self.num_classes],
                                          device=gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores

        anchor_xyxy = anchor_bboxes
        gt_xyxy = gt_bboxes
        ious = batch_iou_similarity(gt_xyxy, anchor_xyxy)

        gt_center = box_center(gt_xyxy)
        ac_center = box_center(anchor_xyxy.unsqueeze(0))

        gt_to_anchor_dists = torch.cdist(gt_center, ac_center, p=2)

        is_in_topk_list = []
        topk_idxs_list = []
        num_anchors_cumsum = [0] + list(torch.tensor(num_anchors_list).cumsum(0).tolist())
        for level_idx in range(len(num_anchors_list)):
            start = num_anchors_cumsum[level_idx]
            end = num_anchors_cumsum[level_idx + 1]
            distances = gt_to_anchor_dists[:, :, start:end]
            k = min(self.topk, distances.shape[-1])
            _, topk_idxs = torch.topk(distances, k, dim=-1, largest=False)
            topk_idxs_list.append(topk_idxs + start)

            is_in_topk = F.one_hot(topk_idxs, distances.shape[-1]).sum(dim=-2).float()
            is_in_topk_list.append(is_in_topk * pad_gt_mask.unsqueeze(-1))

        is_in_topk_all = torch.cat(is_in_topk_list, dim=-1)
        topk_idxs_all = torch.cat(topk_idxs_list, dim=-1)

        topk_ious = torch.gather(ious, -1, topk_idxs_all)
        topk_ious_mean = topk_ious.mean(dim=-1, keepdim=True)
        topk_ious_std = topk_ious.std(dim=-1, keepdim=True)
        iou_thr = topk_ious_mean + topk_ious_std

        is_in_gts = check_points_inside_bboxes(ac_center.squeeze(0), gt_xyxy)

        candidate_mask = is_in_topk_all * (ious >= iou_thr).float() * is_in_gts.float()

        _, max_iou_gt_idx = torch.max(ious, dim=1)
        assigned_labels = torch.full([batch_size, num_anchors], bg_index,
                                     dtype=torch.long, device=gt_bboxes.device)
        assigned_bboxes = torch.zeros([batch_size, num_anchors, 4], device=gt_bboxes.device)
        assigned_scores = torch.zeros([batch_size, num_anchors, self.num_classes],
                                      device=gt_bboxes.device)

        for b in range(batch_size):
            active = candidate_mask[b].sum(dim=0) > 0
            if active.any():
                _, best_gt = torch.max(ious[b], dim=0)
                for gt_idx in range(num_max_boxes):
                    if pad_gt_mask[b, gt_idx].item() == 0:
                        continue
                    pos_mask = candidate_mask[b, gt_idx] > 0
                    pos_mask = pos_mask & (best_gt == gt_idx)
                    if pos_mask.sum() == 0:
                        best_iou, best_idx = ious[b, gt_idx].max(dim=0)
                        if best_iou > 0.1:
                            pos_mask[best_idx] = True
                    assigned_labels[b, pos_mask] = gt_labels[b, gt_idx].item()
                    assigned_bboxes[b, pos_mask] = gt_bboxes[b, gt_idx]
                    assigned_scores[b, pos_mask, gt_labels[b, gt_idx].item()] = ious[b, gt_idx, pos_mask]

        return assigned_labels, assigned_bboxes, assigned_scores


@register
class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anchor_points, num_anchors_list,
                gt_labels, gt_bboxes, pad_gt_mask, bg_index, gt_scores=None):
        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index,
                                         dtype=torch.long, device=gt_bboxes.device)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4], device=gt_bboxes.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes],
                                          device=gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores

        pred_xyxy = pred_bboxes
        gt_xyxy = gt_bboxes
        ious = batch_iou_similarity(gt_xyxy, pred_xyxy)

        pred_scores_t = pred_scores.transpose(1, 2)
        gt_labels_t = gt_labels.squeeze(-1).long()
        batch_idx = torch.arange(batch_size, device=gt_labels.device).unsqueeze(-1)
        gather_idx = torch.stack([
            batch_idx.expand(-1, num_max_boxes),
            gt_labels_t,
        ], dim=-1)
        bbox_cls_scores = pred_scores_t[gather_idx[..., 0], gather_idx[..., 1]]

        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        is_in_gts = check_points_inside_bboxes(anchor_points, gt_xyxy)

        is_in_topk_list = []
        num_anchors_cumsum = [0] + list(torch.tensor(num_anchors_list).cumsum(0).tolist())
        for level_idx in range(len(num_anchors_list)):
            start = num_anchors_cumsum[level_idx]
            end = num_anchors_cumsum[level_idx + 1]
            metrics = alignment_metrics[:, :, start:end]
            in_gts = is_in_gts[:, :, start:end]
            masked_metrics = metrics * in_gts.float()
            k = min(self.topk, metrics.shape[-1])
            _, topk_idxs = torch.topk(masked_metrics, k, dim=-1)
            is_in_topk = F.one_hot(topk_idxs, metrics.shape[-1]).sum(dim=-2).float()
            is_in_topk_list.append(is_in_topk * pad_gt_mask.unsqueeze(-1))

        is_in_topk_all = torch.cat(is_in_topk_list, dim=-1)

        assigned_labels = torch.full([batch_size, num_anchors], bg_index,
                                     dtype=torch.long, device=gt_bboxes.device)
        assigned_bboxes = torch.zeros([batch_size, num_anchors, 4], device=gt_bboxes.device)
        assigned_scores = torch.zeros([batch_size, num_anchors, num_classes],
                                      device=gt_bboxes.device)

        for b in range(batch_size):
            active = is_in_topk_all[b].sum(dim=0) > 0
            if active.any():
                _, best_gt = torch.max(ious[b] * is_in_gts[b].float(), dim=0)
                for gt_idx in range(num_max_boxes):
                    if pad_gt_mask[b, gt_idx].item() == 0:
                        continue
                    pos_mask = (is_in_topk_all[b, gt_idx] > 0) & (best_gt == gt_idx)
                    if pos_mask.sum() > 0:
                        assigned_labels[b, pos_mask] = gt_labels[b, gt_idx].item()
                        assigned_bboxes[b, pos_mask] = gt_bboxes[b, gt_idx]
                        assigned_scores[b, pos_mask, gt_labels[b, gt_idx].item()] = \
                            ious[b, gt_idx, pos_mask]

        return assigned_labels, assigned_bboxes, assigned_scores
