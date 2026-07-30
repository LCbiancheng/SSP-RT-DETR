"""PPYOLOEHead - Dense auxiliary detection head from RT-DETRv3.

Ported from PaddlePaddle to PyTorch.
Provides hierarchical dense positive supervision via ATSS/TaskAligned assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigners import (
    ATSSAssigner,
    TaskAlignedAssigner,
    generate_anchors_for_grid_cell,
    batch_distance2bbox,
)
from .box_ops import box_iou, box_cxcywh_to_xyxy, generalized_box_iou


__all__ = ['PPYOLOEHead']


class ConvBNLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=1, act='swish'):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        if act == 'swish' or act == 'silu':
            self.act = nn.SiLU()
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish'):
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


class PPYOLOEHead(nn.Module):
    __shared__ = ['num_classes']

    def __init__(self,
                 in_channels=None,
                 num_classes=80,
                 fpn_strides=(8, 16, 32),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 static_assigner_epoch=30,
                 use_varifocal_loss=True,
                 loss_weight=None,
                 static_assigner=None,
                 assigner=None):
        super().__init__()
        if in_channels is None:
            in_channels = [256, 256, 256]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = list(fpn_strides)
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.reg_channels = reg_max + 1
        self.static_assigner_epoch = static_assigner_epoch
        self.use_varifocal_loss = use_varifocal_loss
        self.loss_weight = loss_weight or {'class': 1.0, 'iou': 2.5, 'dfl': 0.5}

        if static_assigner is None:
            static_assigner = ATSSAssigner(topk=9, num_classes=num_classes)
        if assigner is None:
            assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)
        self.static_assigner = static_assigner
        self.assigner = assigner

        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act='swish'))
            self.stem_reg.append(ESEAttn(in_c, act='swish'))

        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(nn.Conv2d(in_c, num_classes, 3, padding=1))
            self.pred_reg.append(nn.Conv2d(in_c, 4 * self.reg_channels, 3, padding=1))

        self.proj_conv = nn.Conv2d(self.reg_channels, 1, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for cls_conv, reg_conv in zip(self.pred_cls, self.pred_reg):
            nn.init.constant_(cls_conv.bias, -4.595)
            nn.init.constant_(reg_conv.bias, 1.0)

        proj = torch.linspace(0, self.reg_max, self.reg_channels).reshape(1, self.reg_channels, 1, 1)
        self.proj_conv.weight.data.copy_(proj)
        self.proj_conv.weight.requires_grad = False

    def _bbox_decode(self, anchor_points, pred_dist):
        n, l, _ = pred_dist.shape
        pred_dist = pred_dist.reshape(n, l, 4, self.reg_channels)
        pred_dist = F.softmax(pred_dist, dim=-1)
        pred_dist = pred_dist.permute(0, 3, 1, 2)
        pred_dist = self.proj_conv(pred_dist).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1 = bbox[..., :2]
        x2y2 = bbox[..., 2:]
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], -1).clamp(0, self.reg_max - 0.01)

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(score, label, weight=weight, reduction='sum')
        return loss

    def _df_loss(self, pred_dist, target, lower_bound=0):
        target_left = target.floor().long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left
        pred_flat = pred_dist.reshape(-1, self.reg_channels)
        loss_left = F.cross_entropy(pred_flat, (target_left - lower_bound).reshape(-1),
                                    reduction='none').reshape_as(target)
        loss_right = F.cross_entropy(pred_flat, (target_right - lower_bound).reshape(-1),
                                     reduction='none').reshape_as(target)
        return (loss_left * weight_left + loss_right * weight_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum, num_classes):
        mask_positive = (assigned_labels != num_classes)
        num_pos = mask_positive.sum()

        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).repeat(1, 1, 4)
            pred_bboxes_pos = pred_bboxes[bbox_mask].reshape(-1, 4)
            assigned_bboxes_pos = assigned_bboxes[bbox_mask].reshape(-1, 4)
            bbox_weight = assigned_scores.sum(-1)[mask_positive].unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            giou = generalized_box_iou(pred_bboxes_pos, assigned_bboxes_pos)
            loss_iou = ((1.0 - torch.diag(giou)) * bbox_weight.squeeze(-1)).sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).repeat(1, 1, self.reg_channels * 4)
            pred_dist_pos = pred_dist[dist_mask].reshape(-1, 4, self.reg_channels)
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = assigned_ltrb[bbox_mask].reshape(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([], device=pred_bboxes.device)
            loss_iou = torch.zeros([], device=pred_bboxes.device)
            loss_dfl = torch.zeros([], device=pred_bboxes.device)

        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta, epoch_id=0, aux_pred=None):
        pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta.get('pad_gt_mask', None)
        if pad_gt_mask is None:
            pad_gt_mask = [torch.ones(len(l), 1, device=pred_scores.device) for l in gt_labels]
            pad_gt_mask = self._pad_gt(gt_bboxes, pad_gt_mask)

        gt_labels_t = self._pad_gt(gt_labels).unsqueeze(-1)
        gt_bboxes_t = self._pad_gt(gt_bboxes)
        pad_gt_mask_t = self._pad_gt(pad_gt_mask).unsqueeze(-1)

        if epoch_id < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = self.static_assigner(
                anchors, num_anchors_list, gt_labels_t, gt_bboxes_t,
                pad_gt_mask_t, bg_index=self.num_classes,
                pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels_t, gt_bboxes_t,
                pad_gt_mask_t,
                bg_index=self.num_classes)
            alpha_l = -1

        assigned_bboxes = assigned_bboxes / stride_tensor.unsqueeze(0)

        return self._compute_losses(pred_scores, pred_distri, pred_bboxes,
                                    anchor_points_s, assigned_labels,
                                    assigned_bboxes, assigned_scores, alpha_l)

    @staticmethod
    def _pad_gt(gt_list, pad_val=None):
        max_len = max(len(g) for g in gt_list)
        if pad_val is None:
            pad_val = gt_list[0].new_tensor(0.0)
        padded = []
        for g in gt_list:
            if len(g) < max_len:
                pad = g.new_zeros(max_len - len(g), *g.shape[1:])
                g = torch.cat([g, pad], dim=0)
            padded.append(g)
        return torch.stack(padded, dim=0)

    def _compute_losses(self, pred_scores, pred_distri, pred_bboxes,
                        anchor_points_s, assigned_labels, assigned_bboxes,
                        assigned_scores, alpha_l):
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels.long(), self.num_classes + 1)[..., :-1].float()
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores, one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        assigned_scores_sum = torch.clamp(assigned_scores_sum, min=1.)
        loss_cls = loss_cls / assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = self._bbox_loss(
            pred_distri, pred_bboxes, anchor_points_s,
            assigned_labels, assigned_bboxes, assigned_scores,
            assigned_scores_sum, self.num_classes)

        loss = (self.loss_weight['class'] * loss_cls +
                self.loss_weight['iou'] * loss_iou +
                self.loss_weight['dfl'] * loss_dfl)

        return {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }

    def forward(self, feats, targets=None, epoch_id=None):
        if len(feats) != len(self.fpn_strides):
            feats = feats[-len(self.fpn_strides):]

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            cls_score = cls_logit.sigmoid()
            cls_score_list.append(cls_score.flatten(2).transpose(1, 2))
            reg_distri_list.append(reg_distri.flatten(2).transpose(1, 2))

        cls_score_list = torch.cat(cls_score_list, dim=1)
        reg_distri_list = torch.cat(reg_distri_list, dim=1)

        if self.training and targets is not None:
            gt_meta = self._build_gt_meta(targets, cls_score_list.device)
            return self.get_loss(
                [cls_score_list, reg_distri_list, anchors, anchor_points,
                 num_anchors_list, stride_tensor],
                gt_meta, epoch_id=epoch_id if epoch_id is not None else 0)
        else:
            anchor_points_s = anchor_points / stride_tensor
            pred_bboxes = self._bbox_decode(anchor_points_s, reg_distri_list)
            return cls_score_list, pred_bboxes

    def _build_gt_meta(self, targets, device):
        gt_class = [t['labels'] for t in targets]
        gt_bbox = [box_cxcywh_to_xyxy(t['boxes']) for t in targets]
        pad_gt_mask = [torch.ones(len(t['labels']), 1, dtype=torch.float32, device=device)
                       for t in targets]
        return {
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'pad_gt_mask': pad_gt_mask,
        }
