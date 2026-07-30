"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from typing import Optional

from .box_ops import (
    box_cxcywh_to_xyxy,
    aligned_box_iou,
    box_iou,
    border_giou_loss,
    generalized_box_iou,
    mpdiou_loss,
    focal_mpdiou_loss,
)

from .matcher import _compute_match_costs

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register



@register
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80,
                 bbox_iou_loss_type='giou',
                 focal_mpdiou_gamma=0.5,
                 mpdiou_eps=1e-7,
                 border_giou_lambda=0.05,
                 use_defect_aware_query=False,
                 foreground_loss_weight=0.2,
                 foreground_loss_warmup_epochs=0,
                 foreground_pos_iou_thresh=0.5,
                 foreground_neg_iou_thresh=0.3,
                 foreground_target_type='iou',
                 foreground_center_radius=0.5,
                 use_discriminative_aux_loss=False,
                 aux_loss_type='prototype',
                 aux_loss_weight=0.05,
                 aux_loss_warmup_epochs=0,
                 discriminative_embed_dim=128,
                 prototype_momentum=0.95,
                 prototype_margin=0.2,
                 use_qdqr=False,
                 qdqr_alpha=1.0,
                 qdqr_beta=1.0,
                 qdqr_gamma=0.0,
                 qdqr_min_weight=0.2,
                 qdqr_max_weight=1.0,
                 qdqr_warmup_epochs=0,
                 qdqr_detach_score=True,
                 qdqr_detach_quality=True,
                 qdqr_use_defectness=False,
                 qdqr_eps=1e-6,
                 use_query_quality=False,
                 query_quality_loss_weight=1.0,
                 query_quality_loss_type='l1',
                 query_quality_use_detached_iou_target=True,
                 use_ddnq=False,
                 ddnq_loss_weight=1.0,
                 ddnq_warmup_epochs=3,
                 use_pg_o2m=False,
                 pg_o2m_gt_replicate=4,
                 pg_o2m_loss_weight=0.5,
                 pg_o2m_warmup_epochs=3,
                 pg_o2m_lite_top_m=2,
                 pg_o2m_lite_iou_thr=0.3,
                 pg_o2m_lite_cls_only=True,
                 pg_o2m_lite_exclude_hungarian=True,
                 pg_o2m_lite_decay=True,
                 pg_o2m_lite_decay_start_ratio=0.5,
                 pg_o2m_lite_decay_end_ratio=1.0,
                 pg_o2m_lite_layers=None,
                 use_defect_aware_mal=False,
                 mal_gamma=1.5,
                 mal_eps=1e-6,
                 mal_defect_area_alpha=0.5,
                 mal_defect_area_tau=0.02,
                 mal_defect_shape_beta=0.3,
                 mal_defect_shape_r0=3.0,
                 mal_defect_weight_max=2.0,
                 use_query_perturb=False,
                 query_perturb_loss_weight=0.3,
                 query_perturb_start_epoch=5,
                 query_perturb_end_epoch=-1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma
        self.current_epoch = 0
        self._total_epochs = 100
        self.bbox_iou_loss_type = str(bbox_iou_loss_type).lower()
        self.focal_mpdiou_gamma = float(focal_mpdiou_gamma)
        self.mpdiou_eps = float(mpdiou_eps)
        self.border_giou_lambda = float(border_giou_lambda)
        self.last_box_iou_mean = 0.0
        self.use_defect_aware_query = bool(use_defect_aware_query)
        self.foreground_loss_weight = float(foreground_loss_weight)
        self.foreground_loss_warmup_epochs = int(foreground_loss_warmup_epochs)
        self.foreground_pos_iou_thresh = float(foreground_pos_iou_thresh)
        self.foreground_neg_iou_thresh = float(foreground_neg_iou_thresh)
        self.foreground_target_type = str(foreground_target_type).lower()
        self.foreground_center_radius = float(foreground_center_radius)
        self.use_discriminative_aux_loss = bool(use_discriminative_aux_loss)
        self.aux_loss_type = str(aux_loss_type).lower()
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_loss_warmup_epochs = int(aux_loss_warmup_epochs)
        self.discriminative_embed_dim = int(discriminative_embed_dim)
        self.prototype_momentum = float(prototype_momentum)
        self.prototype_margin = float(prototype_margin)
        self.use_qdqr = bool(use_qdqr)
        self.qdqr_alpha = float(qdqr_alpha)
        self.qdqr_beta = float(qdqr_beta)
        self.qdqr_gamma = float(qdqr_gamma)
        self.qdqr_min_weight = float(qdqr_min_weight)
        self.qdqr_max_weight = float(qdqr_max_weight)
        self.qdqr_warmup_epochs = int(qdqr_warmup_epochs)
        self.qdqr_detach_score = bool(qdqr_detach_score)
        self.qdqr_detach_quality = bool(qdqr_detach_quality)
        self.qdqr_use_defectness = bool(qdqr_use_defectness)
        self.qdqr_eps = float(qdqr_eps)
        self.use_query_quality = bool(use_query_quality)
        self.query_quality_loss_weight = float(query_quality_loss_weight)
        self.query_quality_loss_type = str(query_quality_loss_type).lower()
        self.query_quality_use_detached_iou_target = bool(query_quality_use_detached_iou_target)
        self.use_ddnq = bool(use_ddnq)
        self.ddnq_loss_weight = float(ddnq_loss_weight)
        self.ddnq_warmup_epochs = int(ddnq_warmup_epochs)
        self.use_pg_o2m = bool(use_pg_o2m)
        self.pg_o2m_gt_replicate = int(pg_o2m_gt_replicate)
        self.pg_o2m_loss_weight = float(pg_o2m_loss_weight)
        self.pg_o2m_warmup_epochs = int(pg_o2m_warmup_epochs)
        self.pg_o2m_lite_top_m = int(pg_o2m_lite_top_m)
        self.pg_o2m_lite_iou_thr = float(pg_o2m_lite_iou_thr)
        self.pg_o2m_lite_cls_only = bool(pg_o2m_lite_cls_only)
        self.pg_o2m_lite_exclude_hungarian = bool(pg_o2m_lite_exclude_hungarian)
        self.pg_o2m_lite_decay = bool(pg_o2m_lite_decay)
        self.pg_o2m_lite_decay_start_ratio = float(pg_o2m_lite_decay_start_ratio)
        self.pg_o2m_lite_decay_end_ratio = float(pg_o2m_lite_decay_end_ratio)
        if pg_o2m_lite_layers is None:
            self.pg_o2m_lite_layers = [0, 1, 2]
        elif isinstance(pg_o2m_lite_layers, (list, tuple)):
            self.pg_o2m_lite_layers = [int(x) for x in pg_o2m_lite_layers]
        else:
            self.pg_o2m_lite_layers = [int(pg_o2m_lite_layers)]

        self.use_defect_aware_mal = bool(use_defect_aware_mal)
        self.mal_gamma = float(mal_gamma)
        self.mal_eps = float(mal_eps)
        self.mal_defect_area_alpha = float(mal_defect_area_alpha)
        self.mal_defect_area_tau = float(mal_defect_area_tau)
        self.mal_defect_shape_beta = float(mal_defect_shape_beta)
        self.mal_defect_shape_r0 = float(mal_defect_shape_r0)
        self.mal_defect_weight_max = float(mal_defect_weight_max)

        if self.mal_gamma <= 0:
            raise ValueError('mal_gamma must be > 0')
        if self.mal_eps <= 0:
            raise ValueError('mal_eps must be > 0')
        if self.mal_defect_area_alpha < 0:
            raise ValueError('mal_defect_area_alpha must be >= 0')
        if self.mal_defect_area_tau <= 0:
            raise ValueError('mal_defect_area_tau must be > 0')
        if self.mal_defect_shape_beta < 0:
            raise ValueError('mal_defect_shape_beta must be >= 0')
        if self.mal_defect_weight_max < 1.0:
            raise ValueError('mal_defect_weight_max must be >= 1.0')

        self.use_query_perturb = bool(use_query_perturb)
        self.query_perturb_loss_weight = float(query_perturb_loss_weight)
        self.query_perturb_start_epoch = int(query_perturb_start_epoch)
        self.query_perturb_end_epoch = int(query_perturb_end_epoch)

        if self.query_perturb_loss_weight < 0:
            raise ValueError('query_perturb_loss_weight must be >= 0')
        if self.query_perturb_start_epoch < 0:
            raise ValueError('query_perturb_start_epoch must be >= 0')

        valid_iou_loss_types = {'giou', 'border_giou', 'mpdiou', 'focal_mpdiou'}
        if self.bbox_iou_loss_type not in valid_iou_loss_types:
            raise ValueError(
                f"Unsupported bbox_iou_loss_type: {self.bbox_iou_loss_type}. "
                f"Use one of: {sorted(valid_iou_loss_types)}"
            )
        if self.border_giou_lambda < 0.0:
            raise ValueError('border_giou_lambda must be >= 0.')
        if self.foreground_neg_iou_thresh > self.foreground_pos_iou_thresh:
            raise ValueError('foreground_neg_iou_thresh must be <= foreground_pos_iou_thresh.')
        if self.foreground_loss_warmup_epochs < 0:
            raise ValueError('foreground_loss_warmup_epochs must be >= 0.')
        if self.foreground_target_type not in {'iou', 'center', 'hybrid'}:
            raise ValueError(
                f"Unsupported foreground_target_type: {self.foreground_target_type}. "
                "Use one of: ['iou', 'center', 'hybrid']"
            )
        if self.foreground_center_radius <= 0.0:
            raise ValueError('foreground_center_radius must be > 0.')
        if self.aux_loss_type not in {'prototype', 'center'}:
            raise ValueError(f'Unsupported aux_loss_type: {self.aux_loss_type}. Use one of: [\'prototype\', \'center\']')
        if self.aux_loss_warmup_epochs < 0:
            raise ValueError('aux_loss_warmup_epochs must be >= 0.')
        if self.discriminative_embed_dim <= 0:
            raise ValueError('discriminative_embed_dim must be > 0.')
        if self.use_qdqr and not self.use_discriminative_aux_loss:
            raise ValueError('QDQR requires use_discriminative_aux_loss=true.')
        if self.qdqr_warmup_epochs < 0:
            raise ValueError('qdqr_warmup_epochs must be >= 0.')
        if self.qdqr_min_weight <= 0.0:
            raise ValueError('qdqr_min_weight must be > 0.')
        if self.qdqr_max_weight < self.qdqr_min_weight:
            raise ValueError('qdqr_max_weight must be >= qdqr_min_weight.')
        if self.qdqr_eps <= 0.0:
            raise ValueError('qdqr_eps must be > 0.')
        if self.query_quality_loss_weight < 0.0:
            raise ValueError('query_quality_loss_weight must be >= 0.')
        if self.query_quality_loss_type not in {'l1', 'bce'}:
            raise ValueError("query_quality_loss_type must be one of: ['l1', 'bce']")
        if self.ddnq_loss_weight < 0.0:
            raise ValueError('ddnq_loss_weight must be >= 0.')
        if self.ddnq_warmup_epochs < 0:
            raise ValueError('ddnq_warmup_epochs must be >= 0.')
        if self.pg_o2m_gt_replicate < 1:
            raise ValueError('pg_o2m_gt_replicate must be >= 1.')
        if self.pg_o2m_loss_weight < 0.0:
            raise ValueError('pg_o2m_loss_weight must be >= 0.')
        if self.pg_o2m_warmup_epochs < 0:
            raise ValueError('pg_o2m_warmup_epochs must be >= 0.')
        if self.pg_o2m_lite_top_m < 1:
            raise ValueError('pg_o2m_lite_top_m must be >= 1.')
        if not 0.0 <= self.pg_o2m_lite_iou_thr <= 1.0:
            raise ValueError('pg_o2m_lite_iou_thr must be in [0, 1].')
        if not 0.0 <= self.pg_o2m_lite_decay_start_ratio <= 1.0:
            raise ValueError('pg_o2m_lite_decay_start_ratio must be in [0, 1].')
        if not 0.0 <= self.pg_o2m_lite_decay_end_ratio <= 1.0:
            raise ValueError('pg_o2m_lite_decay_end_ratio must be in [0, 1].')
        if self.pg_o2m_lite_decay_start_ratio > self.pg_o2m_lite_decay_end_ratio:
            raise ValueError('pg_o2m_lite_decay_start_ratio must be <= pg_o2m_lite_decay_end_ratio.')
        for layer_idx in self.pg_o2m_lite_layers:
            if not isinstance(layer_idx, int) or layer_idx < 0:
                raise ValueError(f'pg_o2m_lite_layers must contain non-negative integers, got {layer_idx}.')
        if self.matcher is not None:
            if hasattr(self.matcher, 'bbox_iou_cost_type'):
                matcher_supported_loss_types = {'giou', 'mpdiou', 'focal_mpdiou'}
                if self.bbox_iou_loss_type in matcher_supported_loss_types:
                    self.matcher.bbox_iou_cost_type = self.bbox_iou_loss_type
            if hasattr(self.matcher, 'focal_mpdiou_gamma'):
                self.matcher.focal_mpdiou_gamma = self.focal_mpdiou_gamma
            if hasattr(self.matcher, 'mpdiou_eps'):
                self.matcher.mpdiou_eps = self.mpdiou_eps

        # Diagnostic metrics (updated every forward pass during training)
        self.last_matched_iou_o2o_mean = 0.0
        self.last_foreground_loss = 0.0
        self.last_foreground_valid_count = 0
        self.last_foreground_pos_count = 0
        self.last_foreground_neg_count = 0
        self.last_foreground_ignore_count = 0
        self.last_foreground_pos_ratio = 0.0
        self.last_foreground_ignore_ratio = 0.0
        self.last_foreground_topk_pos_ratio = 0.0
        self.last_foreground_topk_ignore_ratio = 0.0
        self.last_foreground_loss_weight = 0.0
        self.last_discriminative_loss = 0.0
        self.last_discriminative_query_count = 0
        self.last_discriminative_loss_weight = 0.0
        self.last_discriminative_compact_loss = 0.0
        self.last_discriminative_margin_loss = 0.0
        self.last_valid_prototype_count = 0
        self.last_negative_prototype_count_mean = 0.0
        self.last_qdqr_weight_mean = 1.0
        self.last_qdqr_weight_min = 1.0
        self.last_qdqr_weight_max = 1.0
        self.last_qdqr_cls_score_mean = 0.0
        self.last_qdqr_iou_mean = 0.0
        self.last_qdqr_defectness_mean = 0.0
        self.last_query_quality_loss = 0.0
        self.last_query_quality_match_count = 0
        self.last_query_quality_pred_mean = 0.0
        self.last_query_quality_target_mean = 0.0
        self.last_query_quality_loss_weight = 0.0
        self.last_ddnq_loss_weight = 0.0
        self.last_ddnq_num_group = 0
        self.last_ddnq_valid_count = 0
        self.last_ddnq_slender_count = 0

        if self.use_discriminative_aux_loss:
            self.register_buffer(
                'query_prototypes',
                torch.zeros(self.num_classes, self.discriminative_embed_dim),
            )
            self.register_buffer(
                'query_prototype_initialized',
                torch.zeros(self.num_classes, dtype=torch.bool),
            )
        else:
            self.query_prototypes = None
            self.query_prototype_initialized = None

    @staticmethod
    def _is_cls_loss_name(loss_name):
        return loss_name in {'labels', 'bce', 'focal', 'vfl'}

    def _get_warmup_weight(self, base_weight, warmup_epochs):
        base_weight = float(base_weight)
        warmup_epochs = int(warmup_epochs)
        if warmup_epochs <= 0:
            return base_weight
        progress = min(max(float(self.current_epoch + 1) / float(warmup_epochs), 0.0), 1.0)
        return base_weight * progress

    def _get_ddnq_warmup_weight(self):
        return self._get_warmup_weight(self.ddnq_loss_weight, self.ddnq_warmup_epochs)

    def _compute_o2o_matched_iou(self, outputs, targets, indices_o2o):
        """Compute mean IoU of O2O-matched prediction–GT pairs."""
        pred_boxes = outputs['pred_boxes']
        ious = []
        for i, (src_i, tgt_i) in enumerate(indices_o2o):
            if len(src_i) == 0:
                continue
            pred_b = pred_boxes[i][src_i]
            gt_b = targets[i]['boxes'][tgt_i]
            if len(pred_b) == 0:
                continue
            iou_mat, _ = box_iou(
                box_cxcywh_to_xyxy(pred_b),
                box_cxcywh_to_xyxy(gt_b),
            )
            diag_iou = torch.diag(iou_mat)
            if diag_iou.numel() > 0:
                ious.append(diag_iou.mean().item())
        return float(np.mean(ious)) if ious else 0.0

    def _validate_indices(self, targets, indices):
        """Validate that indices do not exceed target sizes and fix if necessary."""
        indices = list(indices)
        for t_idx, (t, (src_i, tgt_i)) in enumerate(zip(targets, indices)):
            if len(tgt_i) == 0:
                continue
                
            # Check if src_i and tgt_i have matching lengths (required for paired filtering)
            if len(src_i) != len(tgt_i):
                print(f"\n[CRITERION ERROR] Mismatched index lengths in batch {t_idx}:")
                print(f"  src_i length: {len(src_i)}")
                print(f"  tgt_i length: {len(tgt_i)}")
                print(f"  This indicates a bug in the matcher implementation")
                # Try to recover by truncating to the shorter length
                min_len = min(len(src_i), len(tgt_i))
                src_i = src_i[:min_len]
                tgt_i = tgt_i[:min_len]
                print(f"  Truncated both to length: {min_len}\n")
            
            num_boxes_in_target = len(t['boxes'])
            num_labels_in_target = len(t['labels'])
            
            max_idx = tgt_i.max().item()
            if max_idx >= num_boxes_in_target or max_idx >= num_labels_in_target:
                print(f"\n[CRITERION WARNING] Invalid target index in batch {t_idx}:")
                print(f"  Max matched index: {max_idx}")
                print(f"  Num boxes: {num_boxes_in_target}, Num labels: {num_labels_in_target}")
                print(f"  Total matches before fix: {len(tgt_i)}")
                
                valid_mask = (tgt_i < num_boxes_in_target) & (tgt_i < num_labels_in_target)
                
                src_i = src_i[valid_mask]
                tgt_i = tgt_i[valid_mask]
                indices[t_idx] = (src_i, tgt_i)
                print(f"  Valid matches after filtering: {len(tgt_i)}\n")
            else:
                indices[t_idx] = (src_i, tgt_i)
        return indices


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # Validate indices first
        indices = self._validate_indices(targets, indices)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) if any(len(i[1]) > 0 for i in indices) else torch.zeros(0, dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        
        # Validate indices first
        indices = self._validate_indices(targets, indices)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) if any(len(i[1]) > 0 for i in indices) else torch.zeros(0, dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # Validate indices first
        indices = self._validate_indices(targets, indices)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) if any(len(i[1]) > 0 for i in indices) else torch.zeros(0, dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none")
        # prob = F.sigmoid(src_logits) # TODO .detach()
        # p_t = prob * target + (1 - prob) * (1 - target)
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        
        # Validate indices first
        indices = self._validate_indices(targets, indices)
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) if any(len(i[1]) > 0 for i in indices) else src_boxes.new_zeros((0, 4))
        
        # Handle empty case for IoU calculation
        if len(target_boxes) == 0:
            ious = src_boxes.new_zeros(0)
        else:
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) if any(len(i[1]) > 0 for i in indices) else torch.zeros(0, dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def _compute_defect_aware_weights(self, target_boxes, device, dtype):
        if len(target_boxes) == 0:
            return target_boxes.new_zeros((0,))

        wh = target_boxes[:, 2:].clamp(min=self.mal_eps)
        area = (wh[:, 0] * wh[:, 1]).clamp(min=self.mal_eps)

        w_area = 1.0 + self.mal_defect_area_alpha * torch.exp(-area / self.mal_defect_area_tau)

        ratio = torch.maximum(wh[:, 0] / wh[:, 1], wh[:, 1] / wh[:, 0])
        w_shape = 1.0 + self.mal_defect_shape_beta * torch.sigmoid(ratio - self.mal_defect_shape_r0)

        w_defect = (w_area * w_shape).clamp(min=1.0, max=self.mal_defect_weight_max)
        return w_defect.to(dtype=dtype, device=device)

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs

        indices = self._validate_indices(targets, indices)
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0
        ) if any(len(i[1]) > 0 for i in indices) else src_boxes.new_zeros((0, 4))

        if len(target_boxes) > 0:
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = src_boxes.new_zeros(0)

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        ) if any(len(i[1]) > 0 for i in indices) else torch.zeros(
            0, dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.pow(self.mal_gamma).clamp(min=0, max=1).to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_labels_defect_aware_mal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs

        indices = self._validate_indices(targets, indices)
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits']

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0
        ) if any(len(i[1]) > 0 for i in indices) else src_boxes.new_zeros((0, 4))

        if len(target_boxes) > 0:
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
            defect_weights = self._compute_defect_aware_weights(
                target_boxes, src_logits.device, src_logits.dtype)
        else:
            ious = src_boxes.new_zeros(0)
            defect_weights = src_boxes.new_zeros(0)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        ) if any(len(i[1]) > 0 for i in indices) else torch.zeros(
            0, dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.pow(self.mal_gamma).clamp(min=0, max=1).to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        if len(defect_weights) > 0:
            defect_weight_t = torch.ones(target_classes.shape, dtype=src_logits.dtype, device=src_logits.device)
            defect_weight_t[idx] = defect_weights.to(defect_weight_t.dtype)
            defect_weight_per_query = defect_weight_t.max(dim=-1).values
            per_query_penalty = 1.0 + (defect_weight_per_query - 1.0).unsqueeze(-1) * target
            weight = weight * per_query_penalty

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the IoU-family loss.
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
        # Validate indices first
        indices = self._validate_indices(targets, indices)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) if any(len(i[1]) > 0 for i in indices) else src_boxes.new_zeros((0, 4))

        losses = {}

        if len(target_boxes) == 0:
            # Handle empty case
            losses['loss_bbox'] = src_boxes.sum() * 0.0  # Zero loss but keeps gradient graph
            losses['loss_iou'] = src_boxes.sum() * 0.0
            self.last_box_iou_mean = 0.0
        else:
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            pred_xyxy = box_cxcywh_to_xyxy(src_boxes)
            target_xyxy = box_cxcywh_to_xyxy(target_boxes)
            pair_iou_diag, _ = box_iou(pred_xyxy, target_xyxy)
            pair_iou_diag = torch.diag(pair_iou_diag).detach()
            self.last_box_iou_mean = float(pair_iou_diag.mean().item()) if pair_iou_diag.numel() > 0 else 0.0

            if self.bbox_iou_loss_type == 'mpdiou':
                image_wh = self._get_matched_image_wh(targets, indices, pred_xyxy.device, pred_xyxy.dtype)
                pred_xyxy_abs = self._to_image_space_xyxy(pred_xyxy, image_wh)
                target_xyxy_abs = self._to_image_space_xyxy(target_xyxy, image_wh)
                loss_iou, _, _ = mpdiou_loss(
                    pred_xyxy_abs,
                    target_xyxy_abs,
                    image_wh=image_wh,
                    eps=self.mpdiou_eps,
                )
            elif self.bbox_iou_loss_type == 'focal_mpdiou':
                image_wh = self._get_matched_image_wh(targets, indices, pred_xyxy.device, pred_xyxy.dtype)
                pred_xyxy_abs = self._to_image_space_xyxy(pred_xyxy, image_wh)
                target_xyxy_abs = self._to_image_space_xyxy(target_xyxy, image_wh)
                loss_iou, _, _ = focal_mpdiou_loss(
                    pred_xyxy_abs,
                    target_xyxy_abs,
                    image_wh=image_wh,
                    gamma=self.focal_mpdiou_gamma,
                    eps=self.mpdiou_eps,
                )
            elif self.bbox_iou_loss_type == 'border_giou':
                # Border-GIoU changes only the training regression loss; matcher cost stays GIoU.
                loss_iou = border_giou_loss(
                    pred_xyxy,
                    target_xyxy,
                    lambda_border=self.border_giou_lambda,
                    eps=self.mpdiou_eps,
                    reduction='none',
                )
            else:
                loss_iou = 1 - torch.diag(generalized_box_iou(pred_xyxy, target_xyxy))
            losses['loss_iou'] = loss_iou.sum() / num_boxes
        
        return losses

    @staticmethod
    def _to_image_space_xyxy(boxes_xyxy: torch.Tensor, image_wh: Optional[torch.Tensor]) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy
        if image_wh is None or image_wh.numel() == 0:
            return boxes_xyxy

        scale = torch.stack(
            [image_wh[:, 0], image_wh[:, 1], image_wh[:, 0], image_wh[:, 1]],
            dim=-1,
        )
        return boxes_xyxy * scale

    @staticmethod
    def _get_target_image_wh(target) -> torch.Tensor:
        if 'size' in target:
            return target['size']
        if 'orig_size' in target:
            return target['orig_size']
        return torch.ones(2, device=target['boxes'].device, dtype=target['boxes'].dtype)

    def _get_matched_image_wh(self, targets, indices, device, dtype) -> torch.Tensor:
        matched_image_wh = []
        for target, (_, tgt_idx) in zip(targets, indices):
            if len(tgt_idx) == 0:
                continue
            image_wh = self._get_target_image_wh(target).to(device=device, dtype=dtype)
            matched_image_wh.append(image_wh.unsqueeze(0).repeat(len(tgt_idx), 1))

        if matched_image_wh:
            return torch.cat(matched_image_wh, dim=0)
        return torch.zeros((0, 2), device=device, dtype=dtype)

    @torch.no_grad()
    def _build_token_foreground_targets(self, outputs, targets):
        if 'enc_foreground_logits' not in outputs or 'enc_token_boxes' not in outputs:
            return None

        token_boxes = outputs['enc_token_boxes'].detach()
        anchor_boxes = outputs.get('enc_token_anchors', token_boxes).detach()
        valid_mask = outputs.get(
            'enc_valid_mask',
            torch.ones(token_boxes.shape[:2], dtype=torch.bool, device=token_boxes.device),
        )

        fg_targets = token_boxes.new_full(token_boxes.shape[:2], -1.0)
        for batch_idx, target in enumerate(targets):
            valid_idx = torch.where(valid_mask[batch_idx])[0]
            if valid_idx.numel() == 0:
                continue

            if len(target['boxes']) == 0:
                fg_targets[batch_idx, valid_idx] = 0.0
                continue

            iou_matrix, _ = box_iou(
                box_cxcywh_to_xyxy(token_boxes[batch_idx, valid_idx]),
                box_cxcywh_to_xyxy(target['boxes']),
            )
            max_iou = iou_matrix.max(dim=1).values
            iou_pos = max_iou >= self.foreground_pos_iou_thresh
            iou_neg = max_iou <= self.foreground_neg_iou_thresh

            anchor_centers = anchor_boxes[batch_idx, valid_idx, :2]
            center_pos, center_inside = self._build_center_foreground_masks(
                anchor_centers,
                target['boxes'],
            )

            if self.foreground_target_type == 'iou':
                pos_mask = iou_pos
                neg_mask = iou_neg
            elif self.foreground_target_type == 'center':
                pos_mask = center_pos
                neg_mask = ~center_inside
            else:
                pos_mask = iou_pos | center_pos
                neg_mask = iou_neg & (~center_inside)

            fg_targets[batch_idx, valid_idx[pos_mask]] = 1.0
            fg_targets[batch_idx, valid_idx[neg_mask]] = 0.0

        return fg_targets

    def _build_center_foreground_masks(self, token_centers, target_boxes):
        if target_boxes.numel() == 0 or token_centers.numel() == 0:
            empty = torch.zeros(token_centers.shape[0], dtype=torch.bool, device=token_centers.device)
            return empty, empty

        delta = (token_centers[:, None, :] - target_boxes[None, :, :2]).abs()
        half_wh = target_boxes[None, :, 2:] * 0.5
        inside_full_box = (delta <= half_wh).all(dim=-1).any(dim=1)

        center_half_wh = half_wh * self.foreground_center_radius
        inside_center_box = (delta <= center_half_wh).all(dim=-1).any(dim=1)
        return inside_center_box, inside_full_box

    def _update_foreground_target_stats(self, fg_targets, outputs):
        pos_mask = fg_targets == 1
        neg_mask = fg_targets == 0
        ignore_mask = fg_targets < 0
        valid_mask = fg_targets >= 0

        pos_count = int(pos_mask.sum().item())
        neg_count = int(neg_mask.sum().item())
        ignore_count = int(ignore_mask.sum().item())
        valid_count = int(valid_mask.sum().item())
        total_count = max(int(fg_targets.numel()), 1)

        self.last_foreground_pos_count = pos_count
        self.last_foreground_neg_count = neg_count
        self.last_foreground_ignore_count = ignore_count
        self.last_foreground_valid_count = valid_count
        self.last_foreground_pos_ratio = float(pos_count) / float(max(valid_count, 1))
        self.last_foreground_ignore_ratio = float(ignore_count) / float(total_count)
        self.last_foreground_topk_pos_ratio = 0.0
        self.last_foreground_topk_ignore_ratio = 0.0

        topk_indices = outputs.get('topk_indices')
        if topk_indices is None:
            return

        topk_targets = fg_targets.gather(1, topk_indices)
        topk_valid = topk_targets >= 0
        topk_count = max(int(topk_targets.numel()), 1)
        topk_valid_count = int(topk_valid.sum().item())
        self.last_foreground_topk_pos_ratio = float((topk_targets == 1).sum().item()) / float(max(topk_valid_count, 1))
        self.last_foreground_topk_ignore_ratio = float((topk_targets < 0).sum().item()) / float(topk_count)

    def _loss_defect_aware_query(self, outputs, targets):
        if not self.use_defect_aware_query:
            return None
        if not self.training:
            return None
        if 'enc_foreground_logits' not in outputs:
            raise RuntimeError(
                'SetCriterion.use_defect_aware_query=true, but model outputs do not contain '
                "'enc_foreground_logits'. Keep RTDETRTransformer.use_defect_aware_query in sync."
            )

        fg_targets = self._build_token_foreground_targets(outputs, targets)
        if fg_targets is None:
            raise RuntimeError(
                'Foreground query loss is enabled, but encoder token metadata is missing from model outputs.'
            )

        fg_logits = outputs['enc_foreground_logits']
        self._update_foreground_target_stats(fg_targets, outputs)
        valid_mask = fg_targets >= 0
        valid_count = int(valid_mask.sum().item())
        self.last_foreground_valid_count = valid_count
        effective_weight = self._get_warmup_weight(
            self.foreground_loss_weight,
            self.foreground_loss_warmup_epochs,
        )
        self.last_foreground_loss_weight = effective_weight

        if valid_count == 0:
            zero = fg_logits.sum() * 0.0
            self.last_foreground_loss = 0.0
            return {'loss_query_fg': zero}

        loss_fg = torchvision.ops.sigmoid_focal_loss(
            fg_logits[valid_mask],
            fg_targets[valid_mask],
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='mean',
        )
        self.last_foreground_loss = float(loss_fg.detach().item())
        return {'loss_query_fg': loss_fg * effective_weight}

    def _get_batch_query_center_stats(self, query_features, target_labels):
        feature_sums = query_features.new_zeros(self.num_classes, query_features.shape[-1])
        feature_counts = query_features.new_zeros(self.num_classes)
        unique_labels = torch.unique(target_labels)
        for class_id in unique_labels.tolist():
            if class_id < 0 or class_id >= self.num_classes:
                continue
            class_mask = target_labels == class_id
            if not class_mask.any():
                continue
            class_features = query_features[class_mask]
            feature_sum = class_features.sum(dim=0)
            if torch.isfinite(feature_sum).all():
                feature_sums[int(class_id)] = feature_sum
                feature_counts[int(class_id)] = float(class_features.shape[0])
        return feature_sums, feature_counts

    @staticmethod
    def _build_batch_centers(feature_sums, feature_counts):
        batch_centers = {}
        valid_classes = torch.where(feature_counts > 0)[0].tolist()
        for class_id in valid_classes:
            center = feature_sums[class_id] / feature_counts[class_id].clamp(min=1.0)
            if torch.isfinite(center).all():
                batch_centers[int(class_id)] = center
        return batch_centers

    @staticmethod
    def _sync_query_center_stats(feature_sums, feature_counts):
        if not is_dist_available_and_initialized():
            return feature_sums, feature_counts

        torch.distributed.all_reduce(feature_sums)
        torch.distributed.all_reduce(feature_counts)
        return feature_sums, feature_counts

    @torch.no_grad()
    def _update_query_prototypes(self, batch_centers):
        if self.query_prototypes is None or self.query_prototype_initialized is None:
            return

        for class_id, center in batch_centers.items():
            center = F.normalize(center.detach(), dim=0)
            if self.query_prototype_initialized[class_id]:
                updated = (
                    self.prototype_momentum * self.query_prototypes[class_id] +
                    (1.0 - self.prototype_momentum) * center
                )
                self.query_prototypes[class_id] = F.normalize(updated, dim=0)
            else:
                self.query_prototypes[class_id] = center
                self.query_prototype_initialized[class_id] = True

    def _get_qdqr_weights(self, outputs, targets, indices, src_idx, target_labels):
        num_queries = int(target_labels.numel())
        if num_queries == 0:
            return outputs['pred_logits'].new_zeros((0,))

        one_weights = outputs['pred_logits'].new_ones((num_queries,))
        if not self.use_qdqr or self.current_epoch < self.qdqr_warmup_epochs:
            self.last_qdqr_weight_mean = 1.0
            self.last_qdqr_weight_min = 1.0
            self.last_qdqr_weight_max = 1.0
            self.last_qdqr_cls_score_mean = 0.0
            self.last_qdqr_iou_mean = 0.0
            self.last_qdqr_defectness_mean = 0.0
            return one_weights

        matched_logits = outputs['pred_logits'][src_idx]
        cls_scores = matched_logits.sigmoid().gather(1, target_labels.unsqueeze(1)).squeeze(1)
        cls_scores = cls_scores.clamp(min=self.qdqr_eps, max=1.0 - self.qdqr_eps)
        if self.qdqr_detach_score:
            cls_scores = cls_scores.detach()

        src_boxes = outputs['pred_boxes'][src_idx]
        target_boxes = torch.cat(
            [t['boxes'][J] for t, (_, J) in zip(targets, indices)],
            dim=0,
        )
        pred_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        iou_scores, _, _ = aligned_box_iou(pred_xyxy, target_xyxy, eps=self.qdqr_eps)
        iou_scores = iou_scores.clamp(min=0.0, max=1.0)
        if self.qdqr_detach_quality:
            iou_scores = iou_scores.detach()

        if self.qdqr_use_defectness:
            defectness = matched_logits.sigmoid().max(dim=1).values.clamp(min=self.qdqr_eps, max=1.0 - self.qdqr_eps)
            if self.qdqr_detach_score:
                defectness = defectness.detach()
        else:
            defectness = matched_logits.new_zeros((num_queries,))

        raw_weight = self.qdqr_alpha * cls_scores + self.qdqr_beta * iou_scores
        if self.qdqr_use_defectness:
            raw_weight = raw_weight + self.qdqr_gamma * defectness
        qdqr_weights = torch.sigmoid(raw_weight).clamp(
            min=self.qdqr_min_weight,
            max=self.qdqr_max_weight,
        )

        self.last_qdqr_weight_mean = float(qdqr_weights.detach().mean().item())
        self.last_qdqr_weight_min = float(qdqr_weights.detach().min().item())
        self.last_qdqr_weight_max = float(qdqr_weights.detach().max().item())
        self.last_qdqr_cls_score_mean = float(cls_scores.detach().mean().item())
        self.last_qdqr_iou_mean = float(iou_scores.detach().mean().item())
        self.last_qdqr_defectness_mean = float(defectness.detach().mean().item())
        return qdqr_weights

    def _loss_query_discriminative(self, outputs, targets, indices):
        if not self.use_discriminative_aux_loss:
            return None
        if not self.training:
            return None
        if 'disc_query_features' not in outputs:
            raise RuntimeError(
                'SetCriterion.use_discriminative_aux_loss=true, but model outputs do not contain '
                "'disc_query_features'. Keep RTDETRTransformer.use_discriminative_aux_loss in sync."
            )

        effective_weight = self._get_warmup_weight(self.aux_loss_weight, self.aux_loss_warmup_epochs)
        self.last_discriminative_loss_weight = effective_weight
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = outputs['disc_query_features'].sum() * 0.0
            self.last_discriminative_loss = 0.0
            self.last_discriminative_query_count = 0
            self.last_discriminative_compact_loss = 0.0
            self.last_discriminative_margin_loss = 0.0
            self.last_negative_prototype_count_mean = 0.0
            self.last_qdqr_weight_mean = 1.0
            self.last_qdqr_weight_min = 1.0
            self.last_qdqr_weight_max = 1.0
            self.last_qdqr_cls_score_mean = 0.0
            self.last_qdqr_iou_mean = 0.0
            self.last_qdqr_defectness_mean = 0.0
            return {'loss_query_disc': zero}

        query_features = outputs['disc_query_features'][idx]
        target_labels = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)],
            dim=0,
        )
        qdqr_weights = self._get_qdqr_weights(outputs, targets, indices, idx, target_labels)
        query_features = F.normalize(query_features, dim=-1)
        feature_sums, feature_counts = self._get_batch_query_center_stats(query_features, target_labels)
        feature_sums, feature_counts = self._sync_query_center_stats(feature_sums, feature_counts)
        batch_centers = self._build_batch_centers(feature_sums, feature_counts)
        self.last_discriminative_query_count = int(query_features.shape[0])
        if not batch_centers:
            zero = outputs['disc_query_features'].sum() * 0.0
            self.last_discriminative_loss = 0.0
            self.last_discriminative_compact_loss = 0.0
            self.last_discriminative_margin_loss = 0.0
            self.last_negative_prototype_count_mean = 0.0
            return {'loss_query_disc': zero}

        if self.aux_loss_type == 'center':
            centers = torch.stack(
                [
                    F.normalize(batch_centers[int(label.item())].detach(), dim=0)
                    for label in target_labels
                ],
                dim=0,
            )
            compact_loss = (query_features - centers).pow(2).sum(dim=-1)
            loss_disc = (compact_loss * qdqr_weights).mean()
            self.last_discriminative_compact_loss = float(compact_loss.mean().detach().item())
            self.last_discriminative_margin_loss = 0.0
            self.last_valid_prototype_count = len(batch_centers)
            self.last_negative_prototype_count_mean = 0.0
        else:
            prototypes = query_features.new_zeros(self.num_classes, query_features.shape[-1])
            valid_proto_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=query_features.device)
            if self.query_prototypes is not None:
                prototypes = self.query_prototypes.to(query_features.device).detach().clone()
                valid_proto_mask = self.query_prototype_initialized.to(query_features.device).clone()
            for class_id, center in batch_centers.items():
                prototypes[class_id] = F.normalize(center.detach(), dim=0)
                valid_proto_mask[class_id] = True

            pos_proto = prototypes[target_labels]
            pos_sim = (query_features * pos_proto).sum(dim=-1)

            all_sim = torch.matmul(query_features, prototypes.transpose(0, 1))
            neg_mask = valid_proto_mask.unsqueeze(0).expand_as(all_sim).clone()
            neg_mask.scatter_(1, target_labels.unsqueeze(1), False)
            masked_neg = all_sim.masked_fill(~neg_mask, -1.0)
            neg_sim = masked_neg.max(dim=1).values
            neg_sim = torch.where(neg_mask.any(dim=1), neg_sim, neg_sim.new_full(neg_sim.shape, -1.0))

            compact_loss = 1.0 - pos_sim
            margin_loss = F.relu(self.prototype_margin + neg_sim - pos_sim)
            loss_disc = ((compact_loss + margin_loss) * qdqr_weights).mean()
            self.last_discriminative_compact_loss = float(compact_loss.mean().detach().item())
            self.last_discriminative_margin_loss = float(margin_loss.mean().detach().item())
            self.last_valid_prototype_count = int(valid_proto_mask.sum().item())
            self.last_negative_prototype_count_mean = float(neg_mask.sum(dim=1).float().mean().detach().item())

        self._update_query_prototypes(batch_centers)
        if not torch.isfinite(loss_disc).all():
            raise RuntimeError('Non-finite discriminative query loss detected.')
        self.last_discriminative_loss = float(loss_disc.detach().item())
        return {'loss_query_disc': loss_disc * effective_weight}

    def _loss_query_quality(self, outputs, targets, indices):
        if not self.use_query_quality:
            return None
        if not self.training:
            return None
        if 'pred_quality' not in outputs:
            raise RuntimeError(
                'SetCriterion.use_query_quality=true, but model outputs do not contain '
                "'pred_quality'. Keep RTDETRTransformer.use_query_quality in sync."
            )

        pred_quality = outputs['pred_quality']
        if pred_quality.dim() == 3 and pred_quality.shape[-1] == 1:
            pred_quality = pred_quality.squeeze(-1)
        if pred_quality.shape != outputs['pred_logits'].shape[:2]:
            raise RuntimeError(
                'pred_quality must have shape [B, num_queries, 1] or [B, num_queries], '
                f'got {tuple(outputs["pred_quality"].shape)}'
            )

        self.last_query_quality_loss_weight = self.query_quality_loss_weight
        indices = self._validate_indices(targets, indices)
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = outputs['pred_quality'].sum() * 0.0
            self.last_query_quality_loss = 0.0
            self.last_query_quality_match_count = 0
            self.last_query_quality_pred_mean = 0.0
            self.last_query_quality_target_mean = 0.0
            return {'loss_query_quality': zero}

        matched_quality = pred_quality[idx].clamp(min=1e-6, max=1.0 - 1e-6)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][J] for t, (_, J) in zip(targets, indices)],
            dim=0,
        )
        iou_targets, _, _ = aligned_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes),
            eps=self.qdqr_eps,
        )
        iou_targets = iou_targets.clamp(min=0.0, max=1.0)
        if self.query_quality_use_detached_iou_target:
            iou_targets = iou_targets.detach()

        if self.query_quality_loss_type == 'bce':
            quality_loss = F.binary_cross_entropy(matched_quality, iou_targets, reduction='mean')
        else:
            quality_loss = F.l1_loss(matched_quality, iou_targets, reduction='mean')

        if not torch.isfinite(quality_loss).all():
            raise RuntimeError('Non-finite query quality loss detected.')

        self.last_query_quality_loss = float(quality_loss.detach().item())
        self.last_query_quality_match_count = int(matched_quality.numel())
        self.last_query_quality_pred_mean = float(matched_quality.detach().mean().item())
        self.last_query_quality_target_mean = float(iou_targets.detach().mean().item())
        return {'loss_query_quality': quality_loss * self.query_quality_loss_weight}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        if len(indices) == 0:
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty
        target_device = None
        for src, _ in indices:
            if len(src) > 0:
                target_device = src.device
                break
        if target_device is None:
            target_device = indices[0][0].device
        norm_src = [src.to(target_device) for (src, _) in indices]
        batch_idx = torch.cat([torch.full_like(src, i) for i, src in enumerate(norm_src)])
        src_idx = torch.cat(norm_src)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        if len(indices) == 0:
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty
        target_device = None
        for _, tgt in indices:
            if len(tgt) > 0:
                target_device = tgt.device
                break
        if target_device is None:
            target_device = indices[0][1].device
        norm_tgt = [tgt.to(target_device) for (_, tgt) in indices]
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, tgt in enumerate(norm_tgt)])
        tgt_idx = torch.cat(norm_tgt)
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'defect_aware_mal': self.loss_labels_defect_aware_mal,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _get_ddnq_cls_loss_name(self):
        for loss_name in self.losses:
            if self._is_cls_loss_name(loss_name):
                return loss_name
        return None

    def _compute_ddnq_layer_losses(self, dn_outputs, targets, indices, num_boxes_dn):
        zero = dn_outputs['pred_logits'].sum() * 0.0
        layer_losses = {
            'loss_dn_cls': zero,
            'loss_dn_bbox': zero,
            'loss_dn_giou': zero,
        }

        cls_loss_name = self._get_ddnq_cls_loss_name()
        if cls_loss_name is not None:
            cls_dict = self.get_loss(cls_loss_name, dn_outputs, targets, indices, num_boxes_dn, log=False)
            cls_loss = zero
            for key, value in cls_dict.items():
                if key in self.weight_dict:
                    cls_loss = cls_loss + value * self.weight_dict[key]
            layer_losses['loss_dn_cls'] = cls_loss

        if 'boxes' in self.losses:
            box_dict = self.get_loss('boxes', dn_outputs, targets, indices, num_boxes_dn, log=False)
            if 'loss_bbox' in box_dict and 'loss_bbox' in self.weight_dict:
                layer_losses['loss_dn_bbox'] = box_dict['loss_bbox'] * self.weight_dict['loss_bbox']
            if 'loss_iou' in box_dict and 'loss_iou' in self.weight_dict:
                layer_losses['loss_dn_giou'] = box_dict['loss_iou'] * self.weight_dict['loss_iou']

        return layer_losses

    def _loss_ddnq_outputs(self, outputs, targets, num_boxes_o2o):
        dn_meta = outputs['dn_meta']
        if not self.use_ddnq:
            raise RuntimeError(
                'Model produced DDNQ outputs, but SetCriterion.use_ddnq is false. '
                'Keep RTDETRTransformer.use_ddnq and SetCriterion.use_ddnq in sync.'
            )

        indices = self.get_cdn_matched_indices(dn_meta, targets)
        num_group = int(dn_meta['dn_num_group'])
        num_boxes_dn = num_boxes_o2o * num_group
        effective_weight = self._get_ddnq_warmup_weight()
        self.last_ddnq_loss_weight = effective_weight
        self.last_ddnq_num_group = num_group
        self.last_ddnq_valid_count = int(dn_meta.get('ddnq_valid_count', 0))
        self.last_ddnq_slender_count = int(dn_meta.get('ddnq_slender_count', 0))

        losses = {}
        dn_outputs_list = outputs.get('dn_aux_outputs', [])
        if not dn_outputs_list:
            return losses

        last_idx = len(dn_outputs_list) - 1
        for layer_idx, dn_outputs in enumerate(dn_outputs_list):
            layer_losses = self._compute_ddnq_layer_losses(
                dn_outputs,
                targets,
                indices,
                num_boxes_dn,
            )
            suffix = '' if layer_idx == last_idx else f'_aux_{layer_idx}'
            for key, value in layer_losses.items():
                losses[f'{key}{suffix}'] = value * effective_weight

        return losses

    def _compute_perturb_loss(self, outputs, targets, num_boxes_o2o, device):
        perturb_outputs = outputs.get('perturb_outputs', [])
        if not perturb_outputs:
            return None

        if self.query_perturb_end_epoch > 0 and self.current_epoch >= self.query_perturb_end_epoch:
            return None

        total_loss = None
        num_branches = 0

        for branch_out in perturb_outputs:
            if hasattr(self.matcher, 'current_epoch'):
                self.matcher.current_epoch = self.current_epoch
            indices_p = self._validate_indices(targets, self.matcher(branch_out, targets))

            branch_loss = {}
            for loss_name in self.losses:
                if loss_name == 'masks':
                    continue
                kwargs = {}
                if loss_name == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss_name, branch_out, targets, indices_p, num_boxes_o2o, **kwargs)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                branch_loss.update(l_dict)

            if total_loss is None:
                total_loss = branch_loss
            else:
                for k in branch_loss:
                    total_loss[k] = total_loss[k] + branch_loss[k]
            num_branches += 1

        if total_loss is None or num_branches == 0:
            return None

        total_loss = {k: v / num_branches for k, v in total_loss.items()}
        loss_sum = sum(total_loss.values())
        effective_weight = self.query_perturb_loss_weight
        return loss_sum * effective_weight

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        device = next(iter(outputs.values())).device

        if hasattr(self.matcher, 'current_epoch'):
            self.matcher.current_epoch = self.current_epoch

        # Standard Hungarian matching
        indices_o2o = self._validate_indices(targets, self.matcher(outputs_without_aux, targets))

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes_o2o = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices_o2o, num_boxes_o2o)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        self.last_foreground_loss = 0.0
        self.last_foreground_valid_count = 0
        self.last_foreground_pos_count = 0
        self.last_foreground_neg_count = 0
        self.last_foreground_ignore_count = 0
        self.last_foreground_pos_ratio = 0.0
        self.last_foreground_ignore_ratio = 0.0
        self.last_foreground_topk_pos_ratio = 0.0
        self.last_foreground_topk_ignore_ratio = 0.0
        self.last_foreground_loss_weight = 0.0
        fg_loss_dict = self._loss_defect_aware_query(outputs_without_aux, targets)
        if fg_loss_dict is not None:
            losses.update(fg_loss_dict)

        self.last_discriminative_loss = 0.0
        self.last_discriminative_query_count = 0
        self.last_discriminative_loss_weight = 0.0
        self.last_discriminative_compact_loss = 0.0
        self.last_discriminative_margin_loss = 0.0
        self.last_valid_prototype_count = 0
        self.last_negative_prototype_count_mean = 0.0
        self.last_qdqr_weight_mean = 1.0
        self.last_qdqr_weight_min = 1.0
        self.last_qdqr_weight_max = 1.0
        self.last_qdqr_cls_score_mean = 0.0
        self.last_qdqr_iou_mean = 0.0
        self.last_qdqr_defectness_mean = 0.0
        disc_loss_dict = self._loss_query_discriminative(outputs_without_aux, targets, indices_o2o)
        if disc_loss_dict is not None:
            losses.update(disc_loss_dict)

        self.last_query_quality_loss = 0.0
        self.last_query_quality_match_count = 0
        self.last_query_quality_pred_mean = 0.0
        self.last_query_quality_target_mean = 0.0
        self.last_query_quality_loss_weight = 0.0
        self.last_ddnq_loss_weight = 0.0
        self.last_ddnq_num_group = 0
        self.last_ddnq_valid_count = 0
        self.last_ddnq_slender_count = 0
        self.last_pg_o2m_loss = 0.0
        self.last_pg_o2m_weight = 0.0
        self.last_pg_o2m_lite_loss = 0.0
        self.last_pg_o2m_lite_weight = 0.0
        self.last_pg_o2m_lite_aux_pos_count = 0
        self.last_pg_o2m_lite_active_layers = 0
        self.last_pg_o2m_lite_iou_mean = 0.0
        query_quality_loss_dict = self._loss_query_quality(outputs_without_aux, targets, indices_o2o)
        if query_quality_loss_dict is not None:
            losses.update(query_quality_loss_dict)

        # Diagnostic: O2O matched IoU (always computed, cheap with detach)
        if self.training:
            with torch.no_grad():
                self.last_matched_iou_o2o_mean = self._compute_o2o_matched_iou(
                    outputs_without_aux, targets, indices_o2o)

        # Auxiliary decoder outputs
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if hasattr(self.matcher, 'current_epoch'):
                    self.matcher.current_epoch = self.current_epoch
                indices_o2o_aux = self._validate_indices(targets, self.matcher(aux_outputs, targets))
                for loss in self.losses:
                    if loss == 'masks': continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_o2o_aux, num_boxes_o2o, log=False)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Denoising auxiliary outputs
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            dn_type = outputs['dn_meta'].get('dn_type', 'cdn')
            if dn_type == 'ddnq':
                losses.update(self._loss_ddnq_outputs(outputs, targets, num_boxes_o2o))
            else:
                indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
                num_boxes_cdn = num_boxes_o2o * outputs['dn_meta']['dn_num_group']

                for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                    for loss in self.losses:
                        if loss == 'masks': continue
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes_cdn, log=False)
                        l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                        l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        if self.training and self.use_pg_o2m and 'pg_o2m_logits' in outputs:
            pg_loss = self.compute_pg_o2m_lite_loss(
                outputs, targets, indices_o2o,
                epoch=self.current_epoch,
                total_epochs=getattr(self, '_total_epochs', 100),
            )
            if pg_loss is not None:
                losses['loss_pg_o2m'] = pg_loss
            else:
                self.last_pg_o2m_lite_loss = 0.0
                self.last_pg_o2m_lite_weight = 0.0
                self.last_pg_o2m_lite_aux_pos_count = 0
                self.last_pg_o2m_lite_active_layers = 0
                self.last_pg_o2m_lite_iou_mean = 0.0

        if self.training and self.use_query_perturb:
            if self.current_epoch >= self.query_perturb_start_epoch:
                p_loss = self._compute_perturb_loss(outputs, targets, num_boxes_o2o, device)
                if p_loss is not None:
                    losses['loss_query_perturb'] = p_loss

        return losses
    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices

    def _get_pg_o2m_lite_decay_weight(self, epoch, total_epochs):
        if not self.pg_o2m_lite_decay:
            return 1.0
        if total_epochs <= 0:
            return 1.0
        decay_start = int(self.pg_o2m_lite_decay_start_ratio * total_epochs)
        decay_end = int(self.pg_o2m_lite_decay_end_ratio * total_epochs)
        if epoch < decay_start:
            return 1.0
        if epoch >= decay_end:
            return 0.0
        progress = (epoch - decay_start) / max(decay_end - decay_start, 1)
        return max(1.0 - progress, 0.0)

    def _get_pg_o2m_lite_cls_loss_name(self):
        for loss_name in self.losses:
            if self._is_cls_loss_name(loss_name):
                return loss_name
        return None

    def compute_pg_o2m_lite_loss(self, outputs, targets, indices_o2o, epoch, total_epochs):
        pg_logits_all = outputs['pg_o2m_logits']
        pg_boxes_all = outputs['pg_o2m_boxes']

        num_layers, B, N, num_q = pg_logits_all.shape[:4]
        device = pg_logits_all.device

        decay_weight = self._get_pg_o2m_lite_decay_weight(epoch, total_epochs)
        if decay_weight <= 0:
            return None

        effective_weight = self.pg_o2m_loss_weight * decay_weight

        cls_loss_name = self._get_pg_o2m_lite_cls_loss_name()
        if cls_loss_name is None:
            return None

        total_cls_loss = pg_logits_all.new_zeros(())
        total_pos_count = 0
        total_iou_sum = 0.0
        active_layer_count = 0

        for layer_idx in range(num_layers):
            if layer_idx not in self.pg_o2m_lite_layers:
                continue

            active_layer_count += 1
            layer_logits = pg_logits_all[layer_idx]
            layer_boxes = pg_boxes_all[layer_idx]

            layer_boxes_xyxy = box_cxcywh_to_xyxy(layer_boxes.reshape(B * N * num_q, 4))

            for b in range(B):
                tgt_labels = targets[b]['labels']
                tgt_boxes = targets[b]['boxes']
                num_gt = len(tgt_labels)

                if num_gt == 0:
                    continue

                hungarian_src = indices_o2o[b][0]
                hungarian_set = set(hungarian_src.tolist()) if self.pg_o2m_lite_exclude_hungarian else set()

                tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)

                offset = b * N * num_q
                cand_xyxy = layer_boxes_xyxy[offset:offset + N * num_q]
                iou_mat, _ = box_iou(cand_xyxy, tgt_xyxy)

                for gt_i in range(num_gt):
                    gt_iou = iou_mat[:, gt_i]

                    valid_mask = gt_iou > self.pg_o2m_lite_iou_thr
                    if self.pg_o2m_lite_exclude_hungarian:
                        for q_idx in range(len(gt_iou)):
                            main_q_idx = q_idx % num_q
                            if main_q_idx in hungarian_set:
                                valid_mask[q_idx] = False

                    if not valid_mask.any():
                        continue

                    valid_iou = gt_iou[valid_mask]
                    valid_indices = torch.where(valid_mask)[0]

                    k = min(self.pg_o2m_lite_top_m, len(valid_iou))
                    _, topk_local = torch.topk(valid_iou, k)
                    topk_global = valid_indices[topk_local]

                    target_class = tgt_labels[gt_i].item()
                    aux_logits = layer_logits[b].reshape(N * num_q, -1)[topk_global]

                    target = aux_logits.new_zeros(aux_logits.shape[0], self.num_classes)
                    target[:, target_class] = 1.0

                    aux_cls_loss = torchvision.ops.sigmoid_focal_loss(
                        aux_logits, target, self.alpha, self.gamma, reduction='mean'
                    )
                    total_cls_loss = total_cls_loss + aux_cls_loss
                    total_pos_count += k
                    total_iou_sum += valid_iou[topk_local].sum().item()

        if total_pos_count == 0:
            return None

        loss_pg_o2m = total_cls_loss / max(active_layer_count, 1)

        self.last_pg_o2m_lite_loss = float(loss_pg_o2m.detach().item())
        self.last_pg_o2m_lite_weight = float(effective_weight)
        self.last_pg_o2m_lite_aux_pos_count = total_pos_count
        self.last_pg_o2m_lite_active_layers = active_layer_count
        self.last_pg_o2m_lite_iou_mean = total_iou_sum / max(total_pos_count, 1)

        return loss_pg_o2m * effective_weight





@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
