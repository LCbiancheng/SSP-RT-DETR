"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register



@register
class BaselineSetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80,
                 mal_gamma=1.5, mal_eps=1e-6,
                 mal_defect_area_alpha=0.5,
                 mal_defect_area_tau=0.02,
                 mal_defect_shape_beta=0.3,
                 mal_defect_shape_r0=3.0,
                 mal_defect_weight_max=2.0,
                 use_pg_o2m=False,
                 pg_o2m_loss_weight=0.1,
                 pg_o2m_lite_top_m=2,
                 pg_o2m_lite_iou_thr=0.3,
                 pg_o2m_lite_cls_only=True,
                 pg_o2m_lite_exclude_hungarian=True,
                 pg_o2m_lite_decay=True,
                 pg_o2m_lite_decay_start_ratio=0.5,
                 pg_o2m_lite_decay_end_ratio=1.0,
                 pg_o2m_lite_layers=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
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
        self.mal_gamma = float(mal_gamma)
        self.mal_eps = float(mal_eps)
        if self.mal_gamma <= 0:
            raise ValueError('mal_gamma must be > 0')
        if self.mal_eps <= 0:
            raise ValueError('mal_eps must be > 0')

        self.mal_defect_area_alpha = float(mal_defect_area_alpha)
        self.mal_defect_area_tau = float(mal_defect_area_tau)
        self.mal_defect_shape_beta = float(mal_defect_shape_beta)
        self.mal_defect_shape_r0 = float(mal_defect_shape_r0)
        self.mal_defect_weight_max = float(mal_defect_weight_max)

        if self.mal_defect_area_alpha < 0:
            raise ValueError('mal_defect_area_alpha must be >= 0')
        if self.mal_defect_area_tau <= 0:
            raise ValueError('mal_defect_area_tau must be > 0')
        if self.mal_defect_shape_beta < 0:
            raise ValueError('mal_defect_shape_beta must be >= 0')

        self.use_pg_o2m = bool(use_pg_o2m)
        self.pg_o2m_loss_weight = float(pg_o2m_loss_weight)
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

        self.current_epoch = 0
        self._total_epochs = 100

        if self.pg_o2m_loss_weight < 0:
            raise ValueError('pg_o2m_loss_weight must be >= 0')
        if self.pg_o2m_lite_top_m < 1:
            raise ValueError('pg_o2m_lite_top_m must be >= 1')


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
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
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
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

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
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

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, log=True):
        """MAL-VFL hybrid: VFL BCE framework + MAL's q^gamma matchability signal.

        Keeps VFL's stable BCE training dynamics, but replaces target_score = iou
        with target_score = iou^mal_gamma (MAL's matchability modulation).
        This avoids MAL's (1-q^gamma)*log(1-p) term which destroys positive
        gradient for low-IoU matches (critical for sparse datasets like NEU-DET).
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

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

    def loss_labels_defect_aware_mal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(target_boxes) > 0:
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
            defect_weights = self._compute_defect_aware_weights(
                target_boxes, target_boxes.device, target_boxes.dtype)
        else:
            ious = src_boxes.new_zeros(0)
            defect_weights = src_boxes.new_zeros(0)

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

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

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
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
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

    @staticmethod
    def _is_cls_loss_name(loss_name):
        return loss_name in {'labels', 'bce', 'focal', 'vfl'}

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
        total_box_loss = pg_logits_all.new_zeros(())
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

                    if not self.pg_o2m_lite_cls_only:
                        aux_boxes = layer_boxes[b].reshape(N * num_q, -1)[topk_global]
                        tgt_box = tgt_boxes[gt_i].unsqueeze(0).expand(k, -1)

                        loss_l1 = F.l1_loss(aux_boxes, tgt_box, reduction='none').sum(dim=-1).mean()

                        aux_xyxy = box_cxcywh_to_xyxy(aux_boxes)
                        tgt_xyxy = box_cxcywh_to_xyxy(tgt_box)
                        loss_iou = (1 - torch.diag(generalized_box_iou(aux_xyxy, tgt_xyxy))).mean()

                        total_box_loss = total_box_loss + loss_l1 + loss_iou

                    total_pos_count += k
                    total_iou_sum += valid_iou[topk_local].sum().item()

        if total_pos_count == 0:
            self.last_pg_o2m_lite_iou_mean = 0.0
            return None

        if not self.pg_o2m_lite_cls_only and total_box_loss > 0:
            loss_pg_o2m = (total_cls_loss + total_box_loss) / max(active_layer_count, 1)
        else:
            loss_pg_o2m = total_cls_loss / max(active_layer_count, 1)

        self.last_pg_o2m_lite_iou_mean = total_iou_sum / max(total_pos_count, 1)
        return loss_pg_o2m * effective_weight

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        device = next(iter(outputs.values())).device

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        num_boxes_o2o = num_boxes

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if self.training and self.use_pg_o2m and 'pg_o2m_logits' in outputs:
            pg_loss = self.compute_pg_o2m_lite_loss(
                outputs, targets, indices,
                epoch=self.current_epoch,
                total_epochs=getattr(self, '_total_epochs', 100),
            )
            if pg_loss is not None:
                losses['loss_pg_o2m'] = pg_loss

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
