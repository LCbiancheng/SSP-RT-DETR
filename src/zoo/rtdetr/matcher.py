"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

by lyuwenyu
"""

import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    box_iou,
    pairwise_mpdiou_cost,
    pairwise_focal_mpdiou_cost,
)

from src.core import register


def _compute_match_costs(
    outputs,
    targets,
    use_focal_loss=False,
    alpha=0.25,
    gamma=2.0,
    bbox_iou_cost_type='giou',
    focal_mpdiou_gamma=0.5,
    mpdiou_eps=1e-7,
):
    bs, num_queries = outputs["pred_logits"].shape[:2]

    if use_focal_loss:
        out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
    else:
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

    out_bbox = outputs["pred_boxes"].flatten(0, 1)
    tgt_ids = torch.cat([v["labels"] for v in targets])
    tgt_bbox = torch.cat([v["boxes"] for v in targets])

    if use_focal_loss:
        out_prob = out_prob[:, tgt_ids]
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class - neg_cost_class
    else:
        cost_class = -out_prob[:, tgt_ids]

    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    pred_xyxy = box_cxcywh_to_xyxy(out_bbox)
    tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
    pair_giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
    pair_iou, _ = box_iou(pred_xyxy, tgt_xyxy)
    cost_giou = -pair_giou
    if bbox_iou_cost_type == 'mpdiou':
        target_image_wh = []
        for target in targets:
            if len(target["boxes"]) == 0:
                continue
            if "size" in target:
                image_wh = target["size"]
            elif "orig_size" in target:
                image_wh = target["orig_size"]
            else:
                image_wh = torch.ones(2, device=out_bbox.device, dtype=out_bbox.dtype)

            image_wh = image_wh.to(device=out_bbox.device, dtype=out_bbox.dtype)
            target_image_wh.append(image_wh.unsqueeze(0).repeat(len(target["boxes"]), 1))

        if target_image_wh:
            target_image_wh = torch.cat(target_image_wh, dim=0)
        else:
            target_image_wh = out_bbox.new_zeros((0, 2))

        cost_loc = pairwise_mpdiou_cost(
            pred_xyxy,
            tgt_xyxy,
            image_wh=target_image_wh,
            eps=mpdiou_eps,
        )
    elif bbox_iou_cost_type == 'focal_mpdiou':
        target_image_wh = []
        for target in targets:
            if len(target["boxes"]) == 0:
                continue
            if "size" in target:
                image_wh = target["size"]
            elif "orig_size" in target:
                image_wh = target["orig_size"]
            else:
                image_wh = torch.ones(2, device=out_bbox.device, dtype=out_bbox.dtype)

            image_wh = image_wh.to(device=out_bbox.device, dtype=out_bbox.dtype)
            target_image_wh.append(image_wh.unsqueeze(0).repeat(len(target["boxes"]), 1))

        if target_image_wh:
            target_image_wh = torch.cat(target_image_wh, dim=0)
        else:
            target_image_wh = out_bbox.new_zeros((0, 2))

        cost_loc = pairwise_focal_mpdiou_cost(
            pred_xyxy,
            tgt_xyxy,
            image_wh=target_image_wh,
            gamma=focal_mpdiou_gamma,
            eps=mpdiou_eps,
        )
    else:
        cost_loc = cost_giou

    return {
        'bs': bs,
        'num_queries': num_queries,
        'cost_class': cost_class,
        'cost_bbox': cost_bbox,
        'cost_giou': cost_giou,
        'cost_loc': cost_loc,
        'pair_iou': pair_iou,
        'out_bbox': out_bbox,
    }


@register
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(
        self,
        weight_dict,
        use_focal_loss=False,
        alpha=0.25,
        gamma=2.0,
        bbox_iou_cost_type='giou',
        focal_mpdiou_gamma=0.5,
        mpdiou_eps=1e-7,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.bbox_iou_cost_type = str(bbox_iou_cost_type).lower()
        self.focal_mpdiou_gamma = float(focal_mpdiou_gamma)
        self.mpdiou_eps = float(mpdiou_eps)

        valid_cost_types = {'giou', 'mpdiou', 'focal_mpdiou'}
        if self.bbox_iou_cost_type not in valid_cost_types:
            raise ValueError(
                f"Unsupported bbox_iou_cost_type: {self.bbox_iou_cost_type}. "
                f"Use one of: {sorted(valid_cost_types)}"
            )

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        total_targets = sum(len(v["boxes"]) for v in targets)
        if total_targets == 0:
            device = outputs["pred_logits"].device
            return [
                (
                    torch.empty(0, dtype=torch.int64, device=device),
                    torch.empty(0, dtype=torch.int64, device=device),
                )
                for _ in targets
            ]

        match_inputs = _compute_match_costs(
            outputs,
            targets,
            use_focal_loss=self.use_focal_loss,
            alpha=self.alpha,
            gamma=self.gamma,
            bbox_iou_cost_type=self.bbox_iou_cost_type,
            focal_mpdiou_gamma=self.focal_mpdiou_gamma,
            mpdiou_eps=self.mpdiou_eps,
        )
        bs = match_inputs['bs']
        num_queries = match_inputs['num_queries']
        cost_class = match_inputs['cost_class']
        cost_bbox = match_inputs['cost_bbox']
        cost_loc = match_inputs['cost_loc']
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_loc
        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=1e8, posinf=1e8, neginf=-1e8)

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
