"""RT-DETRv3 Criterion - faithful port of DINOv3Loss.

Key features from RT-DETRv3:
- O2M GT replication (o2m): replicate GT boxes for one-to-many matching
- Multi-group loss averaging: average losses across all query groups
- PPYOLOEHead auxiliary loss integration
"""

import copy

import torch
import torch.nn.functional as F

from .rtdetr_criterion import SetCriterion
from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register


__all__ = ['SetCriterionV3']


@register
class SetCriterionV3(SetCriterion):
    __share__ = ['num_classes']

    def __init__(self,
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.75,
                 gamma=2.0,
                 eos_coef=1e-4,
                 num_classes=80,
                 o2m=4,
                 **kwargs):
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            eos_coef=eos_coef,
            num_classes=num_classes,
            **kwargs,
        )
        self.o2m = int(o2m)

    def forward(self, outputs, targets):
        device = next(iter(outputs.values())).device

        if hasattr(self.matcher, 'current_epoch'):
            self.matcher.current_epoch = self.current_epoch

        loss = {}
        num_noise_groups = 1

        if 'extra_outputs' in outputs and outputs['extra_outputs']:
            extra_outputs = outputs['extra_outputs']
            num_noise_groups += len(extra_outputs)

            group_output = self._build_single_group_output(outputs)
            group_loss = self._compute_single_group_loss(group_output, targets, device)
            for k, v in group_loss.items():
                loss[k] = v

            for g_idx, extra in enumerate(extra_outputs):
                g_loss = self._compute_single_group_loss(extra, targets, device)
                for k, v in g_loss.items():
                    if '_o2m_branch' not in k and '_aux_o2m' not in k:
                        loss[k] = loss.get(k, 0.0) + v
        else:
            group_loss = self._compute_single_group_loss(outputs, targets, device)
            loss.update(group_loss)

        for k in loss:
            if '_aux' not in k and '_dn' not in k and '_o2m_branch' not in k and '_aux_o2m' not in k:
                loss[k] = loss[k] / num_noise_groups

        return loss

    def _build_single_group_output(self, outputs):
        outputs_without_extra = {k: v for k, v in outputs.items()
                                 if k not in ('extra_outputs',)}
        return outputs_without_extra

    def _compute_single_group_loss(self, outputs, targets, device):
        outputs_without_aux = {k: v for k, v in outputs.items()
                               if 'aux' not in k and 'dn' not in k}

        if self.o2m != 1:
            targets_o2m = [{
                'labels': t['labels'].repeat(self.o2m),
                'boxes': t['boxes'].repeat(self.o2m, 1),
            } for t in targets]
            match_targets = targets_o2m
        else:
            match_targets = targets

        indices_o2o = self._validate_indices(
            match_targets, self.matcher(outputs_without_aux, match_targets))

        if self.o2m != 1:
            for i, (src_idx, tgt_idx) in enumerate(indices_o2o):
                indices_o2o[i] = (src_idx, tgt_idx % len(targets[i]['labels']))

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes_o2o = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss_name in self.losses:
            l_dict = self.get_loss(loss_name, outputs, targets, indices_o2o, num_boxes_o2o)
            l_dict = {k: l_dict[k] * self.weight_dict[k]
                      for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if hasattr(self.matcher, 'current_epoch'):
                    self.matcher.current_epoch = self.current_epoch
                indices_aux = self._validate_indices(
                    match_targets, self.matcher(aux_outputs, match_targets))
                if self.o2m != 1:
                    for j, (src_idx, tgt_idx) in enumerate(indices_aux):
                        indices_aux[j] = (src_idx, tgt_idx % len(targets[j]['labels']))
                for loss_name in self.losses:
                    if loss_name == 'masks':
                        continue
                    l_dict = self.get_loss(loss_name, aux_outputs, targets,
                                           indices_aux, num_boxes_o2o, log=False)
                    l_dict = {k: l_dict[k] * self.weight_dict[k]
                              for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'dn_aux_outputs' in outputs and 'dn_meta' in outputs and outputs['dn_meta'] is not None:
            dn_meta = outputs['dn_meta']
            indices = self.get_cdn_matched_indices(dn_meta, targets)
            num_group = int(dn_meta['dn_num_group'])
            num_boxes_dn = num_boxes_o2o * num_group

            dn_outputs_list = outputs.get('dn_aux_outputs', [])
            for i, aux_outputs in enumerate(dn_outputs_list):
                for loss_name in self.losses:
                    if loss_name == 'masks':
                        continue
                    l_dict = self.get_loss(loss_name, aux_outputs, targets,
                                           indices, num_boxes_dn, log=False)
                    l_dict = {k: l_dict[k] * self.weight_dict[k]
                              for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'aux_o2m_losses' in outputs:
            for k, v in outputs['aux_o2m_losses'].items():
                if isinstance(v, torch.Tensor):
                    losses[k + '_aux_o2m'] = v

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta['dn_positive_idx']
        dn_num_group = dn_meta['dn_num_group']
        dn_match_indices = []
        for i in range(len(targets)):
            num_gt = len(targets[i]['labels'])
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64,
                                      device=targets[i]['labels'].device)
                gt_idx = gt_idx.repeat(dn_num_group)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((
                    torch.zeros(0, dtype=torch.int64, device=targets[i]['labels'].device),
                    torch.zeros(0, dtype=torch.int64, device=targets[i]['labels'].device),
                ))
        return dn_match_indices
