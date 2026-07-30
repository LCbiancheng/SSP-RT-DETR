"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from src.core import register


__all__ = ['RTDETRPostProcessor']


@register
class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category', 'score_threshold', 'nms_threshold']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False, score_threshold=None, nms_threshold=None) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = None
        self.score_threshold = None if score_threshold is None else float(score_threshold)
        self.nms_threshold = None if nms_threshold is None else float(nms_threshold)

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    def _decode_predictions(self, outputs, orig_target_sizes):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        quality_scores = outputs.get('pred_quality', None)
        if quality_scores is not None:
            if quality_scores.dim() == 2:
                quality_scores = quality_scores.unsqueeze(-1)
            quality_scores = quality_scores.to(dtype=logits.dtype).clamp(min=0.0, max=1.0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred = bbox_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            if quality_scores is not None:
                scores = scores * quality_scores
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            if quality_scores is not None:
                scores = scores * quality_scores
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        return labels, boxes, scores

    def forward(self, outputs, orig_target_sizes):
        labels, boxes, scores = self._decode_predictions(outputs, orig_target_sizes)

        if self.deploy_mode == 'raw':
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            if self.score_threshold is not None:
                keep = sco > self.score_threshold
                lab = lab[keep]
                box = box[keep]
                sco = sco[keep]
            
            # Validation/inference should keep boxes from different classes independent.
            if len(lab) > 0 and self.nms_threshold is not None:
                keep = torchvision.ops.batched_nms(box, sco, lab, self.nms_threshold)
                lab = lab[keep]
                box = box[keep]
                sco = sco[keep]
            
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        

    def deploy(self, mode='raw'):
        if mode not in {'raw'}:
            raise ValueError(f'Unsupported deploy mode: {mode!r}.')
        self.eval()
        self.deploy_mode = mode
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )
