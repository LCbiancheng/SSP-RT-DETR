"""by lyuwenyu
"""

import torch 

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh



def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device
    
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # if label_noise_ratio > 0:
    #     input_query_class = input_query_class.flatten()
    #     pad_gt_mask = pad_gt_mask.flatten()
    #     # half of bbox prob
    #     # mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
    #     mask = torch.rand_like(input_query_class) < (label_noise_ratio * 0.5)
    #     chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
    #     # randomly put a new one here
    #     new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=input_query_class.dtype)
    #     # input_query_class.scatter_(dim=0, index=chosen_idx, value=new_label)
    #     input_query_class[chosen_idx] = new_label
    #     input_query_class = input_query_class.reshape(bs, num_denoising)
    #     pad_gt_mask = pad_gt_mask.reshape(bs, num_denoising)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # class_embed = torch.concat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=device)])
    # input_query_class = torch.gather(
    #     class_embed, input_query_class.flatten(),
    #     axis=0).reshape(bs, num_denoising, -1)
    # input_query_class = class_embed(input_query_class.flatten()).reshape(bs, num_denoising, -1)
    input_query_class = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True
    
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])
    
    return input_query_class, input_query_bbox, attn_mask, dn_meta


def _apply_label_noise(labels, pad_gt_mask, num_classes, label_noise_ratio):
    if label_noise_ratio <= 0 or num_classes <= 1:
        return labels

    noise_mask = (torch.rand(labels.shape, device=labels.device) < float(label_noise_ratio)) & pad_gt_mask
    random_offset = torch.randint(
        low=1,
        high=num_classes,
        size=labels.shape,
        dtype=labels.dtype,
        device=labels.device,
    )
    noisy_labels = (labels + random_offset) % num_classes
    return torch.where(noise_mask, noisy_labels, labels)


def _apply_defect_aware_box_noise(
    boxes,
    pad_gt_mask,
    box_noise_scale,
    slender_thr,
    slender_wh_scale,
    eps=1e-6,
):
    if box_noise_scale <= 0:
        return boxes

    noisy_boxes = boxes.clone()
    wh = boxes[..., 2:].clamp(min=eps)
    aspect_ratio = torch.maximum(
        wh[..., 0] / wh[..., 1],
        wh[..., 1] / wh[..., 0],
    )
    slender_mask = aspect_ratio > float(slender_thr)

    center_noise = (torch.rand_like(noisy_boxes[..., :2]) * 2.0 - 1.0)
    wh_noise = (torch.rand_like(noisy_boxes[..., 2:]) * 2.0 - 1.0)

    center_delta = center_noise * wh * (0.5 * float(box_noise_scale))
    wh_scale = torch.full_like(aspect_ratio, float(box_noise_scale))
    wh_scale = torch.where(
        slender_mask,
        wh_scale * float(slender_wh_scale),
        wh_scale,
    )
    wh_delta = wh_noise * wh * wh_scale.unsqueeze(-1)

    noisy_boxes[..., :2] = noisy_boxes[..., :2] + center_delta
    noisy_boxes[..., 2:] = (noisy_boxes[..., 2:] + wh_delta).clamp(min=eps, max=1.0)
    noisy_boxes[..., :2] = noisy_boxes[..., :2].clamp(min=0.0, max=1.0)
    half_wh = noisy_boxes[..., 2:] * 0.5
    noisy_boxes[..., :2] = torch.maximum(
        torch.minimum(noisy_boxes[..., :2], 1.0 - half_wh),
        half_wh,
    )

    noisy_xyxy = box_cxcywh_to_xyxy(noisy_boxes).clamp(min=0.0, max=1.0)
    noisy_boxes = box_xyxy_to_cxcywh(noisy_xyxy)
    noisy_boxes[..., 2:] = noisy_boxes[..., 2:].clamp(min=eps, max=1.0)

    return torch.where(pad_gt_mask.unsqueeze(-1), noisy_boxes, boxes)


def _build_ddnq_attention_mask(num_denoising, num_queries, max_gt_num, num_group, device):
    tgt_size = num_denoising + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True

    for group_idx in range(num_group):
        start = max_gt_num * group_idx
        end = max_gt_num * (group_idx + 1)
        attn_mask[start:end, :start] = True
        attn_mask[start:end, end:num_denoising] = True

    return attn_mask


def get_defect_aware_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_groups=5,
    label_noise_ratio=0.2,
    box_noise_scale=0.2,
    slender_thr=2.0,
    slender_wh_scale=0.5,
):
    """Build DDNQ denoising queries with defect-aware bbox noise.

    DDNQ uses positive GT-based denoising queries only. For slender defects,
    center noise keeps the configured scale while width/height noise is reduced
    to preserve elongated shapes.
    """
    if targets is None or num_groups <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]
    if len(num_gts) == 0:
        return None, None, None, None

    device = targets[0]['labels'].device
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    bs = len(num_gts)
    num_group = int(num_groups)
    num_denoising = int(max_gt_num * num_group)

    input_query_class = torch.full(
        [bs, max_gt_num],
        num_classes,
        dtype=torch.int64,
        device=device,
    )
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for batch_idx, num_gt in enumerate(num_gts):
        if num_gt > 0:
            input_query_class[batch_idx, :num_gt] = targets[batch_idx]['labels'].to(torch.int64)
            input_query_bbox[batch_idx, :num_gt] = targets[batch_idx]['boxes']
            pad_gt_mask[batch_idx, :num_gt] = True

    input_query_class = input_query_class.tile([1, num_group])
    input_query_bbox = input_query_bbox.tile([1, num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, num_group])

    input_query_class = _apply_label_noise(
        input_query_class,
        pad_gt_mask,
        num_classes,
        label_noise_ratio,
    )
    input_query_bbox = _apply_defect_aware_box_noise(
        input_query_bbox,
        pad_gt_mask,
        box_noise_scale,
        slender_thr,
        slender_wh_scale,
    )

    dn_positive_idx = []
    slender_count = 0
    valid_count = 0
    with torch.no_grad():
        for batch_idx, num_gt in enumerate(num_gts):
            if num_gt <= 0:
                dn_positive_idx.append(torch.zeros(0, dtype=torch.int64, device=device))
                continue
            group_offsets = torch.arange(num_group, dtype=torch.int64, device=device) * max_gt_num
            gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
            dn_positive_idx.append((group_offsets[:, None] + gt_idx[None, :]).reshape(-1))

            target_wh = targets[batch_idx]['boxes'][:, 2:].clamp(min=1e-6)
            target_ratio = torch.maximum(
                target_wh[:, 0] / target_wh[:, 1],
                target_wh[:, 1] / target_wh[:, 0],
            )
            slender_count += int((target_ratio > float(slender_thr)).sum().item()) * num_group
            valid_count += int(num_gt) * num_group

    input_query_bbox = inverse_sigmoid(input_query_bbox)
    input_query_class = class_embed(input_query_class)
    attn_mask = _build_ddnq_attention_mask(
        num_denoising,
        num_queries,
        max_gt_num,
        num_group,
        device,
    )

    dn_meta = {
        'dn_type': 'ddnq',
        'dn_positive_idx': dn_positive_idx,
        'dn_num_group': num_group,
        'dn_num_split': [num_denoising, num_queries],
        'ddnq_max_gt_num': max_gt_num,
        'ddnq_valid_count': valid_count,
        'ddnq_slender_count': slender_count,
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta
