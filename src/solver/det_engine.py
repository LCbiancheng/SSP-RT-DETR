"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import csv
import math
import os
import sys
import time
from typing import Iterable

import numpy as np
import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict, dist, is_concise_logging)


QUERY_OUTPUT_METRICS = (
    'query_base_mean',
    'query_base_std',
    'query_base_min',
    'query_base_max',
    'query_fg_mean',
    'query_fg_std',
    'query_fg_min',
    'query_fg_max',
    'query_select_mean',
    'query_select_std',
    'query_select_min',
    'query_select_max',
    'query_topk_base_mean',
    'query_topk_fg_mean',
    'query_topk_select_mean',
    'query_topk_base_overlap',
    'query_topk_fg_overlap',
    'query_beta_eff',
)

CRITERION_DIAGNOSTIC_METRICS = (
    ('fg_pos', 'last_foreground_pos_count'),
    ('fg_neg', 'last_foreground_neg_count'),
    ('fg_ignore', 'last_foreground_ignore_count'),
    ('fg_pos_ratio', 'last_foreground_pos_ratio'),
    ('fg_ignore_ratio', 'last_foreground_ignore_ratio'),
    ('fg_topk_pos_ratio', 'last_foreground_topk_pos_ratio'),
    ('fg_topk_ignore_ratio', 'last_foreground_topk_ignore_ratio'),
    ('fg_loss_weight', 'last_foreground_loss_weight'),
    ('disc_queries', 'last_discriminative_query_count'),
    ('disc_loss_weight', 'last_discriminative_loss_weight'),
    ('disc_compact', 'last_discriminative_compact_loss'),
    ('disc_margin', 'last_discriminative_margin_loss'),
    ('disc_proto_valid', 'last_valid_prototype_count'),
    ('disc_neg_proto_mean', 'last_negative_prototype_count_mean'),
    ('qdqr_w_mean', 'last_qdqr_weight_mean'),
    ('qdqr_w_min', 'last_qdqr_weight_min'),
    ('qdqr_w_max', 'last_qdqr_weight_max'),
    ('qdqr_cls', 'last_qdqr_cls_score_mean'),
    ('qdqr_iou', 'last_qdqr_iou_mean'),
    ('qq_matches', 'last_query_quality_match_count'),
    ('qq_loss_weight', 'last_query_quality_loss_weight'),
    ('qq_pred', 'last_query_quality_pred_mean'),
    ('qq_target', 'last_query_quality_target_mean'),
    ('ddnq_loss_weight', 'last_ddnq_loss_weight'),
    ('ddnq_groups', 'last_ddnq_num_group'),
    ('ddnq_valid', 'last_ddnq_valid_count'),
    ('ddnq_slender', 'last_ddnq_slender_count'),
    ('matched_iou', 'last_matched_iou_o2o_mean'),
)


def _scalar_from_tensor(value):
    if not torch.is_tensor(value):
        return float(value)
    if value.numel() != 1:
        value = value.detach().float().mean()
    return float(value.detach().float().item())


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    # Use scientific notation to avoid showing tiny LR values as 0.000000.
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_vfl', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bbox', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_iou', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    concise_mode = is_concise_logging()
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    device_type = device.type if isinstance(device, torch.device) else str(device)

    if hasattr(criterion, 'current_epoch'):
        criterion.current_epoch = epoch

    # Propagate epoch to decoder for current-stage scheduling hooks.
    model_for_epoch = model.module if hasattr(model, 'module') else model
    if hasattr(model_for_epoch, 'set_epoch'):
        model_for_epoch.set_epoch(epoch)
    elif hasattr(model_for_epoch, 'decoder') and hasattr(model_for_epoch.decoder, 'set_epoch'):
        model_for_epoch.decoder.set_epoch(epoch)
    elif hasattr(model_for_epoch, 'decoder') and hasattr(model_for_epoch.decoder, 'current_epoch'):
        model_for_epoch.decoder.current_epoch = epoch

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=device_type, cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=device_type, enabled=False):
                outputs = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in outputs.items()}
                if 'aux_outputs' in outputs:
                    outputs['aux_outputs'] = [{k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in aux.items()} for aux in outputs['aux_outputs']]
                if 'dn_aux_outputs' in outputs:
                    outputs['dn_aux_outputs'] = [{k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in aux.items()} for aux in outputs['dn_aux_outputs']]
                
                # Debug Check
                if torch.isnan(outputs['pred_boxes']).any():
                    print(f"!!! DET_ENGINE DEBUG: outputs['pred_boxes'] HAS NaN IMMEDIATELY AFTER MODEL() IN BATCH {i} !!!")
                
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            
            # Debug Check
            if torch.isnan(outputs['pred_boxes']).any():
                print(f"!!! DET_ENGINE DEBUG: outputs['pred_boxes'] HAS NaN IMMEDIATELY AFTER MODEL() IN FP32 IN BATCH {i} !!!")
                
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        main_loss_metrics = {
            key: float(loss_dict_reduced[key])
            for key in (
                'loss_vfl',
                'loss_bbox',
                'loss_iou',
                'loss_giou',
                'loss_query_fg',
                'loss_query_disc',
                'loss_query_quality',
                'loss_dn_cls',
                'loss_dn_bbox',
                'loss_dn_giou',
            )
            if key in loss_dict_reduced
        }
        metric_logger.update(**main_loss_metrics)

        query_metrics = {
            key: _scalar_from_tensor(outputs[key])
            for key in QUERY_OUTPUT_METRICS
            if key in outputs
        }
        if query_metrics:
            metric_logger.update(**query_metrics)

        criterion_metrics = {
            metric_name: float(getattr(criterion, attr_name))
            for metric_name, attr_name in CRITERION_DIAGNOSTIC_METRICS
            if hasattr(criterion, attr_name)
        }
        if criterion_metrics:
            metric_logger.update(**criterion_metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if not concise_mode:
        print(f"Epoch [{epoch}] summary | {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    result_csv_name: str = 'experiment_results.csv',
):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    concise_mode = is_concise_logging()

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Timer for FPS
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        infer_start_time = time.time()
        
        outputs = model(samples)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        infer_end_time = time.time()
        metric_logger.update(fps=float(samples.tensors.shape[0] if hasattr(samples, 'tensors') else samples.shape[0]) / (infer_end_time - infer_start_time))

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if not concise_mode:
        print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            coco_stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
            stats['coco_eval_bbox'] = coco_stats
            
            # --- Fine-grained Performance Dashboard ---
            if dist.is_main_process():
                mAP_50_95 = coco_stats[0]
                mAP_50 = coco_stats[1]
                mAP_75 = coco_stats[2]
                AP_S = coco_stats[3]
                AP_M = coco_stats[4]
                AP_L = coco_stats[5]
                
                # Extract Class-Wise APs manually from pycocotools
                ce_bbox = coco_evaluator.coco_eval['bbox']
                # ce_bbox.eval['precision'] is [T, R, K, A, M] 
                # T: iouThr, R: recThr, K: catIds, A: areaRng, M: maxDets
                p = ce_bbox.eval['precision']
                # Calculate AP for each category (A=0 for all areas, M=2 for maxDets=100)
                class_aps = []
                cat_ids = ce_bbox.params.catIds
                for k, cat_id in enumerate(cat_ids):
                    precision_k = p[:, :, k, 0, 2] # All IoUs, All Recalls, Area=all, maxDets=100
                    precision_k = precision_k[precision_k > -1]
                    if len(precision_k) > 0:
                        class_ap = np.mean(precision_k)
                        class_aps.append((cat_id, class_ap))
                    else:
                        class_aps.append((cat_id, -1.0))
                        
                # FPS Calculation
                fps = metric_logger.meters['fps'].global_avg if 'fps' in metric_logger.meters else -1.0
                
                # Parameters & GFLOPs hook (if we want to add thop later, otherwise basic param count)
                try:
                    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6 # in Millions
                except:
                    params = 0.0

                csv_path = os.path.join(str(output_dir), result_csv_name)
                write_header = not os.path.exists(csv_path)
                with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if write_header:
                        header = ['Epoch', 'mAP_50:95', 'mAP_50', 'mAP_75', 'AP_small', 'AP_medium', 'AP_large', 'FPS', 'Params(M)']
                        header.extend([f'Class_{cat_id}_AP' for cat_id, _ in class_aps])
                        writer.writerow(header)
                
                # Inject detailed stats for solver
                stats['AP_detail'] = {
                    'mAP_50_95': mAP_50_95, 'mAP_50': mAP_50, 'mAP_75': mAP_75,
                    'AP_S': AP_S, 'AP_M': AP_M, 'AP_L': AP_L,
                    'FPS': fps, 'Params(M)': params,
                    'Class_APs': {cat_id: cap for cat_id, cap in class_aps}
                }

        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
