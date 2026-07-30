from __future__ import annotations

from pathlib import Path

import yaml


INCLUDE_KEY = '__include__'
EXPECTED_INPUT_SIZE = [512, 512]
EXPECTED_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
EXPECTED_NORMALIZE_STD = [0.229, 0.224, 0.225]


def _merge_dict(dst, src):
    for key, value in src.items():
        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            _merge_dict(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_resolved_config(cfg_path):
    cfg_path = Path(cfg_path).resolve()
    with cfg_path.open(encoding='utf-8') as fh:
        raw_cfg = yaml.safe_load(fh) or {}

    merged = {}
    for include_path in raw_cfg.get(INCLUDE_KEY, []):
        include_path = Path(include_path).expanduser()
        if not include_path.is_absolute():
            include_path = cfg_path.parent / include_path
        _merge_dict(merged, load_resolved_config(include_path))

    local_cfg = {k: v for k, v in raw_cfg.items() if k != INCLUDE_KEY}
    return _merge_dict(merged, local_cfg)


def _get_nested(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _loader_ops(cfg, loader_name):
    return _get_nested(
        cfg,
        loader_name,
        'dataset',
        'transforms',
        'ops',
        default=[],
    ) or []


def _find_first_op(ops, op_type):
    for op in ops:
        if isinstance(op, dict) and op.get('type') == op_type:
            return op
    return None


def _as_number_list(value):
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [value]
    return list(value)


def _as_float_list(value):
    return [float(v) for v in _as_number_list(value)]


def _validate_loader_geometry(cfg, errors):
    for loader_name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        ops = _loader_ops(cfg, loader_name)
        op_types = [op.get('type') for op in ops if isinstance(op, dict)]

        resize = _find_first_op(ops, 'Resize')
        resize_size = _as_number_list(resize.get('size') if resize else None)
        if resize_size != EXPECTED_INPUT_SIZE:
            errors.append(
                f'{loader_name} Resize.size expected {EXPECTED_INPUT_SIZE}, got {resize_size}'
            )

        normalize = _find_first_op(ops, 'Normalize')
        if normalize is None:
            errors.append(f'{loader_name} must include Normalize after ConvertDtype')
        else:
            mean = _as_float_list(normalize.get('mean'))
            std = _as_float_list(normalize.get('std'))
            if mean != EXPECTED_NORMALIZE_MEAN or std != EXPECTED_NORMALIZE_STD:
                errors.append(
                    f'{loader_name} Normalize expected ImageNet mean/std, got mean={mean}, std={std}'
                )

        convert_box = _find_first_op(ops, 'ConvertBox')
        if convert_box is None:
            errors.append(f'{loader_name} must include ConvertBox')
        else:
            if convert_box.get('out_fmt') != 'cxcywh' or bool(convert_box.get('normalize')) is not True:
                errors.append(
                    f'{loader_name} ConvertBox expected cxcywh normalized boxes, got {convert_box}'
                )

        required_order = ['Resize', 'ToImageTensor', 'ConvertDtype', 'Normalize', 'ConvertBox']
        order_positions = []
        for required_op in required_order:
            try:
                order_positions.append(op_types.index(required_op))
            except ValueError:
                order_positions.append(None)
        if all(pos is not None for pos in order_positions):
            if order_positions != sorted(order_positions):
                errors.append(
                    f'{loader_name} transform order should be Resize -> ToImageTensor -> '
                    f'ConvertDtype -> Normalize -> ConvertBox, got {op_types}'
                )

    train_ops = _loader_ops(cfg, 'train_dataloader')
    train_op_types = [op.get('type') for op in train_ops if isinstance(op, dict)]
    if 'RandomIoUCrop' in train_op_types and 'SanitizeBoundingBox' not in train_op_types:
        errors.append('train_dataloader uses RandomIoUCrop but lacks SanitizeBoundingBox')


def _validate_model_geometry(cfg, errors):
    encoder_eval_size = _as_number_list(_get_nested(cfg, 'HybridEncoder', 'eval_spatial_size'))
    decoder_eval_size = _as_number_list(_get_nested(cfg, 'RTDETRTransformer', 'eval_spatial_size'))
    if encoder_eval_size != EXPECTED_INPUT_SIZE:
        errors.append(
            f'HybridEncoder.eval_spatial_size expected {EXPECTED_INPUT_SIZE}, got {encoder_eval_size}'
        )
    if decoder_eval_size != EXPECTED_INPUT_SIZE:
        errors.append(
            f'RTDETRTransformer.eval_spatial_size expected {EXPECTED_INPUT_SIZE}, got {decoder_eval_size}'
        )

    multi_scale = _as_number_list(_get_nested(cfg, 'RTDETR', 'multi_scale', default=[]))
    if not multi_scale:
        errors.append('RTDETR.multi_scale must be configured for training')
    else:
        max_stride = max(_as_number_list(_get_nested(cfg, 'RTDETRTransformer', 'feat_strides', default=[32])))
        for size in multi_scale:
            if int(size) % int(max_stride) != 0:
                errors.append(
                    f'RTDETR.multi_scale size {size} must be divisible by max stride {max_stride}'
                )
        if max(multi_scale) != EXPECTED_INPUT_SIZE[0]:
            errors.append(
                f'RTDETR.multi_scale max expected {EXPECTED_INPUT_SIZE[0]}, got {max(multi_scale)}'
            )


def validate_mainline_stage(
    cfg_path,
    stage_alias,
    *,
    expect_s2,
    expect_adaptive_fusion,
    expect_bbox_iou,
    expect_defect_aware_query=None,
    expect_discriminative_aux=None,
    expect_adaptive_fusion_mode=None,
    expect_qdqr=None,
    expect_query_quality=None,
    expect_ddnq=None,
):
    cfg = load_resolved_config(cfg_path)
    errors = []

    _validate_loader_geometry(cfg, errors)
    _validate_model_geometry(cfg, errors)

    query_select_mode = _get_nested(cfg, 'RTDETRTransformer', 'query_select_mode', default='global')
    if query_select_mode != 'global':
        errors.append(f'query_select_mode must stay global, got {query_select_mode!r}')

    decoder_defect_query = bool(
        _get_nested(cfg, 'RTDETRTransformer', 'use_defect_aware_query', default=False)
    )
    criterion_defect_query = bool(
        _get_nested(cfg, 'SetCriterion', 'use_defect_aware_query', default=False)
    )
    if decoder_defect_query != criterion_defect_query:
        errors.append(
            'RTDETRTransformer.use_defect_aware_query and SetCriterion.use_defect_aware_query must match'
        )

    decoder_disc_aux = bool(
        _get_nested(cfg, 'RTDETRTransformer', 'use_discriminative_aux_loss', default=False)
    )
    criterion_disc_aux = bool(
        _get_nested(cfg, 'SetCriterion', 'use_discriminative_aux_loss', default=False)
    )
    if decoder_disc_aux != criterion_disc_aux:
        errors.append(
            'RTDETRTransformer.use_discriminative_aux_loss and SetCriterion.use_discriminative_aux_loss must match'
        )

    if decoder_defect_query:
        query_score_alpha = float(_get_nested(cfg, 'RTDETRTransformer', 'query_score_alpha', default=1.0))
        query_score_beta = float(_get_nested(cfg, 'RTDETRTransformer', 'query_score_beta', default=1.0))
        if query_score_alpha <= 0.0 and query_score_beta <= 0.0:
            errors.append(
                'Defect-aware query scoring requires query_score_alpha > 0 or query_score_beta > 0'
            )

    if decoder_disc_aux:
        decoder_disc_dim = int(
            _get_nested(cfg, 'RTDETRTransformer', 'discriminative_embed_dim', default=128)
        )
        criterion_disc_dim = int(
            _get_nested(cfg, 'SetCriterion', 'discriminative_embed_dim', default=128)
        )
        if decoder_disc_dim != criterion_disc_dim:
            errors.append(
                'RTDETRTransformer.discriminative_embed_dim and SetCriterion.discriminative_embed_dim must match'
            )

    if expect_defect_aware_query is not None and decoder_defect_query != bool(expect_defect_aware_query):
        errors.append(
            'Defect-aware query expected '
            f'{bool(expect_defect_aware_query)}, got {decoder_defect_query}'
        )

    if expect_discriminative_aux is not None and decoder_disc_aux != bool(expect_discriminative_aux):
        errors.append(
            'Discriminative query auxiliary loss expected '
            f'{bool(expect_discriminative_aux)}, got {decoder_disc_aux}'
        )

    decoder_query_quality = bool(_get_nested(cfg, 'RTDETRTransformer', 'use_query_quality', default=False))
    criterion_query_quality = bool(_get_nested(cfg, 'SetCriterion', 'use_query_quality', default=False))
    if decoder_query_quality != criterion_query_quality:
        errors.append('RTDETRTransformer.use_query_quality and SetCriterion.use_query_quality must match')
    if expect_query_quality is not None and decoder_query_quality != bool(expect_query_quality):
        errors.append(
            'Query quality calibration expected '
            f'{bool(expect_query_quality)}, got {decoder_query_quality}'
        )
    if criterion_query_quality:
        query_quality_loss_weight = float(
            _get_nested(cfg, 'SetCriterion', 'query_quality_loss_weight', default=1.0)
        )
        query_quality_loss_type = str(
            _get_nested(cfg, 'SetCriterion', 'query_quality_loss_type', default='l1')
        ).lower()
        if query_quality_loss_weight < 0.0:
            errors.append(
                f'SetCriterion.query_quality_loss_weight must be >= 0, got {query_quality_loss_weight}'
            )
        if query_quality_loss_type not in {'l1', 'bce'}:
            errors.append(
                "SetCriterion.query_quality_loss_type expected one of ['l1', 'bce'], "
                f'got {query_quality_loss_type!r}'
            )

    decoder_ddnq = bool(_get_nested(cfg, 'RTDETRTransformer', 'use_ddnq', default=False))
    criterion_ddnq = bool(_get_nested(cfg, 'SetCriterion', 'use_ddnq', default=False))
    if decoder_ddnq != criterion_ddnq:
        errors.append('RTDETRTransformer.use_ddnq and SetCriterion.use_ddnq must match')
    if expect_ddnq is not None and decoder_ddnq != bool(expect_ddnq):
        errors.append(f'DDNQ expected {bool(expect_ddnq)}, got {decoder_ddnq}')
    if decoder_ddnq:
        ddnq_num_groups = int(_get_nested(cfg, 'RTDETRTransformer', 'ddnq_num_groups', default=5))
        ddnq_label_noise_ratio = float(
            _get_nested(cfg, 'RTDETRTransformer', 'ddnq_label_noise_ratio', default=0.2)
        )
        ddnq_box_noise_scale = float(
            _get_nested(cfg, 'RTDETRTransformer', 'ddnq_box_noise_scale', default=0.2)
        )
        ddnq_slender_thr = float(_get_nested(cfg, 'RTDETRTransformer', 'ddnq_slender_thr', default=2.0))
        ddnq_slender_wh_scale = float(
            _get_nested(cfg, 'RTDETRTransformer', 'ddnq_slender_wh_scale', default=0.5)
        )
        if ddnq_num_groups <= 0:
            errors.append(f'RTDETRTransformer.ddnq_num_groups must be > 0, got {ddnq_num_groups}')
        if not 0.0 <= ddnq_label_noise_ratio <= 1.0:
            errors.append(
                'RTDETRTransformer.ddnq_label_noise_ratio must be in [0, 1], '
                f'got {ddnq_label_noise_ratio}'
            )
        if ddnq_box_noise_scale < 0.0:
            errors.append(
                f'RTDETRTransformer.ddnq_box_noise_scale must be >= 0, got {ddnq_box_noise_scale}'
            )
        if ddnq_slender_thr <= 0.0:
            errors.append(f'RTDETRTransformer.ddnq_slender_thr must be > 0, got {ddnq_slender_thr}')
        if ddnq_slender_wh_scale <= 0.0:
            errors.append(
                'RTDETRTransformer.ddnq_slender_wh_scale must be > 0, '
                f'got {ddnq_slender_wh_scale}'
            )
        ddnq_loss_weight = float(_get_nested(cfg, 'SetCriterion', 'ddnq_loss_weight', default=1.0))
        ddnq_warmup_epochs = int(_get_nested(cfg, 'SetCriterion', 'ddnq_warmup_epochs', default=3))
        if ddnq_loss_weight < 0.0:
            errors.append(f'SetCriterion.ddnq_loss_weight must be >= 0, got {ddnq_loss_weight}')
        if ddnq_warmup_epochs < 0:
            errors.append(f'SetCriterion.ddnq_warmup_epochs must be >= 0, got {ddnq_warmup_epochs}')

    criterion_qdqr = bool(_get_nested(cfg, 'SetCriterion', 'use_qdqr', default=False))
    if criterion_qdqr and not criterion_disc_aux:
        errors.append('SetCriterion.use_qdqr requires SetCriterion.use_discriminative_aux_loss=true')
    if expect_qdqr is not None and criterion_qdqr != bool(expect_qdqr):
        errors.append(f'QDQR expected {bool(expect_qdqr)}, got {criterion_qdqr}')

    bbox_iou_loss_type = str(_get_nested(cfg, 'SetCriterion', 'bbox_iou_loss_type', default='giou')).lower()
    if bbox_iou_loss_type != expect_bbox_iou:
        errors.append(
            f'SetCriterion.bbox_iou_loss_type expected {expect_bbox_iou!r}, got {bbox_iou_loss_type!r}'
        )
    matcher_iou_cost_type = str(
        _get_nested(cfg, 'SetCriterion', 'matcher', 'bbox_iou_cost_type', default='giou')
    ).lower()
    if bbox_iou_loss_type == 'border_giou':
        if matcher_iou_cost_type != 'giou':
            errors.append(
                'Border-GIoU only changes the training bbox loss; '
                f'matcher bbox_iou_cost_type must remain giou, got {matcher_iou_cost_type!r}'
            )
    elif matcher_iou_cost_type != bbox_iou_loss_type:
        errors.append(
            'SetCriterion.matcher.bbox_iou_cost_type must match '
            f'SetCriterion.bbox_iou_loss_type, got {matcher_iou_cost_type!r} vs {bbox_iou_loss_type!r}'
        )
    border_giou_lambda = float(_get_nested(cfg, 'SetCriterion', 'border_giou_lambda', default=0.05))
    if border_giou_lambda < 0.0:
        errors.append(f'SetCriterion.border_giou_lambda must be >= 0, got {border_giou_lambda}')

    use_s2_adaptive_fusion = bool(_get_nested(cfg, 'HybridEncoder', 'use_s2_adaptive_fusion', default=False))
    if use_s2_adaptive_fusion != bool(expect_adaptive_fusion):
        errors.append(
            f'HybridEncoder.use_s2_adaptive_fusion expected {bool(expect_adaptive_fusion)}, '
            f'got {use_s2_adaptive_fusion}'
        )
    adaptive_fusion_mode = str(_get_nested(cfg, 'HybridEncoder', 'adaptive_fusion_mode', default='softmax')).lower()
    valid_fusion_modes = {'softmax'}
    if adaptive_fusion_mode not in valid_fusion_modes:
        errors.append(
            f'HybridEncoder.adaptive_fusion_mode expected one of {sorted(valid_fusion_modes)}, '
            f'got {adaptive_fusion_mode!r}'
        )
    if expect_adaptive_fusion_mode is not None and adaptive_fusion_mode != str(expect_adaptive_fusion_mode).lower():
        errors.append(
            f'HybridEncoder.adaptive_fusion_mode expected {expect_adaptive_fusion_mode!r}, '
            f'got {adaptive_fusion_mode!r}'
        )

    return_idx = list(_get_nested(cfg, 'PResNet', 'return_idx', default=[]))
    encoder_in_channels = list(_get_nested(cfg, 'HybridEncoder', 'in_channels', default=[]))
    encoder_feat_strides = list(_get_nested(cfg, 'HybridEncoder', 'feat_strides', default=[]))
    decoder_feat_strides = list(_get_nested(cfg, 'RTDETRTransformer', 'feat_strides', default=[]))
    decoder_num_levels = int(_get_nested(cfg, 'RTDETRTransformer', 'num_levels', default=0))

    if expect_s2:
        if return_idx != [0, 1, 2, 3]:
            errors.append(f'PResNet.return_idx expected [0, 1, 2, 3], got {return_idx}')
        if encoder_in_channels != [256, 512, 1024, 2048]:
            errors.append(
                'HybridEncoder.in_channels expected [256, 512, 1024, 2048], '
                f'got {encoder_in_channels}'
            )
        if encoder_feat_strides != [4, 8, 16, 32]:
            errors.append(
                'HybridEncoder.feat_strides expected [4, 8, 16, 32], '
                f'got {encoder_feat_strides}'
            )
        if decoder_feat_strides != [4, 8, 16, 32]:
            errors.append(
                'RTDETRTransformer.feat_strides expected [4, 8, 16, 32], '
                f'got {decoder_feat_strides}'
            )
        if decoder_num_levels != 4:
            errors.append(f'RTDETRTransformer.num_levels expected 4, got {decoder_num_levels}')
    else:
        if return_idx != [1, 2, 3]:
            errors.append(f'PResNet.return_idx expected [1, 2, 3], got {return_idx}')
        if encoder_in_channels != [512, 1024, 2048]:
            errors.append(
                'HybridEncoder.in_channels expected [512, 1024, 2048], '
                f'got {encoder_in_channels}'
            )
        if encoder_feat_strides != [8, 16, 32]:
            errors.append(
                'HybridEncoder.feat_strides expected [8, 16, 32], '
                f'got {encoder_feat_strides}'
            )
        if decoder_feat_strides != [8, 16, 32]:
            errors.append(
                'RTDETRTransformer.feat_strides expected [8, 16, 32], '
                f'got {decoder_feat_strides}'
            )
        if decoder_num_levels != 3:
            errors.append(f'RTDETRTransformer.num_levels expected 3, got {decoder_num_levels}')

    if expect_adaptive_fusion:
        adaptive_layers = int(_get_nested(cfg, 'HybridEncoder', 'adaptive_fusion_num_layers', default=0))
        adaptive_reduction = int(_get_nested(cfg, 'HybridEncoder', 'adaptive_fusion_reduction', default=0))
        if adaptive_layers < 1:
            errors.append(f'HybridEncoder.adaptive_fusion_num_layers must be >= 1, got {adaptive_layers}')
        if adaptive_reduction < 1:
            errors.append(f'HybridEncoder.adaptive_fusion_reduction must be >= 1, got {adaptive_reduction}')

    if errors:
        joined = '\n  - '.join(errors)
        raise ValueError(
            f'Mainline ablation stage {stage_alias} failed config guard for {cfg_path}:\n  - {joined}'
        )
