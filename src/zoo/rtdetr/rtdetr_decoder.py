"""by lyuwenyu
"""

import math 
import copy 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import (
    get_contrastive_denoising_training_group,
    get_defect_aware_denoising_training_group,
)
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob


from src.core import register


__all__ = ['RTDETRTransformer']



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value = value * value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # target self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos_embed),
            reference_points=reference_points,
            value=memory,
            value_spatial_shapes=memory_spatial_shapes,
            value_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt.clamp(min=-65504, max=65504))

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                *args,
                **kwargs):
        memory_level_start_index = args[0]
        bbox_head, score_head, query_pos_head = args[1:]

        attn_mask = kwargs.get('attn_mask', None)
        memory_mask = kwargs.get('memory_mask', None)

        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_features = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            query_pos_embed = query_pos_head(ref_points_detach)

            # Standard RT-DETR layer expects reference points as [B, L, num_levels, 4/2]
            n_levels = len(memory_spatial_shapes)
            ref_points_input = ref_points_detach.unsqueeze(2).expand(-1, -1, n_levels, -1)
            
            output = layer(
                output, ref_points_input, memory, memory_spatial_shapes,
                memory_level_start_index, attn_mask, memory_mask, query_pos_embed,
            )

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            inter_ref_bbox = torch.nan_to_num(inter_ref_bbox, nan=0.5)
            layer_logits = score_head[i](output)

            if self.training:
                dec_out_logits.append(layer_logits)
                # Keep the stored layer outputs consistent with the actual
                # detached reference points used by this decoder layer.
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_features.append(output)
            elif i == self.eval_idx:
                dec_out_logits.append(layer_logits)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_features.append(output)
                break

            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), torch.stack(dec_out_features)


@register
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 use_ddnq=False,
                 ddnq_num_groups=5,
                 ddnq_label_noise_ratio=0.2,
                 ddnq_box_noise_scale=0.2,
                 ddnq_slender_thr=2.0,
                 ddnq_slender_wh_scale=0.5,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 query_select_mode='global',
                 query_select_level_power=0.5,
                 use_defect_aware_query=False,
                 query_score_alpha=1.0,
                 query_score_beta=1.0,
                 query_score_beta_start=0.0,
                 query_score_beta_warmup_epochs=0,
                 defect_query_hidden_ratio=4,
                 use_discriminative_aux_loss=False,
                 discriminative_embed_dim=128,
                 use_query_quality=False,
                 use_pg_o2m=False,
                 pg_o2m_num_groups=3,
                 pg_o2m_mask_ratio=0.3,
                 pg_o2m_noise_std=0.05,
                 pg_o2m_detach_aux_query=True,
                 pg_o2m_center_jitter=0.03,
                 pg_o2m_scale_jitter=0.05,
                 use_query_perturb=False,
                 num_perturb_branch=2,
                 query_noise_std=0.01,
                 ref_noise_scale=0.02,
                 perturb_start_epoch=5,
                 perturb_end_epoch=-1):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.query_select_mode = str(query_select_mode).lower()
        self.query_select_level_power = float(query_select_level_power)
        self.use_defect_aware_query = bool(use_defect_aware_query)
        self.query_score_alpha = float(query_score_alpha)
        self.query_score_beta = float(query_score_beta)
        self.query_score_beta_start = float(query_score_beta_start)
        self.query_score_beta_warmup_epochs = int(query_score_beta_warmup_epochs)
        self.use_discriminative_aux_loss = bool(use_discriminative_aux_loss)
        self.discriminative_embed_dim = int(discriminative_embed_dim)
        self.use_query_quality = bool(use_query_quality)
        self.use_pg_o2m = bool(use_pg_o2m)
        self.pg_o2m_num_groups = int(pg_o2m_num_groups)
        self.pg_o2m_mask_ratio = float(pg_o2m_mask_ratio)
        self.pg_o2m_noise_std = float(pg_o2m_noise_std)
        self.pg_o2m_detach_aux_query = bool(pg_o2m_detach_aux_query)
        self.pg_o2m_center_jitter = float(pg_o2m_center_jitter)
        self.pg_o2m_scale_jitter = float(pg_o2m_scale_jitter)
        self.use_query_perturb = bool(use_query_perturb)
        self.num_perturb_branch = int(num_perturb_branch)
        self.query_noise_std = float(query_noise_std)
        self.ref_noise_scale = float(ref_noise_scale)
        self.perturb_start_epoch = int(perturb_start_epoch)
        self.perturb_end_epoch = int(perturb_end_epoch)
        if self.num_perturb_branch < 1:
            raise ValueError('num_perturb_branch must be >= 1')
        if self.query_noise_std < 0:
            raise ValueError('query_noise_std must be >= 0')
        if self.ref_noise_scale < 0:
            raise ValueError('ref_noise_scale must be >= 0')
        if self.perturb_start_epoch < 0:
            raise ValueError('perturb_start_epoch must be >= 0')
        self.current_epoch = 0

        valid_query_select_modes = {'global', 'level_aware'}
        if self.query_select_mode not in valid_query_select_modes:
            raise ValueError(
                f"Unsupported query_select_mode: {self.query_select_mode}. "
                f"Use one of: {sorted(valid_query_select_modes)}"
            )
        if self.use_defect_aware_query and (self.query_score_alpha <= 0.0 and self.query_score_beta <= 0.0):
            raise ValueError(
                'query_score_alpha and query_score_beta cannot both be <= 0 when use_defect_aware_query=true.'
            )
        if self.query_score_beta_start < 0.0:
            raise ValueError('query_score_beta_start must be >= 0.')
        if self.query_score_beta_warmup_epochs < 0:
            raise ValueError('query_score_beta_warmup_epochs must be >= 0.')
        if defect_query_hidden_ratio <= 0:
            raise ValueError('defect_query_hidden_ratio must be > 0.')
        if self.discriminative_embed_dim <= 0:
            raise ValueError('discriminative_embed_dim must be > 0.')

        self.last_query_select_counts = []
        self.last_query_select_ratios = []
        self.last_query_level_names = []
        self.last_query_select_score_means = []
        self.last_query_select_score_stds = []
        self.last_topk_indices = None
        self.last_topk_level_slices = []
        self.last_topk_level_names = []
        self.last_topk_bboxes = None

        # Standard RT-DETR multi-scale projection.
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim)
            ) for _ in range(num_levels)
        ])

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.use_ddnq = bool(use_ddnq)
        self.ddnq_num_groups = int(ddnq_num_groups)
        self.ddnq_label_noise_ratio = float(ddnq_label_noise_ratio)
        self.ddnq_box_noise_scale = float(ddnq_box_noise_scale)
        self.ddnq_slender_thr = float(ddnq_slender_thr)
        self.ddnq_slender_wh_scale = float(ddnq_slender_wh_scale)
        if self.ddnq_num_groups <= 0:
            raise ValueError('ddnq_num_groups must be > 0.')
        if not 0.0 <= self.ddnq_label_noise_ratio <= 1.0:
            raise ValueError('ddnq_label_noise_ratio must be in [0, 1].')
        if self.ddnq_box_noise_scale < 0.0:
            raise ValueError('ddnq_box_noise_scale must be >= 0.')
        if self.ddnq_slender_thr <= 0.0:
            raise ValueError('ddnq_slender_thr must be > 0.')
        if self.ddnq_slender_wh_scale <= 0.0:
            raise ValueError('ddnq_slender_wh_scale must be > 0.')
        if num_denoising > 0 or self.use_ddnq:
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        if self.use_defect_aware_query:
            fg_hidden_dim = max(1, hidden_dim // int(defect_query_hidden_ratio))
            self.enc_foreground_head = MLP(hidden_dim, fg_hidden_dim, 1, num_layers=2, act='gelu')
        else:
            self.enc_foreground_head = None

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        if self.use_discriminative_aux_loss:
            self.query_discriminative_head = nn.Linear(hidden_dim, self.discriminative_embed_dim)
        else:
            self.query_discriminative_head = None
        if self.use_query_quality:
            self.query_quality_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                get_activation('silu'),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.query_quality_head = None

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        if self.enc_foreground_head is not None:
            init.constant_(self.enc_foreground_head.layers[-1].bias, bias)
        if self.query_discriminative_head is not None:
            init.xavier_uniform_(self.query_discriminative_head.weight)
            init.constant_(self.query_discriminative_head.bias, 0)
        if self.query_quality_head is not None:
            init.xavier_uniform_(self.query_quality_head[0].weight)
            init.constant_(self.query_quality_head[0].bias, 0)
            init.xavier_uniform_(self.query_quality_head[2].weight)
            init.constant_(self.query_quality_head[2].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _get_effective_query_score_beta(self):
        if not self.training or self.query_score_beta_warmup_epochs <= 0:
            return self.query_score_beta

        progress = min(
            max(float(self.current_epoch) / float(self.query_score_beta_warmup_epochs), 0.0),
            1.0,
        )
        return self.query_score_beta_start + progress * (self.query_score_beta - self.query_score_beta_start)

    @staticmethod
    def _masked_stats(scores, valid_mask):
        if valid_mask.shape != scores.shape:
            if (
                valid_mask.dim() == 2
                and scores.dim() == 2
                and valid_mask.shape[0] == 1
                and valid_mask.shape[1] == scores.shape[1]
            ):
                valid_mask = valid_mask.expand_as(scores)
            else:
                raise ValueError(
                    f'valid_mask shape {tuple(valid_mask.shape)} does not match '
                    f'score shape {tuple(scores.shape)}.'
                )
        # Use torch.masked_select to ensure indices stay within bounds
        valid_scores = torch.masked_select(scores, valid_mask)
        if valid_scores.numel() == 0:
            zero = scores.new_tensor(0.0)
            return zero, zero, zero, zero
        return (
            valid_scores.mean(),
            valid_scores.std(unbiased=False),
            valid_scores.min(),
            valid_scores.max(),
        )

    @staticmethod
    def _topk_overlap(topk_a, topk_b):
        if topk_a is None or topk_b is None or topk_a.numel() == 0 or topk_b.numel() == 0:
            return topk_a.new_tensor(0.0) if topk_a is not None else torch.tensor(0.0)
        overlap = (topk_a.unsqueeze(-1) == topk_b.unsqueeze(-2)).any(dim=-1)
        return overlap.to(dtype=torch.float32).mean()

    def _build_query_score_diagnostics(self, base_scores, fg_scores, selection_scores, valid, topk_ind):
        diagnostics = {}
        with torch.no_grad():
            for prefix, scores in (
                ('query_base', base_scores),
                ('query_select', selection_scores),
            ):
                mean, std, min_value, max_value = self._masked_stats(scores.detach(), valid)
                diagnostics[f'{prefix}_mean'] = mean.detach()
                diagnostics[f'{prefix}_std'] = std.detach()
                diagnostics[f'{prefix}_min'] = min_value.detach()
                diagnostics[f'{prefix}_max'] = max_value.detach()

            topk_base = base_scores.detach().gather(1, topk_ind)
            topk_select = selection_scores.detach().gather(1, topk_ind)
            diagnostics['query_topk_base_mean'] = topk_base.mean().detach()
            diagnostics['query_topk_select_mean'] = topk_select.mean().detach()

            base_rank_scores = base_scores.detach().masked_fill(~valid, float('-inf'))
            _, base_topk_ind = torch.topk(base_rank_scores, topk_ind.shape[1], dim=1)
            diagnostics['query_topk_base_overlap'] = self._topk_overlap(topk_ind.detach(), base_topk_ind).detach()

            if fg_scores is not None:
                mean, std, min_value, max_value = self._masked_stats(fg_scores.detach(), valid)
                diagnostics['query_fg_mean'] = mean.detach()
                diagnostics['query_fg_std'] = std.detach()
                diagnostics['query_fg_min'] = min_value.detach()
                diagnostics['query_fg_max'] = max_value.detach()
                diagnostics['query_topk_fg_mean'] = fg_scores.detach().gather(1, topk_ind).mean().detach()

                fg_rank_scores = fg_scores.detach().masked_fill(~valid, float('-inf'))
                _, fg_topk_ind = torch.topk(fg_rank_scores, topk_ind.shape[1], dim=1)
                diagnostics['query_topk_fg_overlap'] = self._topk_overlap(topk_ind.detach(), fg_topk_ind).detach()
                diagnostics['query_beta_eff'] = selection_scores.new_tensor(
                    float(self._get_effective_query_score_beta())
                )

        return diagnostics


    def _get_encoder_input(self, nodes_dict_or_list):
        if isinstance(nodes_dict_or_list, dict):
            if 'multi_scale_features' in nodes_dict_or_list:
                feat_list = list(nodes_dict_or_list['multi_scale_features'])
            else:
                node_keys = ['node_1', 'node_2', 'node_3', 'node_4']
                feat_list = [nodes_dict_or_list[k] for k in node_keys if k in nodes_dict_or_list]
        else:
            feat_list = nodes_dict_or_list

        if not isinstance(feat_list, (list, tuple)):
            raise TypeError(
                f'Decoder expects encoder features as list/tuple, but got {type(feat_list)!r}.'
            )
        if len(feat_list) == 0:
            raise ValueError('Decoder received zero encoder feature levels.')

        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feat_list[:self.num_levels])]

        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feat_list[:self.num_levels][-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        standard_memory_flatten = []
        standard_spatial_shapes = []
        standard_memory_levels = []
        level_start_index = [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            level_flat = feat.flatten(2).permute(0, 2, 1)
            standard_memory_flatten.append(level_flat)
            standard_memory_levels.append(level_flat)
            standard_spatial_shapes.append([h, w])
            level_start_index.append(h * w + level_start_index[-1])

        standard_memory_flatten = torch.concat(standard_memory_flatten, 1)
        level_start_index.pop()

        return standard_memory_flatten, standard_spatial_shapes, level_start_index, standard_memory_levels

    def _compute_level_query_budgets(self, level_lengths):
        if not level_lengths:
            return []

        level_lengths = [int(length) for length in level_lengths]
        weights = torch.tensor(
            [float(max(length, 1)) ** self.query_select_level_power for length in level_lengths],
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp(min=1e-12)

        raw_budgets = weights * float(self.num_queries)
        budgets = torch.floor(raw_budgets).to(torch.int64)
        capacities = torch.tensor(level_lengths, dtype=torch.int64)
        budgets = torch.minimum(budgets, capacities)

        remaining = int(self.num_queries - budgets.sum().item())
        if remaining > 0:
            fractional = raw_budgets - budgets.to(raw_budgets.dtype)
            order = torch.argsort(fractional, descending=True).tolist()
            while remaining > 0:
                progressed = False
                for level_idx in order:
                    if remaining <= 0:
                        break
                    if budgets[level_idx] >= capacities[level_idx]:
                        continue
                    budgets[level_idx] += 1
                    remaining -= 1
                    progressed = True
                if not progressed:
                    break

        return budgets.tolist()

    def _select_topk_indices(self, selection_scores, standard_spatial_shapes):
        if selection_scores.shape[1] < self.num_queries:
            raise ValueError(
                f'Not enough encoder tokens for query selection: '
                f'got {selection_scores.shape[1]}, need num_queries={self.num_queries}. '
                'Use a larger input size or reduce RTDETRTransformer.num_queries.'
            )

        if self.query_select_mode == 'global' or len(standard_spatial_shapes) <= 1:
            _, topk_ind = torch.topk(selection_scores, self.num_queries, dim=1)

            self.last_query_select_counts = [self.num_queries]
            self.last_query_select_ratios = [1.0]
            self.last_query_level_names = ['global']
            self.last_query_select_score_means = [float(selection_scores.gather(1, topk_ind).mean().item())]
            self.last_query_select_score_stds = [
                float(selection_scores.gather(1, topk_ind).std(unbiased=False).item())
            ]
            self.last_topk_indices = topk_ind.detach()
            self.last_topk_level_slices = [(0, selection_scores.shape[1])]
            self.last_topk_level_names = ['global']
            return topk_ind

        level_lengths = [int(h * w) for h, w in standard_spatial_shapes]
        level_names = [
            f's{int(self.feat_strides[idx])}' if idx < len(self.feat_strides) else f'level_{idx}'
            for idx in range(len(level_lengths))
        ]
        level_budgets = self._compute_level_query_budgets(level_lengths)

        selected_indices = []
        selected_scores = []
        selected_score_means = []
        selected_score_stds = []
        level_slices = []
        start = 0
        for level_idx, (level_length, budget) in enumerate(zip(level_lengths, level_budgets)):
            end = start + level_length
            level_slices.append((start, end))
            level_scores = selection_scores[:, start:end]
            if budget > 0:
                level_score, level_index = torch.topk(level_scores, budget, dim=1)
                selected_scores.append(level_score)
                selected_indices.append(level_index + start)
                selected_score_means.append(float(level_score.mean().item()))
                selected_score_stds.append(float(level_score.std(unbiased=False).item()))
            else:
                selected_score_means.append(0.0)
                selected_score_stds.append(0.0)
            start = end

        merged_scores = torch.cat(selected_scores, dim=1)
        merged_indices = torch.cat(selected_indices, dim=1)
        order = torch.argsort(merged_scores, dim=1, descending=True)
        topk_ind = merged_indices.gather(1, order)

        self.last_query_select_counts = [int(v) for v in level_budgets]
        self.last_query_select_ratios = [float(v) / float(self.num_queries) for v in level_budgets]
        self.last_query_level_names = level_names
        self.last_query_select_score_means = selected_score_means
        self.last_query_select_score_stds = selected_score_stds
        self.last_topk_indices = topk_ind.detach()
        self.last_topk_level_slices = level_slices
        self.last_topk_level_names = level_names
        return topk_ind

    def _generate_anchors(self,
                          spatial_shapes=None,
                          level_strides=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        if level_strides is None:
            level_strides = self.feat_strides[:len(spatial_shapes)]
        base_stride = float(min(level_strides)) if level_strides else 1.0
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            stride_scale = float(level_strides[lvl]) / base_stride
            wh = torch.ones_like(grid_xy) * grid_size * stride_scale
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.zeros_like(anchors))

        return anchors, valid_mask


    def _get_decoder_input(self,
                           standard_memory_levels,
                           standard_spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        flatten_for_scoring = torch.concat(standard_memory_levels, dim=1)
        level_strides = self.feat_strides[:len(standard_spatial_shapes)]

        bs, _, _ = flatten_for_scoring.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                standard_spatial_shapes,
                level_strides=level_strides,
                device=flatten_for_scoring.device,
            )
        else:
            anchors, valid_mask = self._generate_anchors(
                standard_spatial_shapes,
                level_strides=level_strides,
                device=flatten_for_scoring.device,
            )

        memory = valid_mask.to(flatten_for_scoring.dtype) * flatten_for_scoring

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        enc_foreground_logits = None
        if self.enc_foreground_head is not None:
            enc_foreground_logits = self.enc_foreground_head(output_memory).squeeze(-1)

        # Mask invalid anchor scores to -inf to ensure topk doesn't select them
        valid = valid_mask.squeeze(-1).expand(bs, -1)
        enc_outputs_class = enc_outputs_class.masked_fill(~valid.unsqueeze(-1), float('-inf'))
        base_scores = torch.sigmoid(enc_outputs_class.max(-1).values)
        fg_scores = None
        if enc_foreground_logits is not None:
            fg_scores = torch.sigmoid(enc_foreground_logits).masked_fill(~valid, 0.0)
            effective_beta = self._get_effective_query_score_beta()
            combined_weight = self.query_score_alpha + effective_beta
            selection_scores = (
                self.query_score_alpha * base_scores +
                effective_beta * fg_scores
            ) / max(combined_weight, 1e-6)
        else:
            selection_scores = base_scores
        selection_scores = selection_scores.masked_fill(~valid, float('-inf'))
        topk_ind = self._select_topk_indices(selection_scores, standard_spatial_shapes)
        
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        self.last_topk_bboxes = enc_topk_bboxes.detach()
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        extra_outputs = {
            'enc_token_boxes': F.sigmoid(enc_outputs_coord_unact),
            'enc_token_anchors': torch.sigmoid(anchors).expand(bs, -1, -1),
            'enc_token_logits': enc_outputs_class,
            'enc_valid_mask': valid.expand(bs, -1),
            'topk_indices': topk_ind,
            'query_selection_scores': selection_scores,
        }
        if enc_foreground_logits is not None:
            extra_outputs['enc_foreground_logits'] = enc_foreground_logits
        extra_outputs.update(
            self._build_query_score_diagnostics(base_scores, fg_scores, selection_scores, valid, topk_ind)
        )

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits, bs, extra_outputs


    def forward(self, nodes_dict, targets=None):

        standard_memory_flatten, standard_spatial_shapes, level_start_index, standard_memory_levels = self._get_encoder_input(nodes_dict)
            
        # Decide what is passed as Memory to Decoder (Multi-scale vs 4 nodes)
        decoder_memory = standard_memory_flatten
        decoder_spatial_shapes = standard_spatial_shapes
        decoder_level_index = level_start_index
        
        # prepare denoising training
        if self.training and self.use_ddnq:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_defect_aware_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_groups=self.ddnq_num_groups,
                    label_noise_ratio=self.ddnq_label_noise_ratio,
                    box_noise_scale=self.ddnq_box_noise_scale,
                    slender_thr=self.ddnq_slender_thr,
                    slender_wh_scale=self.ddnq_slender_wh_scale,
                )
        elif self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, bs, decoder_meta = \
            self._get_decoder_input(standard_memory_levels, standard_spatial_shapes, denoising_class, denoising_bbox_unact)
        
        # decoder
        out_bboxes, out_logits, out_query_features = self.decoder(
            target,
            init_ref_points_unact,
            decoder_memory,
            decoder_spatial_shapes,
            decoder_level_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_query_features, out_query_features = torch.split(
                out_query_features, dn_meta['dn_num_split'], dim=2
            )

            
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if isinstance(nodes_dict, dict):
            if 'f5_raw' in nodes_dict:
                out['student_f5_raw'] = nodes_dict['f5_raw']
                out['student_f5'] = out['student_f5_raw']
            elif 'node_1' in nodes_dict:
                out['student_f5_raw'] = nodes_dict['node_1']
                out['student_f5'] = out['student_f5_raw']

        out['query_select_counts'] = torch.tensor(self.last_query_select_counts, device=out_logits.device, dtype=out_logits.dtype)
        out.update(decoder_meta)
        out['query_features'] = out_query_features[-1]
        if self.query_discriminative_head is not None:
            disc_query_features = self.query_discriminative_head(out_query_features[-1])
            out['disc_query_features'] = F.normalize(disc_query_features, dim=-1)
        if self.query_quality_head is not None:
            out['pred_quality'] = self.query_quality_head(out_query_features[-1])

        # --- Cascading Layer Monitoring Hook ---
        if not self.training:
            out['layer_out_logits'] = out_logits
            out['layer_out_bboxes'] = out_bboxes

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
            
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        if self.training and self.use_pg_o2m:
            if dn_meta is not None:
                _, clean_target = torch.split(target, dn_meta['dn_num_split'], dim=1)
                _, clean_ref = torch.split(init_ref_points_unact, dn_meta['dn_num_split'], dim=1)
            else:
                clean_target = target
                clean_ref = init_ref_points_unact

            if self.pg_o2m_detach_aux_query:
                clean_target = clean_target.detach()
                clean_ref = clean_ref.detach()

            B, N, num_q = clean_target.shape[0], self.pg_o2m_num_groups, self.num_queries

            noise = torch.randn(B, N, num_q, clean_target.shape[-1], device=clean_target.device) * self.pg_o2m_noise_std
            perturbed_all = clean_target.unsqueeze(1) + noise

            clean_ref_sigmoid = F.sigmoid(clean_ref)
            ref_cx, ref_cy, ref_w, ref_h = clean_ref_sigmoid[..., 0], clean_ref_sigmoid[..., 1], clean_ref_sigmoid[..., 2], clean_ref_sigmoid[..., 3]
            ref_cx = ref_cx + (torch.rand_like(ref_cx) * 2 - 1) * self.pg_o2m_center_jitter * ref_w
            ref_cy = ref_cy + (torch.rand_like(ref_cy) * 2 - 1) * self.pg_o2m_center_jitter * ref_h
            ref_w = ref_w * torch.exp((torch.rand_like(ref_w) * 2 - 1) * self.pg_o2m_scale_jitter)
            ref_h = ref_h * torch.exp((torch.rand_like(ref_h) * 2 - 1) * self.pg_o2m_scale_jitter)
            perturbed_ref_sigmoid = torch.stack([ref_cx, ref_cy, ref_w, ref_h], dim=-1).clamp(0.0, 1.0)
            perturbed_ref = inverse_sigmoid(perturbed_ref_sigmoid)

            pg_logits_all = []
            pg_boxes_all = []

            for g in range(N):
                perturbed_target = perturbed_all[:, g, :, :]

                rand_mask = torch.rand(num_q, num_q, device=clean_target.device)
                pg_mask = rand_mask < self.pg_o2m_mask_ratio
                pg_mask.fill_diagonal_(True)

                pg_bboxes, pg_logits, _ = self.decoder(
                    perturbed_target,
                    perturbed_ref,
                    decoder_memory,
                    decoder_spatial_shapes,
                    decoder_level_index,
                    self.dec_bbox_head,
                    self.dec_score_head,
                    self.query_pos_head,
                    attn_mask=pg_mask,
                )

                pg_logits_all.append(pg_logits)
                pg_boxes_all.append(pg_bboxes)

            out['pg_o2m_logits'] = torch.stack(pg_logits_all, dim=2)
            out['pg_o2m_boxes'] = torch.stack(pg_boxes_all, dim=2)

        if self.training and self.use_query_perturb:
            if self.current_epoch >= self.perturb_start_epoch:
                if self.perturb_end_epoch > 0 and self.current_epoch >= self.perturb_end_epoch:
                    pass
                else:
                    if dn_meta is not None:
                        _, clean_target = torch.split(target, dn_meta['dn_num_split'], dim=1)
                        _, clean_ref = torch.split(init_ref_points_unact, dn_meta['dn_num_split'], dim=1)
                    else:
                        clean_target = target
                        clean_ref = init_ref_points_unact

                    perturb_outputs = []
                    for _ in range(self.num_perturb_branch):
                        query_noise = torch.randn_like(clean_target) * self.query_noise_std
                        perturbed_target = clean_target + query_noise

                        clean_ref_sigmoid = F.sigmoid(clean_ref)
                        ref_noise = (torch.rand_like(clean_ref_sigmoid) * 2.0 - 1.0) * self.ref_noise_scale
                        perturbed_ref_sigmoid = (clean_ref_sigmoid + ref_noise).clamp(0.0, 1.0)
                        perturbed_ref = inverse_sigmoid(perturbed_ref_sigmoid)

                        p_bboxes, p_logits, _ = self.decoder(
                            perturbed_target,
                            perturbed_ref,
                            decoder_memory,
                            decoder_spatial_shapes,
                            decoder_level_index,
                            self.dec_bbox_head,
                            self.dec_score_head,
                            self.query_pos_head,
                            attn_mask=None,
                        )

                        perturb_outputs.append({
                            'pred_logits': p_logits[-1],
                            'pred_boxes': p_bboxes[-1],
                        })

                    out['perturb_outputs'] = perturb_outputs

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
