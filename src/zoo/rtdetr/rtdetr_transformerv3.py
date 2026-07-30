"""RT-DETRv3 Transformer - faithful PyTorch port from PaddlePaddle.

Key innovations from RT-DETRv3 (WACV 2025 Oral):
1. Multi-group queries (num_noises) - additional noisy query groups
   sharing decoder, each with independent encoder head.
2. Random sparse attention mask for noise groups (g_id > 0).
3. O2M branch (o2m_branch) - dedicated one-to-many queries with
   full self-attention, no denoising.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .denoising import get_contrastive_denoising_training_group
from .utils import get_activation, inverse_sigmoid
from .utils import bias_init_with_prob

from src.core import register


__all__ = ['RTDETRTransformerv3']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        from .rtdetr_decoder import MSDeformableAttention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

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
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = torch.where(
                attn_mask.bool(),
                torch.zeros_like(attn_mask, dtype=tgt.dtype),
                torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                query_pos_head_inv_sig=False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(inverse_sigmoid(ref_points_detach))

            n_levels = len(memory_spatial_shapes)
            ref_points_input = ref_points_detach.unsqueeze(2).expand(-1, -1, n_levels, -1)

            output = layer(
                output, ref_points_input, memory, memory_spatial_shapes,
                memory_level_start_index, attn_mask, memory_mask, query_pos_embed,
            )

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            inter_ref_bbox = torch.nan_to_num(inter_ref_bbox, nan=0.5)

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register
class RTDETRTransformerv3(nn.Module):
    __share__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=None,
                 feat_strides=None,
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
                 learnt_init_query=False,
                 query_pos_head_inv_sig=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 num_noises=0,
                 num_noise_queries=None,
                 num_noise_denoising=100,
                 o2m_branch=False,
                 num_queries_o2m=450):
        super().__init__()
        if feat_channels is None:
            feat_channels = [512, 1024, 2048]
        if feat_strides is None:
            feat_strides = [8, 16, 32]
        if num_noise_queries is None:
            num_noise_queries = []

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.eval_idx = eval_idx

        self.num_noises = int(num_noises)
        self.num_noise_queries = [int(q) for q in num_noise_queries]
        self.num_noise_denoising = int(num_noise_denoising)

        self.num_queries_list = [num_queries]
        self.num_groups = 1
        if self.num_noises > 0:
            self.num_queries_list.extend(self.num_noise_queries)
            self.num_groups += self.num_noises

        self.o2m_branch = bool(o2m_branch)
        self.num_queries_o2m = int(num_queries_o2m)
        if self.o2m_branch:
            self.num_queries_list.append(num_queries_o2m)
            self.num_groups += 1

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
            ) for _ in range(num_levels)
        ])

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(
            hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.denoising_class_embed = nn.Embedding(
            num_classes + 1, hidden_dim, padding_idx=num_classes)

        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_inv_sig = bool(query_pos_head_inv_sig)

        self.enc_output = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(self.num_groups)
        ])
        self.enc_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(self.num_groups)
        ])
        self.enc_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(self.num_groups)
        ])

        self.map_memory = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self._init_anchors()

        self._reset_parameters()

    def _init_anchors(self):
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors, persistent=False)
            self.register_buffer('valid_mask', valid_mask, persistent=False)
        else:
            self.anchors = None
            self.valid_mask = None

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        for enc_score_head in self.enc_score_head:
            init.constant_(enc_score_head.bias, bias)
        for enc_bbox_head in self.enc_bbox_head:
            init.constant_(enc_bbox_head.layers[-1].weight, 0)
            init.constant_(enc_bbox_head.layers[-1].bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        for enc_output in self.enc_output:
            init.xavier_uniform_(enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _get_encoder_input(self, nodes_dict_or_list):
        if isinstance(nodes_dict_or_list, dict):
            if 'multi_scale_features' in nodes_dict_or_list:
                feat_list = list(nodes_dict_or_list['multi_scale_features'])
            else:
                node_keys = ['node_1', 'node_2', 'node_3', 'node_4']
                feat_list = [nodes_dict_or_list[k] for k in node_keys if k in nodes_dict_or_list]
        else:
            feat_list = nodes_dict_or_list
        feat_list = feat_list[:self.num_levels]

        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feat_list)]
        for i in range(len(proj_feats), self.num_levels):
            proj_feats.append(self.input_proj[i](feat_list[-1]))

        memory_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            level_flat = feat.flatten(2).permute(0, 2, 1)
            memory_flatten.append(level_flat)
            spatial_shapes.append([h, w])
            level_start_index.append(h * w + level_start_index[-1])

        memory_flatten = torch.concat(memory_flatten, 1)
        level_start_index.pop()
        return memory_flatten, spatial_shapes, level_start_index

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        device = self.enc_output[0][0].weight.device
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype, device=device),
                torch.arange(end=w, dtype=dtype, device=device),
                indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.full_like(anchors, float('inf')))
        return anchors, valid_mask

    def forward(self, nodes_dict, targets=None):
        memory_flatten, spatial_shapes, level_start_index = self._get_encoder_input(nodes_dict)
        bs = memory_flatten.shape[0]

        if self.training:
            denoising_classes, denoising_bbox_unacts, attn_masks, dn_metas = [], [], [], []
            for g_id in range(self.num_noises + 1):
                is_o2m = self.o2m_branch and (g_id == self.num_groups - 1)
                if is_o2m:
                    dn_class, dn_bbox_unact, attn_mask, dn_meta = None, None, None, None
                else:
                    num_den = self.num_noise_denoising if g_id > 0 else self.num_denoising
                    dn_class, dn_bbox_unact, attn_mask, dn_meta = \
                        get_contrastive_denoising_training_group(
                            targets, self.num_classes,
                            self.num_queries_list[g_id],
                            self.denoising_class_embed,
                            num_denoising=num_den,
                            label_noise_ratio=self.label_noise_ratio,
                            box_noise_scale=self.box_noise_scale)
                denoising_classes.append(dn_class)
                denoising_bbox_unacts.append(dn_bbox_unact)
                attn_masks.append(attn_mask)
                dn_metas.append(dn_meta)

            target, init_ref, enc_bboxes, enc_logits = self._get_decoder_input(
                memory_flatten, spatial_shapes, bs,
                denoising_classes, denoising_bbox_unacts)

            new_size = target.shape[1]
            new_attn_mask = torch.full((new_size, new_size), False, dtype=torch.bool, device=target.device)
            begin, end = 0, 0
            for g_id in range(self.num_groups):
                is_o2m = self.o2m_branch and (g_id == self.num_groups - 1)
                num_q = self.num_queries_list[g_id]
                if is_o2m:
                    end = end + num_q
                    new_mask = torch.full((num_q, num_q), True, dtype=torch.bool, device=target.device)
                else:
                    attn_mask = attn_masks[g_id]
                    end = end + attn_mask.shape[0]
                    dn_size, q_size = dn_metas[g_id]['dn_num_split']
                    rand_mask = torch.rand(num_q, num_q, device=target.device)
                    if g_id > 0:
                        new_mask = rand_mask > 0.1
                    else:
                        new_mask = rand_mask >= 0.0
                    attn_mask[dn_size:dn_size + q_size, dn_size:dn_size + q_size] = new_mask
                    new_attn_mask[begin:end, begin:end] = attn_mask
                begin = end
            attn_masks_final = new_attn_mask

            out_bboxes, out_logits = self.decoder(
                target, init_ref, memory_flatten, spatial_shapes,
                level_start_index, self.dec_bbox_head, self.dec_score_head,
                self.query_pos_head,
                attn_mask=attn_masks_final,
                query_pos_head_inv_sig=self.query_pos_head_inv_sig)
        else:
            target, init_ref, enc_bboxes, enc_logits = self._get_decoder_input(
                memory_flatten, spatial_shapes, bs, None, None)

            out_bboxes, out_logits = self.decoder(
                target, init_ref, memory_flatten, spatial_shapes,
                level_start_index, self.dec_bbox_head, self.dec_score_head,
                self.query_pos_head,
                query_pos_head_inv_sig=self.query_pos_head_inv_sig)

        out = {
            'out_bboxes': out_bboxes,
            'out_logits': out_logits,
            'enc_topk_bboxes': enc_bboxes,
            'enc_topk_logits': enc_logits,
            'dn_metas': dn_metas if (self.training and any(m is not None for m in dn_metas)) else None,
        }
        return out

    def _get_decoder_input(self, memory_flatten, spatial_shapes, bs,
                           denoising_classes, denoising_bbox_unacts):
        do_denoise = denoising_classes is not None

        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask

        memory = torch.where(valid_mask, memory_flatten, torch.zeros_like(memory_flatten))
        map_memory = self.map_memory(memory_flatten.detach())

        all_targets, all_ref_unact, all_enc_bboxes, all_enc_logits = [], [], [], []

        for g_id in range(self.num_groups):
            is_o2m = self.o2m_branch and (g_id == self.num_groups - 1)
            num_q = self.num_queries_list[g_id]

            output_memory = self.enc_output[g_id](memory)
            enc_outputs_class = self.enc_score_head[g_id](output_memory)
            enc_outputs_coord_unact = self.enc_bbox_head[g_id](output_memory) + anchors

            _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, num_q, dim=1)

            reference_points_unact = enc_outputs_coord_unact.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, 4))
            enc_topk_bbox = F.sigmoid(reference_points_unact)

            if do_denoise and not is_o2m and denoising_bbox_unacts[g_id] is not None:
                reference_points_unact = torch.concat(
                    [denoising_bbox_unacts[g_id], reference_points_unact], 1)
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            if self.learnt_init_query:
                target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            else:
                if g_id == 0:
                    target = output_memory.gather(
                        dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
                    if self.training:
                        target = target.detach()
                else:
                    target = map_memory.gather(
                        dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, map_memory.shape[-1]))

            enc_topk_logit = enc_outputs_class.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

            if do_denoise and not is_o2m and denoising_classes[g_id] is not None:
                target = torch.concat([denoising_classes[g_id], target], 1)

            if not self.training:
                return target, reference_points_unact, enc_topk_bbox, enc_topk_logit

            all_targets.append(target)
            all_ref_unact.append(reference_points_unact)
            all_enc_bboxes.append(enc_topk_bbox)
            all_enc_logits.append(enc_topk_logit)

        targets_cat = torch.cat(all_targets, dim=1)
        ref_unact_cat = torch.cat(all_ref_unact, dim=1)
        enc_bboxes_cat = torch.cat(all_enc_bboxes, dim=1)
        enc_logits_cat = torch.cat(all_enc_logits, dim=1)

        return targets_cat, ref_unact_cat, enc_bboxes_cat, enc_logits_cat
