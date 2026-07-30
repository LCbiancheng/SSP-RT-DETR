'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std



class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
                
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class S2GuidedAdaptiveFusionBlock(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_layers=1,
            reduction=4,
            expansion=1.0,
            act='silu',
            prior_scale_init=1.0,
            fusion_mode='softmax'):
        super().__init__()
        reduction = max(int(reduction), 1)
        hidden = max(hidden_dim // reduction, 32)
        num_blocks = max(int(num_layers), 1)
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode in {'legacy', 'adaptive', 'three_way'}:
            self.fusion_mode = 'softmax'
        if self.fusion_mode not in {'softmax'}:
            raise ValueError(f'Unsupported S2 fusion mode: {fusion_mode}')

        # NEU-DET contains many elongated defects, so the S2 prior keeps isotropic
        # and directional branches before entering the adaptive CCFM gate.
        self.prior_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                get_activation(act),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, (1, 5), 1, (0, 2), groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                get_activation(act),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, (5, 1), 1, (2, 0), groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                get_activation(act),
            ),
        ])
        self.prior_align = nn.Sequential(
            nn.Conv2d(hidden_dim * len(self.prior_branches), hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
        )
        self.prior_scale = nn.Parameter(torch.tensor(float(prior_scale_init), dtype=torch.float32))
        self.gate_stem = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden, 1, 1, 0, bias=True),
            get_activation(act),
            nn.Conv2d(hidden, hidden_dim * 3, 1, 1, 0, bias=True),
        )
        self.spatial_gate = nn.Conv2d(hidden_dim, 3, 3, 1, 1, bias=True)
        self.refine = CSPRepLayer(
            hidden_dim * 3,
            hidden_dim,
            num_blocks=num_blocks,
            expansion=expansion,
            act=act,
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(self, feat_low, feat_high, s2_prior):
        prior_feats = [branch(s2_prior) for branch in self.prior_branches]
        # Keep the learnable S2 injection strength bounded to avoid sign flips
        # or runaway amplification in deep fusion stages.
        prior_scale = self.prior_scale.clamp(min=0.0, max=1.0)
        s2_prior = prior_scale * self.prior_align(torch.concat(prior_feats, dim=1))
        gate_feat = self.gate_stem(torch.concat([feat_low, feat_high, s2_prior], dim=1))

        bs, channels, _, _ = gate_feat.shape
        channel_logits = self.channel_gate(gate_feat).reshape(bs, 3, channels, 1, 1)
        channel_weights = torch.softmax(channel_logits, dim=1)

        spatial_logits = self.spatial_gate(gate_feat).unsqueeze(2)
        spatial_weights = torch.softmax(spatial_logits, dim=1)

        sources = torch.stack([feat_low, feat_high, s2_prior], dim=1)
        weighted_sources = sources * channel_weights * spatial_weights
        fused = weighted_sources.sum(dim=1)
        refined = self.refine(torch.concat([weighted_sources[:, 0], weighted_sources[:, 1], weighted_sources[:, 2]], dim=1))
        return self.out_proj(refined + fused)


# LiteS2Fusion: S2 stride=4 -> P3 stride=8 single-point weak gated residual enhancement
# Only S2 prior is injected as a residual supplement to P3, with zero-init alpha.
class LiteS2Fusion(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 32)

        self.s2_align = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, base_feat, s2_feat):
        s2 = self.s2_align(s2_feat)

        if s2.shape[-2:] != base_feat.shape[-2:]:
            s2 = F.interpolate(s2, size=base_feat.shape[-2:], mode='bilinear', align_corners=False)

        gate = self.gate(torch.cat([base_feat, s2], dim=1))
        out = base_feat + self.alpha * gate * s2
        return out


class S2FastFusion(nn.Module):
    def __init__(self, channels, reduction=4, use_gate=True):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.use_gate = use_gate

        self.s2_align = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=True),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, p3, s2):
        s2_aligned = self.s2_align(s2)

        if s2_aligned.shape[-2:] != p3.shape[-2:]:
            s2_aligned = F.interpolate(
                s2_aligned,
                size=p3.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        if self.use_gate:
            gate = self.gate(torch.cat([p3, s2_aligned], dim=1))
            return p3 + self.alpha * gate * s2_aligned

        return p3 + self.alpha * s2_aligned


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, spatial_shape=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None, spatial_shape=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed, spatial_shape=spatial_shape)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 use_s2_adaptive_fusion=False,
                 adaptive_fusion_num_layers=1,
                 adaptive_fusion_reduction=4,
                 adaptive_fusion_mode='softmax',
                 use_s2_lite_fusion=False,
                 s2_lite_fusion_reduction=4,
                 use_s2_fast=False,
                 s2_fast_reduction=4,
                 s2_fast_use_gate=True):
        super().__init__()
        if len(in_channels) != len(feat_strides):
            raise ValueError('HybridEncoder expects len(in_channels) == len(feat_strides).')
        if any(idx < 0 or idx >= len(in_channels) for idx in use_encoder_idx):
            raise ValueError('HybridEncoder.use_encoder_idx contains an out-of-range feature index.')
        if use_s2_adaptive_fusion and len(in_channels) < 4:
            raise ValueError('HybridEncoder.use_s2_adaptive_fusion requires a 4-level pyramid with S2 inputs.')

        self.use_s2_adaptive_fusion = bool(use_s2_adaptive_fusion)
        self.use_s2_fast = bool(use_s2_fast)

        if self.use_s2_adaptive_fusion and self.use_s2_fast:
            raise ValueError(
                'use_s2_adaptive_fusion and use_s2_fast cannot be enabled at the same time.'
            )

        if self.use_s2_fast and len(in_channels) < 4:
            raise ValueError('HybridEncoder.use_s2_fast requires a 4-level pyramid with S2 inputs.')

        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.adaptive_fusion_num_layers = int(adaptive_fusion_num_layers)
        self.adaptive_fusion_reduction = int(adaptive_fusion_reduction)
        self.adaptive_fusion_mode = str(adaptive_fusion_mode).lower()
        if self.adaptive_fusion_mode in {'legacy', 'adaptive', 'three_way'}:
            self.adaptive_fusion_mode = 'softmax'
        if self.adaptive_fusion_mode not in {'softmax'}:
            raise ValueError(f'Unsupported HybridEncoder.adaptive_fusion_mode: {adaptive_fusion_mode}')
        self.actual_adaptive_fusion_num_layers = max(self.adaptive_fusion_num_layers, 1)

        self.use_s2_lite_fusion = bool(use_s2_lite_fusion)
        self.s2_lite_fusion_reduction = max(int(s2_lite_fusion_reduction), 1)

        self.s2_fast_reduction = max(int(s2_fast_reduction), 1)
        self.s2_fast_use_gate = bool(s2_fast_use_gate)

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        neck_num_blocks = max(round(3 * depth_mult), 1)
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        self.encoder = nn.ModuleList()
        for enc_ind in self.use_encoder_idx:
            encoder_layer = TransformerEncoderLayer(
                hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=enc_act)
            self.encoder.append(TransformerEncoder(encoder_layer, num_encoder_layers))

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, neck_num_blocks, act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, neck_num_blocks, act=act, expansion=expansion)
            )

        if self.use_s2_adaptive_fusion:
            td_blocks = []
            for block_idx in range(len(in_channels) - 1):
                target_stride = feat_strides[len(in_channels) - 2 - block_idx]
                td_blocks.append(
                    S2GuidedAdaptiveFusionBlock(
                        hidden_dim,
                        num_layers=self.actual_adaptive_fusion_num_layers,
                        reduction=self.adaptive_fusion_reduction,
                        expansion=expansion,
                        act=act,
                        prior_scale_init=self._resolve_prior_scale_init(target_stride),
                        fusion_mode=self.adaptive_fusion_mode,
                    )
                )
            self.td_adaptive_fusion_blocks = nn.ModuleList(td_blocks)

            bu_blocks = []
            for block_idx in range(len(in_channels) - 1):
                target_stride = feat_strides[block_idx + 1]
                bu_blocks.append(
                    S2GuidedAdaptiveFusionBlock(
                        hidden_dim,
                        num_layers=self.actual_adaptive_fusion_num_layers,
                        reduction=self.adaptive_fusion_reduction,
                        expansion=expansion,
                        act=act,
                        prior_scale_init=self._resolve_prior_scale_init(target_stride),
                        fusion_mode=self.adaptive_fusion_mode,
                    )
                )
            self.bu_adaptive_fusion_blocks = nn.ModuleList(bu_blocks)
        else:
            self.td_adaptive_fusion_blocks = None
            self.bu_adaptive_fusion_blocks = None

        self._reset_parameters()

        if self.use_s2_lite_fusion:
            self.s2_lite_fusion = LiteS2Fusion(hidden_dim, reduction=self.s2_lite_fusion_reduction)
        else:
            self.s2_lite_fusion = None

        if self.use_s2_fast:
            self.s2_fast_fusion = S2FastFusion(
                hidden_dim,
                reduction=self.s2_fast_reduction,
                use_gate=self.s2_fast_use_gate
            )
        else:
            self.s2_fast_fusion = None

    def _resolve_prior_scale_init(self, target_stride):
        min_stride = float(min(self.feat_strides)) if self.feat_strides else 1.0
        target_stride = float(target_stride)
        return max(0.25, min(1.0, 2.0 * min_stride / target_stride))

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    @staticmethod
    def _resize_s2_prior(s2_feat, target_feat):
        return F.interpolate(s2_feat, size=target_feat.shape[2:], mode='bilinear', align_corners=False)

    def _forward_s2_fast(self, proj_feats):
        s2_proj = proj_feats[0]
        ccfm_inputs = proj_feats[1:]

        f5_raw = ccfm_inputs[-1]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                adjusted_idx = enc_ind - 1
                if adjusted_idx < 0 or adjusted_idx >= len(ccfm_inputs):
                    continue
                h, w = ccfm_inputs[adjusted_idx].shape[2:]
                src_flatten = ccfm_inputs[adjusted_idx].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed, spatial_shape=(h, w))
                ccfm_inputs[adjusted_idx] = memory.permute(0, 2, 1).reshape(
                    -1, self.hidden_dim, h, w).contiguous()
                f5_raw = ccfm_inputs[-1]

        n_ccfm = len(ccfm_inputs)

        inner_outs = [ccfm_inputs[-1]]
        for idx in range(n_ccfm - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = ccfm_inputs[idx - 1]
            lat_idx = len(self.in_channels) - n_ccfm + (n_ccfm - 1 - idx)
            feat_high = self.lateral_convs[lat_idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            fpn_idx = len(self.in_channels) - n_ccfm + (n_ccfm - 1 - idx)
            inner_out = self.fpn_blocks[fpn_idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(n_ccfm - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            ds_idx = len(self.in_channels) - n_ccfm + idx
            downsample_feat = self.downsample_convs[ds_idx](feat_low)
            pan_idx = len(self.in_channels) - n_ccfm + idx
            out = self.pan_blocks[pan_idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        outs[0] = self.s2_fast_fusion(outs[0], s2_proj)

        result = {'f5_raw': f5_raw}
        result['multi_scale_features'] = outs
        result['multi_scale_strides'] = self.out_strides[1:]

        compatibility_nodes = list(reversed(outs[:3]))
        for idx, feat in enumerate(compatibility_nodes, start=1):
            result[f'node_{idx}'] = feat
        return result

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.use_s2_fast:
            return self._forward_s2_fast(proj_feats)

        s2_detail_source = proj_feats[0] if len(proj_feats) >= 4 else None
        f5_raw = proj_feats[-1]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed, spatial_shape=(h, w))
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                f5_raw = proj_feats[-1]

        # CCFM-style cross-scale fusion: top-down aggregation followed by bottom-up refinement.
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            block_idx = len(self.in_channels) - 1 - idx
            if self.td_adaptive_fusion_blocks is not None and s2_detail_source is not None:
                s2_prior = self._resize_s2_prior(s2_detail_source, feat_low)
                inner_out = self.td_adaptive_fusion_blocks[block_idx](feat_low, upsample_feat, s2_prior)
            else:
                inner_out = self.fpn_blocks[block_idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if self.bu_adaptive_fusion_blocks is not None and s2_detail_source is not None:
                s2_prior = self._resize_s2_prior(s2_detail_source, feat_high)
                out = self.bu_adaptive_fusion_blocks[idx](downsample_feat, feat_high, s2_prior)
            else:
                out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        result = {'f5_raw': f5_raw}

        if len(self.in_channels) >= 4:
            # LiteS2Fusion: single-point S2(stride=4)->P3(stride=8) weak gated residual enhancement.
            if self.s2_lite_fusion is not None:
                outs[1] = self.s2_lite_fusion(outs[1], outs[0])

            # Ordered shallow-to-deep pyramid for decoder consumption, e.g. [S2, S3, S4, S5].
            result['multi_scale_features'] = outs
            result['multi_scale_strides'] = list(self.out_strides)

            compatibility_nodes = list(reversed(outs[:4]))
            for idx, feat in enumerate(compatibility_nodes, start=1):
                result[f'node_{idx}'] = feat
            return result

        # Keep the original 3-level node layout for baseline compatibility.
        # inner_outs: [S3/P3 top-down, S4/P4 top-down, S5/P5 top-down]
        node_3 = inner_outs[0] # S3/P3 top-down: high resolution and strong geometry
        node_2 = inner_outs[1] # S4/P4 top-down
        node_1 = inner_outs[2] # S5/P5 top-down: strongest semantics

        # outs: [S3/P3 bottom-up, S4/P4 bottom-up, S5/P5 bottom-up]
        node_4 = outs[1] # S4/P4 bottom-up

        result.update({
            'node_1': node_1,
            'node_2': node_2,
            'node_3': node_3,
            'node_4': node_4,
        })
        return result
