"""RT-DETRv3 Model Architecture - faithful PyTorch port.

Combines RTDETRTransformerv3 (multi-group + O2M branch) with
PPYOLOEHead (dense auxiliary head) for hierarchical dense supervision.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register


__all__ = ['RTDETRV3']


@register
class RTDETRV3(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'aux_o2m_head']

    __share__ = ['o2m_branch', 'num_queries_o2m']

    def __init__(self,
                 backbone: nn.Module,
                 encoder,
                 decoder,
                 aux_o2m_head=None,
                 multi_scale=None,
                 o2m_branch=False,
                 num_queries_o2m=450):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.aux_o2m_head = aux_o2m_head
        self.multi_scale = multi_scale
        self.o2m_branch = o2m_branch
        self.num_queries_o2m = num_queries_o2m
        self.current_epoch = 0

    @staticmethod
    def _count_encoder_levels(encoded):
        if isinstance(encoded, dict):
            if 'multi_scale_features' in encoded:
                return len(encoded['multi_scale_features'])
            return len([key for key in ('node_1', 'node_2', 'node_3', 'node_4')
                        if key in encoded])
        if isinstance(encoded, (list, tuple)):
            return len(encoded)
        raise TypeError(f'Unsupported encoder output type: {type(encoded)!r}')

    def _get_encoder_feats(self, encoded):
        if isinstance(encoded, dict):
            if 'multi_scale_features' in encoded:
                return list(encoded['multi_scale_features'])
            node_keys = ['node_1', 'node_2', 'node_3', 'node_4']
            return [encoded[k] for k in node_keys if k in encoded]
        return list(encoded)

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        x = self.encoder(x)

        if hasattr(self.decoder, "num_levels"):
            available_levels = self._count_encoder_levels(x)
            if available_levels < self.decoder.num_levels:
                raise ValueError(
                    f"Encoder returned {available_levels} feature levels, "
                    f"but decoder requires at least {self.decoder.num_levels}."
                )

        transformer_out = self.decoder(x, targets)

        if self.training:
            dec_out_bboxes = transformer_out['out_bboxes']
            dec_out_logits = transformer_out['out_logits']
            enc_topk_bboxes = transformer_out['enc_topk_bboxes']
            enc_topk_logits = transformer_out['enc_topk_logits']
            dn_metas = transformer_out['dn_metas']

            if dn_metas is not None:
                total_dec_queries = dec_out_bboxes.shape[2]
                total_enc_queries = enc_topk_bboxes.shape[1]
                loss = {}

                split_dec_num = [sum(dn['dn_num_split']) if dn is not None else q
                                 for dn, q in zip(dn_metas, self.decoder.num_queries_list)]
                split_enc_num = [dn['dn_num_split'][1] if dn is not None else q
                                 for dn, q in zip(dn_metas, self.decoder.num_queries_list)]

                dec_out_bboxes_split = torch.split(dec_out_bboxes, split_dec_num, dim=2)
                dec_out_logits_split = torch.split(dec_out_logits, split_dec_num, dim=2)
                enc_bboxes_split = torch.split(enc_topk_bboxes, split_enc_num, dim=1)
                enc_logits_split = torch.split(enc_topk_logits, split_enc_num, dim=1)

                num_noise_groups = self.decoder.num_groups - int(self.o2m_branch)

                for g_id in range(num_noise_groups):
                    dn_meta_g = dn_metas[g_id]
                    if dn_meta_g is not None:
                        dn_split = dn_meta_g['dn_num_split']
                        dn_out_bboxes_g, clean_out_bboxes_g = torch.split(
                            dec_out_bboxes_split[g_id], dn_split, dim=2)
                        dn_out_logits_g, clean_out_logits_g = torch.split(
                            dec_out_logits_split[g_id], dn_split, dim=2)
                    else:
                        clean_out_bboxes_g = dec_out_bboxes_split[g_id]
                        clean_out_logits_g = dec_out_logits_split[g_id]
                        dn_out_bboxes_g = None
                        dn_out_logits_g = None

                    out_bboxes_g = torch.cat([
                        enc_bboxes_split[g_id].unsqueeze(0), clean_out_bboxes_g
                    ])
                    out_logits_g = torch.cat([
                        enc_logits_split[g_id].unsqueeze(0), clean_out_logits_g
                    ])

                    decoder_outputs = self._build_outputs(
                        out_bboxes_g, out_logits_g, enc_bboxes_split[g_id],
                        enc_logits_split[g_id], dn_out_bboxes_g, dn_out_logits_g,
                        dn_meta_g)

                    if self.aux_o2m_head is not None:
                        encoder_feats = self._get_encoder_feats(x)
                        aux_losses = self.aux_o2m_head(
                            encoder_feats, targets, epoch_id=self.current_epoch)
                        decoder_outputs['aux_o2m_losses'] = aux_losses

                    if g_id == 0:
                        output = decoder_outputs
                    else:
                        output['extra_outputs'] = output.get('extra_outputs', [])
                        output['extra_outputs'].append(decoder_outputs)

                return output

            else:
                out_bboxes = torch.cat([enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
                out_logits = torch.cat([enc_topk_logits.unsqueeze(0), dec_out_logits])

                decoder_outputs = {
                    'pred_logits': out_logits[-1],
                    'pred_boxes': out_bboxes[-1],
                    'aux_outputs': [
                        {'pred_logits': a, 'pred_boxes': b}
                        for a, b in zip(out_logits[:-1], out_bboxes[:-1])
                    ],
                }
                if self.aux_o2m_head is not None:
                    encoder_feats = self._get_encoder_feats(x)
                    aux_losses = self.aux_o2m_head(
                        encoder_feats, targets, epoch_id=self.current_epoch)
                    decoder_outputs['aux_o2m_losses'] = aux_losses
                return decoder_outputs
        else:
            dec_out_bboxes = transformer_out['out_bboxes']
            dec_out_logits = transformer_out['out_logits']

            eval_idx = self.decoder.eval_idx
            return {
                'pred_logits': dec_out_logits[-1],
                'pred_boxes': dec_out_bboxes[-1],
                'layer_out_logits': dec_out_logits,
                'layer_out_bboxes': dec_out_bboxes,
            }

    def _build_outputs(self, out_bboxes, out_logits, enc_bboxes, enc_logits,
                       dn_out_bboxes=None, dn_out_logits=None, dn_meta=None):
        dec_aux = [
            {'pred_logits': out_logits[i], 'pred_boxes': out_bboxes[i]}
            for i in range(out_logits.shape[0] - 1)
        ]
        enc_aux = [{'pred_logits': enc_logits, 'pred_boxes': enc_bboxes}]

        outputs = {
            'pred_logits': out_logits[-1],
            'pred_boxes': out_bboxes[-1],
            'aux_outputs': dec_aux + enc_aux,
        }

        if dn_out_bboxes is not None and dn_out_logits is not None:
            outputs['dn_aux_outputs'] = [
                {'pred_logits': dn_out_logits[i], 'pred_boxes': dn_out_bboxes[i]}
                for i in range(dn_out_logits.shape[0])
            ]
            outputs['dn_meta'] = dn_meta

        return outputs

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        if hasattr(self.decoder, 'set_epoch'):
            self.decoder.set_epoch(epoch)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
