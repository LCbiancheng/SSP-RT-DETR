"""by lyuwenyu
"""

import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

    @staticmethod
    def _count_encoder_levels(encoded):
        if isinstance(encoded, dict):
            if 'multi_scale_features' in encoded:
                feat_list = encoded['multi_scale_features']
                if not isinstance(feat_list, (list, tuple)):
                    raise TypeError(
                        "Encoder key 'multi_scale_features' must be a list/tuple of tensors."
                    )
                return len(feat_list)
            return len([key for key in ('node_1', 'node_2', 'node_3', 'node_4') if key in encoded])
        if isinstance(encoded, (list, tuple)):
            return len(encoded)
        raise TypeError(f'Unsupported encoder output type: {type(encoded)!r}')
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        x = self.encoder(x)
        # Strictly align the number of encoder/decoder levels to avoid silent bugs.
        if hasattr(self.decoder, "num_levels"):
            available_levels = self._count_encoder_levels(x)
            if available_levels < self.decoder.num_levels:
                raise ValueError(
                    f"Encoder returned {available_levels} feature levels, "
                    f"but decoder requires at least {self.decoder.num_levels}."
                )
            
        x = self.decoder(x, targets)

        return x

    def set_epoch(self, epoch):
        if hasattr(self.decoder, 'set_epoch'):
            self.decoder.set_epoch(epoch)
        elif hasattr(self.decoder, 'current_epoch'):
            self.decoder.current_epoch = int(epoch)

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
