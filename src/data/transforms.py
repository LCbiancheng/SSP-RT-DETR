""""by lyuwenyu
"""


import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()

# torchvision >= 0.16 renamed `datapoints` to `tv_tensors`
try:
    from torchvision import tv_tensors as datapoints
except ImportError:
    from torchvision import datapoints

# Dynamic type for backward compatibility
BoundingBoxType = getattr(datapoints, 'BoundingBoxes', getattr(datapoints, 'BoundingBox', None))

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image 
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG


__all__ = ['Compose', ]


RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
# RandomIoUCrop = register(T.RandomIoUCrop)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
# Torchvision >= 0.16 API subclass wrappers for name matching in GLOBAL_CONFIG
_ToImageTensorBase = getattr(T, 'ToImageTensor', getattr(T, 'ToImage', getattr(T, 'ToPureTensor', object)))
@register
class ToImageTensor(_ToImageTensorBase):
    pass

_ConvertDtypeBase = getattr(T, 'ConvertDtype', getattr(T, 'ToDtype', object))
@register
class ConvertDtype(_ConvertDtypeBase):
    def __init__(self, dtype=torch.float32, **kwargs):
        # YAML configuration may pass dtype as a string
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.split('.')[-1]) if dtype.startswith('torch.') else getattr(torch, dtype)
        
        try:
            super().__init__(dtype=dtype, **kwargs)
        except TypeError:
            super().__init__(**kwargs)

_SanitizeBoundingBoxBase = getattr(T, 'SanitizeBoundingBox', getattr(T, 'SanitizeBoundingBoxes', object))
@register
class SanitizeBoundingBox(_SanitizeBoundingBoxBase):
    pass

RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)



@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                    # op['type'] = name
                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]
 
        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        BoundingBoxType,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        # Handle tuple vs list difference between torchvision versions
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self._get_params(flat_inputs)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        
        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class DefectAwareCrop(T.Transform):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        BoundingBoxType,
    )

    def __init__(
        self,
        prob=0.5,
        scale_range=(1.5, 3.0),
        min_crop_size=128,
        max_crop_trials=20,
        min_box_area=4,
        min_visibility=0.3,
        keep_empty=False,
    ):
        super().__init__()
        self.prob = prob
        self.scale_range = scale_range
        self.min_crop_size = min_crop_size
        self.max_crop_trials = max_crop_trials
        self.min_box_area = min_box_area
        self.min_visibility = min_visibility
        self.keep_empty = keep_empty

    @staticmethod
    def _get_image_wh(image):
        if isinstance(image, Image.Image):
            return image.size[0], image.size[1]
        if isinstance(image, torch.Tensor):
            return int(image.shape[-1]), int(image.shape[-2])
        spatial = F.get_spatial_size(image)
        return int(spatial[1]), int(spatial[0])

    @staticmethod
    def _crop_image(image, x1, y1, x2, y2):
        if isinstance(image, Image.Image):
            return image.crop((int(x1), int(y1), int(x2), int(y2)))
        if isinstance(image, torch.Tensor):
            return image[:, int(y1):int(y2), int(x1):int(x2)]
        return F.crop(image, int(y1), int(x1), int(y2 - y1), int(x2 - x1))

    def __call__(self, *inputs):
        image, target = inputs if len(inputs) > 1 else (inputs[0], None)

        if target is None:
            return image

        if torch.rand(1).item() >= self.prob:
            return image, target

        if 'boxes' not in target or len(target['boxes']) == 0:
            return image, target

        boxes_obj = target['boxes']
        BoundingBoxCls = getattr(datapoints, 'BoundingBox', None)
        if isinstance(boxes_obj, torch.Tensor) and not isinstance(boxes_obj, (datapoints.BoundingBoxes, BoundingBoxCls)):
            boxes_tensor = boxes_obj
        else:
            boxes_tensor = torch.as_tensor(boxes_obj)

        img_w, img_h = self._get_image_wh(image)

        scale_lo, scale_hi = self.scale_range

        for _ in range(self.max_crop_trials):
            idx = torch.randint(0, len(boxes_tensor), (1,)).item()
            x1, y1, x2, y2 = boxes_tensor[idx].tolist()

            box_w = x2 - x1
            box_h = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            scale = scale_lo + torch.rand(1).item() * (scale_hi - scale_lo)
            aspect = 0.75 + torch.rand(1).item() * (1.33 - 0.75)

            crop_w = max(box_w * scale, self.min_crop_size)
            crop_h = max(box_h * scale, self.min_crop_size)

            crop_w = crop_w * (aspect ** 0.5)
            crop_h = crop_h / (aspect ** 0.5)

            crop_w = min(crop_w, img_w)
            crop_h = min(crop_h, img_h)

            jitter_x = (torch.rand(1).item() * 0.5 - 0.25) * box_w
            jitter_y = (torch.rand(1).item() * 0.5 - 0.25) * box_h

            crop_cx = cx + jitter_x
            crop_cy = cy + jitter_y

            crop_x1 = crop_cx - crop_w / 2.0
            crop_y1 = crop_cy - crop_h / 2.0
            crop_x2 = crop_x1 + crop_w
            crop_y2 = crop_y1 + crop_h

            crop_x1 = max(0.0, crop_x1)
            crop_y1 = max(0.0, crop_y1)
            crop_x2 = min(float(img_w), crop_x2)
            crop_y2 = min(float(img_h), crop_y2)

            crop_w_actual = crop_x2 - crop_x1
            crop_h_actual = crop_y2 - crop_y1
            if crop_w_actual < 1.0 or crop_h_actual < 1.0:
                continue

            new_boxes_list = []
            new_labels_list = []
            new_area_list = []
            new_iscrowd_list = []

            has_area = 'area' in target
            has_iscrowd = 'iscrowd' in target

            for i in range(len(boxes_tensor)):
                ox1, oy1, ox2, oy2 = boxes_tensor[i].tolist()

                nx1 = max(ox1, crop_x1) - crop_x1
                ny1 = max(oy1, crop_y1) - crop_y1
                nx2 = min(ox2, crop_x2) - crop_x1
                ny2 = min(oy2, crop_y2) - crop_y1

                new_w = max(0.0, nx2 - nx1)
                new_h = max(0.0, ny2 - ny1)
                new_area_val = new_w * new_h

                old_w = ox2 - ox1
                old_h = oy2 - oy1
                old_area_val = old_w * old_h
                visibility = new_area_val / max(old_area_val, 1e-7)

                if new_w > 0.0 and new_h > 0.0 and new_area_val >= self.min_box_area and visibility >= self.min_visibility:
                    new_boxes_list.append([nx1, ny1, nx2, ny2])
                    new_labels_list.append(target['labels'][i].item() if torch.is_tensor(target['labels'][i]) else target['labels'][i])
                    if has_area:
                        new_area_list.append(new_area_val)
                    if has_iscrowd:
                        new_iscrowd_list.append(target['iscrowd'][i].item() if torch.is_tensor(target['iscrowd'][i]) else target['iscrowd'][i])

            if len(new_boxes_list) == 0 and not self.keep_empty:
                continue

            cropped_image = self._crop_image(image, crop_x1, crop_y1, crop_x2, crop_y2)

            new_target = {}
            for k, v in target.items():
                if k not in ('boxes', 'labels', 'area', 'iscrowd', 'size'):
                    new_target[k] = v

            new_target['labels'] = torch.tensor(new_labels_list, dtype=torch.int64)

            if has_area:
                new_target['area'] = torch.tensor(new_area_list, dtype=torch.float32) if new_area_list else torch.zeros((0,), dtype=torch.float32)

            if has_iscrowd:
                new_target['iscrowd'] = torch.tensor(new_iscrowd_list, dtype=torch.int64) if new_iscrowd_list else torch.zeros((0,), dtype=torch.int64)

            new_boxes_tensor = torch.tensor(new_boxes_list, dtype=torch.float32) if new_boxes_list else torch.zeros((0, 4), dtype=torch.float32)

            cropped_h, cropped_w = int(crop_h_actual), int(crop_w_actual)
            cropped_size = (cropped_h, cropped_w)

            if hasattr(datapoints, 'BoundingBoxes'):
                new_target['boxes'] = datapoints.BoundingBoxes(
                    new_boxes_tensor,
                    format=datapoints.BoundingBoxFormat.XYXY,
                    canvas_size=cropped_size,
                )
            else:
                bbox_cls = getattr(datapoints, 'BoundingBox', None)
                new_target['boxes'] = bbox_cls(
                    new_boxes_tensor,
                    format=datapoints.BoundingBoxFormat.XYXY,
                    spatial_size=cropped_size,
                )

            new_target['size'] = torch.tensor([cropped_w, cropped_h])

            return cropped_image, new_target

        return image, target


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    _transformed_types = (
        BoundingBoxType,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            # Handle API rename: spatial_size vs canvas_size
            spatial_size = getattr(inpt, 'canvas_size', getattr(inpt, 'spatial_size', None))
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            
            bbox_cls = getattr(datapoints, 'BoundingBoxes', None) or datapoints.BoundingBox
            if hasattr(datapoints, 'BoundingBoxes'):
                inpt = bbox_cls(inpt, format=self.data_fmt[self.out_fmt], canvas_size=spatial_size)
            else:
                inpt = bbox_cls(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)
        
        if self.normalize:
            spatial_size = getattr(inpt, 'canvas_size', getattr(inpt, 'spatial_size', None))
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
