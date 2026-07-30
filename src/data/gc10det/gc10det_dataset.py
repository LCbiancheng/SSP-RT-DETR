"""GC10-DET Dataset Implementation
"""

import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET

try:
    from torchvision import tv_tensors as datapoints
except ImportError:
    from torchvision import datapoints

from src.core import register

__all__ = ['GC10DetDetection']

CLASS_TO_IDX = {
    '1_chongkong': 0,
    '2_hanfeng': 1,
    '3_yueyawan': 2,
    '4_shuiban': 3,
    '5_youban': 4,
    '6_siban': 5,
    '7_yiwu': 6,
    '8_yahen': 7,
    '9_zhehen': 8,
    '10_yaozhe': 9,
    '10_yaozhed': 9,
}
NUM_CLASSES = len(set(CLASS_TO_IDX.values()))


@register
class GC10DetDetection(torch.utils.data.Dataset):
    """GC10-DET Dataset for defect detection"""
    __inject__ = ['transforms']

    def __init__(self, img_folder, ann_folder, transforms=None,
                 use_adaptive_mosaic=False,
                 adaptive_mosaic_prob=0.5,
                 adaptive_mosaic_same_image_threshold=1,
                 adaptive_mosaic_size=640,
                 adaptive_mosaic_close_last_epochs=10):
        super(GC10DetDetection, self).__init__()
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.transforms = transforms
        self.class_to_idx = CLASS_TO_IDX
        self.num_classes = NUM_CLASSES

        self.img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')])
        self.ids = range(len(self.img_files))

        self.use_adaptive_mosaic = bool(use_adaptive_mosaic)
        if self.use_adaptive_mosaic:
            from src.data.adaptive_mosaic import AdaptiveDenseO2OMosaic
            self.mosaic = AdaptiveDenseO2OMosaic(
                dataset=self,
                prob=adaptive_mosaic_prob,
                same_image_threshold=adaptive_mosaic_same_image_threshold,
                mosaic_size=adaptive_mosaic_size,
                close_mosaic_last_epochs=adaptive_mosaic_close_last_epochs,
            )
        else:
            self.mosaic = None

    def set_mosaic_epoch(self, epoch, total_epochs=None):
        if self.mosaic is not None:
            self.mosaic.set_epoch(epoch, total_epochs)

    @staticmethod
    def _get_current_image_wh(image):
        if isinstance(image, torch.Tensor):
            return int(image.shape[-1]), int(image.shape[-2])
        width, height = image.size
        return int(width), int(height)

    def _load_item(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_file)
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size

        boxes, labels, area, iscrowd = self._parse_annotation(idx, img_width, img_height)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
        else:
            boxes = boxes.reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        labels = torch.clamp(labels, min=0, max=NUM_CLASSES - 1)

        if labels.numel() > 0:
            assert labels.min() >= 0 and labels.max() < NUM_CLASSES, \
                f"Bad labels after clamping: min={labels.min().item()} max={labels.max().item()} num_classes={NUM_CLASSES}"

        target = {
            'image_id': torch.tensor([idx]),
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([img_width, img_height]),
            'size': torch.as_tensor([img_width, img_height])
        }

        if hasattr(datapoints, 'BoundingBoxes'):
            target['boxes'] = datapoints.BoundingBoxes(
                target['boxes'],
                format=datapoints.BoundingBoxFormat.XYXY,
                canvas_size=img.size[::-1],
            )
        else:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=img.size[::-1],
            )

        return img, target

    def _parse_annotation(self, idx, img_width, img_height):
        img_file = self.img_files[idx]
        ann_file = img_file.replace('.jpg', '.xml')
        ann_path = os.path.join(self.ann_folder, ann_file)

        boxes = []
        labels = []
        area = []
        iscrowd = []

        tree = ET.parse(ann_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_to_idx:
                continue

            label = self.class_to_idx[class_name]
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_width, xmax)
            ymax = min(img_height, ymax)

            xmin = min(xmin, xmax - 1e-5)
            ymin = min(ymin, ymax - 1e-5)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            area.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(0)

        return boxes, labels, area, iscrowd

    def get_coco_ground_truth(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_file)
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        boxes_xyxy, labels, area, iscrowd = self._parse_annotation(idx, img_width, img_height)
        boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes_xyxy]

        annotations = []
        for i in range(len(boxes_xywh)):
            annotations.append({
                'bbox': boxes_xywh[i],
                'category_id': labels[i],
                'area': area[i],
                'iscrowd': iscrowd[i],
            })

        return {
            'image_id': idx,
            'height': img_height,
            'width': img_width,
            'annotations': annotations,
        }

    def _sanitize_target(self, target):
        labels = target.get('labels')
        boxes = target.get('boxes')

        if labels is None or boxes is None:
            return target

        num_labels = labels.shape[0] if labels.numel() > 0 or labels.ndim > 0 else 0
        num_boxes = boxes.shape[0] if boxes.ndim >= 2 else 0

        if isinstance(boxes, (datapoints.BoundingBoxes, getattr(datapoints, 'BoundingBox', object))):
            num_boxes = boxes.shape[0]
        elif isinstance(boxes, torch.Tensor):
            if boxes.ndim == 1 and boxes.shape[0] == 0:
                num_boxes = 0
            elif boxes.ndim == 2:
                num_boxes = boxes.shape[0]
            else:
                num_boxes = 0

        if num_boxes != num_labels:
            if num_boxes == 0 or num_labels == 0:
                empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
                img_w, img_h = target.get('size', torch.tensor([640, 640]))
                canvas_size = (int(img_h.item()), int(img_w.item())) if img_w is not None else (640, 640)
                if hasattr(datapoints, 'BoundingBoxes'):
                    target['boxes'] = datapoints.BoundingBoxes(
                        empty_boxes, format=datapoints.BoundingBoxFormat.XYXY, canvas_size=canvas_size
                    )
                else:
                    target['boxes'] = datapoints.BoundingBox(
                        empty_boxes, format=datapoints.BoundingBoxFormat.XYXY, spatial_size=canvas_size
                    )
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
                target['area'] = torch.zeros((0,), dtype=torch.float32)
                target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
            else:
                min_count = min(num_boxes, num_labels)
                if isinstance(boxes, (datapoints.BoundingBoxes, getattr(datapoints, 'BoundingBox', object))):
                    target['boxes'] = boxes[:min_count]
                else:
                    target['boxes'] = boxes[:min_count]
                target['labels'] = labels[:min_count]
                if 'area' in target and target['area'].shape[0] > min_count:
                    target['area'] = target['area'][:min_count]
                if 'iscrowd' in target and target['iscrowd'].shape[0] > min_count:
                    target['iscrowd'] = target['iscrowd'][:min_count]

        return target

    def __getitem__(self, idx):
        if self.mosaic is not None and self.mosaic.is_enabled():
            img, target = self.mosaic(idx)
        else:
            img, target = self._load_item(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target = self._sanitize_target(target)

        current_width, current_height = self._get_current_image_wh(img)
        target['size'] = torch.as_tensor([current_width, current_height])

        if 'labels' in target and target['labels'].numel() > 0:
            target['labels'] = torch.clamp(target['labels'], min=0, max=NUM_CLASSES - 1)
            assert target['labels'].min() >= 0 and target['labels'].max() < NUM_CLASSES, \
                f"Bad labels after transforms: min={target['labels'].min().item()} max={target['labels'].max().item()} num_classes={NUM_CLASSES}"

        return img, target

    def __len__(self):
        return len(self.img_files)
