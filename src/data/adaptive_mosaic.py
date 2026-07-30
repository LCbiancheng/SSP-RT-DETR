"""Adaptive Dense O2O Mosaic augmentation for NEU-DET.

When an image has few defects (<= same_image_threshold), uses same-image
light perturbation Mosaic; otherwise standard multi-image random Mosaic.
"""

import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

try:
    from torchvision import tv_tensors as datapoints
except ImportError:
    from torchvision import datapoints


class AdaptiveDenseO2OMosaic:
    def __init__(
        self,
        dataset,
        prob=0.5,
        same_image_threshold=1,
        mosaic_size=640,
        close_mosaic_last_epochs=10,
    ):
        self.dataset = dataset
        self.prob = float(prob)
        self.same_image_threshold = int(same_image_threshold)
        self.mosaic_size = int(mosaic_size)
        self.close_mosaic_last_epochs = int(close_mosaic_last_epochs)

        self.current_epoch = 0
        self.total_epochs = 100

    def set_epoch(self, epoch, total_epochs=None):
        self.current_epoch = int(epoch)
        if total_epochs is not None:
            self.total_epochs = int(total_epochs)

    def is_enabled(self):
        if self.prob <= 0:
            return False
        if self.current_epoch >= self.total_epochs - self.close_mosaic_last_epochs:
            return False
        return random.random() < self.prob

    @staticmethod
    def _load_image(dataset, idx):
        if isinstance(idx, int):
            img_path = os.path.join(dataset.img_folder, dataset.img_files[idx])
        else:
            img_path = idx
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        boxes, labels, area, iscrowd = dataset._parse_annotation(idx, width, height)
        return img, boxes, labels, area, iscrowd, width, height

    @staticmethod
    def _light_augment(img):
        factor = 0.9 + random.random() * 0.2
        img = ImageEnhance.Brightness(img).enhance(factor)
        factor = 0.9 + random.random() * 0.2
        img = ImageEnhance.Contrast(img).enhance(factor)
        if random.random() < 0.1:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 1.5, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        if random.random() < 0.3:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def _random_resize(img, target_w, target_h):
        orig_w, orig_h = img.size
        scale = random.uniform(0.5, 1.5)
        new_w = int(target_w * scale)
        new_h = int(target_h * scale)
        img = img.resize((max(1, new_w), max(1, new_h)), Image.BILINEAR)
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        return img, scale_x, scale_y

    def _make_bounding_boxes(self, boxes_tensor, canvas_h, canvas_w):
        if hasattr(datapoints, 'BoundingBoxes'):
            return datapoints.BoundingBoxes(
                boxes_tensor,
                format=datapoints.BoundingBoxFormat.XYXY,
                canvas_size=(canvas_h, canvas_w),
            )
        return datapoints.BoundingBox(
            boxes_tensor,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=(canvas_h, canvas_w),
        )

    def _same_image_mosaic(self, main_img, main_boxes, main_labels):
        sub_w = self.mosaic_size // 2

        canvas = Image.new('RGB', (self.mosaic_size, self.mosaic_size), (114, 114, 114))
        all_boxes = []
        all_labels = []

        scale_x = sub_w / main_img.width
        scale_y = sub_w / main_img.height

        for i in range(4):
            sub_img = self._light_augment(main_img.copy())
            sub_img = sub_img.resize((sub_w, sub_w), Image.BILINEAR)

            offset_x = (i % 2) * sub_w
            offset_y = (i // 2) * sub_w
            canvas.paste(sub_img, (offset_x, offset_y))

            if len(main_boxes) > 0:
                boxes_np = main_boxes.copy()
                boxes_np[:, 0] = boxes_np[:, 0] * scale_x + offset_x
                boxes_np[:, 1] = boxes_np[:, 1] * scale_y + offset_y
                boxes_np[:, 2] = boxes_np[:, 2] * scale_x + offset_x
                boxes_np[:, 3] = boxes_np[:, 3] * scale_y + offset_y

                boxes_np[:, 0::2] = np.clip(boxes_np[:, 0::2], 0, self.mosaic_size - 1)
                boxes_np[:, 1::2] = np.clip(boxes_np[:, 1::2], 0, self.mosaic_size - 1)

                valid = (boxes_np[:, 2] > boxes_np[:, 0]) & (boxes_np[:, 3] > boxes_np[:, 1])
                if valid.any():
                    all_boxes.append(boxes_np[valid])
                    all_labels.append(main_labels[valid])

        if all_boxes:
            merged_boxes = np.concatenate(all_boxes, axis=0)
            merged_labels = np.concatenate(all_labels, axis=0)
        else:
            merged_boxes = np.zeros((0, 4), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int64)

        return canvas, merged_boxes, merged_labels

    def _multi_image_mosaic(self, main_img, main_boxes, main_labels, main_idx):
        sub_w = self.mosaic_size // 2

        n_total = len(self.dataset)
        quadrants = [(0, 0), (sub_w, 0), (0, sub_w), (sub_w, sub_w)]
        quadrant_images = [None, None, None, None]
        quadrant_boxes = [None, None, None, None]
        quadrant_labels = [None, None, None, None]

        main_quadrant = random.randint(0, 3)
        quadrant_images[main_quadrant] = main_img
        quadrant_boxes[main_quadrant] = main_boxes
        quadrant_labels[main_quadrant] = main_labels

        for q in range(4):
            if quadrant_images[q] is not None:
                continue
            rand_idx = random.randint(0, n_total - 1)
            while rand_idx == main_idx:
                rand_idx = random.randint(0, n_total - 1)

            img, boxes, labels, _, _, _, _ = self._load_image(self.dataset, rand_idx)
            quadrant_images[q] = img
            quadrant_boxes[q] = boxes
            quadrant_labels[q] = labels

        canvas = Image.new('RGB', (self.mosaic_size, self.mosaic_size), (114, 114, 114))
        all_boxes = []
        all_labels = []

        for q in range(4):
            img = quadrant_images[q]
            boxes_np = np.array(quadrant_boxes[q], dtype=np.float32).copy()
            labels_np = np.array(quadrant_labels[q], dtype=np.int64).copy()

            orig_w, orig_h = img.size
            scale_x = sub_w / orig_w
            scale_y = sub_w / orig_h
            img = img.resize((sub_w, sub_w), Image.BILINEAR)

            offset_x, offset_y = quadrants[q]
            canvas.paste(img, (offset_x, offset_y))

            if len(boxes_np) > 0:
                boxes_np[:, 0] = boxes_np[:, 0] * scale_x + offset_x
                boxes_np[:, 1] = boxes_np[:, 1] * scale_y + offset_y
                boxes_np[:, 2] = boxes_np[:, 2] * scale_x + offset_x
                boxes_np[:, 3] = boxes_np[:, 3] * scale_y + offset_y

                boxes_np[:, 0::2] = np.clip(boxes_np[:, 0::2], 0, self.mosaic_size - 1)
                boxes_np[:, 1::2] = np.clip(boxes_np[:, 1::2], 0, self.mosaic_size - 1)

                valid = (boxes_np[:, 2] > boxes_np[:, 0]) & (boxes_np[:, 3] > boxes_np[:, 1])
                if valid.any():
                    all_boxes.append(boxes_np[valid])
                    all_labels.append(labels_np[valid])

        if all_boxes:
            merged_boxes = np.concatenate(all_boxes, axis=0)
            merged_labels = np.concatenate(all_labels, axis=0)
        else:
            merged_boxes = np.zeros((0, 4), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int64)

        return canvas, merged_boxes, merged_labels

    def __call__(self, idx):
        main_img, boxes, labels, area, iscrowd, img_w, img_h = self._load_image(
            self.dataset, idx
        )
        boxes_np = np.array(boxes, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int64)

        num_gt = len(boxes_np)

        if num_gt <= self.same_image_threshold:
            mosaic_img, merged_boxes, merged_labels = self._same_image_mosaic(
                main_img, boxes_np, labels_np
            )
        else:
            mosaic_img, merged_boxes, merged_labels = self._multi_image_mosaic(
                main_img, boxes_np, labels_np, idx
            )

        merged_boxes = torch.as_tensor(merged_boxes, dtype=torch.float32)
        merged_labels = torch.as_tensor(merged_labels, dtype=torch.int64)
        merged_area = (merged_boxes[:, 2] - merged_boxes[:, 0]) * (
            merged_boxes[:, 3] - merged_boxes[:, 1]
        )
        merged_iscrowd = torch.zeros(len(merged_labels), dtype=torch.int64)

        target = {
            'image_id': torch.tensor([idx]),
            'boxes': merged_boxes,
            'labels': merged_labels,
            'area': merged_area,
            'iscrowd': merged_iscrowd,
            'orig_size': torch.as_tensor([self.mosaic_size, self.mosaic_size]),
            'size': torch.as_tensor([self.mosaic_size, self.mosaic_size]),
        }

        target['boxes'] = self._make_bounding_boxes(
            target['boxes'], self.mosaic_size, self.mosaic_size
        )

        return mosaic_img, target
