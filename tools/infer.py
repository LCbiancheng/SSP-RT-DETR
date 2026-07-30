import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as T
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
from src.core import YAMLConfig 


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def resolve_eval_size(cfg, default=512):
    encoder_cfg = cfg.yaml_cfg.get('HybridEncoder', {})
    eval_size = encoder_cfg.get('eval_spatial_size', None)
    if isinstance(eval_size, (list, tuple)) and len(eval_size) == 2:
        return int(eval_size[0]), int(eval_size[1])
    return default, default


def build_infer_transform(input_size):
    return T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def finalize_predictions(labels, boxes, scores, score_threshold=None, nms_threshold=None):
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.int64)
    if not torch.is_tensor(boxes):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if not torch.is_tensor(scores):
        scores = torch.as_tensor(scores, dtype=torch.float32)

    labels = labels.reshape(-1).to(torch.int64)
    boxes = boxes.reshape(-1, 4).to(torch.float32)
    scores = scores.reshape(-1).to(torch.float32)

    if score_threshold is not None:
        keep = scores > float(score_threshold)
        labels = labels[keep]
        boxes = boxes[keep]
        scores = scores[keep]

    if labels.numel() > 0 and nms_threshold is not None:
        keep = torchvision.ops.batched_nms(boxes, scores, labels, float(nms_threshold))
        labels = labels[keep]
        boxes = boxes[keep]
        scores = scores[keep]

    return [labels.cpu()], [boxes.cpu()], [scores.cpu()]


def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size

    slices = []
    coordinates = []
    step_x = max(1, int(slice_width * (1 - overlap_ratio)))
    step_y = max(1, int(slice_height * (1 - overlap_ratio)))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates


def merge_predictions(predictions, slice_coordinates, orig_image_size, threshold=None):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, prediction in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        labels = prediction['labels'].detach().cpu().reshape(-1)
        boxes = prediction['boxes'].detach().cpu().reshape(-1, 4).clone()
        scores = prediction['scores'].detach().cpu().reshape(-1)
        if threshold is not None:
            valid_indices = scores > float(threshold)
            valid_labels = labels[valid_indices]
            valid_boxes = boxes[valid_indices]
            valid_scores = scores[valid_indices]
        else:
            valid_labels = labels
            valid_boxes = boxes
            valid_scores = scores
        for j, box in enumerate(valid_boxes):
            box[0] = torch.clamp(box[0] + x_shift, min=0, max=orig_width)
            box[1] = torch.clamp(box[1] + y_shift, min=0, max=orig_height)
            box[2] = torch.clamp(box[2] + x_shift, min=0, max=orig_width)
            box[3] = torch.clamp(box[3] + y_shift, min=0, max=orig_height)
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)

    if merged_labels:
        labels = torch.stack([label.to(torch.int64) for label in merged_labels])
        boxes = torch.stack(merged_boxes).to(torch.float32)
        scores = torch.stack(merged_scores).to(torch.float32)
        return labels, boxes, scores

    return (
        torch.zeros(0, dtype=torch.int64),
        torch.zeros((0, 4), dtype=torch.float32),
        torch.zeros(0, dtype=torch.float32),
    )


def draw(images, labels, boxes, scores, thrh = 0.6, path = ""):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        for j,b in enumerate(box):
            draw.rectangle(b.tolist(), outline='red',)
            draw.text((float(b[0].item()), float(b[1].item())), text=f"label: {lab[j].item()} {round(scrs[j].item(),2)}", font=ImageFont.load_default(), fill='blue')
        if path == "":
            im.save(f'results_{i}.jpg')
        else:
            im.save(path)
            
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    postprocessor = cfg.postprocessor

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            
        def forward(self, images):
            return self.model(images)
    
    model = Model().to(args.device)
    model.eval()
    postprocessor.eval()
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = build_infer_transform(resolve_eval_size(cfg))
    im_data = transforms(im_pil)[None].to(args.device)
    if args.sliced:
        num_boxes = max(1, int(args.numberofboxes))

        aspect_ratio = w / h
        num_cols = max(1, int(np.sqrt(num_boxes * aspect_ratio)))
        num_rows = max(1, int(np.ceil(num_boxes / num_cols)))
        slice_height = max(1, h // num_rows)
        slice_width = max(1, w // num_cols)
        overlap_ratio = 0.2
        slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
        predictions = []
        device_type = 'cuda' if args.device.startswith('cuda') else 'cpu'
        for i, slice_img in enumerate(slices):
            slice_tensor = transforms(slice_img)[None].to(args.device)
            with torch.no_grad(), torch.autocast(device_type=device_type, enabled=args.device.startswith('cuda')):
                output = model(slice_tensor)
                result = postprocessor(
                    output,
                    torch.tensor([[slice_img.size[0], slice_img.size[1]]], device=args.device),
                )[0]
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
            predictions.append(result)
        
        merged_labels, merged_boxes, merged_scores = merge_predictions(
            predictions,
            coordinates,
            (h, w),
            threshold=args.score_threshold if args.score_threshold is not None else postprocessor.score_threshold,
        )
        labels, boxes, scores = finalize_predictions(
            merged_labels,
            merged_boxes,
            merged_scores,
            score_threshold=args.score_threshold if args.score_threshold is not None else postprocessor.score_threshold,
            nms_threshold=args.nms_threshold if args.nms_threshold is not None else postprocessor.nms_threshold,
        )
    else:
        with torch.no_grad():
            output = model(im_data)
            result = postprocessor(output, orig_size)[0]
        labels, boxes, scores = finalize_predictions(
            result['labels'],
            result['boxes'],
            result['scores'],
            score_threshold=args.score_threshold if args.score_threshold is not None else postprocessor.score_threshold,
            nms_threshold=args.nms_threshold if args.nms_threshold is not None else postprocessor.nms_threshold,
        )
        
    draw([im_pil], labels, boxes, scores, 0.6)
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-s', '--sliced', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    parser.add_argument('--score-threshold', type=float, default=None)
    parser.add_argument('--nms-threshold', type=float, default=None)
    args = parser.parse_args()
    main(args)




