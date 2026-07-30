# SSP-RT-DETR: Steel Surface Defect Detection

Training-stage improvements based on RT-DETR (ResNet-18 backbone, 640x640) for steel surface defect detection. Validated on both **NEU-DET** and **GC10-DET** datasets.

**Core Principle**: All innovations are enabled only during training, with zero additional overhead during inference.

---

## Directory Structure

```
SSP-RT-DETR/
├── README.md                  # Project description
├── requirements.txt           # Python dependencies
│
├── configs/                   # Configuration files
│   ├── runtime.yml
│   ├── dataset/
│   │   ├── neudet_detection.yml   
│   │   └── gc10det_detection.yml   
│   └── rtdetr/
│       ├── include/                # Shared base configurations
│       └── ablation_200e_4090/     # Ablation experiment configurations
│
├── src/                       # Source code
│   ├── core/                  # Core configuration
│   ├── data/                  # Data processing (coco/neudet/gc10det)
│   ├── nn/                    # Neural networks (backbone/criterion)
│   ├── misc/                  # Utilities
│   ├── optim/                 # Optimizer
│   ├── solver/                # Training solver
│   └── zoo/rtdetr/            # RT-DETR model
│
├── tools/                     # Utility scripts
│   ├── train.py               # Training entry point
│   ├── infer.py               # Inference
│   ├── export_onnx.py         # ONNX export
│   └── ablation_guard.py      # Ablation experiment manager
│
├── scripts/                   # Dataset splitting
│   ├── split_neudet_dataset.py
│   └── split_gc10det_dataset.py
---

## Datasets

### NEU-DET

crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches. A total of 1800 images, 3351 annotated boxes, VOC XML format.

### GC10-DET

chongkong, hanfeng, yueyawan, shuiban, youban, siban, yiwu, yahen, zhehen, yaozhe. A total of 2070 images, VOC XML format.

---
## Dataset Splitting

Dataset splitting scripts for NEU-DET and GC10-DET are located in `scripts/`.

---
## Innovations

Steel surface defect detection is a critical component of industrial quality inspection, and its detection accuracy directly impacts steel product quality and production safety. To address issues in RT-DETR for steel surface defect detection, such as sparse defect target distribution, easy loss of fine-grained defect details, and insufficient positive sample supervision, this paper proposes an improved RT-DETR model for steel surface defect detection.

| No. | Name | Description | Location |
|------|------|------|---------|
| **E1** | SA-Mosaic (Sparse-Aware Mosaic) | Dynamically selects SA-Mosaic or standard Mosaic augmentation based on the number of real defect boxes in the image, increasing effective defect supervision density in sparse samples | `Dataset.__getitem__` |
| **E2** | SCAF (Shallow Cross-Attention Fusion) | Fuses S2 shallow detail features with S3 semantic features via cross-attention, enhancing the model's perception of fine-grained defects such as tiny cracks and scratches | `BaselineHybridEncoder` |
| **E3** | PG-O2MAS (Perturbation-Guided One-to-Many Auxiliary Supervision) | Generates multiple groups of perturbed queries during training, supplementing high-quality auxiliary positive samples for each defect via IoU-guided Top-M filtering, alleviating supervision sparsity caused by one-to-one matching | `Decoder.forward()` + `Criterion` |

---
## Experiments

### NEU-DET Ablation Experiments (200 epoch, bs=16, RTX 4090)

Complete ablation study on the NEU-DET dataset: 8 experiments = B0 + 3 single-item + 3 two-item + full combination.

| Experiment | Configuration |
|------|------|
| **B0** Baseline | `baseline` |
| **E1** +SA-Mosaic | `mosaic_only` |
| **E2** +SCAF | `s2_cross_attn` |
| **E3** +PG-O2MAS | `query_perturb_only` |
| **E12** E1 + E2 | `mosaic_s2` |
| **E13** E1 + E3 | `mosaic_pgo2m` |
| **E23** E2 + E3 | `pgo2m_s2` |
| **E123** All | `all_innovations` |

### GC10-DET Generalization Experiments (200 epoch, bs=16)

Comparative validation of Baseline vs. full innovation combination on the GC10-DET dataset.

## Environment Setup & Usage

### Installation

```bash
conda create -n SSP-RT-DETR python=3.10 -y && conda activate SSP-RT-DETR
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Training

```bash
# Single experiment
python tools/train.py -c configs/rtdetr/ablation_200e_4090/rtdetr_r50vd_6x_neudet_baseline_200e.yml --seed 42 --amp
```

### Inference / Export

```bash
python tools/infer.py -c <config> -r <checkpoint> --source <img>
python tools/export_onnx.py -c <config> -r <checkpoint>
```
---

## Training Hyperparameters

| Parameter | NEU-DET (4090) | GC10-DET |
|------|--------------|----------|
| Backbone | ResNet-18 | ResNet-18 |
| Input Size | 640x640 | 640x640 |
| Batch Size | 16 | 16 |
| LR / BLR | 1e-4 / 1e-5 | 1e-4 / 1e-5 |
| Optimizer | AdamW (0.9, 0.999) | AdamW (0.9, 0.999) |
| Epochs | 200 | 200 |
