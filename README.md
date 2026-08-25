# SSP-RT-DETR: an improved RT-DETR model for steel surface defect detection

---

## Description

SSP-RT-DETR is an improved RT-DETR model for steel surface defect detection, built on RT-DETR (ResNet-18 backbone, 640×640). It addresses issues in steel surface defect detection such as sparse defect distribution, easy loss of fine-grained defect details, and insufficient positive-sample supervision. The code includes complete data processing, model definition, training solver, inference, and ONNX export scripts, and is validated on both the **NEU-DET** and **GC10-DET** datasets.

---

## Dataset Information

### NEU-DET

A steel surface defect dataset with 6 defect classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.

- Number of images: 1800
- Number of annotated boxes: 3351
- Annotation format: VOC XML
- Source: <https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/>

### GC10-DET

A metallic surface defect dataset with 10 defect classes: `chongkong`, `hanfeng`, `yueyawan`, `shuiban`, `youban`, `siban`, `yiwu`, `yahen`, `zhehen`, `yaozhe`.

- Number of images: 2070
- Annotation format: VOC XML
- Source: <https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets>

### Dataset Splitting

The train / validation / test split uses an 8:1:1 ratio. The splitting scripts are located in `scripts/`:

- `scripts/split_neudet_dataset.py`: splits the NEU-DET dataset from `./NEU-DET` into `./NEU-DET-split`.
- `scripts/split_gc10det_dataset.py`: splits the GC10-DET dataset from `./GC10-DET` into `./GC10-DET-split`.

---

## Code Information

```
SSP-RT-DETR/
├── README.md                  # Project description
├── requirements.txt           # Python dependencies
│
├── configs/                   # Configuration files
│   ├── runtime.yml            # Runtime configuration (AMP / EMA / early stopping, etc.)
│   ├── dataset/
│   │   ├── neudet_detection.yml    # NEU-DET dataset and augmentation config
│   │   └── gc10det_detection.yml   # GC10-DET dataset and augmentation config
│   └── rtdetr/
│       ├── include/                # Shared base configs (backbone / dataloader / optimizer)
│       └── ablation_200e_4090/     # Ablation experiment configs
│
├── src/                       # Source code
│   ├── core/                  # Core config parsing
│   ├── data/                  # Data processing (coco / neudet / gc10det, augmentation)
│   ├── nn/                    # Neural networks (backbone / criterion)
│   ├── misc/                  # Utilities (logging, distributed, visualization)
│   ├── optim/                 # Optimizer (AMP / EMA / optimizer wrappers)
│   ├── solver/                # Training solver
│   └── zoo/rtdetr/            # RT-DETR model
│
├── tools/                     # Entry-point scripts
│   ├── train.py               # Training entry point
│   ├── infer.py               # Inference
│   ├── export_onnx.py         # ONNX export
│   └── ablation_guard.py      # Ablation experiment manager
│
├── scripts/                   # Dataset splitting scripts
│   ├── split_neudet_dataset.py
│   └── split_gc10det_dataset.py
│
└── figure_scripts/            # Visualization scripts
    ├── fig2_draw_mosaic_neudet.py  # Mosaic vs. SA-Mosaic comparison figure
    ├── fig6_model_heatmap.py       # Model heatmap comparison figure
    └── fig7_neudet_comparison.py   # NEU-DET multi-model detection comparison figure
```

---

## Usage Instructions

### Requirements

- Python >= 3.10
- PyTorch >= 2.1.0 (install the CUDA build matching your environment)
- Other dependencies listed in `requirements.txt`:

```text
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
onnx==1.14.0
onnxruntime==1.15.1
pycocotools
PyYAML
scipy
tensorboard
```

### Installation

```bash
conda create -n SSP-RT-DETR python=3.10 -y && conda activate SSP-RT-DETR
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Dataset Preparation

1. Download the NEU-DET / GC10-DET datasets into the project directory (see the source links above).
2. Run the splitting scripts to generate the `train / val / test` splits:

```bash
python scripts/split_neudet_dataset.py
python scripts/split_gc10det_dataset.py
```

Dataset loading: the `train_dataloader` / `val_dataloader` / `test_dataloader` sections in the config files specify the dataset via `type` (`NeuDetDetection` / `GC10DetDetection`) and the image / annotation directories via `img_folder` / `ann_folder`. During training, the dataset classes under `src/data/` automatically load the VOC XML annotations and apply data augmentation.

### Training

```bash
python tools/train.py -c configs/rtdetr/ablation_200e_4090/rtdetr_r50vd_6x_neudet_baseline_200e.yml --seed 42 --amp
```

---

## Methodology

SSP-RT-DETR improves RT-DETR for steel surface defect detection through three components:

1. **Sparse-Aware Mosaic (SA-Mosaic)**: According to the number of ground-truth defect boxes in each image, dynamically selects either same-image perturbation Mosaic or standard cross-image Mosaic, increasing the density of effective supervision in sparse-defect samples.
2. **Shallow Cross-Attention Fusion (SCAF)**: Performs cross-attention-based fusion between the shallow detail features from S2 and the semantic features from S3, enhancing detection of small defects such as fine cracks and scratches.
3. **Perturbation-Guided One-to-Many Auxiliary Supervision (PG-O2MAS)**: Generates multiple groups of perturbed queries during training and uses an IoU-guided Top-M selection strategy to assign high-quality auxiliary positive samples to each defect, alleviating the sparse-supervision problem caused by one-to-one assignment.

On NEU-DET, SSP-RT-DETR achieves 47.50% AP and 77.01% AP50, exceeding the RT-DETR baseline by 3.69 and 4.11 percentage points. On GC10-DET, AP and AP50 improve by 1.93 and 2.39 percentage points.

---

## Citations

```bibtex
@article{lv2020gc10,
  author  = {Lv, X. and Duan, F. and Jiang, J.-J. and Fu, X. and Gan, L.},
  title   = {Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network},
  journal = {Sensors},
  volume  = {20},
  number  = {6},
  pages   = {1562},
  year    = {2020},
  doi     = {10.3390/s20061562},
  note    = {\doi{10.3390/s20061562}}
}

@article{song2013neu,
  author  = {Song, K. and Yan, Y.},
  title   = {A Noise Robust Method Based on Completed Local Binary Patterns for Hot-Rolled Steel Strip Surface Defects},
  journal = {Applied Surface Science},
  volume  = {285},
  pages   = {858--864},
  year    = {2013},
  doi     = {10.1016/j.apsusc.2013.09.002},
  note    = {\doi{10.1016/j.apsusc.2013.09.002}}
}
```

---

## License & Contribution Guidelines

- Baozhang Liu: Conceptualization, Data Curation, Methodology, Validation, Software, Formal analysis, Investigation, Writing-original draft, Visualization.
- Wei Shi: Conceptualization, Methodology, Resources, Writing-review & editing, Project administration, Supervision.
- Jingyang Wang: Conceptualization, Methodology, Resources, Writing-review & editing, Project administration, Supervision, Funding acquisition.
