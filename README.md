# A²GC: Asymmetric Aggregation with Geometric Constraints for Locally Aggregated Descriptors

A PyTorch implementation of A²GC (Asymmetric Aggregation with Geometric Constraints) for Visual Place Recognition (VPR), featuring support for DINOv2 backbones.

## 🚀 Features

- **Multiple Backbone Support**: DINOv2 (ViT-B/14, ViT-L/14, ViT-G/14), DINOv3 (ViT-S/16), and ResNet
- **Asymmetric Aggregation with Geometric Constraints**: A²GC aggregator for robust feature aggregation
- **Comprehensive Evaluation**: Support for multiple VPR benchmarks (Pittsburgh, MSLS, Nordland, SPED, SF-XL)
- **Flexible Training**: PyTorch Lightning-based training with various loss functions and optimizers
- **Visualization Tools**: Feature matching and heatmap visualization utilities

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Citation](#citation)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.12+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/CV4RA/A2GC.git
cd A2GC
```

2. Create a conda environment:
```bash
conda env create -f environment.yml
conda activate A2GC
```

3. Install additional dependencies:
```bash
pip install pytorch-lightning faiss-cpu  # or faiss-gpu for GPU support
```

## 🚀 Quick Start

### Evaluation with Pre-trained Model

```bash
python eval.py \
    --ckpt_path weights/your_best.ckpt \
    --backbone_arch dinov2_vitb14 \
    --val_datasets pitts30k_test msls_val \
    --faiss_gpu
```

### Training

```bash
python main.py
```

Modify `main.py` to customize training parameters, backbone architecture, and aggregator configuration.

## 📦 Dataset Preparation

### Supported Datasets

- **Pittsburgh 30k/250k**: Download from [VPR-Bench](https://github.com/MubarizZaffar/VPR-Bench)
- **MSLS (Mapillary Street Level Sequences)**: Download from [MSLS website](https://www.mapillary.com/datasets/places)
- **Nordland**: Download from [Nordland dataset](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W)
- **SPED**: Download from [SPED dataset](https://github.com/ahmetozlu/vehicle_counting_tensorflow)
- **SF-XL**: Download from [SF-XL dataset](https://github.com/gmberton/CosPlace)

### Dataset Structure

```
data/
├── Pittsburgh/
│   ├── queries_real/
│   └── [000-010]/
├── mapillary/
│   ├── train_val/
│   └── test/
├── Nordland/
│   ├── query/
│   └── ref/
└── ...

datasets/
├── Pittsburgh/
│   ├── pitts30k_test_dbImages.npy
│   ├── pitts30k_test_qImages.npy
│   └── pitts30k_test_gt.npy
└── ...
```

## 🏋️ Training

### Configuration

Edit `main.py` to configure:

- **Backbone**: `backbone_arch` (e.g., `'dinov2_vitb14'`)
- **Aggregator**: `agg_arch` (e.g., `'ASYOT'` for A²GC)
- **Training parameters**: learning rate, batch size, optimizer, etc.

### Training on GSV-Cities

The default training uses GSV-Cities dataset. Ensure the dataset is properly set up in `data/GSVCities/`.

## 📈 Results

### Performance on Standard Benchmarks

| Dataset | R@1 | R@5 | R@10 |
|---------|-----|-----|------|
| Pitts30k | 96.7 | 99.8 | 100.0 |
| MSLS-val | 96.4 | 97.9 | 98.6 |

*Results with DINOv2-ViT-B/14 backbone at 588×588 input resolution*

### Impact of Input Resolution

| Input Size | Pitts30k (R@1/5/10) | MSLS-val (R@1/5/10) |
|------------|---------------------|---------------------|
| 224×224 | 94.9/98.5/99.5 | 90.4/95.3/96.1 |
| 364×364 | 94.9/99.1/99.6 | 91.0/96.0/96.6 |
| 406×406 | 95.2/99.2/99.8 | 93.2/96.7/97.2 |
| 588×588 | **96.7/99.8/100** | **96.4/97.9/98.6** |

## 🎨 Visualization

### Feature Matching Visualization

Visualize feature matches between query and reference images:

```bash
python tools/visualize_feature_matching.py \
    --query path/to/query.jpg \
    --ref path/to/reference.jpg \
    --ckpt weights/a2gc.ckpt \
    --backbone dinov2_vitb14 \
    --top-k 200 \
    --threshold 0.3 \
    --out ./viz_matching
```

This generates:
- `feature_matching_lines.png`: Matching points with color-coded similarity scores
- `similarity_matrix.png`: Full similarity matrix visualization
- `feature_heatmap_comparison.png`: Feature activation heatmaps

### Feature Map Visualization

```bash
python tools/visualize_feature_maps.py \
    --image path/to/image.jpg \
    --ckpt weights/a2gc.ckpt \
    --backbone dinov2_vitb14 \
    --out ./viz_features
```

## 🏗️ Architecture

### Model Components

1. **Backbone**: Feature extraction (DINOv2/DINOv3/ResNet)
2. **Aggregator**: Feature aggregation (ASYOT for A²GC/MixVPR/ConvAP)
3. **Loss Function**: Metric learning loss (MultiSimilarityLoss, TripletMarginLoss, etc.)

### A²GC Aggregator

The Asymmetric Aggregation with Geometric Constraints (A²GC) aggregator uses asymmetric optimal transport with geometric constraints to aggregate spatial features, providing robust place descriptors that are invariant to viewpoint changes and partial occlusions.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{a2gc2024,
  title={A²GC: Asymmetric Aggregation with Geometric Constraints for Locally Aggregated Descriptors},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) for the vision transformer backbone
- [DINOv3](https://github.com/facebookresearch/dinov3) for the latest vision transformer
- [CosPlace](https://github.com/gmberton/CosPlace) for dataset preparation utilities
- [PyTorch Lightning](https://www.pytorchlightning.ai/) for the training framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Note**: This is a research implementation. For production use, additional optimizations and testing may be required.

