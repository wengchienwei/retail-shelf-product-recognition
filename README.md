# Retail Shelf Product Recognition

Two-stage detection and classification pipeline for automated grocery product recognition using YOLOv5 and ResNet-18. Achieves **88.9% detection mAP** and **78.31% classification accuracy** with modular architecture enabling independent model optimization.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/yolov5)

**Datasets:** [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) (11,762 shelf images) | [Grocery Store](https://github.com/marcusklasson/GroceryStoreDataset) (5,125 images, 81 classes)

---

## Overview

Complete end-to-end pipeline for automated grocery product recognition combining YOLOv5 detection with ResNet-18 classification.
```
Shelf Image → [Stage 1: Detection] → Product Regions → [Stage 2: Classification] → Product Labels
              YOLOv5s                                   ResNet-18
```

## Design Rationale

**Two-stage approach advantages:**
- **Modularity:** Independent model upgrades without full pipeline retraining
- **Dataset leverage:** Uses SKU-110K (11K shelf images) for detection and Grocery Store (5K images, 81 classes) for classification without unified labels
- **Fine-grained focus:** ResNet-18 optimized for subtle product distinctions vs single-stage detector classifiers

## Pipeline Stages

### Stage 1: Product Detection
- **Model:** YOLOv5s (9.1M parameters)
- **Dataset:** SKU-110K (11,762 shelf images)
- **Task:** Single-class object detection
- **Performance:** 88.9% mAP@0.5, 89.6% precision, 81.8% recall
- **Output:** Cropped product regions

### Stage 2: Product Classification  
- **Model:** ResNet-18 fine-tuned (11.2M parameters)
- **Dataset:** Grocery Store (2,640 train / 2,485 test, 81 categories)
- **Task:** Fine-grained product classification
- **Performance:** 78.31% accuracy, 77.75% macro F1
- **Output:** Product category predictions

### Stage 3: End-to-End Integration
- **Input:** 7,990 crops from 50 shelf images
- **Processing:** 165 images/second
- **Finding:** 3.69% avg confidence (expected due to domain gap)
- **Validation:** Pipeline functionality confirmed

## Key Results

| Stage | Metric | Value |
|-------|--------|-------|
| Detection | mAP@0.5 | 88.9% |
| Detection | Precision/Recall | 89.6% / 81.8% |
| Classification | Test Accuracy | 78.31% |
| Classification | Macro F1 | 77.75% |
| Integration | Processing Speed | 165 img/s |
| Integration | Avg Confidence | 3.69% (domain gap) |

## Models

**Included:**
- `01_detection/sku110k_batch_3.pt` - YOLOv5 detector (18.5MB)
- `02_classification/best_resnet18_grocery.pth` - ResNet-18 classifier (42.8MB)
- `02_classification/class_names.json` - 81 class labels (required for Stage 3)

## Repository Structure
```
retail-shelf-product-recognition/
├── README.md
├── requirements.txt
├── LICENSE
│
├── 01_detection/
│   ├── product_detection_YOLO.ipynb
│   └── sku110k_batch_3.pt
│
├── 02_classification/
│   ├── product_classification_ResNet18.ipynb
│   ├── best_resnet18_grocery.pth          
│   └── class_names.json                          
│
├── 03_integration/
│   └── end_to_end_demo.ipynb
│
├── results/
│   └── figures/
│       ├── 01/                      # Detection visualizations
│       ├── 02/                      # Classification visualizations
│       └── 03/                      # Integration analysis
│
├── sample_outputs/
│   ├── 01_detection/                # Example detection outputs
│   └── 02_classification/           # Example classification results        
│
└── docs/
    ├── proposal.pdf
    └── final_report.pdf
```

## Quick Start
```bash
git clone https://github.com/wengchienwei/retail-shelf-product-recognition.git
cd retail-shelf-product-recognition
pip install -r requirements.txt
```

**Run notebooks sequentially:**
1. `01_detection/product_detection_YOLO.ipynb`
2. `02_classification/product_classification_ResNet18.ipynb`
3. `03_integration/end_to_end_demo.ipynb`

## Tech Stack

**Core:** Python 3.11+, PyTorch 2.0+, torchvision  
**Detection:** YOLOv5 (Ultralytics), SKU-110K dataset  
**Classification:** ResNet-18, Grocery Store dataset  
**Visualization:** matplotlib, seaborn  
**Experiment Tracking:** Weights & Biases

## Documentation
- [Project Proposal](docs/proposal.pdf) - Initial project design
- [Final Report](docs/final_report.pdf) - Complete technical report (LNCS format)

## Datasets

- **SKU-110K:** 11,762 dense shelf images for detection ([Goldman et al., CVPR 2019](https://github.com/eg4000/SKU110K_CVPR19))
- **Grocery Store:** 2,640 train / 2,485 test images, 81 fine-grained categories ([Klasson et al., WACV 2019](https://github.com/marcusklasson/GroceryStoreDataset))

## References

- Goldman, E., et al. (2019). Precise Detection in Densely Packed Scenes. CVPR.
- Klasson, M., et al. (2019). A Hierarchical Grocery Store Image Dataset. WACV.
- Ultralytics YOLOv5: [github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Authors

**Team Members:** Chien-Wei Weng, Yuhong Li, Ke Chen, Yingzhou Fang

MSc Data Sciences and Business Analytics  
CentraleSupélec, Université Paris-Saclay

---

**Academic Project | Foundations of Deep Learning (Fall 2025/2026)**
