# Two-Stage Grocery Product Recognition

MSc Data Sciences and Business Analytics - CentraleSupélec, Université Paris-Saclay

**Team Members:** Chien-Wei Weng, Ke Chen, Yuhong Li, Yingzhou Fang

---

## Overview

A modular pipeline combining YOLOv5 detection with ResNet-18 classification for automated grocery product recognition on retail shelves.
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
- **Dataset:** SKU-110K (11,762 images, 147 products/image avg)
- **Task:** Single-class object detection
- **Performance:** 88.9% mAP@0.5, 89.6% precision, 81.8% recall
- **Output:** Cropped product regions

### Stage 2: Product Classification  
- **Model:** ResNet-18 fine-tuned (11.2M parameters)
- **Dataset:** Grocery Store (2,640 train / 2,485 test images, 81 categories)
- **Task:** Fine-grained product classification
- **Performance:** 78.31% accuracy, 77.75% macro F1
- **Finding:** 7 vegetable classes show circular misclassification pattern

## Repository Structure
```
grocery-product-recognition/
├── README.md
├── requirements.txt
│
├── 01_detection/
│   └── product_detection_YOLO.ipynb
│
├── 02_classification/
│   └── product_classification_ResNet18.ipynb
│
├── results/
│   └── figures/
│       ├── 01/                      # Detection visualizations
│       └── 02/                      # Classification visualizations
│
├── sample_outputs/
│   ├── 01_detection/                # Example detection outputs
│   └── 02_classification/           # Example classification results
│
└── docs/
    └── proposal.pdf
```

## Installation
```bash
git clone https://github.com/wengchienwei/retail-shelf-product-recognition.git
cd retail-shelf-product-recognition
pip install -r requirements.txt
```

## Usage

### Stage 1: Detection
```python
# Open in Google Colab
01_detection/product_detection_YOLO.ipynb
```

### Stage 2: Classification
```python
# Open in Google Colab
02_classification/product_classification_ResNet18.ipynb
```

## Results Summary

| Stage | Metric | Value |
|-------|--------|-------|
| Detection | mAP@0.5 | 88.9% |
| Detection | Precision/Recall | 89.6% / 81.8% |
| Classification | Test Accuracy | 78.31% |
| Classification | Macro F1 | 77.75% |

## Datasets

**SKU-110K** (Detection)
- Source: [GitHub](https://github.com/eg4000/SKU110K_CVPR19)
- 11,762 shelf images with dense product annotations
- Single-class object detection task

**Grocery Store** (Classification)
- Source: [Klasson et al., 2019](https://doi.org/10.1109/WACV.2019.00058)
- 5,125 images across 81 fine-grained product categories
- Real-world retail shelf photography

## References

- Goldman, E., et al. (2019). Precise Detection in Densely Packed Scenes. CVPR.
- Klasson, M., et al. (2019). A Hierarchical Grocery Store Image Dataset. WACV.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Ultralytics YOLOv5: [github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

---

**Course:** Deep Learning - Fall 2025/2026  
**Institution:** CentraleSupélec, Université Paris-Saclay
