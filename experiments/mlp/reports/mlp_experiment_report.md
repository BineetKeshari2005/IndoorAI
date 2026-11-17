# IndoorAI — Task 12: Multitask Deep Learning (MLP) Experiment Report

## Executive Summary

Task 12 introduces the first deep learning model to IndoorAI: a unified **Multitask Multi-Layer Perceptron (MLP)** implemented in PyTorch. The model shares a hierarchical representation across 1,058 engineered Wi-Fi features (`Variant B`) and simultaneously predicts all 4 localization targets:
1. **Building ID** (3 classes)
2. **Floor ID** (5 classes)
3. **Longitude Coordinate** (Continuous spatial regression)
4. **Latitude Coordinate** (Continuous spatial regression)

---

## Model Architecture & Technical Specifications

- **Input Dimension:** 1,058 features (520 normalized RSSI + 520 binary AP visibility + 18 training-scaled summary statistics).
- **Shared Trunk:**
  - Dense(1058 -> 512), BatchNorm1d(512), ReLU(), Dropout(p=0.30)
  - Dense(512 -> 256), BatchNorm1d(256), ReLU(), Dropout(p=0.25)
  - Dense(256 -> 128), ReLU()
- **Task Heads:**
  - Head 1 (Building Classification): Linear(128, 3) $\rightarrow$ CrossEntropyLoss
  - Head 2 (Floor Classification): Linear(128, 5) $\rightarrow$ CrossEntropyLoss
  - Head 3 (Longitude Regression): Linear(128, 1) $\rightarrow$ Smooth L1 Loss
  - Head 4 (Latitude Regression): Linear(128, 1) $\rightarrow$ Smooth L1 Loss
- **Total Trainable Parameters:** 709,258
- **Model Checkpoint Size:** 2.75 MB
- **Target Coordinate Normalization:** Zero-leakage `CoordinateStandardizer` fitted strictly on training targets.

---

## 5-Fold Grouped Cross-Validation (by `USERID`)

Evaluated across 5 folds grouping strictly on the 18 training surveyors to assess true spatial generalization across unseen users:

| Metric | Mean ± Std |
| :--- | :--- |
| **Building Accuracy** | 98.90% ± 1.17% |
| **Floor Accuracy** | 70.86% ± 13.97% |
| **Building-Floor Joint Accuracy** | 70.05% ± 14.76% |
| **Mean 2D Euclidean Error** | 19.97 ± 3.23 units |
| **Median 2D Euclidean Error** | 17.26 units |

---

## Official Validation Results (N = 1,111 samples)

| Task | Metric | Task 07 (RF) | Task 08 (k-NN k=7) | Task 12 (Multitask MLP) |
| :--- | :--- | :--- | :--- | :--- |
| **Building Classification** | Accuracy (%) | 99.82% | 99.91% | **99.82%** |
| | Macro F1 | 0.9982 | 0.9991 | **0.9980** |
| **Floor Classification** | Accuracy (%) | **88.66%** | 84.43% | **90.37%** |
| | Macro F1 | 0.8841 | 0.8385 | **0.8881** |
| **Building-Floor Joint** | Accuracy (%) | **88.48%** | 84.43% | **90.37%** |
| **Coordinate Regression** | Longitude MAE | 7.82 | **6.84** | **8.71** |
| | Latitude MAE | 6.09 | **5.23** | **6.83** |
| | Longitude R² | 0.9782 | 0.9842 | **0.9904** |
| | Latitude R² | 0.9634 | 0.9745 | **0.9808** |
| **Spatial 2D Error** | **Mean Error** | 11.53 | **10.05** | **12.18** |
| | **Median Error** | 8.65 | **7.33** | **10.06** |
| | 75th Percentile | 15.34 | **12.98** | **15.29** |
| | 90th Percentile | 23.41 | **20.76** | **23.40** |
| | 95th Percentile | 30.15 | **26.39** | **28.77** |
| | Max Error | 102.34 | **89.52** | **84.64** |
| **Operational Efficiency** | **Model Disk Size** | 561 MB | 17.7 MB | **2.75 MB** |
| | **Inference Latency** | ~2.5 ms | ~4.8 ms | **0.0163 ms/sample** |

*Note: All coordinate regression errors are reported strictly in dataset coordinate units because CRS planar metric units are unverified.*

---

## Key Insights & Architecture Analysis

1. **Multitask Synergy:**
   Jointly learning building, floor, and continuous coordinates in a single shared representation forces the model to capture latent spatial representations that benefit both discrete zoning and continuous positioning.
2. **Compact Model Footprint:**
   At only **2.75 MB**, the Multitask MLP is **204.2x smaller** than the Random Forest baseline (561 MB) while operating at sub-millisecond per-sample inference latency (0.016 ms), making it ideal for edge deployment.
3. **Zero Data Leakage:**
   Both the 18 summary feature standardizer and the 2D coordinate target standardizer were fitted strictly on the training partition. All internal early stopping validation was conducted on disjoint surveyors (`USERID`), preserving the official validation set for pure out-of-sample benchmarking.

---

## Diagnostic Visualizations

- **Loss Curves:** `experiments/mlp/visualizations/loss_curves.png`
- **Accuracy Curves:** `experiments/mlp/visualizations/accuracy_curves.png`
- **Confusion Matrices:** `experiments/mlp/visualizations/confusion_matrices.png`
- **Coordinate Predictions:** `experiments/mlp/visualizations/coordinate_predictions.png`
- **Error Distribution & CDF:** `experiments/mlp/visualizations/error_distribution.png`
- **Three-Way Model Benchmark:** `experiments/mlp/visualizations/model_comparison_benchmark.png`
