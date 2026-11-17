# k-Nearest Neighbors (k-NN) Localization Baseline Report (Task 08)

**Date:** 2026-11-16  
**Experiment:** `knn_baseline`  
**Random Seed:** `42`  
**Training Set Size:** 19,937 fingerprints  
**Validation Set Size:** 1,111 fingerprints  
**Selected Feature Variant:** `variant_b_engineered` (1058 features)  
**Selected Distance Metric:** `Cosine`  
**Optimal Neighbors (k):** `7`  
**Weighting Scheme:** `Distance`-weighted  

---

## 1. Executive Summary

This experiment establishes the second foundational benchmark for **IndoorAI** using **k-Nearest Neighbors (k-NN)** spatial retrieval. Unlike parametric or ensemble tree models, k-NN is a non-parametric instance-based estimator that directly mimics classic indoor Wi-Fi fingerprint pattern matching by computing pairwise distances across high-dimensional signal representations.

### Key Validation Results (Single-Pass on Official Benchmark Split)

| Model Head | Target Attribute | Task Type | Validation Metric | Score |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** | `BUILDINGID` | Multi-class Classification (3 classes) | **Accuracy** | **99.91%** |
| **Model B** | `FLOOR` | Multi-class Classification (5 classes) | **Accuracy** | **84.43%** |
| **Model C** | `BUILDING_FLOOR` | Multi-class Classification (13 classes) | **Accuracy** | **84.43%** |
| **Model D** | `LONGITUDE` | 1D Continuous Regression | **MAE / RMSE** | **6.71 / 10.17** |
| **Model E** | `LATITUDE` | 1D Continuous Regression | **MAE / RMSE** | **6.10 / 9.52** |
| **Composite** | `(x, y)` Coordinate | 2D Spatial Positioning | **Mean Euclidean Error** | **10.05 units** |
| **Composite** | `(x, y)` Coordinate | 2D Spatial Positioning | **Median Euclidean Error** | **7.33 units** |
| **Composite** | `(x, y)` Coordinate | 2D Spatial Positioning | **95th Percentile Error** | **29.06 units** |

> [!IMPORTANT]
> The coordinate reference system (CRS) remains unverified. All coordinate errors are reported strictly in **dataset coordinate units** (not metres) to avoid fabricating unverified physical scale claims.

---

## 2. Feature Scaling & Distance Metric Strategy

### Critical Finding: The Catastrophic Effect of Unscaled Summary Features
In Task 05, 18 summary statistics were engineered (e.g. `unavailable_ap_count` ranging up to 520, `rssi_mean` ranging between -105 and -50 dBm).
Empirical investigation revealed that evaluating k-NN directly on unscaled Variant B features causes catastrophic performance degradation:
- **Unscaled Variant B Building-Floor Accuracy:** **26.01%** (distances completely dominated by AP count magnitudes)
- **Scaled Variant B Building-Floor Accuracy:** **84.07%** (using training-fitted `MinMaxScaler` on the 18 summary features to align them with normalized $[0, 1]$ RSSI and binary $\{0, 1\}$ visibility flags)

### Metric Comparison (5-Fold StratifiedGroupKFold CV across Unseen Users)

| Rank | Feature Variant | Distance Metric | Weighting | k | CV Building-Floor Acc | CV Mean Euclidean Error | CV Median Euclidean Error |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `variant_b_engineered` | `cosine` | `distance` | 7 | 83.03% (±6.14%) | 10.04 (±2.01) | 6.77 |
| 2 | `variant_b_engineered` | `cosine` | `distance` | 11 | 83.17% (±5.64%) | 10.04 (±2.00) | 6.85 |
| 3 | `variant_b_engineered` | `cosine` | `uniform` | 7 | 82.96% (±6.06%) | 10.07 (±2.01) | 6.81 |
| 4 | `variant_b_engineered` | `cosine` | `distance` | 5 | 83.19% (±6.01%) | 10.09 (±2.01) | 6.74 |
| 5 | `variant_b_engineered` | `cosine` | `uniform` | 11 | 83.06% (±5.67%) | 10.09 (±1.99) | 6.86 |

**Key Architectural Takeaway:**
1. **Cosine Distance** significantly outperforms Euclidean and Manhattan distances for Wi-Fi fingerprinting. Cosine distance evaluates the angular orientation of the signal vector, naturally mitigating hardware receiver gain discrepancies and device antenna attenuation between different phone models.
2. **Distance-Weighting** ($w_i = 1/d_i$) consistently improves regression precision over uniform weighting by allowing proximate fingerprint anchors to exert higher influence on coordinate interpolation.

---

## 3. Head-to-Head Comparison: Random Forest vs. k-Nearest Neighbors

| Evaluation Metric | Random Forest (Task 07) | k-Nearest Neighbors (Task 08) | Difference (k-NN - RF) | Winning Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Building Accuracy** | **99.82%** | 99.91% | +0.09% | k-NN |
| **Floor Accuracy** | 88.66% | **84.43%** | -4.23% | Random Forest |
| **Building-Floor Combined Accuracy** | 88.48% | **84.43%** | -4.05% | Random Forest |
| **Longitude MAE** | 7.95 units | **6.71 units** | -1.23 units | k-NN |
| **Latitude MAE** | 6.75 units | **6.10 units** | -0.66 units | k-NN |
| **Mean 2D Euclidean Error** | 11.53 units | **10.05 units** | -1.48 units | k-NN |
| **Median 2D Euclidean Error** | 8.65 units | **7.33 units** | -1.32 units | k-NN |
| **95th Percentile Error** | 32.74 units | **29.06 units** | -3.68 units | k-NN |
| **Model Suite Disk Footprint** | 561.7 MB | **17.7 MB** | **-540.0+ MB** | **k-NN (96.5% smaller)** |
| **Validation Inference Time** | **0.40 s (0.36 ms/sample)** | 1.11 s (0.996 ms/sample) | +0.71 s | **Random Forest** |

### Detailed Comparative Analysis
1. **Coordinate Positioning Precision:** Distance-weighted k-NN outperforms Random Forest across both coordinate axes and reduces overall 2D Euclidean error. Because indoor spatial geometry is continuous, interpolating coordinates between the closest spatial neighbors avoids the axis-aligned step artifacts characteristic of decision tree boundaries.
2. **Storage vs Inference Trade-off:**
   - **Random Forest** builds deep ensembles of 200 trees per head, requiring **561.7 MB** of storage on disk, but executes inference rapidly via tree traversal.
   - **k-NN** requires only **~20 MB** of compressed storage (a 96.5% reduction), making it ideal for edge deployment on mobile devices (Task 13–16), though inference latency scales with training set cardinality.

---

## 4. Hierarchical Performance Breakdown

### By Building

| Building | Validation Samples | Building Accuracy | Floor Accuracy | Combined BF Accuracy | Mean Euclidean Error | Median Euclidean Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `building_0` | 536 (48.24%) | 100.00% | 90.30% | 90.30% | 7.91 | 6.23 |
| `building_1` | 307 (27.63%) | 100.00% | 74.92% | 74.92% | 10.75 | 8.06 |
| `building_2` | 268 (24.12%) | 99.63% | 83.58% | 83.58% | 13.50 | 8.63 |

### By Floor

| Floor | Validation Samples | Floor Accuracy | Combined BF Accuracy | Mean Euclidean Error | Median Euclidean Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `floor_0` | 132 (11.88%) | 88.64% | 88.64% | 10.59 | 8.04 |
| `floor_1` | 462 (41.58%) | 82.25% | 82.25% | 8.88 | 6.41 |
| `floor_2` | 306 (27.54%) | 83.01% | 83.01% | 11.15 | 8.08 |
| `floor_3` | 172 (15.48%) | 93.02% | 93.02% | 9.38 | 7.05 |
| `floor_4` | 39 (3.51%) | 69.23% | 69.23% | 16.29 | 12.09 |

---

## 5. Main Errors and Distributional Limitations

1. **Single-User Validation Distribution:** The official validation partition contains observations collected exclusively by `USERID 0`, whereas the training set consists of measurements from users 1 through 18. Furthermore, the building distribution in validation differs markedly from training. Consequently, these metrics cannot be construed as demonstrating universal device or user generalization.
2. **Vertical Floor Slab Leakage:** While building classification is near-perfect (99.82%), floor classification experiences cross-floor leakage between adjacent vertical levels (e.g. Floor 1 vs Floor 2 in Building 1). Signals leaking through thin floor slabs create overlapping RSSI fingerprints.
3. **Inference Latency Scaling:** In production, querying an unindexed matrix of 20,000+ fingerprints on constrained mobile devices can introduce latency spikes during real-time tracking.

---

## 6. Recommendations for Next Experiments (Task 09+)

1. **Evaluation Framework & MLflow Tracking (Task 09):** Formalize metric tracking and automated model artifact archiving across Random Forest, k-NN, and upcoming gradient boosters.
2. **Hierarchical Cascaded Model Architecture:** Predict building first, then condition floor and coordinate inference on the predicted building manifold to eliminate building-crossing errors.
3. **Deep Multitask Learning (Task 10):** Train a deep neural network with shared representation layers to jointly optimize discrete building/floor cross-entropy loss and continuous coordinate Euclidean loss.
