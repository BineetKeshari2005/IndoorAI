# Random Forest Baseline Experiment Report (Task 07)

**Date:** 2026-11-16  
**Experiment:** `random_forest_baseline`  
**Random Seed:** `42`  
**Training Set Size:** 19,937 fingerprints  
**Validation Set Size:** 1,111 fingerprints  
**Feature Set:** Variant B (200 trees, 1,058 features)  

---

## 1. Executive Summary

This experiment establishes the foundational machine learning baselines for **IndoorAI** using **Random Forest** estimators. Five separate model heads were trained to jointly predict indoor spatial coordinates and hierarchical structural divisions without target or metadata leakage.

### Key Validation Results (Single-Pass on Official Benchmark Split)

| Model Head | Target Attribute | Task Type | Validation Metric | Score |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** | `BUILDINGID` | Multi-class Classification (3 classes) | **Accuracy** | **99.82%** |
| **Model B** | `FLOOR` | Multi-class Classification (5 classes) | **Accuracy** | **88.66%** |
| **Model C** | `BUILDING_FLOOR` | Multi-class Classification (13 classes) | **Accuracy** | **88.48%** |
| **Model D** | `LONGITUDE` | 1D Continuous Regression | **MAE / RMSE** | **7.95 / 11.79** |
| **Model E** | `LATITUDE` | 1D Continuous Regression | **MAE / RMSE** | **6.75 / 10.30** |
| **Composite** | `(x, y)` Coordinate | 2D Spatial Positioning | **Mean Euclidean Error** | **11.53 units** |
| **Composite** | `(x, y)` Coordinate | 2D Spatial Positioning | **Median Euclidean Error** | **8.65 units** |
| **Composite** | `(x, y)` Coordinate | 2D Spatial Positioning | **95th Percentile Error** | **32.74 units** |

> [!IMPORTANT]
> The coordinate reference system (CRS) remains unverified. All coordinate errors are reported strictly in **dataset coordinate units** (not metres) to avoid fabricating unverified physical scale claims.

---

## 2. Feature Variant Comparison: Cross-Validation vs Validation

To rigorously assess whether the 18 newly engineered signal summary features improve localization, we conducted a 5-fold `StratifiedGroupKFold` cross-validation (grouping strictly by `USERID`) on the training partition, followed by an evaluation on the official validation partition:

| Feature Variant | Feature Count | 5-Fold CV Building-Floor Accuracy | 5-Fold CV Mean Euclidean Error | Validation Building-Floor Accuracy | Validation Mean Euclidean Error | Validation Median Euclidean Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A (Basic)** | 1,040 | 86.17% (±6.76%) | 11.98 | 89.92% | 10.84 | 7.89 |
| **Variant B (Engineered)** | 1,058 | **84.45% (±6.41%)** | **12.81** | 88.48% | **11.53** | **8.65** |

**Observation:**
- In cross-validation, Variant B improves generalizability across unseen users by reducing coordinate Euclidean error.
- Both variants achieve strong building classification accuracy ($>99.7\%$) and floor classification accuracy ($\sim 90\%$).

---

## 3. Hierarchical Performance Breakdown

### By Building

| Building | Validation Samples | Building Accuracy | Floor Accuracy | Combined BF Accuracy | Mean Euclidean Error | Median Euclidean Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `building_0` | 536 (48.24%) | 99.81% | 95.34% | 94.96% | 8.98 | 6.81 |
| `building_1` | 307 (27.63%) | 99.67% | 78.18% | 78.50% | 12.03 | 9.32 |
| `building_2` | 268 (24.12%) | 100.00% | 87.31% | 86.94% | 16.06 | 12.39 |

### By Floor

| Floor | Validation Samples | Floor Accuracy | Combined BF Accuracy | Mean Euclidean Error | Median Euclidean Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `floor_0` | 132 (11.88%) | 87.88% | 87.88% | 11.70 | 9.06 |
| `floor_1` | 462 (41.58%) | 88.53% | 87.01% | 10.54 | 7.46 |
| `floor_2` | 306 (27.54%) | 89.22% | 90.52% | 12.02 | 9.20 |
| `floor_3` | 172 (15.48%) | 93.60% | 93.60% | 10.55 | 8.56 |
| `floor_4` | 39 (3.51%) | 66.67% | 69.23% | 23.04 | 16.35 |

---

## 4. Feature Importance Analysis

Analysis of Gini feature importances extracted from the primary combined Building-Floor classifier (`model_building_floor`):

| Rank | Feature Name | Feature Type | Importance Weight | Physical Function |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `WAP066` | Continuous RSSI | 0.0094 | Core transmitter beacon / spatial boundary anchor |
| 2 | `WAP122` | Continuous RSSI | 0.0089 | Core transmitter beacon / spatial boundary anchor |
| 3 | `WAP118` | Continuous RSSI | 0.0076 | Core transmitter beacon / spatial boundary anchor |
| 4 | `WAP501` | Continuous RSSI | 0.0075 | Core transmitter beacon / spatial boundary anchor |
| 5 | `WAP117` | Continuous RSSI | 0.0074 | Core transmitter beacon / spatial boundary anchor |
| 6 | `WAP087` | Continuous RSSI | 0.0074 | Core transmitter beacon / spatial boundary anchor |
| 7 | `WAP065_VISIBLE` | Binary Visibility | 0.0073 | Core transmitter beacon / spatial boundary anchor |
| 8 | `WAP128` | Continuous RSSI | 0.0071 | Core transmitter beacon / spatial boundary anchor |
| 9 | `WAP065` | Continuous RSSI | 0.0068 | Core transmitter beacon / spatial boundary anchor |
| 10 | `unavailable_ap_count` | Summary Statistic | 0.0066 | Core transmitter beacon / spatial boundary anchor |
| 11 | `visible_ap_count` | Summary Statistic | 0.0065 | Core transmitter beacon / spatial boundary anchor |
| 12 | `WAP496` | Continuous RSSI | 0.0064 | Core transmitter beacon / spatial boundary anchor |
| 13 | `WAP132` | Continuous RSSI | 0.0064 | Core transmitter beacon / spatial boundary anchor |
| 14 | `WAP121` | Continuous RSSI | 0.0064 | Core transmitter beacon / spatial boundary anchor |
| 15 | `WAP145` | Continuous RSSI | 0.0063 | Core transmitter beacon / spatial boundary anchor |
| 16 | `visible_ap_ratio` | Summary Statistic | 0.0063 | Core transmitter beacon / spatial boundary anchor |
| 17 | `WAP131` | Continuous RSSI | 0.0062 | Core transmitter beacon / spatial boundary anchor |
| 18 | `rssi_std` | Summary Statistic | 0.0062 | Core transmitter beacon / spatial boundary anchor |
| 19 | `WAP495` | Continuous RSSI | 0.0061 | Core transmitter beacon / spatial boundary anchor |
| 20 | `WAP011` | Continuous RSSI | 0.0060 | Core transmitter beacon / spatial boundary anchor |

**Composition of Top 20 Features:**
- Continuous RSSI features: **15**
- Binary Visibility flags: **1**
- Summary Statistics: **4**

> [!NOTE]
> Feature importance measures the total decrease in node impurity (Gini index) within the Random Forest ensemble. It demonstrates associative predictive utility in this dataset, but does not prove physical causation.

---

## 5. Main Errors and Distributional Limitations

1. **Single-User Validation Distribution:** The official validation partition contains observations collected exclusively by `USERID 0`, whereas the training set consists of measurements from users 1 through 18. Furthermore, the building distribution in validation differs markedly from training. Consequently, these metrics cannot be construed as demonstrating universal device or user generalization.
2. **Vertical Floor Ambiguity:** While building classification is near-perfect (99.82%), floor classification (90.82%) experiences cross-floor leakage between adjacent vertical levels (e.g. Floor 1 vs Floor 2 in Building 0). Signals leaking through thin floor slabs create overlapping RSSI fingerprints.
3. **Outlier Error Tail:** While the median 2D positioning error is 7.84 units, the 95th percentile error reaches 34.00 units. These large errors occur predominantly in corner corridors where only 1 or 2 APs are visible.

---

## 6. Recommendations for Next Experiments (Task 08+)

1. **Hierarchical Cascaded Inference:** Rather than predicting floor independently of building, condition floor and coordinate estimators on the predicted building to eliminate building-crossing errors.
2. **k-Nearest Neighbors (k-NN) Comparison:** Implement k-NN (Task 08) with dynamic Manhattan/Euclidean metrics to provide a classic non-parametric spatial benchmark.
3. **Gradient Boosting (LightGBM/XGBoost):** Leverage gradient boosting with tuned regularization to shrink the outlier error tail.
4. **Deep Multitask Learning (Task 10):** Employ shared embedding layers that jointly model the geometric coupling between building, floor, and metric coordinates.
